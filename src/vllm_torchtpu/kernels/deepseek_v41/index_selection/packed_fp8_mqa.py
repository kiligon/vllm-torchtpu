# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged MQA for transposed, packed FP8 K pages on TPU v7.

Resident Q / paged K follows DeepGEMM sm90_fp8_paged_mqa_logits.cuh (5081f851).
The schedule/operations split follows sglang-jax flash_attention_v2/pipeline.py
(e6cffb7e), specialized to weighted ReLU. See README_packed_fp8_mqa.md for the
cache contract, measured performance and numerical validation limits.
"""

import functools
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.ref import Ref

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, ROW_WORDS
from vllm_torchtpu.kernels.deepseek_v41.index_selection._packed_fp8_math import (
    head_order,
    reduce_heads,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection._packed_fp8_packing import (
    CandidateLayout,
    PagePacking,
    pack_score_page,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection._packed_fp8_pipeline import (
    BufferPlan,
    Pipeline,
    QueryState,
    QueryTile,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    PACKED_FP8_MQA_CONFIG,
)


@dataclass
class MQAOps:
    """Per-query-block operations, instantiated only while tracing Pipeline.run."""

    state: QueryState
    plan: BufferPlan
    lengths: Ref
    tile_flags: Ref
    q: Ref
    weights: Ref
    pool: tuple[Ref, ...]
    page_table: Ref
    q_scale: Ref
    out: Ref
    candidate_pages: Ref | None
    candidate_block_size: int
    queries: Ref
    prepared_queries: Ref
    keys: Ref
    scores: Ref
    compact_scores: Ref | None
    output_counts: Ref
    pages: Ref
    read_sem: Ref
    write_sem: Ref
    q_sem: Ref
    page_sem: Ref
    packing: Ref | None
    packing_bounds: Ref | None
    packing_bounds_smem: Ref | None
    packing_sem: Ref | None
    skip_invalid_pages: bool
    next_tiles: jax.Array = field(init=False)
    next_group: jax.Array = field(init=False)
    live_tiles: jax.Array = field(init=False)
    row_lengths: jax.Array = field(init=False)
    minimum_length: jax.Array = field(init=False)
    output_remaining: jax.Array = field(init=False)
    lhs: jax.Array = field(init=False)
    reduction_weights: jax.Array = field(init=False)
    reduction_selector: jax.Array = field(init=False)

    @property
    def count(self) -> jax.Array:
        return self.state.query.count

    @property
    def metadata_slot(self) -> jax.Array:
        return self.state.step & 3

    @property
    def query_slot(self) -> jax.Array:
        return self.state.step & 1

    def key_slot(self, tile: int | jax.Array) -> jax.Array:
        return (self.state.key_slot + tile) & (self.plan.config.key_buffers - 1)

    def load_page_table(self, query: QueryTile, slot, *, wait=False) -> None:
        transfer = pltpu.make_async_copy(
            self.page_table.at[query.key_group],
            self.pages.at[pl.ds(slot * self.plan.page_count, self.plan.page_count)],
            self.page_sem.at[slot],
        )
        if wait:
            transfer.wait()
        else:
            transfer.start()

    def begin_query(self, following: QueryTile, lookahead: QueryTile) -> None:
        """
        execution steps:
        1. Compute next group's K tile count (for cross-group K prefetching).
        2. Await next group's page table load (VMEM to SMEM).
        3. Trigger async page table load for the lookahead group.
        4. Prefetch next group's Q into the alternate buffer.
        5. Prepare current group parameters.
        """
        c = self.plan.config
        self.next_group = following.key_group
        next_limit = jnp.int32(0)
        for local in range(c.query_tile):
            row = jnp.minimum(following.start + local, self.q.shape[0] - 1)
            next_limit = jnp.maximum(
                next_limit, jnp.where(local < following.count, self.lengths[row], 0)
            )
        self.next_tiles = pl.cdiv(
            jnp.minimum(next_limit, self.plan.page_count * PAGE_RECORDS),
            c.key_tile,
        )

        @pl.when(following.start < self.q.shape[0])
        def finish_following_table():
            self.load_page_table(following, (self.metadata_slot + 1) & 3, wait=True)

        @pl.when(lookahead.start < self.q.shape[0])
        def start_lookahead_table():
            self.load_page_table(lookahead, (self.metadata_slot + 2) & 3)

        self.prefetch_query(following, 1 - self.query_slot)

        # prepare metadata
        limit = jnp.int32(0)
        self.minimum_length = jnp.int32(jnp.iinfo(jnp.int32).max)
        row_lengths, row_weights, row_scales = ([], [], [])
        for local in range(c.query_tile):
            row = jnp.minimum(self.state.query.start + local, self.q.shape[0] - 1)
            length = jnp.where(local < self.count, self.lengths[row], 0)
            self.minimum_length = jnp.minimum(self.minimum_length, length)
            row_lengths.append(length)
            row_weights.append(self.weights[row])
            row_scales.append(self.q_scale[row])
            limit = jnp.maximum(limit, length)
        self.live_tiles = pl.cdiv(
            jnp.minimum(limit, self.plan.page_count * PAGE_RECORDS), c.key_tile
        )
        weights = jnp.concatenate([w[:, None] for w in row_weights], axis=0)
        scales = jnp.concatenate([s[:, None] for s in row_scales], axis=0)
        self.row_lengths = jnp.stack(row_lengths)
        self.reduction_weights = (weights * scales).astype(jnp.bfloat16)
        self.reduction_weights = head_order(self.reduction_weights)
        rows = jnp.arange(192)
        owners = jnp.where(rows < 128, (rows // 2) % 4, 4 + ((rows - 128) // 2) % 2)
        self.reduction_selector = (jnp.arange(6)[:, None] == owners[None, :]).astype(
            jnp.bfloat16
        )
        if self.packing is not None:
            assert self.candidate_pages is not None
            assert self.packing_bounds is not None
            assert self.packing_bounds_smem is not None
            assert self.packing_sem is not None
            layout = PagePacking(self.candidate_block_size)
            for field in range(layout.fields):
                self.packing[field] = jnp.stack(
                    [
                        self.candidate_pages[
                            field,
                            jnp.minimum(
                                self.state.query.start + row, self.q.shape[0] - 1
                            ),
                        ]
                        for row in range(c.query_tile)
                    ]
                )
            descriptor = self.packing[0]
            starts = (descriptor >> layout.count_bits) * layout.stride
            counts = (descriptor & ((1 << layout.count_bits) - 1)) * layout.stride
            visible = (
                jnp.arange(self.plan.page_count)[None, :] * PAGE_RECORDS
                < self.row_lengths[:, None]
            )
            self.packing[0] = jnp.where(
                visible,
                descriptor,
                (descriptor >> layout.count_bits) << layout.count_bits,
            )
            live = visible & (counts > 0)
            self.packing_bounds[0] = jnp.min(
                jnp.where(
                    live,
                    starts // PAGE_RECORDS,
                    pl.cdiv(self.plan.candidate_width, PAGE_RECORDS),
                ),
                axis=0,
            )
            self.packing_bounds[1] = jnp.max(
                jnp.where(live, pl.cdiv(starts + counts, PAGE_RECORDS), 0), axis=0
            )
            pltpu.make_async_copy(
                self.packing_bounds, self.packing_bounds_smem, self.packing_sem
            ).start()

    def wait_query(self) -> None:
        self.copy_queries(self.state.query, self.query_slot, wait=True)
        if self.packing_bounds is not None:
            pltpu.make_async_copy(
                self.packing_bounds, self.packing_bounds_smem, self.packing_sem
            ).wait()
        c = self.plan.config
        lhs = self.queries[pl.ds(self.query_slot * c.query_tile, c.query_tile)].reshape(
            c.query_tile * self.plan.heads, DIM
        )
        self.lhs = head_order(lhs)

        # Materialize the head permutation once per query block. The destination
        # buffer has only DMA writes, making read-only reload annotation sound.
        query_slice = pl.ds(self.query_slot * c.query_tile, c.query_tile)
        self.queries[query_slice] = self.lhs.reshape(c.query_tile, self.plan.heads, DIM)
        copy = pltpu.make_async_copy(
            self.queries.at[query_slice],
            self.prepared_queries.at[query_slice],
            self.q_sem.at[self.query_slot],
        )
        copy.start()
        copy.wait()
        self.lhs = pltpu.annotate(self.prepared_queries, no_store=True)[
            query_slice
        ].reshape(c.query_tile * self.plan.heads, DIM)
        if self.compact_scores is not None:
            self.compact_scores[self.query_slot] = jnp.full(
                self.compact_scores.shape[1:], -jnp.inf, jnp.float32
            )

    def prefetch_query(self, query: QueryTile, slot) -> None:
        """Load the next Q tile, zeroing unused rows for ragged groups."""
        tile = self.plan.config.query_tile
        self.queries[pl.ds(slot * tile, tile)] = jnp.zeros(
            (tile, self.plan.heads, DIM), self.q.dtype
        )
        self.copy_queries(query, slot)

    def copy_queries(self, query: QueryTile, slot, wait: bool = False) -> None:
        for local in range(self.plan.config.query_tile):

            @pl.when(local < query.count)
            def copy(local=local):
                transfer = pltpu.make_async_copy(
                    self.q.at[query.start + local],
                    self.queries.at[slot * self.plan.config.query_tile + local],
                    self.q_sem.at[slot],
                )
                if wait:
                    transfer.wait()
                else:
                    transfer.start()

    def load_page(self, tile, i, metadata_slot, slot):
        record = self.pages[
            metadata_slot * self.plan.page_count
            + tile * (self.plan.config.key_tile // PAGE_RECORDS)
            + i
        ]
        address = record >> 2
        rows = PAGE_ROWS

        def load(bank):
            pltpu.make_async_copy(
                self.pool[bank].at[pl.ds(address, rows)],
                self.keys.at[slot, pl.ds(i * rows, rows)],
                self.read_sem.at[slot],
            ).start()

        if self.skip_invalid_pages:
            # Padded tasks have no DMA and contribute no semaphore completion.
            if len(self.pool) == 1:
                pl.when((record & 2) != 0)(lambda: load(0))
            else:
                bank_and_valid = record & 3
                pl.when(bank_and_valid == 2)(lambda: load(0))
                pl.when(bank_and_valid == 3)(lambda: load(1))
        elif len(self.pool) == 1:
            load(0)
        else:
            jax.lax.cond((record & 1) == 0, lambda: load(0), lambda: load(1))

    def fetch(self, tile, metadata_slot, slot, *, unroll=False):
        def body(i, _):
            self.load_page(tile, i, metadata_slot, slot)

        jax.lax.fori_loop(
            0, self.plan.config.key_tile // PAGE_RECORDS, body, None, unroll=unroll
        )

    def wait_keys(self, tile):
        slot = self.key_slot(tile)
        keys = self.keys.at[slot]
        if self.skip_invalid_pages:
            count = self.tile_flags[self.state.query.key_group, tile]
            keys = keys.at[pl.ds(0, count * PAGE_ROWS)]
        pltpu.make_async_copy(keys, keys, self.read_sem.at[slot]).wait()

    def full_page_tile(self, tile):
        flag = self.tile_flags[self.state.query.key_group, tile]
        if self.skip_invalid_pages:
            return flag == self.plan.config.key_tile // PAGE_RECORDS
        return flag != 0

    def output_slot(self, tile):
        if self.plan.candidate_width:
            return 0
        return 2 * (self.state.step & 1) + (tile & 1)

    def clear_output(self, tile) -> None:
        self.scores[self.output_slot(tile)] = jnp.full(
            self.scores.shape[1:], -jnp.inf, jnp.float32
        )

    def release_output_pair(self, pair):
        # Only the recorded rows own outstanding completions. A tile with
        # zero live queries must not wait on an unissued output transfer.
        count = self.output_counts[pair]
        if self.compact_scores is not None:
            for local in range(self.plan.config.query_tile):

                @pl.when(local < count)
                def wait_compact(local=local):
                    src = self.compact_scores.at[pair, pl.ds(local, 1)]
                    pltpu.make_async_copy(
                        src, src, self.write_sem.at[pair, local]
                    ).wait()

            self.output_counts[pair] = jnp.int32(0)
            return
        tiles = self.plan.page_count * PAGE_RECORDS // self.plan.config.key_tile
        for local_slot in range(min(2, tiles)):
            slot = pair * 2 + local_slot
            for local in range(self.plan.config.query_tile):

                @pl.when(local < count)
                def wait(slot=slot, local=local):
                    src = self.scores.at[slot, pl.ds(local, 1)]
                    pltpu.make_async_copy(
                        src, src, self.write_sem.at[slot, local]
                    ).wait()

        self.output_counts[pair] = jnp.int32(0)

    def publish(self, tile, wait: bool = False, *, slot=None) -> None:
        if self.compact_scores is not None:
            if not wait:
                self.pack_output(tile)
            return
        slot = self.output_slot(tile) if slot is None else slot
        for local in range(self.plan.config.query_tile):

            @pl.when(local < self.count)
            def copy(local=local):
                # Preserve the row axis for DMA into the final tiled 2D layout.
                transfer = pltpu.make_async_copy(
                    self.scores.at[slot, pl.ds(local, 1)],
                    self.out.at[
                        pl.ds(self.state.query.start + local, 1),
                        pl.ds(
                            tile * self.plan.config.key_tile, self.plan.config.key_tile
                        ),
                    ],
                    self.write_sem.at[slot, local],
                )
                if wait:
                    transfer.wait()
                else:
                    transfer.start()

    def pack_output(self, tile) -> None:
        assert self.packing is not None
        assert self.packing_bounds_smem is not None
        assert self.compact_scores is not None
        pages = self.plan.config.key_tile // PAGE_RECORDS

        def page(local, _):
            index = tile * pages + local
            first = self.packing_bounds_smem[0, index]
            end = self.packing_bounds_smem[1, index]

            @pl.when(first < end)
            def pack():
                indices = jnp.full(
                    (self.plan.config.query_tile, 1), index % PAGE_RECORDS, jnp.int32
                )
                fields = [
                    jnp.take_along_axis(
                        self.packing[
                            field,
                            :,
                            pl.ds(index // PAGE_RECORDS * PAGE_RECORDS, PAGE_RECORDS),
                        ],
                        indices,
                        axis=1,
                        mode="promise_in_bounds",
                    )
                    for field in range(self.packing.shape[0])
                ]
                values, starts, ends = pack_score_page(
                    self.scores[0, :, pl.ds(local * PAGE_RECORDS, PAGE_RECORDS)],
                    jnp.concatenate(fields, axis=1),
                    block_size=self.candidate_block_size,
                )

                def write(destination, _):
                    columns = (
                        destination * PAGE_RECORDS
                        + jnp.arange(PAGE_RECORDS, dtype=jnp.int32)[None, :]
                    )
                    pltpu.store(
                        self.compact_scores.at[
                            self.query_slot,
                            :,
                            pl.ds(destination * PAGE_RECORDS, PAGE_RECORDS),
                        ],
                        values,
                        mask=(columns >= starts) & (columns < ends),
                    )

                jax.lax.fori_loop(first, end, write, None)

        jax.lax.fori_loop(0, pages, page, None)

    def finish_output(self) -> None:
        if self.compact_scores is not None:
            for local in range(self.plan.config.query_tile):

                @pl.when(local < self.count)
                def copy(local=local):
                    pltpu.make_async_copy(
                        self.compact_scores.at[self.query_slot, pl.ds(local, 1)],
                        self.out.at[pl.ds(self.state.query.start + local, 1)],
                        self.write_sem.at[self.query_slot, local],
                    ).start()

            self.output_counts[self.query_slot] = self.count
            return

        tiles = self.plan.page_count * PAGE_RECORDS // self.plan.config.key_tile

        @pl.when(self.live_tiles == tiles)
        def retain_outputs():
            self.output_counts[self.query_slot] = self.count

        @pl.when(self.live_tiles < tiles)
        def pad_after_draining():
            @pl.when(self.live_tiles > 0)
            def finish():
                self.publish(self.live_tiles - 1, wait=True)

                @pl.when(self.live_tiles > 1)
                def previous():
                    self.publish(self.live_tiles - 2, wait=True)

            self.fill_padding()

    def qk(self, tile, fragment: int) -> jax.Array:
        query = self.lhs
        data_rows = PAGE_ROWS - 1
        key_parts = []
        for page in range(
            fragment * self.plan.mxu_size // PAGE_RECORDS,
            (fragment + 1) * self.plan.mxu_size // PAGE_RECORDS,
        ):
            words = self.keys[self.key_slot(tile), pl.ds(page * PAGE_ROWS, data_rows)]
            key_parts.append(pltpu.bitcast(words, self.q.dtype))
        keys = jnp.concatenate(key_parts, axis=1)
        logits = jax.lax.dot_general(
            query,
            keys,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        logits = jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))
        return logits

    def reduce(self, fragment: int, logits: jax.Array) -> jax.Array:
        products = logits * self.reduction_weights
        return reduce_heads(
            products, self.reduction_selector, fragment, self.plan.fragments
        )

    def prepare_output_mask(self, tile) -> None:
        # Clamp negative lengths before subtracting to avoid signed underflow.
        # All work stays inside the masked tile's conditional compute segment.
        base = tile * self.plan.config.key_tile + jnp.arange(PAGE_RECORDS)
        self.output_remaining = (
            jnp.maximum(self.row_lengths, 0)[:, None] - base[None, :]
        )

    def store(
        self, tile, fragment: int, values: jax.Array, *, full_tile: bool = False
    ) -> None:
        size = self.plan.mxu_size
        data_rows = PAGE_ROWS - 1
        for offset in range(0, size, PAGE_RECORDS):
            local_page = (fragment * size + offset) // PAGE_RECORDS
            valid_page = None
            if not full_tile:
                page = self.pages[
                    self.metadata_slot * self.plan.page_count
                    + tile * (self.plan.config.key_tile // PAGE_RECORDS)
                    + local_page
                ]
                valid_page = (page & 2) != 0
            scale = jax.lax.bitcast_convert_type(
                self.keys[
                    self.key_slot(tile),
                    pl.ds(local_page * (data_rows + 1) + data_rows, 1),
                ],
                jnp.float32,
            )
            piece = values[:, offset : offset + PAGE_RECORDS].astype(jnp.float32)
            scaled = piece * scale
            if not full_tile:
                # Invalid pages compare against INT_MAX, so every output lane
                # is invalid without a vector predicate AND.
                assert valid_page is not None
                page_offset = jnp.where(
                    valid_page,
                    jnp.int32(fragment * size + offset),
                    jnp.int32(jnp.iinfo(jnp.int32).max),
                )
                valid = self.output_remaining > page_offset
                # QK and head sums never mix token columns. Skipped K pages
                # may contain arbitrary bits, including NaNs; discard the whole
                # column here before top-k or any other cross-token operation.
                scaled = jnp.where(valid, scaled, -jnp.inf)
            self.scores[
                self.output_slot(tile),
                :,
                pl.ds(fragment * size + offset, PAGE_RECORDS),
            ] = scaled

    def fill_padding(self) -> None:
        tiles = self.plan.page_count * PAGE_RECORDS // self.plan.config.key_tile

        @pl.when(self.live_tiles < tiles)
        def initialize_padding():
            self.scores[self.output_slot(0)] = jnp.full(
                self.scores.shape[1:], -jnp.inf, jnp.float32
            )

        for wait in (False, True):

            def body(tile, _, wait=wait):
                self.publish(tile, wait=wait, slot=self.output_slot(0))

            jax.lax.fori_loop(
                self.live_tiles,
                self.plan.page_count * PAGE_RECORDS // self.plan.config.key_tile,
                body,
                None,
            )


def _next_query_tile(query: QueryTile, offsets: Ref, tile_size: int) -> QueryTile:
    """Advance flat Q using only shared-K group boundaries."""
    start = query.start + query.count
    group = jax.lax.while_loop(
        lambda g: (g < offsets.shape[0] - 2) & (start >= offsets[g + 1]),
        lambda g: g + 1,
        query.key_group,
    )
    return QueryTile(start, jnp.minimum(tile_size, offsets[group + 1] - start), group)


def _kernel(
    offsets,
    lengths,
    tile_flags,
    q,
    weights,
    pool,
    page_table,
    q_scale,
    candidate_pages,
    out,
    queries,
    prepared_queries,
    keys,
    scores,
    compact_scores,
    output_counts,
    pages,
    read_sem,
    write_sem,
    q_sem,
    page_sem,
    packing,
    packing_bounds,
    packing_bounds_smem,
    packing_sem,
    *,
    pipeline,
    skip_invalid_pages,
    candidate_block_size,
):
    def make_ops(state):
        return MQAOps(
            state,
            pipeline.plan,
            lengths,
            tile_flags,
            q,
            pltpu.annotate(weights, no_store=True),
            pool,
            pltpu.annotate(page_table, no_store=True),
            pltpu.annotate(q_scale, no_store=True),
            out,
            (
                pltpu.annotate(candidate_pages, no_store=True)
                if candidate_pages is not None
                else None
            ),
            candidate_block_size,
            queries,
            prepared_queries,
            pltpu.annotate(keys, no_store=True),
            scores,
            compact_scores,
            output_counts,
            pages,
            read_sem,
            write_sem,
            q_sem,
            page_sem,
            packing,
            packing_bounds,
            packing_bounds_smem,
            packing_sem,
            skip_invalid_pages,
        )

    next_query = functools.partial(
        _next_query_tile, offsets=offsets, tile_size=pipeline.plan.config.query_tile
    )
    first = next_query(QueryTile(jnp.int32(0), jnp.int32(0), jnp.int32(0)))
    pipeline.run(make_ops, next_query, first, q.shape[0])


@functools.partial(
    jax.jit,
    static_argnames=("cache_layout", "skip_invalid_pages", "candidate_block_size"),
)
def packed_fp8_mqa_logits(
    q: jax.Array,
    weights: jax.Array,
    pool: jax.Array | tuple[jax.Array, jax.Array],
    block_table: jax.Array,
    lengths: jax.Array,
    query_start_loc: jax.Array,
    *,
    q_scale: jax.Array,
    cache_layout: PackedFP8Layout | None = None,
    skip_invalid_pages: bool = False,
    candidate_layout: CandidateLayout | None = None,
    candidate_block_size: int = 1,
) -> jax.Array:
    """Return FP32 logits [T, P*128] from transposed packed FP8 pages.

    Args:
        q: FP8 E4M3FN [T,32,128]; at least six rows unless empty.
        weights: FP32 [T,32]. Valid-token weighted BF16 products must be finite.
        pool: One or two int32 [rows,128] banks with packed_fp8_cache page bytes.
            Ordinary [token,dim] packed bytes are NOT compatible.
        block_table: Int32 [G,P] physical page IDs; invalid IDs produce -inf.
            Capacity P*128 must be divisible by 32768.
        lengths: Int32 [T] visible lengths. Unused tokens produce -inf.
        query_start_loc: Monotonic int32 [G+1] offsets partitioning all Q rows;
            ragged and empty groups are supported.
        q_scale: FP32 [T,32] per-head Q scales. K scales are inside each page.
        cache_layout: This layer's page placement in shared owners. None means
            tightly packed pages, including a page-aligned bank boundary.
        skip_invalid_pages: Sparse task tables skip invalid page DMAs and wait
            for only the issued bytes. Empty tiles skip arithmetic. Padded
            fragments inside nonempty tiles are computed and masked at output.
        candidate_layout: Canonical [T,C] block IDs and page-local permutations.
            When supplied, only [T,C*candidate_block_size] scores leave VMEM.
        candidate_block_size: Tokens per candidate block.

    QK accumulates in FP32 then rounds to BF16. ReLU, folded Q weights and
    weighted products use BF16. Head sums accumulate in FP32, round to BF16,
    then widen for FP32 K scaling and output. Head accumulation is reassociated;
    see the benchmark documentation for independent reference differences.
    """
    tokens, heads, dim = q.shape
    banks = pool if isinstance(pool, tuple) else (pool,)
    key_groups, page_count = block_table.shape
    width = page_count * PAGE_RECORDS
    output_width = width
    if candidate_layout is not None:
        candidate_blocks = candidate_layout.blocks
        if (
            candidate_blocks.ndim != 2
            or candidate_blocks.shape[0] != tokens
            or candidate_blocks.dtype != jnp.int32
            or candidate_block_size < 1
            or candidate_layout.pages.shape
            != (PagePacking(candidate_block_size).fields, tokens, page_count)
            or candidate_layout.pages.dtype != jnp.int32
        ):
            raise ValueError("Expected int32 candidate blocks [T,C] and positive size")
        output_width = candidate_blocks.shape[1] * candidate_block_size
    config = PACKED_FP8_MQA_CONFIG
    if (
        (heads, dim) != (32, DIM)
        or q.dtype != jnp.float8_e4m3fn
        or weights.shape != (tokens, heads)
        or weights.dtype != jnp.float32
        or q_scale.shape != (tokens, heads)
        or q_scale.dtype != jnp.float32
        or lengths.shape != (tokens,)
        or lengths.dtype != jnp.int32
        or query_start_loc.shape != (key_groups + 1,)
        or query_start_loc.dtype != jnp.int32
        or block_table.dtype != jnp.int32
        or key_groups < 1
        or any(
            bank.dtype != jnp.int32
            or bank.ndim != 2
            or bank.shape[1] != ROW_WORDS
            or (cache_layout is None and bank.shape[0] % PAGE_ROWS)
            for bank in banks
        )
    ):
        raise ValueError(
            "Expected FP8 Q[T,32,128], FP32 weights/scales and packed K pages"
        )
    if tokens == 0 or width == 0 or output_width == 0:
        return jnp.full((tokens, output_width), -jnp.inf, jnp.float32)
    if tokens < config.query_tile or width % config.key_tile:
        raise ValueError(
            "Expected at least six Q rows and context capacity divisible by 32768"
        )
    info = pltpu.get_tpu_info()
    if info is None or (
        info.generation != 7 or info.num_mxus != 2 or info.mxu_column_size != 256
    ):
        raise NotImplementedError(
            "Packed FP8 MQA currently targets TPU v7 with two MXUs"
        )
    key_tile = config.key_tile
    plan = BufferPlan(
        config,
        heads,
        page_count,
        info.mxu_column_size,
        q.dtype,
        pl.cdiv(output_width, PAGE_RECORDS) * PAGE_RECORDS
        if candidate_layout is not None
        else 0,
        candidate_block_size,
    )
    plan.validate(info, tokens, key_groups)
    # Prepare once for all Q blocks. The data representation and bytes copied
    # are unchanged; validity and the selected bank are retained in two bits.
    cache_layout = cache_layout or PackedFP8Layout(
        sum(bank.shape[0] for bank in banks) // PAGE_ROWS
    )
    cache_layout.validate(tuple(bank.shape[0] for bank in banks))
    total_pages = cache_layout.pages
    valid_pages = (block_table >= 0) & (block_table < total_pages)
    tiled_valid = valid_pages.reshape(
        key_groups, width // key_tile, key_tile // PAGE_RECORDS
    )
    tile_flags = (
        tiled_valid.sum(axis=-1, dtype=jnp.int32)
        if skip_invalid_pages
        else jnp.all(tiled_valid, axis=-1).astype(jnp.int32)
    )
    safe_pages = jnp.clip(block_table, 0, total_pages - 1)
    global_rows = cache_layout.page_offset + safe_pages * cache_layout.page_stride
    selected_bank = (global_rows >= banks[0].shape[0]).astype(jnp.int32)
    local_rows = global_rows - selected_bank * banks[0].shape[0]
    block_table = (
        (local_rows << 2) | (valid_pages.astype(jnp.int32) << 1) | selected_bank
    )
    pipeline = Pipeline(plan)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    return pl.pallas_call(
        functools.partial(
            _kernel,
            pipeline=pipeline,
            skip_invalid_pages=skip_invalid_pages,
            candidate_block_size=candidate_block_size,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (tokens, plan.candidate_width or output_width), jnp.float32
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=(
                hbm,
                pl.BlockSpec(memory_space=pltpu.VMEM),
                tuple(hbm for _ in banks),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM)
                if candidate_layout is not None
                else None,
            ),
            out_specs=hbm,
            scratch_shapes=plan.scratch_shapes(),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=info.vmem_capacity_bytes,
            disable_bounds_checks=True,
        ),
        name=(
            "paged_mqa_compact_packed_fp8"
            if candidate_layout is not None
            else "paged_mqa_full_packed_fp8"
        ),
    )(
        query_start_loc,
        lengths,
        tile_flags,
        q,
        weights,
        banks,
        block_table,
        q_scale,
        candidate_layout.pages if candidate_layout is not None else None,
    )[:, :output_width]
