# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense paged MQA with per-query page tables and request IDs.

Copied from packed_fp8_mqa.py at 842880870 for independent dense validation.
The row/request contract follows DeepGEMM's SM100 paged MQA scheduler:
https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh
Request grouping happens inside the kernel with a model-configured static tile.
TPU FP8 page storage is unchanged; this module has no candidate filtering.
"""

import functools
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.ref import Ref

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, ROW_WORDS
from vllm_torchtpu.kernels.deepseek_v41.index_selection._packed_fp8_dense_math import (
    head_order,
    reduce_heads,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    PACKED_FP8_MAX_QUERY_TILE,
    PACKED_FP8_MQA_CONFIG,
)

from ._packed_fp8_dense_pipeline import BufferPlan, Pipeline, QueryState, QueryTile


def _encode_page_table(
    page_ids, layout: PackedFP8Layout, first_bank_rows: int, pages_per_tile: int
):
    """Encode one Q tile's shared page row; invalid IDs retain safe addresses."""
    valid = (page_ids >= 0) & (page_ids < layout.pages)
    safe = jnp.clip(page_ids, 0, layout.pages - 1)
    global_rows = layout.page_offset + safe * layout.page_stride
    bank = (global_rows >= first_bank_rows).astype(jnp.int32)
    local_rows = global_rows - bank * first_bank_rows
    full_tiles = []
    for start in range(0, page_ids.size, pages_per_tile):
        tile_valid = valid[start : start + pages_per_tile][None, :]
        full = jnp.all(tile_valid, axis=1, keepdims=True).astype(jnp.int32)
        full_tiles.append(jnp.broadcast_to(full, tile_valid.shape).reshape(-1))
    full = jnp.concatenate(full_tiles)
    # Keep the tile predicate in SMEM with its pages. Reading it must not
    # synchronize the scalar controller with a VPU reduction in the K loop.
    return (local_rows << 3) | (full << 2) | (valid.astype(jnp.int32) << 1) | bank


@dataclass
class MQAOps:
    """Per-query-block operations, instantiated only while tracing Pipeline.run."""

    state: QueryState
    plan: BufferPlan
    cache_layout: PackedFP8Layout
    lengths: Ref
    q: Ref
    weights: Ref
    pool: tuple[Ref, ...]
    page_table: Ref
    q_scale: Ref
    out: Ref
    queries: Ref
    prepared_queries: Ref
    keys: Ref
    scores: Ref
    output_counts: Ref
    pages: Ref
    page_staging: Ref
    read_sem: Ref
    write_sem: Ref
    q_sem: Ref
    page_sem: Ref
    next_tiles: jax.Array = field(init=False)
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

    def page_table_copy(self, query: QueryTile, slot):
        return pltpu.make_async_copy(
            self.page_table.at[query.page_row, pl.ds(0, self.plan.page_count)],
            self.page_staging.at[slot, 0],
            self.page_sem.at[slot],
        )

    def load_page_table(self, query: QueryTile, slot) -> None:
        self.page_table_copy(query, slot).start()

    def encoded_page_copy(self, slot):
        return pltpu.make_async_copy(
            self.page_staging.at[slot, 0],
            self.pages.at[pl.ds(slot * self.plan.page_count, self.plan.page_count)],
            self.page_sem.at[slot],
        )

    def prepare_page_table(self, query: QueryTile, slot) -> None:
        self.page_table_copy(query, slot).wait()
        self.page_staging[slot, 0] = _encode_page_table(
            self.page_staging[slot, 0],
            self.cache_layout,
            self.pool[0].shape[0],
            self.plan.config.key_tile // PAGE_RECORDS,
        )
        self.encoded_page_copy(slot).start()

    def wait_page_table(self, slot) -> None:
        self.encoded_page_copy(slot).wait()

    def begin_query(self, following: QueryTile, lookahead: QueryTile) -> None:
        c = self.plan.config
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
        def prepare_following_table():
            self.prepare_page_table(following, (self.metadata_slot + 1) & 3)

        @pl.when(lookahead.start < self.q.shape[0])
        def start_lookahead_table():
            self.load_page_table(lookahead, (self.metadata_slot + 2) & 3)

        self.prefetch_query(following, 1 - self.query_slot)
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
        self.reduction_weights = head_order(
            self.reduction_weights, c.query_tile, self.plan.sublanes
        )
        owners = jnp.repeat(jnp.arange(c.query_tile), self.plan.heads)[:, None]
        owners = head_order(owners, c.query_tile, self.plan.sublanes)[:, 0]
        self.reduction_selector = (
            jnp.arange(c.query_tile)[:, None] == owners[None, :]
        ).astype(jnp.bfloat16)

        @pl.when(following.start < self.q.shape[0])
        def finish_following_table():
            self.wait_page_table((self.metadata_slot + 1) & 3)

    def wait_query(self) -> None:
        self.copy_queries(self.state.query, self.query_slot, wait=True)
        c = self.plan.config
        lhs = self.queries[pl.ds(self.query_slot * c.query_tile, c.query_tile)].reshape(
            c.query_tile * self.plan.heads, DIM
        )
        self.lhs = head_order(lhs, c.query_tile, self.plan.sublanes)

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
        address = jax.lax.shift_right_logical(record, jnp.int32(3))
        rows = PAGE_ROWS

        def load(bank):
            pltpu.make_async_copy(
                self.pool[bank].at[pl.ds(address, rows)],
                self.keys.at[slot, pl.ds(i * rows, rows)],
                self.read_sem.at[slot],
            ).start()

        if len(self.pool) == 1:
            load(0)
        else:
            jax.lax.cond((record & 1) == 0, lambda: load(0), lambda: load(1))

    def fetch(self, tile, metadata_slot, slot):
        def body(i, _):
            self.load_page(tile, i, metadata_slot, slot)

        jax.lax.fori_loop(0, self.plan.config.key_tile // PAGE_RECORDS, body, None)

    def wait_keys(self, tile):
        slot = self.key_slot(tile)
        keys = self.keys.at[slot]
        pltpu.make_async_copy(keys, keys, self.read_sem.at[slot]).wait()

    def full_page_tile(self, tile):
        pages = self.plan.config.key_tile // PAGE_RECORDS
        encoded = self.pages[self.metadata_slot * self.plan.page_count + tile * pages]
        return (encoded & 4) != 0

    def output_slot(self, tile):
        return 2 * (self.state.step & 1) + (tile & 1)

    def release_output_pair(self, pair):
        # Only the recorded rows own outstanding completions. A tile with
        # zero live queries must not wait on an unissued output transfer.
        count = self.output_counts[pair]
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
            products,
            self.reduction_selector,
            sublanes=self.plan.sublanes,
            use_mxu=bool(fragment % 2)
            and fragment < self.plan.fragments - self.plan.config.issue_depth,
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


def _next_query_tile(
    query: QueryTile, indices: Ref | jax.Array, tile_size: int
) -> QueryTile:
    """Group consecutive queries with the same request ID, up to one Q tile."""
    start = query.start + query.count
    rows = indices.shape[0]
    page_row = jnp.minimum(start, rows - 1)
    request = indices[page_row]
    end = jax.lax.while_loop(
        lambda row: (row < jnp.minimum(start + tile_size, rows))
        & (indices[jnp.minimum(row, rows - 1)] == request),
        lambda row: row + 1,
        jnp.minimum(start + 1, rows),
    )
    return QueryTile(start, end - start, page_row)


def _kernel(
    indices,
    lengths,
    q,
    weights,
    pool,
    page_table,
    q_scale,
    out,
    queries,
    prepared_queries,
    keys,
    scores,
    output_counts,
    pages,
    page_staging,
    read_sem,
    write_sem,
    q_sem,
    page_sem,
    *,
    pipeline,
    cache_layout,
):
    def make_ops(state):
        return MQAOps(
            state,
            pipeline.plan,
            cache_layout,
            lengths,
            q,
            pltpu.annotate(weights, no_store=True),
            pool,
            page_table,
            pltpu.annotate(q_scale, no_store=True),
            out,
            queries,
            prepared_queries,
            pltpu.annotate(keys, no_store=True),
            scores,
            output_counts,
            pages,
            page_staging,
            read_sem,
            write_sem,
            q_sem,
            page_sem,
        )

    next_query = functools.partial(
        _next_query_tile, indices=indices, tile_size=pipeline.plan.config.query_tile
    )
    first = next_query(QueryTile(jnp.int32(0), jnp.int32(0), jnp.int32(0)))
    pipeline.run(make_ops, next_query, first, q.shape[0])


@functools.partial(
    jax.jit,
    static_argnames=("cache_layout", "max_model_len", "query_size"),
)
def packed_fp8_paged_mqa_logits(
    q: jax.Array,
    kv_cache: jax.Array | tuple[jax.Array, jax.Array],
    weights: jax.Array,
    context_lens: jax.Array,
    block_table: jax.Array,
    *,
    indices: jax.Array,
    max_model_len: int,
    q_scale: jax.Array,
    query_size: int,
    cache_layout: PackedFP8Layout | None = None,
) -> jax.Array:
    """Return dense FP32 logits [T, max_model_len], in logical token order.

    Args:
        q: FP8 E4M3FN [T,32,128], one query token per row.
        kv_cache: One or two int32 [rows,128] banks in packed_fp8_cache format.
        weights: FP32 [T,32]. Valid-token weighted BF16 products must be finite.
        context_lens: Int32 [T] effective history lengths in cache tokens.
        block_table: Int32 [T,P] physical page IDs. Rows of queries from the
            same request must be identical. Caller pads capacity to cover
            max_model_len rounded up to the 32768-token compute tile.
            Invalid page IDs and invisible token positions produce -inf.
        indices: Int32 [T] request IDs; queries of a request are consecutive.
            IDs label runs; they do not index block_table or the cache.
        max_model_len: Logical output width, independent of cache pool size.
        q_scale: FP32 [T,32] per-head Q scales. K scales are inside each page.
        query_size: Static model-configured queries per decode request, normally
            num_speculative_tokens + 1, or 1 without speculation. The kernel
            uses min(query_size, 8); longer request runs are split into tiles.
        cache_layout: This layer's page placement in shared owners. None means
            tightly packed pages, including a page-aligned bank boundary.

    This matches vLLM's SM100 row/request semantics, not its MXFP4 cache bytes
    or BF16 output dtype. Arithmetic retains the 842880870 TPU FP8 path.
    """
    if q.ndim != 3 or q.shape[1:] != (32, DIM):
        raise ValueError("Expected FP8 Q[T,32,128] with one query per row")
    tokens, heads, _ = q.shape
    banks = kv_cache if isinstance(kv_cache, tuple) else (kv_cache,)
    if type(query_size) is not int or query_size < 1:
        raise ValueError("query_size must be a positive static model configuration")
    query_tile = min(query_size, PACKED_FP8_MAX_QUERY_TILE)
    config = replace(
        PACKED_FP8_MQA_CONFIG, query_tile=query_tile, query_group=query_tile
    )
    if (
        q.dtype != jnp.float8_e4m3fn
        or weights.shape != (tokens, heads)
        or weights.dtype != jnp.float32
        or q_scale.shape != (tokens, heads)
        or q_scale.dtype != jnp.float32
        or context_lens.shape != (tokens,)
        or context_lens.dtype != jnp.int32
        or indices.shape != (tokens,)
        or indices.dtype != jnp.int32
        or block_table.ndim != 2
        or block_table.shape[0] != tokens
        or block_table.dtype != jnp.int32
        or max_model_len < 0
        or any(
            bank.dtype != jnp.int32
            or bank.ndim != 2
            or bank.shape[1] != ROW_WORDS
            or (cache_layout is None and bank.shape[0] % PAGE_ROWS)
            for bank in banks
        )
    ):
        raise ValueError(
            "Expected per-query int32 tables/lengths/indices, FP32 weights/scales "
            "and packed FP8 K banks; max_model_len must be nonnegative"
        )
    width = pl.cdiv(max_model_len, config.key_tile) * config.key_tile
    page_count = width // PAGE_RECORDS
    if block_table.shape[1] < page_count:
        raise ValueError(
            "Pad each query's block_table to cover max_model_len rounded up "
            "to 32768 tokens"
        )
    if tokens == 0 or width == 0:
        return jnp.empty((tokens, max_model_len), jnp.float32)
    info = pltpu.get_tpu_info()
    if info is None or (
        info.generation != 7 or info.num_mxus != 2 or info.mxu_column_size != 256
    ):
        raise NotImplementedError(
            "Packed FP8 MQA currently targets TPU v7 with two MXUs"
        )
    plan = BufferPlan(
        config, heads, page_count, info.mxu_column_size, info.num_sublanes, q.dtype
    )
    plan.validate(info, tokens)
    cache_layout = cache_layout or PackedFP8Layout(
        sum(bank.shape[0] for bank in banks) // PAGE_ROWS
    )
    cache_layout.validate(tuple(bank.shape[0] for bank in banks))
    pipeline = Pipeline(plan)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    return pl.pallas_call(
        functools.partial(_kernel, pipeline=pipeline, cache_layout=cache_layout),
        out_shape=jax.ShapeDtypeStruct((tokens, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=(
                hbm,
                pl.BlockSpec(memory_space=pltpu.VMEM),
                tuple(hbm for _ in banks),
                hbm,
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ),
            out_specs=hbm,
            scratch_shapes=plan.scratch_shapes(),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=info.vmem_capacity_bytes,
            disable_bounds_checks=True,
        ),
        name="paged_mqa_dense_packed_fp8",
    )(
        indices,
        context_lens,
        q,
        weights,
        banks,
        block_table,
        q_scale,
    )[:, :max_model_len]
