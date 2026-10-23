# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resource ownership and scheduling for paged MQA.

Separates the static schedule from numerical operations, following sglang-jax
PR #1800, flash_attention_v2/{config,pipeline}.py (e6cffb7e). The separate K,
accumulator and output rings adapt DeepGEMM's sparse MQA pipeline to TPU DMA
and MRB. MQA needs no softmax/PV stages.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, ROW_WORDS
from vllm_torchtpu.kernels.deepseek_v41.index_selection._packed_fp8_packing import (
    PagePacking,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    PagedMQAConfig,
)

if TYPE_CHECKING:
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_mqa import (
        MQAOps,
    )


class QueryTile(NamedTuple):
    start: jax.Array
    count: jax.Array
    key_group: jax.Array


class QueryState(NamedTuple):
    query: QueryTile
    step: jax.Array
    key_slot: jax.Array


@dataclass(frozen=True)
class BufferPlan:
    """Static allocation plan; compiler temporaries/spills are additional."""

    config: PagedMQAConfig
    heads: int
    page_count: int
    mxu_size: int
    q_dtype: jnp.dtype
    candidate_width: int = 0
    candidate_block_size: int = 1

    @property
    def fragments(self) -> int:
        return self.config.key_tile // self.mxu_size

    def scratch_shapes(self) -> tuple:
        c = self.config
        return (
            pltpu.VMEM((2 * c.query_tile, self.heads, DIM), self.q_dtype),
            pltpu.VMEM((2 * c.query_tile, self.heads, DIM), self.q_dtype),
            pltpu.VMEM(
                (c.key_buffers, c.key_tile // PAGE_RECORDS * PAGE_ROWS, ROW_WORDS),
                jnp.int32,
            ),
            pltpu.VMEM(
                (1 if self.candidate_width else 4, c.query_tile, c.key_tile),
                jnp.float32,
            ),
            (
                pltpu.VMEM((2, c.query_tile, self.candidate_width), jnp.float32)
                if self.candidate_width
                else None
            ),
            pltpu.SMEM((2,), jnp.int32),
            pltpu.SMEM((4 * self.page_count,), jnp.int32),
            pltpu.SemaphoreType.DMA((c.key_buffers,)),
            pltpu.SemaphoreType.DMA((2 if self.candidate_width else 4, c.query_tile)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((4,)),
            (
                pltpu.VMEM(
                    (
                        PagePacking(self.candidate_block_size).fields,
                        c.query_tile,
                        self.page_count,
                    ),
                    jnp.int32,
                )
                if self.candidate_width
                else None
            ),
            pltpu.VMEM((2, self.page_count), jnp.int32)
            if self.candidate_width
            else None,
            pltpu.SMEM((2, self.page_count), jnp.int32)
            if self.candidate_width
            else None,
            pltpu.SemaphoreType.DMA if self.candidate_width else None,
        )

    def vmem_bytes(self, info: pltpu.TpuInfo, tokens: int, key_groups: int) -> int:
        """Tiled explicit VMEM, including resident weights and page tables."""

        def size(shape, dtype):
            rows = info.get_sublane_tiling(jnp.dtype(dtype))
            return (
                pl.cdiv(shape[-2], rows)
                * rows
                * pl.cdiv(shape[-1], info.num_lanes)
                * info.num_lanes
                * jnp.dtype(dtype).itemsize
                * (shape[0] if len(shape) == 3 else 1)
            )

        scratch = self.scratch_shapes()
        return (
            sum(size(s.shape, s.dtype) for s in scratch[:4])
            + (size(scratch[4].shape, scratch[4].dtype) if scratch[4] else 0)
            + 2 * size((tokens, self.heads), jnp.float32)
            + size((key_groups, self.page_count), jnp.int32)
            + (
                PagePacking(self.candidate_block_size).fields
                * size((tokens, self.page_count), jnp.int32)
                + size(scratch[11].shape, scratch[11].dtype)
                + size(scratch[12].shape, scratch[12].dtype)
                if self.candidate_width
                else 0
            )
        )

    def smem_bytes(self, tokens: int, key_groups: int) -> int:
        return (
            4 * self.page_count
            + tokens
            + 2
            + key_groups
            + 1
            + key_groups * self.page_count * PAGE_RECORDS // self.config.key_tile
            + (2 * self.page_count if self.candidate_width else 0)
        ) * 4

    def validate(self, info: pltpu.TpuInfo, tokens: int, key_groups: int) -> None:
        if self.vmem_bytes(info, tokens, key_groups) >= info.vmem_capacity_bytes:
            raise ValueError("Paged MQA buffers exceed VMEM; reduce batch/context")
        if self.smem_bytes(tokens, key_groups) >= info.smem_capacity_bytes:
            raise ValueError("Paged MQA metadata exceeds SMEM; reduce batch/context")


@dataclass(frozen=True)
class Pipeline:
    """Trace-time producer/consumer schedule; no runtime Python dispatch.

    K slots stay resident through the final scale read. QK results pass through
    BF16 rounding before head reduction without a VMEM logit ring. MXU
    scheduling is compiler-managed. Output DMA starts after compute and must
    finish before the slot is overwritten. Q has two slots, metadata has four
    slots (two-query lookahead), and scores have two slots per query parity.
    Scores may remain in flight across a query boundary; release their pair
    before reuse, and drain both pairs at kernel exit.
    """

    plan: BufferPlan

    def prefill(self, ops: "MQAOps") -> None:
        buffers = self.plan.config.key_buffers
        for tile in range(buffers - 1):

            @pl.when((ops.state.step == 0) & (tile < ops.live_tiles))
            def current(tile=tile):
                ops.fetch(tile, ops.metadata_slot, ops.key_slot(tile))

            @pl.when((ops.live_tiles + tile < buffers - 1) & (tile < ops.next_tiles))
            def following(tile=tile):
                ops.fetch(
                    tile,
                    (ops.metadata_slot + 1) & 3,
                    ops.key_slot(ops.live_tiles + tile),
                )

    def emit_tile(
        self, ops: "MQAOps", tile, *, prefetch: bool, full_tile: bool
    ) -> None:
        following_tile, metadata, following_slot = self.prefetch_target(ops, tile)
        if not full_tile:
            ops.prepare_output_mask(tile)
        # Trace every producer and consumer, including startup and drain.
        # Eight pending fragments allow QK, selector sums and VPU sums to overlap.
        lag = self.plan.config.issue_depth
        pending_logits = {}
        for stage in range(self.plan.fragments + lag):
            if stage < self.plan.fragments:
                if prefetch:
                    ops.load_page(following_tile, stage * 2, metadata, following_slot)
                pending_logits[stage] = ops.qk(tile, stage)
            if stage >= lag:
                fragment = stage - lag
                value = ops.reduce(fragment, pending_logits.pop(fragment))
                ops.store(tile, fragment, value, full_tile=full_tile)
            if stage < self.plan.fragments and prefetch:
                ops.load_page(following_tile, stage * 2 + 1, metadata, following_slot)

    def prefetch_target(self, ops: "MQAOps", tile) -> tuple:
        following = tile + self.plan.config.key_buffers - 1
        following_tile = jnp.where(
            following < ops.live_tiles, following, following - ops.live_tiles
        )
        metadata = jnp.where(
            following < ops.live_tiles, ops.metadata_slot, (ops.metadata_slot + 1) & 3
        )
        return following_tile, metadata, ops.key_slot(following)

    def step(self, ops: "MQAOps", tile) -> None:
        """Process one 32K K-tile for the current Q group with overlapped K prefetching.

        1. Await current K tile readiness (`wait_keys(tile)`).
        2. Await output buffer reuse (from `tile >= 2`, wait on `tile - 2` DMA).
        3. Schedule future K prefetch (`tile + 3` via 4-buffer ring, or next group's K).
        4. Compute 128x256-key fragments: QK -> ReLU -> scale/sum -> mask.
        5. Publish scores: DMA to HBM (dense) or pack to VMEM (compact).
        """
        ops.wait_keys(tile)

        @pl.when(tile >= 2)
        def release_output():
            ops.publish(tile - 2, wait=True)

        pending = (
            tile + self.plan.config.key_buffers - 1 < ops.live_tiles + ops.next_tiles
        )

        def emit(full_tile):
            jax.lax.cond(
                pending,
                lambda: self.emit_tile(ops, tile, prefetch=True, full_tile=full_tile),
                lambda: self.emit_tile(ops, tile, prefetch=False, full_tile=full_tile),
            )

        def compute():
            jax.lax.cond(
                ((tile + 1) * self.plan.config.key_tile <= ops.minimum_length)
                & ops.full_page_tile(tile),
                lambda: emit(True),
                lambda: emit(False),
            )

        if ops.skip_invalid_pages:

            def empty():
                @pl.when(pending)
                def prefetch():
                    target, metadata, slot = self.prefetch_target(ops, tile)
                    group = jnp.where(
                        tile + self.plan.config.key_buffers - 1 < ops.live_tiles,
                        ops.state.query.key_group,
                        ops.next_group,
                    )
                    pl.when(ops.tile_flags[group, target] > 0)(
                        lambda: ops.fetch(target, metadata, slot, unroll=True)
                    )

                ops.clear_output(tile)

            jax.lax.cond(
                ops.tile_flags[ops.state.query.key_group, tile] > 0, compute, empty
            )
        else:
            compute()

        ops.publish(tile)

    def run(
        self,
        make_ops: Callable[[QueryState], "MQAOps"],
        next_query: Callable[[QueryTile], QueryTile],
        first_query: QueryTile,
        tokens: int,
    ) -> None:
        def query_block(state):
            """Process one query group across all its visible K tiles.

            1. Setup pipeline state (`begin_query`).
            2. Await output buffer reuse (`release_output_pair`).
            3. Prefill K tiles (`prefill`) and await Q readiness (`wait_query`).
            4. Loop `live_tiles` (`step`): QK compute, reduction, and output.
            5. Finalize output (`finish_output`) and return next state.

            Hierarchy: Kernel -> query_block (<=6 Qs) -> step (32,768 K/tile).
            """
            ops = make_ops(state)
            following_query = next_query(state.query)
            lookahead_query = next_query(following_query)
            ops.begin_query(following_query, lookahead_query)
            ops.release_output_pair(state.step & 1)
            self.prefill(ops)
            ops.wait_query()

            def body(tile, _):
                self.step(ops, tile)

            jax.lax.fori_loop(0, ops.live_tiles, body, None)
            ops.finish_output()
            return QueryState(
                following_query, state.step + 1, ops.key_slot(ops.live_tiles)
            )

        initial = QueryState(first_query, jnp.int32(0), jnp.int32(0))
        initial_ops = make_ops(initial)
        for pair in range(2):
            initial_ops.output_counts[pair] = jnp.int32(0)
        initial_ops.load_page_table(first_query, 0)
        second_query = next_query(first_query)

        @pl.when(second_query.start < tokens)
        def initial_following_table():
            initial_ops.load_page_table(second_query, 1)

        initial_ops.prefetch_query(first_query, 0)
        initial_ops.load_page_table(first_query, 0, wait=True)
        jax.lax.while_loop(
            lambda state: state.query.start < tokens, query_block, initial
        )
        initial_ops.release_output_pair(0)
        initial_ops.release_output_pair(1)
