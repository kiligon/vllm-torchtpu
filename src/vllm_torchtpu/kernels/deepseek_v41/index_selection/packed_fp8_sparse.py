# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate page masks and canonical block IDs for compact score selection."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._packed_fp8_compact import candidate_membership, candidate_topk
from ._packed_fp8_packing import candidate_layout
from .packed_fp8_cache import PAGE_RECORDS, PackedFP8Layout
from .packed_fp8_mqa import packed_fp8_mqa_logits
from .tuned_block_sizes import PACKED_FP8_MQA_CONFIG


class CandidatePages(NamedTuple):
    physical: jax.Array
    query_offsets: jax.Array
    lengths: jax.Array
    block_mask: jax.Array
    candidates: jax.Array


def packed_fp8_sparse_mqa_logits(
    q: jax.Array,
    weights: jax.Array,
    pool: jax.Array | tuple[jax.Array, jax.Array],
    page_table: jax.Array,
    lengths: jax.Array,
    requests: jax.Array,
    candidates: jax.Array,
    query_start_loc: jax.Array,
    *,
    q_scale: jax.Array,
    cache_layout: PackedFP8Layout,
    block_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Return FP32 [T,C*block_size] scores and canonical [T,C] block IDs.

    Queries are grouped by request. Token column j maps to
    ``canonical[row,j // block_size] * block_size + j % block_size``.
    Invalid or invisible columns are -inf. Scores are packed in TensorCore VMEM
    before writing the candidate-width output to HBM.
    """
    tasks = candidate_pages(
        candidates,
        lengths,
        requests,
        page_table,
        block_size=block_size,
        cache_pages=cache_layout.pages,
        query_start_loc=query_start_loc,
    )
    scores = packed_fp8_mqa_logits(
        q,
        weights,
        pool,
        tasks.physical,
        tasks.lengths,
        tasks.query_offsets,
        q_scale=q_scale,
        cache_layout=cache_layout,
        skip_invalid_pages=True,
        candidate_layout=candidate_layout(
            tasks.candidates,
            tasks.block_mask,
            block_size=block_size,
            pages=tasks.physical.shape[1],
        ),
        candidate_block_size=block_size,
    )
    return scores, tasks.candidates


def packed_fp8_sparse_mqa_topk(
    q: jax.Array,
    weights: jax.Array,
    pool: jax.Array | tuple[jax.Array, jax.Array],
    page_table: jax.Array,
    lengths: jax.Array,
    requests: jax.Array,
    candidates: jax.Array,
    query_start_loc: jax.Array,
    *,
    k: int,
    q_scale: jax.Array,
    cache_layout: PackedFP8Layout,
    block_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Return ascending logical TopK IDs, -1 padded, and canonical block IDs.

    SparseCore selects directly from TensorCore's compact candidate scores.
    """
    scores, canonical = packed_fp8_sparse_mqa_logits(
        q,
        weights,
        pool,
        page_table,
        lengths,
        requests,
        candidates,
        query_start_loc,
        q_scale=q_scale,
        cache_layout=cache_layout,
        block_size=block_size,
    )
    k = min(k, canonical.shape[1] * block_size)
    return candidate_topk(scores, canonical, k=k, block_size=block_size), canonical


def candidate_pages(
    candidates: jax.Array,
    lengths: jax.Array,
    requests: jax.Array,
    page_table: jax.Array,
    *,
    block_size: int,
    cache_pages: int,
    query_start_loc: jax.Array | None = None,
) -> CandidatePages:
    """Mark pages needed by at most six adjacent queries of the same request.

    Queries must be sorted by request. Page slots retain logical positions for
    scoring; ascending unique candidate IDs define compact score-column order.
    """
    rows = candidates.shape[0]
    query_tile = PACKED_FP8_MQA_CONFIG.query_tile
    logical_pages = page_table.shape[1]
    if min(rows, block_size, logical_pages, page_table.shape[0]) < 1:
        raise ValueError("Candidate page tasks require nonempty queries and pages")
    if query_start_loc is None:
        counts = jnp.bincount(requests, length=page_table.shape[0])
        query_start_loc = jnp.concatenate(
            (jnp.zeros((1,), jnp.int32), jnp.cumsum(counts))
        )
    max_groups = min(
        rows, (rows + query_tile - 1) // query_tile + page_table.shape[0] - 1
    )
    request_counts = query_start_loc[1:] - query_start_loc[:-1]
    group_ends = jnp.cumsum((request_counts + query_tile - 1) // query_tile)
    group_ids = jnp.arange(max_groups, dtype=jnp.int32)
    owners = (group_ids[:, None] >= group_ends[None, :]).sum(axis=1)
    owners = jnp.minimum(owners, page_table.shape[0] - 1)
    preceding = jnp.where(owners > 0, group_ends[jnp.maximum(owners - 1, 0)], 0)
    starts = query_start_loc[owners] + (group_ids - preceding) * query_tile
    starts = jnp.where(group_ids < group_ends[-1], starts, rows)
    offsets = jnp.concatenate((starts, jnp.full((1,), rows, jnp.int32)))

    width = logical_pages * PAGE_RECORDS
    blocks = (width + block_size - 1) // block_size
    block_mask, canonical = candidate_membership(candidates, blocks)
    # Only expand to the common block/page boundary, not individual tokens.
    stride = math.gcd(PAGE_RECORDS, block_size)
    covered = jnp.repeat(block_mask, block_size // stride, axis=1)[:, : width // stride]
    covered *= jnp.arange(width // stride)[None, :] * stride < lengths[:, None]
    query_pages = covered.reshape(rows, logical_pages, PAGE_RECORDS // stride).max(-1)
    queries = starts[:, None] + jnp.arange(query_tile, dtype=jnp.int32)
    needed = (
        query_pages[jnp.minimum(queries, rows - 1)]
        * (queries < offsets[1:, None])[:, :, None]
    ).max(axis=1)
    physical = page_table[owners]
    physical = jnp.where(
        (needed != 0) & (physical >= 0) & (physical < cache_pages), physical, -1
    )
    page_tile = PACKED_FP8_MQA_CONFIG.key_tile // PAGE_RECORDS
    padded = (logical_pages + page_tile - 1) // page_tile * page_tile
    physical = jnp.pad(
        physical, ((0, 0), (0, padded - logical_pages)), constant_values=-1
    )
    lengths = jnp.where(query_pages.any(axis=1), lengths, 0)
    return CandidatePages(physical, offsets, lengths, block_mask, canonical)


def mask_candidate_scores(
    scores: jax.Array,
    block_mask: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int,
    width: int,
) -> jax.Array:
    """Broadcast query-local block membership over logically ordered scores."""
    blocks = (scores.shape[1] + block_size - 1) // block_size
    block_mask = jnp.pad(block_mask, ((0, 0), (0, blocks - block_mask.shape[1])))
    allowed = jnp.repeat(block_mask != 0, block_size, axis=1)[:, : scores.shape[1]]
    allowed &= (
        jnp.arange(scores.shape[1])[None, :] < jnp.minimum(lengths, width)[:, None]
    )
    return jnp.where(allowed, scores, -jnp.inf)
