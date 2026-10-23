# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact candidate columns preserve dense masked TopK semantics."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.index_selection._packed_fp8_compact import (
    candidate_membership,
    candidate_topk,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection._packed_fp8_packing import (
    CandidateLayout,
    candidate_layout,
    pack_score_page,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_mqa import (
    MQAOps,
    packed_fp8_mqa_logits,
)


@pytest.mark.cpu_test
def test_selection_handles_empty_rows_ties_and_short_rows():
    positions = np.arange(98304, dtype=np.int32)
    scores = np.stack(
        (
            np.full(98304, -np.inf),
            np.zeros(98304),
            np.where(positions < 17, 0.0, -np.inf),
            positions % 17,
        )
    ).astype(np.float32)
    candidates = np.broadcast_to(np.arange(2048), (4, 2048)).copy()
    candidates[0] = -1
    candidates[3] *= 3
    expected = np.full((4, 512), -1, np.int32)
    for row in range(4):
        blocks = candidates[row][candidates[row] >= 0]
        logical = (blocks[:, None] * 8 + np.arange(8)).reshape(-1)
        logical = logical[np.isfinite(scores[row, logical])]
        chosen = sorted(logical, key=lambda i: (-scores[row, i], i))[:512]
        expected[row, : len(chosen)] = np.sort(chosen)
    logical = (candidates[:, :, None] * 8 + np.arange(8)).reshape(4, -1)
    compact = np.take_along_axis(scores, np.maximum(logical, 0), axis=1)
    compact[logical < 0] = -np.inf
    actual = jax.jit(
        lambda values, ids: candidate_topk(values, ids, k=512, block_size=8)
    )(jnp.asarray(compact), jnp.asarray(candidates, jnp.int32))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.cpu_test
def test_sparse_entry_point_returns_candidate_width(monkeypatch):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import packed_fp8_sparse
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
        PackedFP8Layout,
    )

    rows, width, capacity = 6, 98304, 2048
    candidates = jnp.tile(jnp.arange(capacity)[None, ::-1] * 3, (rows, 1))
    lengths = jnp.arange(rows, dtype=jnp.int32) * (width // rows)

    def scorer(q, weights, pool, pages, lengths, offsets, **kwargs):
        ids, size = kwargs["candidate_layout"].blocks, kwargs["candidate_block_size"]
        logical = (ids[:, :, None] * size + jnp.arange(size)).reshape(rows, -1)
        return jnp.where(
            (logical >= 0) & (logical < lengths[:, None]), logical, -jnp.inf
        ).astype(jnp.float32)

    monkeypatch.setattr(packed_fp8_sparse, "packed_fp8_mqa_logits", scorer)
    scores, canonical = packed_fp8_sparse.packed_fp8_sparse_mqa_logits(
        jnp.zeros((rows, 32, 128), jnp.float8_e4m3fn),
        jnp.ones((rows, 32), jnp.float32),
        jnp.empty((1, 128), jnp.int32),
        jnp.arange(width // 128, dtype=jnp.int32)[None, :],
        lengths,
        jnp.zeros((rows,), jnp.int32),
        candidates,
        jnp.array([0, rows], jnp.int32),
        q_scale=jnp.ones((rows, 32), jnp.float32),
        cache_layout=PackedFP8Layout(width // 128),
        block_size=8,
    )
    assert scores.shape == (rows, 16384)
    expected_ids = np.tile(np.arange(capacity, dtype=np.int32) * 3, (rows, 1))
    np.testing.assert_array_equal(canonical, expected_ids)
    logical = (expected_ids[:, :, None] * 8 + np.arange(8)).reshape(rows, -1)
    expected = np.where(logical < np.asarray(lengths)[:, None], logical, -np.inf)
    np.testing.assert_array_equal(scores, expected)


@pytest.mark.cpu_test
@pytest.mark.parametrize("block_size", [8, 96, 192])
def test_compact_scores_keep_logical_ties_and_remove_duplicates(block_size):
    rng = np.random.default_rng(581)
    rows, width, capacity = 7, 98304, 16
    count = (width + block_size - 1) // block_size
    candidates = rng.integers(-2, count + 3, (rows, capacity), dtype=np.int32)
    candidates[0] = -1
    candidates[1, :5] = [1, 1, 0, 0, np.iinfo(np.int32).max]
    candidates[3, :4] = [
        32768 // block_size - 1,
        32768 // block_size,
        65536 // block_size,
        count - 1,
    ]
    scores = rng.integers(-3, 4, (rows, width)).astype(np.float32)
    scores[2, 19:] = -np.inf
    mask, canonical = candidate_membership(jnp.asarray(candidates), count)
    layout = candidate_layout(
        canonical, mask, block_size=block_size, pages=width // 128
    )
    packed, starts, ends = jax.jit(
        jax.vmap(
            lambda values, meta: pack_score_page(values, meta.T, block_size=block_size),
            in_axes=(1, 2),
            out_axes=(1, 1, 1),
        )
    )(jnp.asarray(scores.reshape(rows, -1, 128)), layout.pages)
    packed, starts, ends = map(np.asarray, (packed, starts, ends))
    compact = np.full((rows, capacity * block_size), -np.inf, np.float32)
    for row in range(rows):
        for page in range(width // 128):
            start, end = int(starts[row, page, 0]), int(ends[row, page, 0])
            indices = np.arange(start, end)
            compact[row, indices] = packed[row, page, indices % 128]
    compact = jnp.asarray(compact)
    logical = (
        np.asarray(canonical)[:, :, None] * block_size + np.arange(block_size)
    ).reshape(rows, -1)
    expected_scores = np.take_along_axis(scores, np.clip(logical, 0, width - 1), axis=1)
    expected_scores[(logical < 0) | (logical >= width)] = -np.inf
    np.testing.assert_array_equal(compact, expected_scores)
    assert compact.shape == (rows, capacity * block_size)
    values, selected = jax.lax.top_k(compact, 16)
    blocks = jnp.take_along_axis(canonical, selected // block_size, axis=1)
    actual = jnp.where(
        jnp.isfinite(values), blocks * block_size + selected % block_size, -1
    )
    expected = np.full((rows, 16), -1, np.int32)
    for row in range(rows):
        logical = np.arange(width)
        valid = np.asarray(mask[row])[logical // block_size] != 0
        valid &= np.isfinite(scores[row])
        chosen = sorted(np.flatnonzero(valid), key=lambda i: (-scores[row, i], i))[:16]
        expected[row, : len(chosen)] = chosen
    np.testing.assert_array_equal(actual, expected)
    result = candidate_topk(compact, canonical, k=16, block_size=block_size)
    for row in range(rows):
        valid = expected[row][expected[row] >= 0]
        expected[row, : valid.size] = np.sort(valid)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.cpu_test
@pytest.mark.parametrize("capacity", [0, 16, 2048])
def test_scorer_allocates_candidate_width_output(capacity):
    shape = jax.ShapeDtypeStruct
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("core",),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        result = jax.eval_shape(
            packed_fp8_mqa_logits,
            shape((6, 32, 128), jnp.float8_e4m3fn),
            shape((6, 32), jnp.float32),
            shape((16 * 33, 128), jnp.int32),
            shape((1, 768), jnp.int32),
            shape((6,), jnp.int32),
            shape((2,), jnp.int32),
            q_scale=shape((6, 32), jnp.float32),
            candidate_layout=CandidateLayout(
                shape((6, capacity), jnp.int32), shape((3, 6, 768), jnp.int32)
            ),
            candidate_block_size=8,
        )
    assert result.shape == (6, capacity * 8)
    assert result.dtype == jnp.float32


@pytest.mark.cpu_test
def test_empty_candidates_have_no_topk_columns():
    result = candidate_topk(
        jnp.empty((6, 0), jnp.float32),
        jnp.empty((6, 0), jnp.int32),
        k=0,
        block_size=8,
    )
    assert result.shape == (6, 0)


@pytest.mark.cpu_test
def test_page_packing_writes_only_each_rows_candidate_columns():
    rng = np.random.default_rng(412)
    rows, width, capacity = 6, 32768, 256
    candidates = np.stack(
        [np.sort(rng.choice(width // 8, capacity, replace=False)) for _ in range(rows)]
    ).astype(np.int32)
    candidates[0] = -1
    candidates[1] = np.arange(capacity)
    candidates[2] = np.arange(width // 8 - capacity, width // 8)
    presence, canonical = candidate_membership(jnp.asarray(candidates), width // 8)
    layout = candidate_layout(canonical, presence, block_size=8, pages=width // 128)
    descriptor = layout.pages[0]
    starts, counts = (descriptor >> 5) * 8, (descriptor & 31) * 8
    bounds = jnp.stack(
        (
            jnp.where(counts > 0, starts // 128, capacity * 8 // 128).min(0),
            jnp.where(counts > 0, (starts + counts + 127) // 128, 0).max(0),
        )
    )
    scores = jnp.asarray(rng.normal(size=(1, rows, width)), jnp.float32)

    def kernel(bounds, metadata, scores, out):
        out[...] = jnp.full(out.shape, -jnp.inf, jnp.float32)
        ops = SimpleNamespace(
            packing=metadata,
            packing_bounds_smem=bounds,
            compact_scores=out,
            scores=scores,
            query_slot=jnp.int32(0),
            candidate_block_size=8,
            plan=SimpleNamespace(
                config=SimpleNamespace(key_tile=width, query_tile=rows)
            ),
        )
        MQAOps.pack_output(ops, jnp.int32(0))

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, rows, capacity * 8), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(pl.BlockSpec(memory_space=pltpu.VMEM),) * 2,
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        interpret=True,
    )
    actual = call(bounds, layout.pages, scores)[0]
    logical = (np.asarray(canonical)[:, :, None] * 8 + np.arange(8)).reshape(rows, -1)
    expected = np.take_along_axis(np.asarray(scores[0]), np.maximum(logical, 0), 1)
    expected[logical < 0] = -np.inf
    np.testing.assert_array_equal(actual, expected)


def test_compact_scorer_matches_dense_scores_on_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 MQA requires TPU v7")
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
        pack_fp8_pages,
    )

    rng = np.random.default_rng(325)
    rows, width, capacity = 12, 98304, 2048
    q = jnp.asarray(rng.normal(0, 0.25, (rows, 32, 128)), jnp.float8_e4m3fn)
    k = jnp.asarray(rng.normal(0, 0.25, (16 * 128, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.normal(0, 0.02, (rows, 32)), jnp.float32)
    packed = pack_fp8_pages(k, jnp.ones((16 * 128,), jnp.float32))
    banks = tuple(jnp.split(packed, 2))
    pages = rng.integers(0, 16, (3, width // 128), dtype=np.int32)
    pages[:, 2] = -1
    pages[1, 256:512] = -1
    lengths = jnp.array(
        [0, 127, 128, 32769, 65537, width, 2048, 16, 65536, width - 1, width, 32768],
        jnp.int32,
    )
    candidates = np.stack(
        [np.sort(rng.choice(width // 8, capacity, replace=False)) for _ in range(rows)]
    ).astype(np.int32)
    candidates[-1] = -1
    candidates = jnp.asarray(candidates)
    args = (q, weights, banks, jnp.asarray(pages), lengths, jnp.array([0, 6, 7, 12]))
    kwargs = dict(q_scale=jnp.ones((rows, 32), jnp.float32), skip_invalid_pages=True)
    dense = packed_fp8_mqa_logits(*args, **kwargs)
    presence, canonical = candidate_membership(candidates, width // 8)
    layout = candidate_layout(canonical, presence, block_size=8, pages=width // 128)
    compact = packed_fp8_mqa_logits(
        *args, **kwargs, candidate_layout=layout, candidate_block_size=8
    )
    logical = (candidates[:, :, None] * 8 + jnp.arange(8)).reshape(rows, -1)
    expected = jnp.take_along_axis(dense, jnp.maximum(logical, 0), axis=1)
    expected = jnp.where(logical >= 0, expected, -jnp.inf)
    np.testing.assert_array_equal(compact, expected)
