# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model cache ownership and query routing, with a CPU scorer stand-in."""

import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "request_sizes,interleaved,expected_rows",
    [([6, 6], False, 6), ([5, 7], False, 12), ([6, 6], True, 12)],
)
def test_query_chunks_preserve_six_query_head_groups(
    monkeypatch, request_sizes, interleaved, expected_rows
):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
        PackedFP8Layout,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import packed_fp8_indexer as model

    # The production table targets the large hardware fixture. Exercise the
    # same dispatch contract with a small scorer spy; this is not a numeric
    # replacement for the TPU scorer.
    monkeypatch.setitem(model.PACKED_FP8_INDEXER_QUERY_ROWS, (12, 256, 8, 8, 8, 2), 6)

    def scorer(
        q, weights, pool, pages, lengths, owners, candidates, offsets, *, k, **kw
    ):
        return jnp.full((q.shape[0], k), q.shape[0], jnp.int32), candidates

    monkeypatch.setattr(model, "packed_fp8_sparse_mqa_topk", scorer)
    # lax.cond traces the unused dense arm as well; this test only checks dispatch.
    monkeypatch.setattr(
        model,
        "packed_fp8_mqa_logits",
        lambda q, *args, **kw: jnp.zeros((q.shape[0], 256), jnp.float32),
    )
    requests = jnp.repeat(jnp.arange(2), jnp.asarray(request_sizes))
    if interleaved:
        requests = requests.reshape(2, 6).T.reshape(-1)
    candidates = jnp.tile(jnp.arange(8), (12, 1))
    selected, published = model._score_and_select(
        jnp.zeros((12, 32, 128), jnp.bfloat16),
        (jnp.zeros((1, 128), jnp.int32),) * 2,
        jnp.tile(jnp.arange(256), (2, 1)),
        jnp.ones((12, 32), jnp.float32),
        jnp.full((12,), 256, jnp.int32),
        candidates,
        requests,
        jnp.array([256], jnp.int32),
        PackedFP8Layout(2),
        8,
        8,
        False,
        True,
        True,
    )
    np.testing.assert_array_equal(selected, np.full((12, 8), expected_rows))
    np.testing.assert_array_equal(published, candidates)


@pytest.fixture
def cpu_cache_ops(monkeypatch):
    """Run the production JAX callbacks without the TPU/Torch bridge."""
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import packed_fp8_indexer

    def to_jax(value):
        if not isinstance(value, torch.Tensor):
            return value
        if value.dtype == torch.bfloat16:
            return jnp.asarray(value.float().numpy(), jnp.bfloat16)
        return jnp.asarray(value.numpy())

    def to_torch(value):
        if value.dtype == jnp.bfloat16:
            return torch.from_numpy(np.asarray(value, np.float32).copy()).bfloat16()
        return torch.from_numpy(np.asarray(value).copy())

    def bridge(callback):
        def operation(*args):
            return jax.tree.map(to_torch, callback(*(to_jax(arg) for arg in args)))

        return operation

    for name, callback in (
        ("_read_op", packed_fp8_indexer._read),
        ("_read_2bank_op", packed_fp8_indexer._read_two),
        ("_write_op", packed_fp8_indexer._write),
        ("_write_2bank_op", packed_fp8_indexer._write_two),
    ):
        monkeypatch.setattr(packed_fp8_indexer, name, bridge(callback))


def _cache(two_banks):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.packed_fp8_indexer import (
        PackedFP8IndexCacheRows,
    )

    raw = (
        (torch.zeros((80, 128), dtype=torch.int32),) * 2
        if two_banks
        else torch.zeros((160, 128), dtype=torch.int32)
    )
    if two_banks:
        raw = tuple(bank.clone() for bank in raw)
    return PackedFP8IndexCacheRows(raw, ((4, 128, 132), torch.uint8, 40 * 512, 5 * 512))


@pytest.mark.parametrize("two_banks", [False, True])
def test_model_cache_incremental_write_and_empty_deferred_write(
    cpu_cache_ops, two_banks
):
    cache = _cache(two_banks)
    records = torch.tensor([0, 127, 128, 255, 256, 511, -1])
    values = torch.arange(7, dtype=torch.bfloat16)[:, None].expand(7, 128)
    cache.write(records, values, after=torch.tensor([1]))
    torch.testing.assert_close(cache[records[:6] // 128, records[:6] % 128], values[:6])
    before = cache.raw.clone()
    before_hi = None if cache.raw_hi is None else cache.raw_hi.clone()
    cache.write(records[:0], values[:0])
    torch.testing.assert_close(cache.raw, before, rtol=0, atol=0)
    if before_hi is not None:
        torch.testing.assert_close(cache.raw_hi, before_hi, rtol=0, atol=0)
    torch.testing.assert_close(cache.row_indices(records), records)
    assert torch.count_nonzero(cache[torch.tensor([4]), torch.tensor([0])]) == 0


def _routing_scorer(
    q,
    weights,
    pool,
    pages,
    lengths,
    offsets,
    *,
    q_scale,
    cache_layout,
    skip_invalid_pages=False,
    candidate_layout=None,
    candidate_block_size=1,
):
    """Test grouping and page interpretation; does not emulate TPU arithmetic."""
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
        read_fp8_records,
    )

    records = (pages[:, :2, None] * 128 + jnp.arange(128)).reshape(-1, 256)
    keys = read_fp8_records(
        pool if isinstance(pool, tuple) else (pool,), records, cache_layout
    )
    owners = (jnp.arange(q.shape[0])[:, None] >= offsets[None, 1:]).sum(-1)
    scores = (q[:, 0, 0].astype(jnp.float32) * q_scale[:, 0])[:, None] * keys[
        owners, :, 0
    ].astype(jnp.float32) + weights[:, :1]
    scores = jnp.where(jnp.arange(256)[None, :] < lengths[:, None], scores, -jnp.inf)
    if candidate_layout is not None:
        logical = (
            candidate_layout.blocks[:, :, None] * candidate_block_size
            + jnp.arange(candidate_block_size)
        ).reshape(q.shape[0], -1)
        selected = jnp.take_along_axis(scores, jnp.clip(logical, 0, 255), axis=1)
        return jnp.where((logical >= 0) & (logical < 256), selected, -jnp.inf)
    return jnp.pad(
        scores, ((0, 0), (0, pages.shape[1] * 128 - 256)), constant_values=-jnp.inf
    )


@pytest.mark.parametrize("mode", ["none", "publish", "filter", "short_filter"])
@pytest.mark.parametrize("write_count", [0, 2])
def test_chunked_ragged_queries_restore_order_and_global_candidate_policy(
    monkeypatch, mode, write_count
):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
        PackedFP8Layout,
        read_fp8_records,
        write_fp8_records,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        attention_kernels,
        packed_fp8_indexer,
    )

    monkeypatch.setattr(packed_fp8_indexer, "packed_fp8_mqa_logits", _routing_scorer)
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import packed_fp8_sparse

    monkeypatch.setattr(packed_fp8_sparse, "packed_fp8_mqa_logits", _routing_scorer)
    monkeypatch.setattr(attention_kernels, "_INDEX_UNTILED_SCORE_ELEMENTS", 6 * 32768)
    layout = PackedFP8Layout(4, 40, 5)
    values = jnp.asarray(
        np.random.default_rng(54).normal(size=(512, 128)), jnp.bfloat16
    )
    banks = write_fp8_records(
        (jnp.zeros((80, 128), jnp.int32),) * 2, jnp.arange(512), values, layout
    )
    page_ids = jnp.array([[0, 2], [3, 1], [2, 0]], jnp.int32)
    records = (page_ids[:, :, None] * 128 + jnp.arange(128)).reshape(3, 256)
    requests = jnp.array([2, 0, 2, 1, -1, 0, 1, 5], jnp.int32)
    lengths = jnp.array([0, 1, 129, 255, 100, 256, 25, 60], jnp.int32)
    if mode == "short_filter":
        lengths = jnp.minimum(lengths, 3)
    q_values = jnp.array([1, -1, 2, -2, 3, -3, 4, -4], jnp.bfloat16)
    q = jnp.broadcast_to(q_values[:, None, None], (8, 32, 128))
    weights = jnp.broadcast_to(jnp.arange(8)[:, None], (8, 32)).astype(jnp.float32)
    candidates = jnp.array([[2], [3], [0], [2], [1], [-1], [0], [1]], jnp.int32)
    write_records = jnp.array([17, 300], jnp.int32)[:write_count]
    new_keys = -values[write_records]
    active_lengths = jnp.where((requests >= 0) & (requests < 3), lengths, 0)
    batch_max = jnp.max(active_lengths).reshape(1)
    publish, filtering = mode == "publish", mode.endswith("filter")
    lo, hi, selected, published = packed_fp8_indexer._select_two(
        q,
        *banks,
        records,
        weights,
        lengths,
        candidates,
        write_records,
        new_keys,
        requests,
        batch_max,
        4,
        40,
        5,
        3,
        64,
        publish,
        filtering,
        False,
    )
    expected_banks = write_fp8_records(banks, write_records, new_keys, layout)
    for actual, expected in zip((lo, hi), expected_banks):
        np.testing.assert_array_equal(actual, expected)
    keys = read_fp8_records(expected_banks, records, layout).astype(jnp.float32)
    expected_scores = (
        q_values.astype(jnp.float32)[:, None] * keys[jnp.clip(requests, 0, 2), :, 0]
        + weights[:, :1]
    )
    expected = attention_kernels.select_indices_jax(
        expected_scores,
        active_lengths,
        candidates,
        3,
        64,
        publish,
        filtering,
        False,
        batch_max,
    )
    for actual, wanted in zip((selected, published), expected):
        np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("two_banks", [False, True])
def test_torch_dispatch_passes_logical_slots_and_preserves_native_owners(
    monkeypatch, two_banks
):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        attention_kernels,
        packed_fp8_indexer,
    )

    cache = _cache(two_banks)
    function = (
        packed_fp8_indexer._select_two if two_banks else packed_fp8_indexer._select
    )
    captured = {}
    selected = torch.tensor([[17, 19, -1]], dtype=torch.int32)
    candidates = torch.tensor([[0]], dtype=torch.int32)

    def operation(*args):
        captured.update(inspect.signature(function).bind(*args).arguments)
        return ((cache.raw, cache.raw_hi) if two_banks else (cache.raw,)) + (
            selected,
            candidates,
        )

    monkeypatch.setattr(
        packed_fp8_indexer, "_select_2bank_op" if two_banks else "_select_op", operation
    )
    updated, actual, published = attention_kernels.paged_indexer(
        torch.zeros((1, 32, 128), dtype=torch.bfloat16),
        cache,
        torch.arange(256).reshape(1, 256),
        torch.ones((1, 32)),
        torch.tensor([[20]]),
        candidates,
        torch.tensor([17]),
        torch.ones((1, 128), dtype=torch.bfloat16),
        torch.tensor([0]),
        topk=3,
        block_size=64,
        candidate_write=False,
        candidate_filter=True,
    )
    assert captured["pages"] == 4
    assert captured["stride"] == 40
    assert captured["offset"] == 5
    assert captured["write_records"].dtype == torch.int32
    assert captured["write_records"].tolist() == [17]
    assert captured["batch_max"].tolist() == [20]
    if two_banks:
        assert updated[0] is cache.raw and updated[1] is cache.raw_hi
    else:
        assert updated is cache.raw
    assert actual is selected and published is candidates


def test_registered_jax_callbacks_have_concrete_schema_annotations():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import packed_fp8_indexer

    for name in (
        "_write",
        "_write_two",
        "_read",
        "_read_two",
        "_select",
        "_select_two",
    ):
        signature = inspect.signature(vars(packed_fp8_indexer)[name])
        assert signature.return_annotation is not inspect.Signature.empty
        assert all(
            parameter.annotation in (jax.Array, int, bool)
            for parameter in signature.parameters.values()
        )


@pytest.mark.parametrize("two_banks", [False, True])
@pytest.mark.parametrize("csa2", [False, True])
def test_shared_allocator_keeps_index_scale_inside_page(
    monkeypatch, cpu_cache_ops, two_banks, csa2
):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as cache_module
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.packed_fp8_indexer import (
        PackedFP8IndexCacheRows,
    )

    # Upstream byte views describe one region with main KV and Index-K pages.
    owner = torch.empty((160 * 512,), dtype=torch.uint8, device="meta")
    views = {
        "main": owner.as_strided((4, 1, 4, 512), (40 * 512, 4 * 512, 512, 1)),
        "index": owner.as_strided(
            (4, 1, 128, 132), (40 * 512, 128 * 132, 132, 1), 5 * 512
        ),
    }

    def allocate(config, device, layout):
        return views

    monkeypatch.setattr(cache_module, "allocate_upstream_kv_cache", allocate)
    if two_banks:
        monkeypatch.setattr(cache_module, "SPARSECORE_MAX_POOL_BYTES", 80 * 512)
    config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(size=160 * 512)], num_blocks=4
    )
    result = cache_module.materialize_upstream_kv_cache(
        config,
        torch.device("cpu"),
        SimpleNamespace(is_layer_compact=False),
        word_rows=True,
        csa2_layers=frozenset({"main"}) if csa2 else frozenset(),
    )
    assert "index" not in result.rope_pools
    assert "index" not in result.scale_pools
    assert set(result.scale_pools if csa2 else result.rope_pools) == {"main"}
    assert result.native_raw_cache["index"] is result.native_raw_cache["main"]
    cache = PackedFP8IndexCacheRows(
        result.native_raw_cache["index"], result.native_cache_layouts["index"]
    )
    records = torch.tensor([0, 255, 256, 511])
    keys = torch.arange(4, dtype=torch.bfloat16)[:, None].expand(4, 128)
    cache.write(records, keys)
    torch.testing.assert_close(cache[records // 128, records % 128], keys)
    banks = (cache.raw,) if cache.raw_hi is None else (cache.raw, cache.raw_hi)
    for bank in banks:
        assert not bank.reshape(-1, 40, 128)[:, :5].any()
        assert not bank.reshape(-1, 40, 128)[:, 38:].any()
