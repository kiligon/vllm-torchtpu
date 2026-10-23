# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense scheduling and page reuse, with a TPU comparison to the grouped scorer."""

import functools
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    _packed_fp8_dense_math as dense_math,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    _packed_fp8_dense_pipeline as dense_pipeline,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    packed_fp8_dense as dense,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PackedFP8Layout,
    pack_fp8_pages,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_dense import (
    _next_query_tile,
    packed_fp8_paged_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    PACKED_FP8_MAX_QUERY_TILE,
    PACKED_FP8_MQA_CONFIG,
)


@pytest.mark.cpu_test
def test_page_flags_preserve_large_bank_addresses():
    layout = PackedFP8Layout(2, page_stride=1 << 28)
    bank_rows = (1 << 28) + 33
    layout.validate((bank_rows,))
    encoded = dense._encode_page_table(
        jnp.array([0, 1, -1, 2], jnp.int32), layout, bank_rows, pages_per_tile=2
    )
    np.testing.assert_array_equal(
        jax.lax.shift_right_logical(encoded, jnp.int32(3)),
        [0, 1 << 28, 0, 1 << 28],
    )
    np.testing.assert_array_equal((encoded & 4) != 0, [True, True, False, False])
    np.testing.assert_array_equal((encoded & 2) != 0, [True, True, False, False])


@pytest.mark.cpu_test
@pytest.mark.parametrize("counts", [(1,), (9,), (2, 8, 1, 11), (1, 1, 1)])
@pytest.mark.parametrize("query_size", [1, 6, 8, 12])
def test_kernel_groups_query_rows_without_crossing_requests(counts, query_size):
    """Exercise the actual SMEM scan, including short runs and tail lookahead."""
    rows = sum(counts)
    tile_size = min(query_size, PACKED_FP8_MAX_QUERY_TILE)
    ids = jnp.asarray(np.repeat([91, 3, 77, 8][: len(counts)], counts), jnp.int32)

    def kernel(indices, out):
        out[...] = jnp.full(out.shape, -1, jnp.int32)

        def next_tile(query):
            return _next_query_tile(query, indices, tile_size)

        def record(query):
            for local in range(tile_size):

                @pl.when(local < query.count)
                def write(local=local):
                    out[query.start + local] = query.page_row

            return next_tile(query)

        first = next_tile(
            dense_pipeline.QueryTile(jnp.int32(0), jnp.int32(0), jnp.int32(0))
        )
        last = jax.lax.while_loop(lambda query: query.start < rows, record, first)
        # The DMA pipeline asks for both following and lookahead tiles at EOF.
        last = next_tile(next_tile(last))
        out[rows] = last.start
        out[rows + 1] = last.count

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows + 2,), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        ),
        interpret=True,
    )
    expected = []
    start = 0
    for count in counts:
        for offset in range(0, count, tile_size):
            expected.extend([start + offset] * min(tile_size, count - offset))
        start += count
    np.testing.assert_array_equal(call(ids), [*expected, rows, 0])


@pytest.mark.cpu_test
def test_dense_entry_rejects_request_level_page_table():
    with pytest.raises(ValueError, match="per-query"):
        packed_fp8_paged_mqa_logits(
            jnp.zeros((6, 32, 128), jnp.float8_e4m3fn),
            jnp.zeros((33, 128), jnp.int32),
            jnp.ones((6, 32), jnp.float32),
            jnp.ones((6,), jnp.int32),
            jnp.zeros((1, 256), jnp.int32),
            indices=jnp.zeros((6,), jnp.int32),
            max_model_len=32768,
            q_scale=jnp.ones((6, 32), jnp.float32),
            query_size=6,
        )


@pytest.fixture
def interpret_dense(monkeypatch):
    """Run the real pipeline on CPU with smaller tiles and BF16 bit operations.

    Pallas interprets DMA and TPU bitcasts, but elementwise pack/unpack lack
    CPU lowering. Substitute only those two primitives, preserving their bits.
    """
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("core",),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        info = pltpu.get_tpu_info()

    def pack(xs, *, packed_dtype):
        assert packed_dtype == jnp.bfloat16 and len(xs) == 2
        bits = [
            jax.lax.bitcast_convert_type(x.astype(packed_dtype), jnp.uint16).astype(
                jnp.uint32
            )
            for x in xs
        ]
        return bits[0] | (bits[1] << 16)

    def unpack(x, *, index, packed_dtype, unpacked_dtype):
        assert packed_dtype == jnp.bfloat16 and unpacked_dtype == jnp.float32
        bits = (x >> (16 * index)).astype(jnp.uint16)
        return jax.lax.bitcast_convert_type(bits, packed_dtype).astype(unpacked_dtype)

    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(pltpu, "pack_elementwise", pack)
    monkeypatch.setattr(pltpu, "unpack_elementwise", unpack)
    monkeypatch.setattr(
        pl, "pallas_call", functools.partial(pl.pallas_call, interpret=True)
    )
    monkeypatch.setattr(
        dense, "PACKED_FP8_MQA_CONFIG", replace(PACKED_FP8_MQA_CONFIG, key_tile=256)
    )
    packed_fp8_paged_mqa_logits.clear_cache()
    with jax.default_device(jax.devices("cpu")[0]):
        yield packed_fp8_paged_mqa_logits
    packed_fp8_paged_mqa_logits.clear_cache()


@pytest.mark.cpu_test
@pytest.mark.parametrize("query_tile", range(1, 9))
def test_static_query_tile_reductions_match_head_sum(interpret_dense, query_tile):
    """Both MXU and VPU paths retain every head under each static permutation."""
    heads, width = 32, 256
    rng = np.random.default_rng(631)
    products = jnp.asarray(
        rng.integers(-256, 257, (query_tile * heads, width)) / 128, jnp.bfloat16
    )
    sublanes = pltpu.get_tpu_info().num_sublanes

    def kernel(src, vpu_out, mxu_out):
        values = dense_math.head_order(src[...], query_tile, sublanes)
        owners = jnp.repeat(jnp.arange(query_tile), heads)[:, None]
        owners = dense_math.head_order(owners, query_tile, sublanes)[:, 0]
        selector = (jnp.arange(query_tile)[:, None] == owners[None]).astype(
            jnp.bfloat16
        )
        for out, use_mxu in ((vpu_out, False), (mxu_out, True)):
            out[...] = dense_math.reduce_heads(
                values, selector, sublanes=sublanes, use_mxu=use_mxu
            )

    shape = jax.ShapeDtypeStruct((query_tile, width), jnp.float32)
    actual = pl.pallas_call(kernel, out_shape=(shape, shape))(products)
    expected = np.asarray(products, np.float32).reshape(query_tile, heads, width)
    expected = expected.sum(axis=1).astype(jnp.bfloat16).astype(np.float32)
    for result in actual:
        np.testing.assert_array_equal(result, expected)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("counts", "bank_count", "query_size"),
    [((1,), 1, 1), ((9, 1, 11, 2, 8), 1, 6)]
    + [((9, 1, 11, 2, 8), 2, size) for size in (*range(1, 9), 12)],
)
def test_page_staging_preserves_query_lengths_and_cache_addresses(
    interpret_dense, counts, bank_count, query_size
):
    _check_dense_query_tiles(interpret_dense, counts, bank_count, query_size)


@pytest.mark.parametrize("query_size", [1, 6, 7, 8, 12])
def test_model_configured_query_tiles_on_tpu(query_size):
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 MQA requires TPU v7")
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_mqa import (
        packed_fp8_mqa_logits,
    )

    # Native TPU FP8 dots need not match CPU matmul bitwise. Check scheduling
    # and DMA against the original TPU scorer, with an independent mask check.
    _check_dense_query_tiles(
        packed_fp8_paged_mqa_logits,
        (9, 1, 11, 2, 8),
        2,
        query_size,
        reference=packed_fp8_mqa_logits,
    )


def _check_dense_query_tiles(score, counts, bank_count, query_size, *, reference=None):
    """Wrap all four metadata slots, with empty groups, holes and strided banks."""
    rows, width, pages = sum(counts), 1001, 16
    owners = np.repeat(np.arange(len(counts)), counts)
    q_values = (np.arange(rows, dtype=np.float32) % 7 + 1)[:, None] / 8
    q_values = q_values * np.resize(np.array([1, 0.5, 0.25, -1], np.float32), 32)
    q = jnp.asarray(
        np.broadcast_to(q_values[:, :, None], (rows, 32, 128)), jnp.float8_e4m3fn
    )
    k_values = (np.arange(pages, dtype=np.float32) + 1) / 16
    k = jnp.asarray(
        np.broadcast_to(np.repeat(k_values, 128)[:, None], (pages * 128, 128)),
        jnp.float8_e4m3fn,
    )
    k_scales = np.where(np.arange(pages) % 2, 0.5, 1).astype(np.float32)
    packed = np.asarray(pack_fp8_pages(k, jnp.asarray(np.repeat(k_scales, 128))))
    layout = PackedFP8Layout(
        pages, page_stride=33 * bank_count, page_offset=33 * (bank_count - 1)
    )
    pool = np.zeros((pages * layout.page_stride, 128), np.int32)
    for page in range(pages):
        start = layout.page_offset + page * layout.page_stride
        pool[start : start + 33] = packed[page * 33 : (page + 1) * 33]
    banks = tuple(jnp.asarray(bank) for bank in np.split(pool, bank_count))
    # Two extra table columns must not change the requested output capacity.
    key_tile = dense.PACKED_FP8_MQA_CONFIG.key_tile
    table_columns = pl.cdiv(width, key_tile) * key_tile // 128 + 2
    table = (np.arange(len(counts))[:, None] * 5 + np.arange(table_columns)) % pages
    table = table.astype(np.int32)
    table[0, [1, 3, 4, 7]] = [-1, pages, np.iinfo(np.int32).min, np.iinfo(np.int32).max]
    if len(counts) > 1:
        table[1] = -1
        table[2, 3] = -1
    lengths = np.resize(
        np.array(
            [
                np.iinfo(np.int32).min,
                -1,
                0,
                1,
                127,
                129,
                256,
                257,
                511,
                512,
                513,
                width,
                999,
                np.iinfo(np.int32).max,
                1024,
            ],
            dtype=np.int32,
        ),
        rows,
    )
    if rows == 1:
        lengths[0] = 129
    else:
        lengths[(owners == 1) | (owners == 3)] = 0
        lengths[owners == 4] = 1024
    q_scales = np.broadcast_to(
        np.where(np.arange(rows) % 2, 2, 1).astype(np.float32)[:, None], (rows, 32)
    )
    weights = np.broadcast_to(
        (np.arange(32, dtype=np.float32) % 3 + 1) / 32, (rows, 32)
    )
    actual = score(
        q,
        banks,
        jnp.asarray(weights),
        jnp.asarray(lengths),
        jnp.asarray(table[owners]),
        indices=jnp.asarray(np.array([91, 3, 77, 8, 12], np.int32)[owners]),
        max_model_len=width,
        q_scale=jnp.asarray(q_scales),
        query_size=query_size,
        cache_layout=layout if bank_count == 2 else None,
    )
    positions = np.arange(width)
    page_ids = table[owners[:, None], positions[None, :] // 128]
    safe = np.clip(page_ids, 0, pages - 1)
    logits = (128 * q_values[:, :, None] * k_values[safe][:, None]).astype(jnp.bfloat16)
    products = (
        np.maximum(logits.astype(np.float32), 0)
        * (weights * q_scales).astype(jnp.bfloat16).astype(np.float32)[:, :, None]
    ).astype(jnp.bfloat16)
    expected = products.astype(np.float32).sum(axis=1).astype(jnp.bfloat16)
    expected = expected.astype(np.float32) * k_scales[safe]
    visible = (
        (positions[None, :] < lengths[:, None]) & (page_ids >= 0) & (page_ids < pages)
    )
    if reference is None:
        expected = np.where(visible, expected, -np.inf)
    else:
        expected = np.asarray(
            reference(
                q,
                jnp.asarray(weights),
                banks,
                jnp.asarray(table[:, : table_columns - 2]),
                jnp.asarray(lengths),
                jnp.asarray(np.r_[0, np.cumsum(counts)], jnp.int32),
                q_scale=jnp.asarray(q_scales),
                cache_layout=layout,
            )
        )[:, :width]
    np.testing.assert_array_equal(np.isneginf(actual), ~visible)
    np.testing.assert_array_equal(actual, expected)


def test_per_query_dense_scores_match_grouped_scorer_on_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 MQA requires TPU v7")
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_mqa import (
        packed_fp8_mqa_logits,
    )

    rng = np.random.default_rng(428)
    rows, width = 17, 98304
    owners = np.repeat(np.arange(3), [7, 1, 9])
    indices = jnp.asarray(np.array([91, 3, 77])[owners], jnp.int32)
    q = jnp.asarray(rng.normal(0, 0.25, (rows, 32, 128)), jnp.float8_e4m3fn)
    k = jnp.asarray(rng.normal(0, 0.25, (16 * 128, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.normal(0, 0.02, (rows, 32)), jnp.float32)
    q_scale = jnp.asarray(rng.uniform(0.5, 1.5, (rows, 32)), jnp.float32)
    k_scale = jnp.asarray(rng.uniform(0.5, 1.5, (16 * 128,)), jnp.float32)
    banks = tuple(jnp.split(pack_fp8_pages(k, k_scale), 2))
    pages = rng.integers(0, 16, (3, width // 128), dtype=np.int32)
    pages[:, 2] = -1
    pages[1] = -1
    lengths = jnp.asarray(
        [0, 127, 128, 32769, 65537, width - 1, width, 0]
        + [1, 129, 32768, 32769, 65536, 65537, width - 2, width - 1, width],
        jnp.int32,
    )
    expected = packed_fp8_mqa_logits(
        q,
        weights,
        banks,
        jnp.asarray(pages),
        lengths,
        jnp.array([0, 7, 8, rows], jnp.int32),
        q_scale=q_scale,
    )
    actual = packed_fp8_paged_mqa_logits(
        q,
        banks,
        weights,
        lengths,
        jnp.asarray(pages[owners]),
        indices=indices,
        max_model_len=width,
        q_scale=q_scale,
        query_size=6,
    )
    np.testing.assert_array_equal(actual, expected)

    # A standalone decode and an output width below physical page capacity.
    single = packed_fp8_paged_mqa_logits(
        q[8:9],
        banks,
        weights[8:9],
        lengths[8:9],
        jnp.asarray(pages[2:3]),
        indices=indices[8:9],
        max_model_len=1001,
        q_scale=q_scale[8:9],
        query_size=1,
    )
    np.testing.assert_array_equal(single, expected[8:9, :1001])
