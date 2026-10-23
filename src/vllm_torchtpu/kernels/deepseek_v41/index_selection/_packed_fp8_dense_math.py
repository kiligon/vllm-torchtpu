# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 head layout for a statically configured dense query tile."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def _head_groups(queries: int, heads: int, sublanes: int):
    """Partition queries into power-of-two groups that fill one vector tile."""
    start = 0
    while start < queries:
        count = 1 << ((queries - start).bit_length() - 1)
        copies = sublanes // count
        # Each uint32 word carries two BF16 head products.
        groups = heads // (2 * copies)
        yield start, count, groups, copies
        start += count


def head_order(x, queries: int, sublanes: int):
    heads, width = x.shape[0] // queries, x.shape[-1]
    x = x.reshape(queries, heads, width)
    parts = []
    for start, count, groups, copies in _head_groups(queries, heads, sublanes):
        part = x[start : start + count].reshape(count, 2, groups, copies, width)
        parts.append(part.transpose(2, 3, 0, 1, 4).reshape(count * heads, width))
    return jnp.concatenate(parts)


def reduce_heads(products, selector, *, sublanes: int, use_mxu: bool):
    """Accumulate finite BF16 products in FP32 and round the completed sum."""
    if use_mxu:
        values = jax.lax.dot_general(
            selector,
            products,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
    else:
        queries, width = selector.shape[0], products.shape[-1]
        heads = products.shape[0] // queries
        values = None
        for start, count, groups, _ in _head_groups(queries, heads, sublanes):
            packed = pltpu.bitcast(
                products[start * heads : (start + count) * heads], jnp.uint32
            ).reshape(groups, sublanes, width)
            partial = None
            for group in range(groups):
                for half in range(2):
                    value = pltpu.unpack_elementwise(
                        packed[group],
                        index=half,
                        packed_dtype=jnp.bfloat16,
                        unpacked_dtype=jnp.float32,
                    )
                    partial = value if partial is None else partial + value
            assert partial is not None
            offset = sublanes // 2
            while offset >= count:
                partial = partial + pltpu.roll(partial, offset, axis=0)
                offset //= 2
            # Each group's query pattern repeats across sublanes. Its start
            # is a multiple of count, so selecting rows needs no rotation.
            values = (
                partial
                if values is None
                else jnp.where(jnp.arange(sublanes)[:, None] < start, values, partial)
            )
        assert values is not None
        values = values[:queries]
    rounded = pltpu.pack_elementwise((values, values), packed_dtype=jnp.bfloat16)
    return pltpu.unpack_elementwise(
        rounded, index=0, packed_dtype=jnp.bfloat16, unpacked_dtype=jnp.float32
    )
