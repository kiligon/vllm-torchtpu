# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page-local permutations for candidate-width score output."""

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .packed_fp8_cache import PAGE_RECORDS


class CandidateLayout(NamedTuple):
    blocks: jax.Array
    pages: jax.Array


@dataclass(frozen=True)
class PagePacking:
    block_size: int

    @property
    def stride(self) -> int:
        return math.gcd(PAGE_RECORDS, self.block_size)

    @property
    def units(self) -> int:
        return PAGE_RECORDS // self.stride

    @property
    def bits(self) -> int:
        return 1 << (max(1, (self.units - 1).bit_length()) - 1).bit_length()

    @property
    def entries(self) -> int:
        return jnp.dtype(jnp.uint32).itemsize * 8 // self.bits

    @property
    def fields(self) -> int:
        return 1 + (self.units + self.entries - 1) // self.entries

    @property
    def count_bits(self) -> int:
        return self.units.bit_length()


def candidate_layout(
    blocks: jax.Array,
    presence: jax.Array,
    *,
    block_size: int,
    pages: int,
) -> CandidateLayout:
    """Encode each page's output offset, count and local inverse permutation."""
    layout = PagePacking(block_size)
    covered = jnp.repeat(presence, block_size // layout.stride, axis=1)
    width = pages * layout.units
    covered = jnp.pad(
        covered[:, :width], ((0, 0), (0, max(0, width - covered.shape[1])))
    )
    covered = covered.reshape(blocks.shape[0], pages, layout.units)
    counts = covered.sum(axis=-1, dtype=jnp.int32)
    starts = jnp.cumsum(counts, axis=-1) - counts
    ranks = jnp.cumsum(covered, axis=-1) - 1
    offsets = jnp.arange(layout.units, dtype=jnp.uint32)
    shifted = offsets << ((ranks % layout.entries) * layout.bits).astype(jnp.uint32)
    fields = [(starts << layout.count_bits) | counts]
    for word in range(layout.fields - 1):
        selected = (covered != 0) & (ranks // layout.entries == word)
        fields.append(jnp.where(selected, shifted, 0).sum(-1, dtype=jnp.uint32))
    return CandidateLayout(blocks, jnp.stack(fields).astype(jnp.int32))


def pack_score_page(
    scores: jax.Array, metadata: jax.Array, *, block_size: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gather within one VREG and rotate into aligned output-column positions."""
    layout = PagePacking(block_size)
    descriptor = metadata[:, :1]
    starts = (descriptor >> layout.count_bits) * layout.stride
    counts = (descriptor & ((1 << layout.count_bits) - 1)) * layout.stride
    positions = jnp.arange(PAGE_RECORDS, dtype=jnp.int32)[None, :]
    relative = (positions - starts) % PAGE_RECORDS
    ranks = relative // layout.stride
    words = jnp.take_along_axis(
        metadata[:, 1:].astype(jnp.uint32),
        ranks // layout.entries,
        axis=1,
        mode="promise_in_bounds",
    )
    offsets = (words >> ((ranks % layout.entries) * layout.bits)) & (layout.units - 1)
    indices = (offsets * layout.stride + relative % layout.stride).astype(jnp.int32)
    values = jnp.take_along_axis(scores, indices, axis=1, mode="promise_in_bounds")
    return values, starts, starts + counts
