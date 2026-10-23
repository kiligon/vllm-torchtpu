# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the maintained packed FP8 MQA entry point (TPU v7, P-state 7)."""

import argparse
import functools
import importlib.metadata
import json
import os
import types
from pathlib import Path

import bench_packed_mqa_compute as compute
import bench_packed_mqa_metadata as metadata
import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PAGE_RECORDS,
    pack_fp8_pages,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_dense import (
    packed_fp8_paged_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_mqa import (
    packed_fp8_mqa_logits,
)


def compile_kernel(
    inputs: tuple, out: Path, *, query_size: int, variant: str = "grouped"
) -> jax.stages.Compiled:
    def score(q, weights, pool, pages, lengths, offsets, scales):
        if variant == "per-query":
            return packed_fp8_paged_mqa_logits(
                q,
                pool,
                weights,
                lengths,
                pages,
                indices=offsets,
                max_model_len=pages.shape[1] * PAGE_RECORDS,
                q_scale=scales,
                query_size=query_size,
            )
        return packed_fp8_mqa_logits(
            q, weights, pool, pages, lengths, offsets, q_scale=scales
        )

    score.__name__ = f"metadata_trial_full_{variant.replace('-', '_')}"
    print("COMPILE", variant, flush=True)
    executable = jax.jit(score).lower(*inputs).compile()
    (out / f"{variant}.hlo.txt").write_text(executable.as_text())
    memory = executable.memory_analysis()
    (out / f"{variant}.memory.json").write_text(
        json.dumps(
            dict(
                argument_bytes=memory.argument_size_in_bytes,
                output_bytes=memory.output_size_in_bytes,
                temporary_bytes=memory.temp_size_in_bytes,
            ),
            indent=2,
        )
    )
    return executable


def per_query_inputs(inputs: tuple) -> tuple:
    """Expand request tables outside the timed scorer, as the caller would."""
    q, weights, pool, pages, lengths, offsets, scales = inputs
    indices = jnp.repeat(
        jnp.arange(pages.shape[0], dtype=jnp.int32),
        jnp.diff(offsets),
        total_repeat_length=q.shape[0],
    )
    return (
        q,
        weights,
        pool,
        pages[indices],
        lengths,
        indices,
        scales,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--variant", choices=("grouped", "per-query", "compare"), default="grouped"
    )
    parser.add_argument("--pstate", type=int, choices=(7,), default=7)
    args = parser.parse_args()
    variants = (
        ("grouped", "per-query") if args.variant == "compare" else (args.variant,)
    )
    args.out.mkdir(parents=True, exist_ok=False)
    if args.compile_only:
        shape = jax.ShapeDtypeStruct
        bank = shape((608256, 128), jnp.int32)
        inputs = (
            shape((288, 32, 128), jnp.float8_e4m3fn),
            shape((288, 32), jnp.float32),
            (bank, bank),
            shape((48, 768), jnp.int32),
            shape((288,), jnp.int32),
            shape((49,), jnp.int32),
            shape((288, 32), jnp.float32),
        )
        per_query = (
            inputs[0],
            inputs[1],
            inputs[2],
            shape((288, 768), jnp.int32),
            inputs[4],
            shape((288,), jnp.int32),
            inputs[6],
        )
        for variant in variants:
            data = per_query if variant == "per-query" else inputs
            compile_kernel(
                data,
                args.out,
                query_size=inputs[0].shape[0] // inputs[3].shape[0],
                variant=variant,
            )
        return
    flags = os.environ.get("LIBTPU_INIT_ARGS", "")
    assert [f for f in flags.split() if f.startswith("--xla_tpu_dvfs_p_state=")] == [
        "--xla_tpu_dvfs_p_state=7"
    ]
    versions = {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")}
    (args.out / "environment.json").write_text(
        json.dumps(
            dict(
                versions=versions,
                flags=flags,
                device=str(jax.devices()[0]),
                variants=variants,
                queries=288,
                requests=48,
                width=98304,
                page_table_expansion_timed=False,
            ),
            indent=2,
        )
    )
    packed = types.SimpleNamespace(pack_pages=pack_fp8_pages)
    executables = {}
    with jax.default_device(jax.devices()[0]):
        assert jax.default_backend() == "tpu"
        for seed in (2026, 2027):
            inputs = metadata.make_inputs(seed, packed)
            calls = {}
            for variant in variants:
                data = per_query_inputs(inputs) if variant == "per-query" else inputs
                if variant not in executables:
                    executables[variant] = compile_kernel(
                        data,
                        args.out,
                        query_size=inputs[0].shape[0] // inputs[3].shape[0],
                        variant=variant,
                    )
                calls[variant.replace("-", "_")] = functools.partial(
                    executables[variant], *data
                )
            correctness = None
            if args.variant == "compare":
                expected = np.asarray(calls["grouped"]())
                actual = np.asarray(calls["per_query"]())
                np.testing.assert_array_equal(
                    actual.view(np.uint32), expected.view(np.uint32)
                )
                correctness = True
                record = dict(seed=seed, bitwise=True, shape=list(actual.shape))
                (args.out / f"correctness_s{seed}.json").write_text(json.dumps(record))
                print("PASS", json.dumps(record), flush=True)
            compute.measure(
                calls,
                (),
                seed,
                args.out,
                args.pstate,
                correctness=correctness,
            )


if __name__ == "__main__":
    main()
