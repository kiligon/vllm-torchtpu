# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare BF16 and packed FP8 Indexer calls, including writes and selection.

Runs the production JAX entry points on one TPU core. Projection, RoPE, Torch
dispatch and TP collectives are outside this benchmark. Inputs are shared logical
Q/K values; each backend owns its native two-bank cache. Cache donation matches
the serving bridge, so repeated calls do not copy the whole cache.
"""

import argparse
import functools
import gzip
import importlib.metadata
import importlib.util
import json
import logging
import os
import statistics
import sys
import time
import types
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_model():
    """Bypass serving registration; import the actual cache/scorer functions."""
    for suffix in (
        "",
        ".models",
        ".models.vllm",
        ".models.vllm.deepseek_v41",
        ".models.vllm.deepseek_v41.v2",
    ):
        name = "vllm_torchtpu" + suffix
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / "src" / name.replace(".", "/"))]
        sys.modules.setdefault(name, package)
    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.utils"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules.setdefault(name, package)

    def no_allocator(*args, **kwargs):
        raise RuntimeError("This benchmark supplies native cache owners directly")

    sys.modules["vllm.v1.worker.utils"].allocate_kv_cache = no_allocator
    logger_module = types.ModuleType("vllm_torchtpu.logger")
    logger = logging.getLogger("indexer_layer")
    logger.warning_once = functools.lru_cache(None)(logger.warning)
    logger_module.init_logger = lambda name: logger
    sys.modules[logger_module.__name__] = logger_module
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        attention_kernels,
        packed_fp8_indexer,
    )

    return attention_kernels, packed_fp8_indexer


def load_sparse_baseline(source: Path):
    """Load historical scorer sources supplied for an exact FP8 A/B."""
    originals = {}
    try:
        for suffix in (
            "kernels.deepseek_v41.index_selection._packed_fp8_compact",
            "kernels.deepseek_v41.index_selection._packed_fp8_select",
            "kernels.deepseek_v41.index_selection._packed_fp8_pipeline",
            "kernels.deepseek_v41.index_selection.packed_fp8_mqa",
            "kernels.deepseek_v41.index_selection.packed_fp8_sparse",
            "models.vllm.deepseek_v41.v2.packed_fp8_indexer",
        ):
            name = "vllm_torchtpu." + suffix
            path = source / "src" / (name.replace(".", "/") + ".py")
            spec = importlib.util.spec_from_file_location(name + "_baseline", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            originals[name] = sys.modules.get(name)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        for name, original in originals.items():
            if original is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original


def round_fp8(values):
    from vllm_torchtpu.kernels.deepseek_v41.index_quantization import (
        quantize_index_rows,
    )

    values, scales, _ = quantize_index_rows(values)
    return (values.astype(jnp.float32) * scales[..., None]).astype(jnp.bfloat16)


def abstract_inputs(requests, queries, width, variant):
    shape = jax.ShapeDtypeStruct
    tokens = requests * queries
    bank_rows = (
        requests * width // 4 if variant == "bf16" else requests * width // 256 * 33
    )
    bank = shape((bank_rows, 128), jnp.int32)
    return (
        shape((tokens, 32, 128), jnp.bfloat16),
        bank,
        bank,
        shape((requests, width), jnp.int32),
        shape((tokens, 32), jnp.float32),
        shape((tokens,), jnp.int32),
        shape((tokens, 2048), jnp.int32),
        shape((requests * 2,), jnp.int32),
        shape((requests * 2, 128), jnp.bfloat16),
        shape((tokens,), jnp.int32),
        shape((1,), jnp.int32),
    )


def fixture(requests, queries, width, seed):
    from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout
    from vllm_torchtpu.kernels.deepseek_v41.index_quantization import (
        quantize_index_rows,
    )
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
        pack_fp8_pages,
    )

    rng = np.random.default_rng(seed)
    count, tokens = requests * width, requests * queries
    q_key, k_key, w_key, new_key = jax.random.split(jax.random.key(seed), 4)
    q = (jax.random.normal(q_key, (tokens, 32, 128)) * 0.25).astype(jnp.bfloat16)
    keys = (jax.random.normal(k_key, (count, 128)) * 0.25).astype(jnp.bfloat16)
    fp8, scales, _ = quantize_index_rows(keys)
    packed = pack_fp8_pages(fp8, scales)
    rounded = (fp8.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)
    words = index_key_layout.encode_records(rounded)
    words = (words[::2] | (words[1::2] << 16)).astype(jnp.int32)
    pools = {
        name: tuple(jnp.split(value, 2))
        for name, value in (("bf16", words), ("packed_fp8", packed))
    }
    pages = rng.permutation(count // 128).astype(np.int32).reshape(requests, -1)
    records = jnp.asarray(
        (pages[:, :, None] * 128 + np.arange(128)).reshape(requests, width), jnp.int32
    )
    lengths = jnp.tile(
        width - (queries - 1 - jnp.arange(queries)) // 4, requests
    ).astype(jnp.int32)
    owners = jnp.repeat(jnp.arange(requests, dtype=jnp.int32), queries)
    weights = jax.random.normal(w_key, (tokens, 32)) * 0.02
    write_records = records[:, -2:].reshape(-1)
    new_keys = (jax.random.normal(new_key, (requests * 2, 128)) * 0.25).astype(
        jnp.bfloat16
    )
    common = (
        q,
        records,
        weights,
        lengths,
        write_records,
        new_keys,
        owners,
        lengths.max().reshape(1),
    )
    # Scattered candidates generally touch most pages; clustered candidates
    # expose the savings when the page union occupies one 32768-key tile.
    scattered = np.stack(
        [rng.choice(width // 8, 2048, replace=False) for _ in range(tokens)]
    )
    starts = rng.integers(0, width // 8 - 2048 + 1, size=requests)
    clustered = np.repeat(starts, queries)[:, None] + np.arange(2048)
    candidates = {
        "scattered": jnp.asarray(scattered, jnp.int32),
        "clustered": jnp.asarray(clustered, jnp.int32),
    }
    return pools, common, candidates


def compile_arm(modules, variant, mode, requests, queries, width, out):
    baseline, candidate = modules
    if variant in (
        "packed_fp8_baseline",
        "packed_fp8_full_scores",
        "packed_fp8_separate_selection",
        "packed_fp8_unsplit",
    ):
        candidate = baseline
    write, filtering = mode == "publish", mode == "filter"

    def run(
        q,
        lo,
        hi,
        records,
        weights,
        lengths,
        candidates,
        write_records,
        keys,
        owners,
        batch_max,
    ):
        if variant == "bf16":
            return baseline.update_and_select_2bank_jax(
                round_fp8(q),
                lo,
                hi,
                records,
                weights,
                lengths,
                candidates,
                write_records,
                round_fp8(keys),
                owners,
                batch_max,
                512,
                8,
                write,
                filtering,
                True,
                128,
            )
        return candidate._select_two(
            q,
            lo,
            hi,
            records,
            weights,
            lengths,
            candidates,
            write_records,
            keys,
            owners,
            batch_max,
            requests * width // 128,
            33,
            0,
            512,
            8,
            write,
            filtering,
            True,
        )

    run.__name__ = f"indexer_layer_{variant}_{mode}"
    started = time.monotonic()

    def compile_run():
        return (
            jax.jit(run, donate_argnums=(1, 2))
            .lower(*abstract_inputs(requests, queries, width, variant))
            .compile()
        )

    if variant == "packed_fp8_scatter":
        from vllm_torchtpu.kernels.deepseek_v41.index_selection import packed_fp8_cache

        def scatter(bank, bases, columns, values):
            rows = bases[:, None] + jnp.arange(values.shape[1])[None, :]
            valid = (bases >= 0) & (bases + values.shape[1] <= bank.shape[0])
            rows = jnp.where(valid[:, None], rows, bank.shape[0])
            return bank.at[rows, columns[:, None]].set(values, mode="drop")

        with mock.patch.object(packed_fp8_cache, "write_packed_pages", scatter):
            compiled = compile_run()
    else:
        compiled = compile_run()
    (out / f"{variant}_{mode}.hlo.txt").write_text(compiled.as_text())
    memory = compiled.memory_analysis()
    assert memory is not None
    (out / f"{variant}_{mode}.memory.json").write_text(
        json.dumps(
            dict(
                argument_bytes=memory.argument_size_in_bytes,
                output_bytes=memory.output_size_in_bytes,
                alias_bytes=memory.alias_size_in_bytes,
                temporary_bytes=memory.temp_size_in_bytes,
            ),
            indent=2,
        )
    )
    print(
        json.dumps(dict(compiled=run.__name__, seconds=time.monotonic() - started)),
        flush=True,
    )
    return compiled


def profile_rows(path, expected):
    (trace,) = path.rglob("*.trace.json.gz")
    with gzip.open(trace, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    devices = {
        e["pid"]
        for e in events
        if e.get("name") == "process_name" and e["args"].get("name") == "/device:TPU:0"
    }
    lanes = {
        (e["pid"], e["tid"]): e["args"].get("name")
        for e in events
        if e.get("name") == "thread_name"
    }
    modules, operations = [], []
    for event in events:
        if event.get("pid") not in devices or "dur" not in event:
            continue
        lane = lanes.get((event["pid"], event["tid"]))
        if lane == "XLA Modules" and "indexer_layer_" in event["name"]:
            modules.append(event)
        elif lane == "XLA Ops":
            operations.append(event)
    modules.sort(key=lambda e: e["ts"])
    if len(modules) != len(expected):
        raise ValueError(
            f"Incomplete trace: got {len(modules)}, expected {len(expected)}"
        )
    rows = []
    for module, variant in zip(modules, expected, strict=True):
        assert f"indexer_layer_{variant}_" in module["name"], module["name"]
        enclosed = [
            e
            for e in operations
            if module["ts"] <= e["ts"]
            and e["ts"] + e["dur"] <= module["ts"] + module["dur"]
        ]
        rows.append(
            dict(
                variant=variant,
                module_us=int(module["args"]["device_duration_ps"]) / 1e6,
                operations=[
                    dict(
                        name=e["name"],
                        duration_us=int(e["args"]["device_duration_ps"]) / 1e6,
                        trace_start_us=e["ts"],
                        trace_duration_us=e["dur"],
                    )
                    for e in enclosed
                    if "device_duration_ps" in e["args"]
                ],
            )
        )
    return rows


def validate_outputs(outputs, lengths, candidates, filtering):
    sets = {}
    for variant, result in outputs.items():
        selected = np.asarray(result[2])
        valid = selected >= 0
        assert np.all(~valid | (selected < np.asarray(lengths)[:, None]))
        sets[variant] = [set(row[row >= 0].tolist()) for row in selected]
        assert all(len(row) == 512 for row in sets[variant])
        if filtering:
            assert all(
                {i // 8 for i in row} <= set(allowed.tolist())
                for row, allowed in zip(sets[variant], np.asarray(candidates))
            )
    reference, candidate = sets.values()
    overlap = [len(a & b) / 512 for a, b in zip(reference, candidate)]
    for baseline in (
        "packed_fp8_baseline",
        "packed_fp8_scatter",
        "packed_fp8_full_scores",
        "packed_fp8_separate_selection",
        "packed_fp8_unsplit",
    ):
        if baseline in outputs:
            for before, after in zip(outputs[baseline], outputs["packed_fp8"]):
                assert bool(jnp.array_equal(before, after)), "FP8 output changed"
    return dict(
        topk_overlap_mean=float(np.mean(overlap)), topk_overlap_min=min(overlap)
    )


def measure(runs, pools, common, candidates, case, seed, out, profile):
    q, records, weights, lengths, write_records, keys, owners, batch_max = common
    outputs = {}

    def call(variant):
        result = jax.block_until_ready(
            runs[variant](
                q,
                *pools[variant],
                records,
                weights,
                lengths,
                candidates,
                write_records,
                keys,
                owners,
                batch_max,
            )
        )
        pools[variant] = result[:2]
        outputs[variant] = result

    reference, candidate = runs
    order = [reference, candidate, candidate, reference]
    for variant in order * 3:
        call(variant)
    correctness = validate_outputs(
        outputs, lengths, candidates, case.startswith("filter")
    )
    samples = []
    sentinel = jax.jit(lambda x: x + 1)
    value = jnp.int32(0)
    jax.block_until_ready(sentinel(value))
    for capture in range(3):
        if profile:
            for attempt in range(3):
                path = out / "profiling/xprof" / f"{case}_s{seed}_{capture}_{attempt}"
                with jax.profiler.trace(str(path)):
                    time.sleep(0.1)
                    jax.block_until_ready(sentinel(value))
                    for variant in order * 5:
                        call(variant)
                        jax.block_until_ready(sentinel(value))
                    time.sleep(0.1)
                try:
                    samples.extend(profile_rows(path, order * 5))
                    break
                except ValueError:
                    if attempt == 2:
                        raise
        else:
            for variant in order * 5:
                start = time.perf_counter()
                call(variant)
                samples.append(
                    dict(variant=variant, module_us=(time.perf_counter() - start) * 1e6)
                )
    (out / "benchmark" / f"{case}_s{seed}_samples.json").write_text(json.dumps(samples))
    for variant in runs:
        values = [r["module_us"] for r in samples if r["variant"] == variant]
        row = dict(
            variant=variant,
            case=case,
            seed=seed,
            latency_ms=statistics.median(values) / 1000,
            p10_us=float(np.percentile(values, 10)),
            p90_us=float(np.percentile(values, 90)),
            samples=len(values),
            tokens=q.shape[0],
            batch_size=records.shape[0],
            seq_len=records.shape[1],
            head_dim=128,
            pstate=7,
            cache_banks=2,
            timing="XProf device_duration_ps" if profile else "host wall time",
            **correctness,
        )
        with (out / "benchmark/metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--width", type=int, default=98304)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compare-cache-writers", action="store_true")
    parser.add_argument("--sparse-baseline", type=Path)
    args = parser.parse_args()
    modules = load_model()
    if args.sparse_baseline is not None:
        modules = (load_sparse_baseline(args.sparse_baseline), modules[1])
    for directory in ("benchmark", "compiler", "profiling"):
        (args.out / directory).mkdir(parents=True, exist_ok=True)
    variants = (
        "packed_fp8_baseline"
        if args.sparse_baseline is not None
        else "packed_fp8_scatter"
        if args.compare_cache_writers
        else "bf16",
        "packed_fp8",
    )
    compiled = {
        mode: {
            variant: compile_arm(
                modules,
                variant,
                mode,
                args.requests,
                args.queries,
                args.width,
                args.out / "compiler",
            )
            for variant in variants
        }
        for mode in (
            ("filter",)
            if args.sparse_baseline is not None
            else ("full", "publish", "filter")
        )
    }
    dependencies = sorted(
        str(Path(m.__file__).relative_to(ROOT))
        for name, m in sys.modules.items()
        if name.startswith("vllm_torchtpu.")
        and getattr(m, "__file__", None)
        and Path(m.__file__).is_relative_to(ROOT)
    )
    (args.out / "dependencies.json").write_text(json.dumps(dependencies, indent=2))
    if args.compile_only:
        return
    versions = {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")}
    assert versions == {"jax": "0.11.2", "jaxlib": "0.11.2", "libtpu": "0.0.50"}
    assert "--xla_tpu_dvfs_p_state=7" in os.environ.get("LIBTPU_INIT_ARGS", "").split()
    (args.out / "profiling/runtime.json").write_text(
        json.dumps(
            dict(
                versions=versions,
                device=str(jax.devices()[0]),
                devices_used=1,
                cache_donation=True,
            ),
            indent=2,
        )
    )
    with jax.default_device(jax.devices()[0]):
        for seed in args.seeds:
            pools, common, candidates = fixture(
                args.requests, args.queries, args.width, seed
            )
            if args.compare_cache_writers or args.sparse_baseline is not None:
                del pools["bf16"]
                pools[variants[0]] = tuple(x.copy() for x in pools["packed_fp8"])
            for mode, distribution in (
                ("full", "scattered"),
                ("publish", "scattered"),
                ("filter", "clustered"),
                ("filter", "scattered"),
            ):
                if mode not in compiled:
                    continue
                case = f"{mode}_{distribution}" if mode == "filter" else mode
                measure(
                    compiled[mode],
                    pools,
                    common,
                    candidates[distribution],
                    case,
                    seed,
                    args.out,
                    args.profile,
                )
            del pools, common, candidates


if __name__ == "__main__":
    main()
