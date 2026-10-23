# Packed FP8 paged MQA

The maintained entry point is `packed_fp8_mqa_logits` in
`src/vllm_torchtpu/kernels/deepseek_v41/index_selection/packed_fp8_mqa.py`.
It replaces the `head42_boundary_overlap` experiment as the implementation to
import and modify. Historical snapshots remain available for comparisons.
The v2 model has an opt-in native-cache route. BF16 remains the default while
model quality and end-to-end performance are being validated.

## Model integration

Set `TPU_DEEPSEEK_V41_INDEXER_BACKEND=packed_fp8` before model initialization.
The selector controls the indexer's cache, independently of the main attention
KV-cache dtype. Use `bf16` in a separate process for the existing comparison
route; a live cache cannot be reinterpreted between the two formats.

The packed route requires TPU v7, H32/D128 index queries, and 128 compressed
index keys per physical page. Set the scheduler block size to
`128 * index compression ratio`; startup rejects a mismatch. For a checkpoint
whose index compression ratio is four:

```bash
# Add the same model, parallelism and serving options used in the baseline.
TPU_DEEPSEEK_V41_IMPL=v2 TPU_DEEPSEEK_V41_INDEXER_BACKEND=packed_fp8 \
vllm serve /path/to/checkpoint --block-size 512
```

Allocation retains upstream shared-owner placements and supports one or two
native banks. The index view reserves 132 bytes per key (FP8 values plus FP32
scale), and the adapter maps this view to the transposed page format below.
Incremental writes quantize only incoming K vectors and update their 33 words
in the corresponding token columns. TPU writes group updates by physical page,
read each affected page once, merge its columns in VMEM, and DMA it back into
the aliased cache. Eight page slots are staged together; untouched pages and
neighboring shared-owner regions are preserved. Query projection/rotation feeds the
scorer without the old FP8-to-BF16 round trip; the JAX bridge quantizes Q once
and passes its scales to the maintained kernel.

The paged model route preserves deferred writes, TP query splitting/gathering,
candidate publication and logical top-k IDs. Ragged queries are grouped by
request and restored to their original order. Score chunks use the existing
512-MiB score-element budget. The route does not materialize the BF16 candidate
K pool: it reads the original FP8 cache directly. Candidate IDs remain necessary.
Candidate consumers mark pages needed by each group of at most six queries from
the same request. Page slots retain their original logical positions; unused
pages are marked `-1`. Only valid pages issue HBM DMAs, and the receive wait counts
only the bytes actually requested. SparseCore builds unique, ascending candidate
block IDs. Page metadata records the compact output offset, count and inverse
permutation. TensorCore gathers scores within each 128-token page, writes them
into aligned compact VMEM windows, and transfers only the compact result to HBM.
The public sparse entry point,
`packed_fp8_sparse_mqa_logits`, returns `[queries, candidate_blocks * block_size]`
FP32 scores and their canonical block IDs. For 2048 eight-token blocks this is
16384 columns. Top-k maps compact column indices back to logical token IDs;
duplicate candidate IDs do not duplicate tokens in the result.
The model's unordered selection uses `packed_fp8_sparse_mqa_topk`: the same
16K candidate scores are read directly by SparseCore for selection, and only
logical TopK IDs are written back. Ordered selection uses the same compact logits.
Candidate publishers still score the full visible context. The whole-batch
short-context bypass and the existing all-blocks-fit identity are preserved.

Read granularity remains a 128-token physical page: a smaller candidate block
can bring other tokens on that page into the computation. Candidates that touch
every page still read every page. The arithmetic pipeline retains its unrolled
32768-key tiles, including page holes and padding in a partial tile. Preserving
logical page positions can compute more tiles than compacting selected pages;
the change removes metadata overhead while retaining the compute pipeline.
Entirely empty tiles skip arithmetic; partial tiles retain their unrolled
compute. Candidate consumers allocate one 32K VMEM score tile and two compact
VMEM output buffers. They allocate no full-context HBM score array and need no
separate full-width gather. For 2048 eight-token candidates at 96K context, the
FP32 score output is 64 KiB per query instead of 384 KiB.
Sparse calls leave skipped K pages unwritten and allocate no zero buffer.
QK and both head-reduction paths operate independently on each token column;
invalid pages are masked to `-inf` before top-k or any cross-token operation.
Their arbitrary values, including NaNs, cannot contribute to valid columns.
Do not add a cross-token reduction before this mask. Sparse pages can select a
different head-reduction path; bitwise agreement with dense scoring has not been
established. See the model integration measurements below for the logical-page
route's hardware timing.

To exercise the scorer, use a context exceeding `index_topk` compressed keys
and a model maximum length beyond that threshold; the existing all-indices-fit
shortcut deliberately bypasses scoring. The supported paged path also requires
the model's SWA window to be divisible by 64. The legacy eager path can decode
selected FP8 records, but uses its original FP32 scoring arithmetic and does
not exercise the optimized kernel.

Local integration checks cover incremental writes, shared ownership, empty
deferred writes, candidate publication/filtering, ragged query order, and the
Torch wrapper's dispatch arguments. CPU routing tests replace the TPU scorer
with a stand-in and are not numerical validation of the kernel. The earlier full-context-scratch route passed a sim-pjrt compile for the
complete JAX two-bank writer, native-page scorer, candidate-page preparation
and SparseCore top-k path at P-state 7.
Metadata tests compare page coverage, per-query masks and logical top-k IDs with
a token-level reference, including page holes, duplicates, empty candidates and
blocks crossing physical page boundaries. DMA descriptor tests verify that
three valid page tasks issue three HBM reads and wait for three pages, rather
than the full 256-page tile. Fragment tests poison skipped pages with FP8 NaNs,
infinite scales and random bits, including holes and fully invalid fragments.
They compare zero-filled and poisoned inputs through the production QK,
VPU/MXU reduction and store methods; valid outputs are identical and invalid
outputs are `-inf`. These CPU tests emulate the unsupported TPU pack/unpack/roll
primitives and are not hardware validation. Hardware validation of the JAX
Indexer stages is reported below; a complete model forward remains untested.
The local check
environment lacks vLLM/torch-tpu, so real custom-op registration and a model
forward still need the full runtime; the existing startup test now includes
the packed ops. Independent kernel-reference differences remain as documented
below.

## Calling the kernel

```python
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    pack_fp8_pages,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_mqa import (
    packed_fp8_mqa_logits,
)

# Encode when creating cache pages, outside the repeated scorer invocation.
# k0/k1: FP8 [complete_pages * 128, 128]; s0/s1: FP32 [complete_pages * 128].
cache = (pack_fp8_pages(k0, s0), pack_fp8_pages(k1, s1))
scores = packed_fp8_mqa_logits(
    q, weights, cache, page_ids, lengths, query_offsets, q_scale=q_scale
)
```

The measured policy is TPU v7, H32/D128, six resident Q rows, 32768 keys/tile,
eight pending fragments and four K buffers. It accepts one or two cache banks;
two banks were benchmarked. Total Q rows must be at least six unless empty.
Individual groups may be empty or have fewer than six queries. Context capacity
must be a multiple of 32768. The kernel returns dense FP32 scores, including
`-inf` for page holes and tokens outside each query's visible length.

## Cache bytes and partial pages

Each page occupies int32 `[33,128]` (16,896 bytes):

- `word[d // 4, token]`, byte `d % 4`, contains the FP8 bits of `K[token,d]`.
- `word[32, token]` contains that token's FP32 scale bits.

After the native Pallas bitcast, K has shape `[dim,token]`. Two pages concatenate
to `[128,256]`, matching the RHS of the QK dot without a separate transpose.
Page IDs, bank boundaries and the order of scales are unaffected.
Ordinary `[token,dim]` packed pages have the same shape but incompatible bytes;
the caller must use the matching writer.

A page with 17 valid tokens still occupies a complete page. Pad its unused K
rows with finite zeros and scales with one before calling `pack_fp8_pages`, and
set `lengths` to the actual visible context length. The scorer still transfers
and computes the full page, then masks unused tokens. `pack_fp8_pages` encodes
complete pages. `write_fp8_records` is the model's incremental writer; it preserves
other token columns and all neighboring cache placements. Native owners are
zero-initialized at allocation. Do not convert the entire cache on every scorer
call.

## Where to change things

| File | Responsibility |
| --- | --- |
| `packed_fp8_cache.py` | Page placement, byte format, complete-page encoder and incremental writer |
| `_packed_fp8_write.py` | Grouped page read/modify/write with batched DMA |
| `packed_fp8_mqa.py` | Public API, metadata preparation, per-query operations and DMA descriptors |
| `packed_fp8_sparse.py` | Candidate page validity and per-query block masks |
| `_packed_fp8_packing.py` | Page-local candidate permutations and compact output offsets |
| `_packed_fp8_math.py` | Head permutation and BF16 products / FP32 head accumulation |
| `_packed_fp8_pipeline.py` | Buffer allocation, fragment schedule and DMA ownership |
| `tuned_block_sizes.py:PACKED_FP8_MQA_CONFIG` | Measured static policy |
| `models/vllm/deepseek_v41/v2/packed_fp8_indexer.py` | Native-owner adapter, Q quantization, ragged grouping and top-k bridge |

The schedule and numerical helpers execute at tracing time. Keep fragment loops
as Python `range` loops: all 128 fragments, including startup and drain, are
unrolled. Keep the order of operations in `emit_tile`: first next-page DMA,
QK issue, older-fragment reduction/store, second next-page DMA. Extracting Python
helpers is safe only when the generated code is checked.

Q has two buffers. Page metadata has four buffers with two-query lookahead.
Dense output has two tile buffers per query parity (four total). Candidate
output reuses one compute tile and has one compact buffer per query parity.
Compact packing uses runtime loops over source pages and destination windows;
it does not unroll the 2048 candidate blocks. The destination range covers all
six query rows, so rows with different output offsets can cause empty masked
stores. Static bundle counts alone do not bound this runtime overhead.
The final output transfers of Q block n overlap Q block n+1. Reusing a parity
waits for its recorded rows first; kernel exit drains both parities. Dense
short-context padding first drains the current outputs. These extra buffers
are VMEM/SMEM buffers, not additional HBM banks.

Head order arranges four queries in the first 128 rows and two in the last 64.
Odd fragments before the final eight use an MXU selector reduction; other
fragments use VPU reductions. This division overlaps resources and is deliberate.
Weighted BF16 products for valid tokens must be finite: dense zero/one selectors
do not preserve independent-query NaN/Inf behavior within a token column.

## Validation and retained performance

Falcon experiment `exp-4onsyjb3t0`, P-state 7, JAX/jaxlib 0.11.2 and libtpu 0.0.50,
compared the former best with the boundary-overlap implementation:

| Seed | Former best, kernel µs | Boundary overlap, kernel µs |
| --- | ---: | ---: |
| 2026 | 395.0174 | 323.4880 |
| 2027 | 395.3746 | 323.4394 |

Each row contains 40 samples per variant in an interleaved comparison. Cache
encoding is outside timing. Kernel time excludes the separate metadata-preparation
kernels; complete module medians were about 402.6 µs and 330.8 µs respectively.
All six full-output comparisons (two seeds, normal/edge/ragged inputs) were
bitwise equal between the two scheduling variants.

Independent numerical correctness is still unresolved: both scheduling variants
have the same sampled reference failures (maximum absolute errors approximately
0.1762 and 0.1839 for the two seeds). Their masks and finite-value classification
matched. Equivalence to the previous implementation is not a claim that this
kernel passes its independent reference or model-level evaluation.

The refactor was checked with local sim-pjrt. After removing source comments,
virtual SSA identifiers and debug marker IDs, all 17,545 static scheduled bundles
matched the measured boundary-overlap snapshot exactly, including physical
registers, operands, branch targets and empty bundles. Complete compute paths,
including their entry/exit branches and verified delay slots:

| Path | Bundles |
| --- | ---: |
| Full tile, prefetch | 4093 |
| Full tile, no prefetch | 3971 |
| Masked tile, prefetch | 4127 |
| Masked tile, no prefetch | 3998 |

Each path retains 768 FP8 matrix instructions and 3072 BF16 weighting instructions.
Prefetch paths retain 512 predicated bank alternatives for 256 logical page DMAs.
These are scheduled bundle counts, not measured stall cycles. The 3500-bundle
optimization target remains unmet. The refactor has not been rerun on hardware;
the 323 µs result belongs to the equivalent pre-refactor source.

The candidate integration was also compiled at P-state 7 with identical
benchmark shapes: Q `[288,32,128]`, page table `[48,768]`, and two packed banks
of `[608256,128]` words. Its dense specialization remains instruction-identical
to the measured baseline after removing debug identities. The sparse
specialization combines the page-valid bit and bank bit into one tag; tag 2
issues bank 0 and tag 3 issues bank 1. This avoids nested predicate combinations.
Assembly-verified selected paths, including branch delay slots:

| Path | Baseline | Sparse |
| --- | ---: | ---: |
| Full tile, prefetch | 4093 | 4084 |
| Full tile, no prefetch | 3971 | 3971 |
| Masked tile, prefetch | 4127 | 4113 |
| Masked tile, no prefetch | 3998 | 3998 |

These compute bodies contain no DMA waits; the two prefetch bodies contain no
internal control-flow branches. In that revision, sparse calls had the same
explicit scratch buffers as dense calls, with no K zero-initialization or VMEM
copy loops.
The entire sparse kernel contains 17,536 static bundles; the dense kernel still
matches the measured baseline's 17,545 bundles instruction-for-instruction
after debug normalization. Static totals include mutually exclusive paths and
are not execution-cycle counts. Candidate metadata preparation, DMA stalls and
model integration still require hardware timing; fewer compute bundles alone
do not establish end-to-end speedup.

## Model integration timing

Measured on TPU v7x at P-state 7, using one core, 48 requests with six queries
each, 98,304 context keys, H32/D128, two native cache banks, top-k 512, and 2,048
candidate blocks of eight keys. Each cell is the mean of the two seed medians
(2026/2027), with 30 XProf device-duration samples per seed and variant.

The previous FP8 adapter is commit `c2c55f990`; the logical-page adapter is
`a7225b918`. The BF16 column is the same-run control for the latter.

| Indexer call | Previous FP8 (ms) | Logical-page FP8 (ms) | BF16 (ms) |
| --- | ---: | ---: | ---: |
| Full context | 1.901 | 1.897 | 1.047 |
| Publish candidates | 2.242 | 2.237 | 1.104 |
| Filter clustered candidates | 1304.173 | 3.840 | 6.714 |
| Filter scattered candidates | 1306.597 | 3.942 | 6.764 |

The sparse path is 1.75x/1.72x faster than the BF16 control. The previous
per-token Boolean membership gather took about 1.296 seconds; it is absent
from the logical-page route. MQA scoring remains about 321 microseconds.
Candidate block-mask scatter still costs about 1.880 ms, and the two incremental
cache-write scatters together cost about 0.79 ms. Full-context and publication
calls remain slower than BF16; this change targets candidate filtering.

Timing includes K writes, Q quantization, metadata, scoring and top-k. It excludes
projections, RoPE, Torch dispatch and TP collectives. CPU token-reference tests
and hardware ID constraints passed; this is not a model-quality evaluation.
The scorer, math and pipeline sources are unchanged. Same-shape sim-pjrt LLO
comparisons retain all 17,536 sparse and 17,545 dense bundles instruction-for-
instruction after removing debug identities.

### Scatter follow-up measured and reverted

Commit `18c83ac12` tried batching candidate membership construction by query and
expressing each cache record as one 33-word column scatter window. It passed
42 CPU tests and retained identical scorer LLO (17536 sparse / 17545 dense
bundles), but P-state-7 experiment `exp-x62w5dthec` showed a regression:

| Case | Prior best (ms) | Window scatter (ms) | Regression |
| --- | ---: | ---: | ---: |
| full | 1.897 | 3.364 | +77.3% |
| publish | 2.237 | 3.729 | +66.7% |
| filter_clustered | 3.840 | 5.840 | +52.1% |
| filter_scattered | 3.942 | 5.943 | +50.8% |

Values are means of two seed medians, with 30 device samples per seed and case,
from analysis `an-fywgpzhzek`. Candidate mask scatter remained about 1.880 ms;
the cache update trace added loops and large copies. The two scatter helpers
were restored byte-for-byte to measured best commit `a7225b918` before the
page-DMA follow-up below. The SparseCore mask prototype has only compiled under
sim-pjrt and is not integrated.

### Page-DMA cache writer retained

Commit `626631300` replaces the TPU cache scatter with explicit Pallas page reads
and writes, grouping updates to the same page before merging them in VMEM.
P-state-7 experiment `exp-eh1vxy4a2j`, analyzed by `an-u9mla6c1nh`, compares the
previous scatter writer with this writer in the same run:

| Indexer call | Scatter writer (ms) | Page-DMA writer (ms) | Reduction |
| --- | ---: | ---: | ---: |
| Full context | 1.899 | 1.161 | 38.8% |
| Publish candidates | 2.238 | 1.501 | 32.9% |
| Filter clustered candidates | 3.840 | 3.106 | 19.1% |
| Filter scattered candidates | 3.944 | 3.209 | 18.6% |

These are means of seed-2026/2027 medians, with 30 XProf device-duration samples
per seed and variant at the shape above. In the full-context trace, the two cache
write operators together fall from 791.8 to 34.5 microseconds. This operator-only
number excludes grouping and preparation; the complete call saves about
735 microseconds. Candidate mask scatter remains about 1.880 ms in the filter
path, and MQA scoring remains about 321 microseconds. The scorer sources and
same-shape compiled LLO instructions are unchanged.

For clustered candidates, analysis `an-3lhtrw91or` exposes the complete recorded
operation table. Averaging per-operation medians across the two seeds gives:

| Component | Time (ms) |
| --- | ---: |
| Candidate block-mask scatter | 1.880 |
| MQA scoring | 0.321 |
| Two SparseCore top-k calls | 0.368 |
| SparseCore data-format call | 0.093 |
| `bitcast-convert_bitcast_fusion.1` and `copy.44` | 0.156 |
| `sort.14` | 0.036 |
| Two page-DMA cache writers | 0.035 |
| Other recorded operations | 0.184 |
| Module latency minus the sum of operation medians | 0.033 |
| Complete Indexer call | 3.106 |

The `cond.5.clone.1` event encloses child operations and is excluded from the sum.
This is an approximate accounting of operation medians, not an exclusive timeline
attribution; the 33-microsecond residual is not established to be wait time.
The sparse scorer skips unused KV pages but retains full logical-position logits,
so candidate masking and top-k still process full-width arrays.

Validation passed 42 CPU tests and eight applicable TPU cache tests. Four local
allocator tests require the unavailable upstream vLLM interface. Two exhaustive
CPU FP8 bit-pattern tests were excluded from the TPU retry after the first run
showed negative-NaN canonicalization on TPU; the tests and packing implementation
are unchanged. Both cache banks, selected IDs and published candidates match
exactly between writers in all eight case/seed pairs. Local interpreted
byte-preservation tests and complete filter-path sim-pjrt compilation also pass.

Use `bench_packed_fp8_layer.py --compare-cache-writers --profile --out <dir>`
to compare both FP8 writers in the same run. The benchmark checks exact equality
of both cache banks, selected IDs and published candidates before measuring.

### Compact candidate scores

P-state-7 experiments `exp-vhy0bbtozb` and `exp-j4e5uvig2m` compare against the
page-DMA writer baseline at commit `c590a5009`. They retain the same 288 queries,
96K context, 16K candidate tokens, two cache banks, and two random seeds.
`filter_clustered` uses one contiguous 16K candidate range per request;
`filter_scattered` independently distributes each query's 16K candidates over
the 96K context. These are synthetic coverage cases, not measured model
candidate distributions. Both have a 16K-column compact score output. These
labels describe logical token positions: both fixtures randomize physical page
placement. Scattered candidates can touch almost every 128-token page despite
selecting only 16K tokens; page reads and compact score width are different
quantities.

| Variant | Clustered (microseconds) | Scattered (microseconds) |
| --- | ---: | ---: |
| Full-width baseline, second run | 3107.511 | 3211.651 |
| Compact scores, first run | 880.809 | 880.567 |
| Compact scores and empty-tile bypass, second run | 872.366 | 881.827 |
| Metadata and empty-tile prefetch fixes, third run | 618.784 | 735.342 |
| Fused compaction and selection, fourth run | 596.299 | 715.920 |
| Fused selection with vector copy and loop unrolling, fifth run | 552.375 | 672.085 |

Numbers are means of the two seed medians, each with 30 XProf device samples.
Declared operator analyses are `an-07cf7q2xvw` and `an-9e885t80wd`; stage-sample
analyses are `an-86ea8dsk2b` and `an-6fyttey0kp`. All four outputs (two cache
banks, selected IDs and candidate IDs) match the historical implementation
exactly for every case and seed. The first compact version's normalized scorer
LLO instructions match the baseline byte-for-byte. The four compute regions
contain 4081/3968/4110/4051 bundles; with empty-tile bypass the region boundaries
contain 4081/3968/4110/3995 bundles. These are static region counts, not complete
runtime cycles or a prediction of latency.

The second run's clustered seed-2026 trace has 315.1 microseconds of MQA scoring,
61.3 of candidate membership, 83.6 of score compaction and 88.2 of top-k. Two
remaining metadata scatters account for 64.5 microseconds, sorting for 35.7 and
logical-ID remapping for 13.5. The third experiment, `exp-sksivkv699`, removes
the metadata scatters, fuses sorting/remapping into TopK, unrolls SC loops and
empty-tile prefetch, and skips metadata walks for empty future tiles. Analyses
`an-m2mdkt2fjt` and `an-9anszs9xns` confirm its results. All four outputs match
the baseline exactly for both candidate distributions and both seeds. The
clustered seed-2026 trace has 194.9 microseconds of scoring, 28.6 of membership,
73.5 of compaction and 145.6 of fused TopK/sort/remapping. The two clustered
seed medians are 607.5 and 630.1 microseconds, so report both rather than only
the faster seed. The compute-region bundle counts remain unchanged from the
second run. Commit `0ebac7883` adds the public sparse API wrapper to measured
commit `f004121e5`; its normalized scorer LLO is identical.

Experiments `exp-5whv0q7crg` and `exp-n4clwbtmj2` test fusion of score compaction
and TopK, then replace serial candidate packing with a vector copy and unroll
the selection loops. The fifth run's seed medians are 542.5/562.3 microseconds
for clustered candidates and 669.8/674.4 for scattered candidates. Analyses
`an-t044oyh8al` and `an-b1asisa722` expose its metrics and stage samples. Its
clustered seed-2026 trace has 195.4 microseconds of scoring and 158.5 of fused
candidate selection. Both cache banks, selected IDs and candidates match the
third version exactly; separate TPU checks cover empty rows, score ties,
short rows and noncontiguous block IDs.

Commit `d4b5c6c19` moves this selector into the kernel package and connects the
model's unordered path directly, removing the benchmark's temporary wiring.
Forty CPU tests and the relevant pre-commit hooks pass. The same-shape TensorCore
scorer LLO and SparseCore dispatch LLO match the measured source after stripping
comments. A full model forward still requires the real vLLM/torch-tpu runtime.
The 500-microsecond target has **not** been demonstrated. These timings cover JAX
cache writes, Q quantization, metadata, scoring and selection; they exclude
projections, RoPE, Torch dispatch and TP collectives.

Use `bench_packed_fp8_layer.py --sparse-baseline <baseline-checkout> --profile
--out <dir>` for the same-run comparison. The baseline checkout must contain
the historical full-width scorer, pipeline, candidate metadata, selection helpers
and model entry point. The benchmark loads these dependencies together to avoid
mixing variants.

## Reproduce

The independent dense copy in `packed_fp8_dense.py` starts from commit
`842880870`. Its `packed_fp8_paged_mqa_logits` entry follows vLLM's Blackwell
row/request contract: Q `[T,32,128]`, page tables `[T,P]`, context lengths
`[T]`, and request IDs `indices[T]`. Consecutive queries with the same request
ID share a page-table row inside the kernel. The required static `query_size`
comes from the model's configured decode width: `num_speculative_tokens + 1`,
or 1 without speculation. The tile size is `min(query_size, 8)`; longer runs
are split at that boundary. Buffers and head reductions follow this size.
The caller supplies identical page-table rows for queries of one request.
The kernel prefetches only the first page-table row of each Q tile, encodes it
once, and shares it across that tile's queries. Four staging rows replace the
resident full-batch encoded table. Each query retains its own history length.
Q scales are `[T,32]`; the singleton axes of the GPU API are omitted.
No candidate list, candidate-page union, or compact-score packing is involved.

Output is FP32 `[T,max_model_len]` in logical token order, with invalid positions
set to `-inf`. This preserves the TPU FP8 cache layout and arithmetic; it does
not adopt NVIDIA's MXFP4 bytes or BF16 output. Physical page-table capacity must
cover `max_model_len` rounded up to 32768 tokens. The dense copy is available for
independent validation; model dispatch still uses the existing implementation.
Use `bench_packed_fp8_mqa.py --variant per-query --out <dir>` to benchmark it,
or add `--compile-only` to compile without execution. Benchmark page-table
expansion happens before the timed call. This homogeneous benchmark supplies
its fixture's six queries per request; the kernel has no six-query default.

Use `--variant compare` for an interleaved grouped/per-query A/B on the same
device. Both seeds must produce bitwise-identical FP32 scores before timing.
The benchmark saves each variant's HLO and memory sizes alongside raw XProf
samples. To compare against a historical grouped implementation, apply only
the dense copy and benchmark changes on that exact baseline checkout.

```bash
# Layout tests can run on CPU.
JAX_PLATFORMS=cpu pytest -q tests/kernels/test_dsv41_packed_fp8_cache.py

# On a TPU v7 host; use a fresh output directory.
LIBTPU_INIT_ARGS='--xla_tpu_dvfs_p_state=7' \
python scripts/dsv41_bench/bench_packed_fp8_mqa.py --out /tmp/packed-fp8-mqa
```

The maintained benchmark imports the source directly and needs no experiment
source directory. Single-variant runs report timing only (`correctness: null`);
`--variant compare` verifies bitwise parity. Use
`--compile-only` under sim-pjrt to compile the fixed shape without device data.
