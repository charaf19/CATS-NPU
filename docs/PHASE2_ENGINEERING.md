# Phase 2 performance engineering — 2026-10-23

This document records implementation evidence. It is not a manuscript and the
measurements are not confirmatory statistics. Raw artifacts are under
`results/phase2_audit/` and `results/phase2_pilot/`.

## Audit diagnosis

The original `--batch 1` NPU benchmark submits one `[1,16]` model for every
tile. At N=4096 this produces 256 synchronous NPU calls. Existing observations
showed that native hierarchical Top-K spent a median 86.0 ms of 93.0 ms in
backend calls; adaptive spent 80.8 ms of 86.5 ms there. The residual host work
was about 5.7–7.0 ms. This directly confirms that repeated small synchronous
submissions dominate the original implementation.

An isolated native TopK profile measured one prepared model at B=1, 8 and 32.
Median call latency was 0.603, 0.635 and 0.702 ms respectively. The amount of
work increased by 32x while call latency increased by only about 16%. Median
per-tile call cost fell from about 603 us to 22 us. This supports batching as a
systems optimization. It does not establish a new selection algorithm.

Compilation is excluded from steady-state latency. The first cold B=1 compile
in that diagnostic took about 2.66 s; warm-cache B=8/B=32 compiles took about
66–72 ms. These values are recorded separately and are cache-order dependent.

## Correctness and baseline repairs

### Portable C++ baseline

The machine-wide Visual Studio installation lacks a complete registered C++
workload/Windows SDK. The project now provides a portable build route through
project-local CMake, Ninja and Zig:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[native]'
.\scripts\08_build_cpp.ps1
```

The script configures CMake with Zig C++17, builds `native/build/cats_topk.dll`,
and immediately runs the native correctness test. The loader accepts both
`cats_topk.dll` and the older `libcats_topk.dll` name. Tests cover FP16/FP32,
normal, tied, near-tied and adversarial inputs, plus a deterministic tied
boundary case.

### Short final tiles

The selector previously capped a short tile's target before choosing a model K,
which generated unprepared K shapes during execution. All tiles in a refinement
round now use one shared request size. Short tiles are padded to the static
width and retain only valid local indices. Physical-NPU smoke tests passed for
N=513, K=10 with hierarchical selection and adaptive initial budget 3.

### Bitonic backend

The original boolean-valued `Select` lowered its data operands to i8, which the
NPU driver compiler rejects. The graph now converts the dynamic comparison
result and static direction mask to i32, then uses `Equal` only to create the
boolean condition for FP16/i32 score/index `Select` operations.

This change repairs graph compilation but not the backend:

- CPU bitonic regression tests pass.
- NPU `DRIVER` compiles the graph but returns `[0,0,0,0]` for the ascending
  smoke input, so it is numerically invalid.
- NPU `PLUGIN` passes the ordinary smoke input.
- NPU `PLUGIN` changes a selected positive FP16 subnormal
  `2.5331974e-05` to `0.0`; strict score/index validation rejects it.

The application-facing strict NPU backend refuses all FP32 inputs. An FP16 row
containing a nonzero subnormal receives measured exact CPU validation after
device inference; any score or ranking disagreement rejects the device result
without correction. This safety gate is based on reproduced failures on this
runtime/driver stack. It prevents an incorrect result from receiving an exact
certificate, but it adds host validation work and does not establish native
full-domain exactness or change the project-wide mathematical contract.

The failing physical-NPU case is retained as an opt-in strict-xfail regression
using `CATS_NPU_RUN_NPU_TESTS=1`. Validation was not weakened. Bitonic remains
experimental and must not be used for performance claims.

## Instrumentation

Each tile-selection result can now record:

- model compilation time, separately from steady-state timing;
- host tile staging time;
- backend input cast/contiguity time;
- synchronous backend execution time;
- output retrieval/copy time;
- score/index validation time;
- count of rows receiving exact CPU validation because they contain FP16
  subnormals;
- candidate extraction and tie-repair time;
- global merge time;
- certification/refinement scheduling time;
- backend calls and physical-device submissions;
- staged input bytes and retrieved output bytes;
- total host-input-to-host-output latency.

`backend_execute_sync_ns` is a host wall-clock interval around synchronous
OpenVINO inference. It may include submission, transfers, device execution and
waiting. It is not a hardware kernel timer. Input and output transfer costs are
not claimed separately unless the runtime exposes a directly measured boundary.

Benchmark summaries now carry `completed` and `valid_for_inference`. Preflight
mismatches save `.npz` inputs/expected/obtained indices, and warm-up/runtime
failures are structured skip records. Analysis preserves correct observations
from failed sessions for descriptive inspection but excludes the entire failed
session from paired inference.

## Exact batched implementations

Explicit methods separate the original and optimized execution modes:

- `ov_npu_native_sequential_hier`
- `ov_npu_native_batched_hier`
- `ov_npu_native_sequential_adaptive`
- `ov_npu_native_batched_adaptive`

For one score vector, the batched backend packs up to B score tiles into a
static `[B,W]` tensor and applies TopK along W. One output row remains associated
with one original tile, so adding the tile start recovers original indices.
The last tile batch and last score tile use `-inf` padding; padded indices are
never retained.

For fixed hierarchical selection, every real tile returns exact local
`min(K,tile_length)` winners under the project ordering. Any global winner must
be among its tile's local K winners, so an exact stable global merge of their
union returns the global result. Batching changes only how independent local
TopK rows are submitted; it does not change this argument.

Adaptive batching groups active tiles that request the same static K. The
certificate remains conditional on exact local selection and valid next-score
bounds. Equal bounds refine because termination requires strict `u_j < tau`.
Every active budget grows geometrically and is capped at K, so termination is
finite. Numeric device counterexamples still limit the declared domain; pilot
success does not remove that limitation.

`--batch` remains a **tile batch for one query**, not a query batch. Multi-query
batch sizes 8/32 are not implemented and must not be reported from this work.

## Development pilot

The final strict NPU pilot used FP16, tile width 16, tile batch 32, N in
{512,4096,16384}, K in {4,10}, four distributions, two blocks, five measured
iterations per block and PLUGIN compilation. It produced 1,920 observations,
zero skips and zero wrong results. It includes the targeted exact CPU validation
for any tile row containing FP16 subnormals and is stored under
`results/phase2_pilot/npu_plugin_b32_strict/`. The stable Torch/CUDA pilot
produced another 1,920 valid observations. These are one session per device
workflow and support descriptive conclusions only.

Normal FP16, K=10 medians:

| N | C++ exact | CUDA stable | NPU seq hier | NPU batch hier | NPU seq adaptive | NPU batch adaptive |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.0308 ms | 0.194 ms | 15.93 ms | 1.056 ms | 15.41 ms | 1.387 ms |
| 4096 | 0.0679 ms | 0.206 ms | 96.34 ms | 8.214 ms | 96.35 ms | 5.960 ms |
| 16384 | 0.141 ms | 0.245 ms | 382.98 ms | 29.82 ms | 375.95 ms | 26.24 ms |

Batching improved hierarchical NPU latency by 15.1x, 11.7x and 12.8x at the
three sizes. It improved adaptive latency by 11.1x, 16.2x and 14.3x. At N=512,
adaptive required two submissions and was slower than fixed batching. At
N=4096 and N=16384, the normal input certified after one batched round and its
smaller candidate/output set was 1.38x and 1.14x faster than batched fixed
hierarchical selection.

At N=4096, batched adaptive spent about 2.80 ms in synchronous backend work and
2.11 ms extracting candidates; total latency was 5.96 ms. At N=16384 those
components were about 13.17 ms and 8.72 ms of a 26.24 ms total. The normal
cases recorded a median one exact subnormal-row validation at N=512/4096 and
two at N=16384. Candidate
handling is now material alongside device execution, rather than launch count
being the only bottleneck.

Across all 24 NPU cases, hierarchical batching speedup ranged from 9.82x to
18.13x with a 12.71x median. Adaptive batching ranged from 4.82x to 16.27x
with a 13.12x median. Batched adaptive was faster than batched hierarchical in
15/24 cases, so the negative cases remain part of the result.

The C++ and stable Torch CPU baselines remain far faster. Host-input/output CUDA
also remains far faster for this matrix. The stable Torch methods canonicalize
returned indices on the host and use a measured CPU exact repair when a TopK
boundary tie is ambiguous. Raw `torch.topk` failures are preserved separately.

## Remaining limitations and next experiment

1. Native NPU full-domain exactness remains unresolved for FP16 subnormals and
   FP32 near ties. FP32 is explicitly rejected; FP16 subnormal rows require
   measured exact CPU validation and are rejected on disagreement. A supported
   order-preserving representation is required before a general native-NPU
   exactness claim.
2. The batch-32 result is a systems improvement over the old sequential path,
   not evidence of competitiveness with CPU/CUDA and not algorithmic novelty.
3. Python candidate extraction now accounts for a substantial share of batched
   latency. Vectorized extraction and larger tile widths should be evaluated.
4. Adaptive selection helps only when it saves output/candidate work without
   adding refinement submissions. The N=512 negative result must be retained.
5. Query batching and a device-resident input/output protocol remain absent.
6. One exploratory session cannot support confidence intervals or significance
   claims. Final experiments require separately launched, time-separated and
   counterbalanced sessions after parameters are frozen.

The next development experiment should ablate tile widths 16/64/128/256 and
tile batches 8/32 at N=4096 and N=16384, while recording static model support,
submissions, bytes, synchronous backend time and candidate extraction time.
Freeze the best supported configuration before collecting independent sessions.

