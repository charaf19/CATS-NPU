# Reassessment: a representation-aware exact Top-K path

**Closed on 2026-10-28:** `docs/FINAL_RESEARCH_CONTRIBUTION.md` records the final NO-GO after the mechanism-equivalence and compiler-source audit. The conditional assessment below is retained as the 2026-10-27 research record.

**2026-10-27 — Research status: CONDITIONAL GO for a bounded systems investigation, not a publication claim.** The earlier `STATIC_NPU_EXECUTION_FEASIBILITY.md` NO-GO applies to a plain FP16 two-level graph. New physical probes identify a narrower possible contribution: lowering finite-FP16 stable Top-K into an integer-ordered, static NPU graph when the native FP16 and some wider/narrower integer paths do not preserve the required order. No manuscript or TeX work has begun.

## Proposed novelty frame

**Working research question:** Can a compiler/runtime-aware representation change make exact, deterministic Top-K both numerically faithful and synchronization-efficient on a static-shape consumer NPU?

**Candidate contribution, stated at the right level:** A semantics-aware lowering of stable FP16 Top-K to one static NPU graph. A host pass maps each finite binary16 score to an order-equivalent signed 32-bit integer code; two stable NPU Top-K stages perform local and final selection, carrying original indices through the graph. The host does only O(N) representation conversion, one submission/synchronization, O(K) output sorting, and an O(N) fail-closed rank check. It never selects or repairs winners on the CPU. The technical problem is to retain the exact score/index order under the compiler's actual type behavior, not to invent hierarchical Top-K.

This is a **candidate systems contribution** rather than a new selection algorithm. The interesting result would be a validated, generalizable rule for choosing and proving a safe execution representation under NPU compiler constraints, together with a measured end-to-end benefit and a map of the domain where that rule works. The present evidence is promising on one hardware/compiler stack but does not yet establish such generality or literature priority.

## Why the earlier frame was too weak

The previous plain FP16 static graph removed 488 of 489 hybrid submissions at N=1,000,000, K=10, but it used standard local-then-global selection and still failed the known FP16-subnormal case. Also, direct NPU Top-K already uses one submission, so the advantage against direct Top-K comes from selection structure, not synchronization count. [DeepSelect](https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.md) explicitly describes local and global selection inside a GPU cluster, and its [repository](https://github.com/deepseek-ai/DeepSelect) now includes Ascend NPU kernels. Local/global fusion or NPU execution alone cannot carry a novelty claim.

The new physical probes expose a more specific compiler constraint:

| Input representation on Intel NPU 3720, OpenVINO 2026.4.1 | Observed result | Raw evidence |
|---|---|---|
| Native FP16 Top-K | Known positive/negative subnormal case selected a boundary of rank 64 at K=4; host rank check rejected it. | `results/phase9/static_graph_edge_cases.json` |
| Signed int16 order codes | Small subnormal case passed, but at SIFT1M N=1,000,000 the NPU returned altered values (for example −7852 where the input code at that selected ID was −7853). Winner IDs happened to match on that query; value corruption forbids an exactness claim. | `results/phase9_novelty/stable_i16_n1m_failure_diagnostic.json` |
| Signed int64 composite keys | A graph compiled and executed on `NPU`, but high bits were lost. Across 2³², the returned IDs were `[511,510,509,508]` where exact Top-K was `[3,2,1,0]`. | `results/phase9_novelty/i64_wrap_probe.json` |
| Signed int32 order codes | Correct on the known subnormal case, a vector containing every finite FP16 representation, a boundary-tie case, three real development queries, and 100 additional same-process adversarial/tied inputs at N=512, K=10. These tests do not prove every possible input/compiler build correct. | `results/phase9_novelty/stable_i32_hier_*.json`, `stable_hier_stress_100.json` |

These are **physical observations of this stack**, not claims that all NPUs implement these types similarly. The official NPU page lists F32/F16 and quantized U8 as inference precisions and says hardware computation precision is FP16; it does not promise faithful Top-K ordering for every integer type. OpenVINO's general [TopK-11 specification](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/sort/top-k-11.html) does define `stable=true`, but compilation and `["NPU"]` placement did not make the int16/int64 numerical paths correct in these probes. [NPU device documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html).

## Mechanism and proof boundary

For a finite FP16 score with raw 16-bit word `b`, normalize `−0` to `+0`, then define `q(b)=(~b)&0xffff` for negative values and `q(b)=b xor 0x8000` for nonnegative values. For finite binary16 numbers, this code strictly preserves numeric order, except that the two zeros intentionally share one code. This familiar sign-bit transform is also documented for floating-point radix sorting in [NVIDIA CUB](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh); it is **not** claimed as new.

With tile-major input order, an exact `stable=true` local Top-K on `q` chooses the highest K score/index pairs in each tile. Concatenating tile outputs in tile order preserves ascending global index order among equal scores. An exact `stable=true` final Top-K over this candidate sequence therefore returns the global stable Top-K by the standard local-K union lemma. The host sorts only the returned K IDs into score/index order. The graph uses one explicit `NPU` inference request and emits K int32 codes plus K int32 IDs. Input staging is 4N bytes plus final padding. The host rank check proves every *successful* returned set exactly equals the required stable Top-K; it raises on disagreement and performs no CPU winner selection. Total success over all finite FP16 inputs still depends on the physical compiler/runtime honoring the int32 and stable-TopK semantics.

The more elaborate **split-key** graph was also tested: it formed local score/index keys, computed the global Kth score on the device, and used another integer Top-K to resolve boundary ties. It passed the subnormal and real development probes. However, OpenVINO already documents stable Top-K, and the simpler stable-int32 graph passed the same checked cases and was faster in the exploratory SIFT1M runs. The split-key graph is retained as an ablation, **not presented as the novelty**. A range-bounded packed-int32 variant also passed, but its explicit score-span restriction and higher SIFT1M latency make it a weaker primary method. Their source and raw results are under `scripts/probe_split_key_npu.py`, `scripts/probe_order_encoded_npu.py`, and `results/phase9_novelty/`.

## Performance evidence and its limit

One *interleaved, single-process* pilot used the existing SIFT1M development query 0 at N=1,000,000, K=10. It used host FP16 scores and returned host stable indices, excluded compilation and first inference, and checked every answer against the independent exact reference. Medians from five observations per method were:

| Method | Median end-to-end ms | Physical device submissions | Exact on this query? |
|---|---:|---:|---|
| Stable int32 static NPU | **30.860** | 1 | Yes |
| Plain FP16 static NPU | 27.145 | 1 | Yes; known subnormal case fails elsewhere |
| Split-key int32 static NPU | 38.657 | 1 | Yes |
| Direct native FP16 NPU | 131.692 | 1 | Yes; known subnormal case fails elsewhere |
| Optimized C++ CPU | 20.403 | 0 | Yes |
| Stable CUDA | 3.019 | 1 | Yes |

Thus the stable-int32 method was **4.27× faster than direct NPU Top-K** on this development input while costing 13.7% more than the plain FP16 static graph. The latter is faster but has a reproduced numerical failure. For stable int32, median score encoding cost 5.282 ms, staging plus synchronous inference 20.289 ms, and output copy plus verification 3.322 ms. These medians are descriptive, need not sum to the total median, and are not independent-session statistics. The CPU and CUDA controls remained faster. Raw observations are in `results/phase9_novelty/paired_sift_dev_q0000_n1000000_k10_with_stable.json`.

Separate exploratory runs also returned correct results on SciFact development query 0 (5,183 scores, 0.966 ms stable-int32 median) and FiQA development query 0 (57,638 scores, 3.761 ms). They are not paired with the prior direct-NPU timings and should not be converted into effect-size claims. The previously used evaluation/confirmatory holdouts were not touched. No energy or peak-memory metric was used.

## Prior-art boundary and research decision

The mathematical building blocks are established: order-preserving floating-point bit transforms ([CUB](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh)), stable Top-K semantics ([OpenVINO TopK-11](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/sort/top-k-11.html)), fused hierarchical selection ([DeepSelect](https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.md)), and explicit boundary-tie policies in accelerator Top-K ([NVIDIA cuDNN DSA Top-K](https://docs.nvidia.com/cudnn/fe-oss-apis/dsa/)). A paper that merely combines those elements on Intel hardware would likely be judged an implementation report.

The **potentially distinctive systems question** is whether a compiler-aware lowering can formally preserve a stable FP16 order through a static, one-submission consumer-NPU graph despite demonstrated type-dependent compiler behavior, and whether that yields a reproducible performance/correctness region that prior accelerator algorithms do not characterize. The current search found no primary source describing this exact Intel-3720/OpenVINO behavior or this exact lowering, but absence of a match does not establish priority. The intellectual claim must be the validated semantic/cost model and transformation, not “first NPU Top-K,” “new hierarchy,” or “new bit transform.”

**Conditional GO for a bounded follow-up only.** A larger confirmatory campaign and manuscript remain unjustified. The next decisive work is to (1) isolate where int16 values change and int64 bits are lost using compiler/per-layer evidence; (2) test stable-int32 correctness across K, tile widths, adversarial ties, and independent sessions; (3) compare direct-int32, stable-int32 hierarchy, plain FP16 hierarchy, and CPU/CUDA in the same protocol; (4) reduce or explain host encoding/verification cost; and (5) search for equivalent precision-aware lowering in current compiler and Top-K literature. If the stable-int32 path fails correctness, loses its end-to-end advantage, or proves to be an already established lowering, return to **NO-GO**. A journal suitability judgment comes only after those gates.

All new raw files are separated in `results/phase9_novelty/`. The historical Phase 7 and Phase 8 records and their decisions remain unchanged.
