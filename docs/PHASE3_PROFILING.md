# Phase 3 performance characterization

Date: 2026-10-24  
Status: development characterization complete; confirmatory inference not attempted

## Scope and evidence

Phase 3 measured the strict OpenVINO `NPU` implementation on the Intel AI Boost 3720 with OpenVINO 2026.4.1. Compilation and model initialization were outside the timed region. Every recorded result was checked against the deterministic exact reference, including score-descending and index-ascending tie semantics.

The complete development sweep is under `results/phase3/ablation_baseline/20261024T205003Z`. It contains 20 configurations, 1,920 timed observations, and no skipped or incorrect observations. Each configuration ran in its own process and retained its manifest, log, raw JSONL, and summary. These processes are development configurations, not independent confirmatory sessions.

## Controlled ablation

The sweep crossed:

- `N`: 4,096 and 16,384
- `K`: 4 and 10
- tile width: 16, 32, 64, 128, and 256
- tile batch: 1, 8, 32, and 64
- distributions: normal, tied, clustered, and adversarial
- algorithms: fixed hierarchical and geometric adaptive

The repository currently executes one score vector per API call. The requested batch dimension therefore varied **tile batch**, which controls how many tiles share one static NPU graph submission. Query batch remained one and is explicitly recorded as unsupported. No result should be described as a query-batch ablation.

All 20 configurations were fully valid. The development rule selected the fully valid configuration with the lowest geometric mean slowdown relative to the per-case oracle. Both algorithms selected tile width 256 and tile batch 32:

| Algorithm | Recommended configuration | Geometric-mean slowdown from per-case oracle | Median case latency |
|---|---:|---:|---:|
| Fixed hierarchical | T256, tile batch 32 | 1.095 | 2.342 ms |
| Geometric adaptive | T256, tile batch 32 | 1.213 | 3.777 ms |

This recommendation was tuned on the same development matrix. It is frozen for later held-out work but is not itself confirmatory evidence.

Relative to the Phase 2 setting T16/B32, T256/B32 improved all 16 matched cases for both methods. Fixed speedup ranged from 5.54x to 14.31x, with median 8.87x. Adaptive speedup ranged from 1.49x to 10.31x, with median 4.37x. At normal `N=16,384`, `K=10`, fixed submissions fell from 32 to 2 and latency fell from 38.400 ms to 3.983 ms; adaptive submissions fell from 32 to 3 and latency fell from 32.290 ms to 3.837 ms.

## Timing boundaries

The benchmark records these direct host-wall-clock components:

1. host input preparation and tile staging;
2. backend input conversion;
3. synchronous `InferRequest` execution;
4. output materialization;
5. strict backend validation;
6. candidate extraction and canonical ordering;
7. candidate merge;
8. correctness certification;
9. total end-to-end latency;
10. submissions, staged bytes, retrieved bytes, and refinement rounds.

`backend_execute_sync_ns` is a directly measured host interval around synchronous OpenVINO inference. It combines submission, driver/runtime work, transfers that OpenVINO performs internally, NPU execution, and waiting. OpenVINO did not expose a reliable cycle-level split for those parts, so statements about their individual costs would be inference rather than measurement. CPU-side tiling is included in host input preparation. Adaptive refinement is represented by rounds, active-tile routing, and additional submissions.

At the baseline recommended configuration, representative normal-score medians were:

| Algorithm and case | Total | Synchronous backend | Candidate extraction | Validation | Submissions |
|---|---:|---:|---:|---:|---:|
| Fixed, N=4,096, K=10 | 1.395 ms | 0.964 ms | 0.177 ms | 0.087 ms | 1 |
| Fixed, N=16,384, K=10 | 3.983 ms | 1.990 ms | 1.246 ms | 0.325 ms | 2 |
| Adaptive, N=4,096, K=10 | 2.333 ms | 1.737 ms | 0.201 ms | 0.147 ms | 2 |
| Adaptive, N=16,384, K=10 | 3.837 ms | 2.572 ms | 0.595 ms | 0.283 ms | 3 |

Synchronous OpenVINO execution was the largest measured component. Candidate extraction was also material at larger `N`, which justified one further host optimization.

## Candidate extraction optimization

Candidate validation, uniqueness checks, score gathering, and canonical ordering were vectorized across rows sharing a static output shape. Exact scalar tie repair remains for ambiguous local boundaries.

The fresh T256/B32 before/after comparison is in `results/phase3/vectorized_before_after.json`.

| Algorithm | Median end-to-end speedup | Cases faster | Median extraction speedup | Extraction cases faster |
|---|---:|---:|---:|---:|
| Fixed | 1.078x | 12/16 | 1.780x | 15/16 |
| Geometric adaptive | 1.026x | 11/16 | 1.186x | 11/16 |

For fixed `N=16,384`, `K=10`, normal scores, candidate extraction fell from 1.246 ms to 0.407 ms and total latency from 3.983 ms to 3.154 ms. Some small cases regressed: fixed `N=4,096`, `K=10`, normal scores changed from 1.395 ms to 1.446 ms despite lower extraction time. The optimization is retained because it improves the large cases and preserves exactness, but the evidence does not support a universal latency improvement claim.

## Remaining performance gap

Using the optimized implementation on the same 16 cases:

- fixed NPU was 10.92x to 23.16x slower than optimized C++, median 20.30x;
- geometric adaptive NPU was 13.05x to 105.47x slower than optimized C++, median 34.50x;
- fixed NPU was 2.13x to 15.74x slower than stable CUDA, median 6.60x;
- geometric adaptive NPU was 2.07x to 32.22x slower than stable CUDA, median 11.89x.

At normal `N=16,384`, `K=10`, the medians were 0.138 ms for C++, 0.245 ms for stable CUDA, 3.154 ms for fixed NPU, and 3.391 ms for adaptive NPU.

The measured evidence points to synchronous OpenVINO execution and the required submissions as the main remaining cost, with host extraction secondary at larger `N`. Because the synchronous interval includes transfer, runtime, execution, and waiting, the experiment cannot attribute the gap to NPU silicon alone.

## Gate decision

The implementation is reasonably optimized for the available high-level OpenVINO path: compiled graphs are reused, tiles are batched, submissions were reduced, static shapes are prepared outside timing, and material host extraction was vectorized. The Phase 3 gate is **passed for engineering characterization** and **failed for CPU/GPU competitiveness**. A different scheduling mechanism can still be tested, but ordinary host micro-optimization is unlikely to erase a 10x–100x C++ gap.

## Reproduction

```powershell
.\scripts\13_phase3_ablation.ps1
.\scripts\14_analyze_phase3.ps1 -ResultsPath results/phase3/ablation_baseline
```

The plotting and aggregation implementation is `scripts/analyze_phase3_ablation.py`. Generated figures and all research outputs remain under `results/`.
