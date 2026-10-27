# Phase 7 statistical validation

Date analyzed: 2026-10-24. Source revision: `7fd0a104f4da2de0d2d5c1f72bd5e7f230c9f6bd`.

The [locked protocol](CONFIRMATORY_PROTOCOL.md) was committed before the SIFT1M confirmatory score arrays were benchmarked. Its untouched query indices 8–15, `N=1,000,000`, `K=10`, FP16, tile width 256, tile batch 32, and `PLUGIN` compiler were used unchanged. The raw run is `results/phase7/confirmatory/20261024T212141Z`; the analysis is `results/phase7/confirmatory_analysis.json`. Six separately initiated sessions contain 2,880 observations: eight queries × six methods × ten timed calls × six sessions. All 48 query runs are complete, with zero skipped or incorrect observations. Every query has one stable input digest across sessions. Compiled OpenVINO models report `NPU` as their execution device.

## Prespecified primary comparison

The session is the unit of inference. Within each session, the analysis takes the median latency for each method and query, forms eight paired log ratios of native direct NPU latency over geometric certified NPU latency, and averages them. The estimate is the exponential mean of six session effects. The two-sided exact sign-flip test has 64 permutations; its smallest attainable two-sided p-value with six identically directed effects is 0.03125. The interval is a 50,000-resample bootstrap of the six session effects. Repeated calls and query rows are not treated as independent experimental sessions.

| Measure | Result |
|---|---:|
| Geometric-mean paired speedup | 1.666× |
| 95% session-cluster bootstrap interval | 1.646–1.685× |
| Two-sided exact sign-flip p-value | 0.03125 |
| Session-level speedups | 1.682, 1.648, 1.659, 1.626, 1.679, 1.702× |
| Paired query/session median wins | 48/48 |
| Predeclared practical threshold | 1.10× |

This passes the protocol's statistical and practical gates for the narrow statement that geometric certified selection is faster than native direct NPU TopK on these SIFT1M scores and this laptop under the measured host-input/host-output contract. The result does not establish an algorithmic novelty claim or performance superiority over CPU or CUDA.

## Latency and throughput

These descriptive values pool all 480 raw observations per method, including all eight queries. Throughput is one million scores divided by measured latency; the P05 throughput corresponds to P95 latency. The method-wise P95 includes score-dependent tie repair and should not be interpreted as a pure device-tail measure.

| Method | Median ms | P95 ms | Median million scores/s |
|---|---:|---:|---:|
| Optimized C++ | 14.183 | 27.783 | 70.507 |
| Stable CUDA | 2.663 | 471.994 | 375.516 |
| Native direct NPU | 135.004 | 635.064 | 7.407 |
| Fixed hierarchical NPU | 122.836 | 154.506 | 8.141 |
| Geometric certified NPU | 96.556 | 117.419 | 10.357 |
| Submission-aware certified NPU | 97.644 | 125.112 | 10.241 |

The secondary comparisons are descriptive by design. Geometric versus fixed NPU has a 1.256× session-level geometric-mean speedup, interval 1.238–1.271×. Submission-aware versus geometric is 0.990×, interval 0.976–1.002×. Geometric versus CUDA has a speedup ratio of 0.0512×, and versus optimized C++ 0.1393×. The secondary intervals are exploratory; there are no adjusted significance claims.

## Boundary-tie audit

Query 15 creates an FP16 tie at the global Top-K boundary. On that query, native direct NPU's median total latency is about 613 ms, of which about 477 ms is measured host candidate extraction and exact tie repair; its synchronous OpenVINO interval is about 123 ms. Stable CUDA's median is about 466 ms, of which about 459 ms is host extraction/tie repair and about 2.4 ms is its synchronous backend interval. Geometric NPU's median is about 91 ms with zero global tie repairs and about 13 ms candidate extraction. All methods still return the same exact deterministic indices. The tie is a real property of the stored FP16 workload and its host repair belongs in the end-to-end contract.

A post-hoc sensitivity check excluding query 15 yields 1.370× geometric-mean speedup, 95% session bootstrap interval 1.351–1.389×, exact two-sided sign-flip p=0.03125. This check is *not* the prespecified primary analysis. It shows that the within-NPU gain persists without the exceptional tie, while the primary 1.666× estimate is materially amplified by it. Raw P95 values for direct NPU and CUDA likewise reflect this one query's repair cost.

### Subsequent software-cause audit (2026-10-27)

The historical source revision routed global tie repairs through `topk_with_ids(scores, arange(N), K)`. That helper checks uniqueness of all `N` IDs with `np.unique`, sorting one million IDs even though sequential indices are known to be unique. This is an avoidable *software cost*, not an intrinsic requirement of exact Top-K or of the NPU. Revision `fdb5e1c` specializes `exact_topk` for implicit sequential IDs while retaining the same stable score/index ordering. The same-process, alternating microbenchmark in `results/phase8/tie_repair_reference_audit.json` found median 305.79 ms for the historical helper and 4.70 ms for the specialized reference on SIFT query 15; the outputs were identical. The dedicated post-repair hardware run in `results/phase8/tie_repair_exploratory/` had 80 correct observations, zero skips, and medians of 147.37 ms for direct NPU, 109.89 ms for geometric NPU, 9.30 ms for stable CUDA, and 10.90 ms for optimized C++. That run is exploratory and was collected in another session, so it must not be spliced into the locked Phase 7 estimate.

Consequently, **1.666× describes the historical implementation, not a durable algorithm-only advantage**. The old source and raw data remain intact. The post-hoc no-tie result is evidence that the NPU-internal gain did not depend entirely on this repair, but a fresh, separately locked study is needed for any confirmatory claim about the repaired implementation.

## Limits and reproducibility

- The six sessions ran sequentially on one laptop over roughly the duration of this collection, not across different machines or days. The narrow bootstrap interval reflects variation between these six sessions for the same eight queries; it does not capture platform or dataset variation.
- Method preparation order is rotated, and timed block order is shuffled. The rotation formula combines session index and query position, which causes some repeated method orders for a given query; it is weaker counterbalancing than a full Latin-square design. Future studies must fix this in a new protocol without altering this record.
- The primary exact sign-flip test assumes exchangeable signs of paired session effects under its null. Slow thermal or OS correlations and shared query inputs can weaken that assumption.
- Compilation and score generation are excluded. Host staging, device calls, synchronization, certification, candidate handling, and exact tie repair are included.
- There is no basis here for device-resident input, energy, peak-memory, multi-query batching, or general consumer-NPU claims.

Reproduce the historical analysis without rerunning the hardware experiment using `./scripts/26_analyze_confirmatory.ps1 -ResultsPath results/phase7/confirmatory/20261024T212141Z`. Figures and CSV tables are in `results/phase7/confirmatory_analysis_*`. Preserve the raw records and the locked source revision when extending the project.
