# Phase 13 correctness-aware performance results

**Checked FP32 meets the prespecified practical and statistical benefit rule against INT32 on SIFT1M. CPU and CUDA remain faster in the paired median-latency comparisons.** The size and confidence of the FP32/INT32 difference depend on the dataset and configuration; no universal superiority is established.

## Confirmatory design and complete cost

The protocol and analysis were frozen at `d97544ac4514bafbb81b0cf38044982f43fb1379` before fresh scores existed. Eight new query IDs per dataset were selected by a prespecified hash ordering after excluding every previously prepared query. SIFT uses N=1,000,000, SciFact N=5,183 and FiQA N=57,638. Base configurations use K=10/W=256; two queries per dataset support targeted sensitivity cases. There are **24 unique fresh queries, 48 workload cases and 15 distinct (N,K,W) configurations**.

All **12 separately initiated performance processes** completed. Six balanced timed repetitions per case/method and two warm-ups yield **20,736 timed calls and 6,912 warm-up calls**. Every output agrees with the independent exact source-score reference; there are **zero rejections, accepted errors, construction failures, invalid environment sessions or invalid case/method rows**. Raw timed and warm-up IDs were independently rechecked. Placement checks report strict NPU for NPU methods; each method occupied each of the six timing positions exactly once per case/session. No outcome was excluded and no failed attempt was replaced.

Input/output contract: **immutable host-resident finite FP16 scores to stable host-resident IDs, or explicit rejection**, score descending/index ascending. Timing includes validation, encoding/preparation, staging, submission/wait, output copying, sorting/certification and any existing host tie handling. Compilation/preparation, two warm-ups and the independent full-sort reference are outside steady-state timing. Per-group aggregate preparation time and every warm-up latency are saved separately. Input FP16 quantization and upstream similarity/embedding computation are outside selection timing.

The methods are unchanged existing implementations. The checked FP32/INT32 methods verify device proposals and reject errors without selecting replacement CPU answers. The FP16 static and native wrappers include their existing host checks/repair logic; the underlying FP16 graph is numerically unsafe in the separate adversarial tests. Its successful retrieval timings do not imply equivalent total-acceptance coverage of the broader finite-FP16 domain. The raw numerical suite explicitly requests stable=true; the performance FP16/native controls use their existing graph semantics and wrapper ordering contract. Legal unstable operator ties are not classified as compiler defects.

## Latency and sequential service rate

The median below is the median of 96 query/session medians (eight queries x twelve sessions, each median from six calls). P95 pools the 576 timed attempts for each dataset/method and is **descriptive**, without an independent-tail confidence interval. The reciprocal-median rate is the implied sequential selection service rate, excluding inter-call oracle and experiment bookkeeping; it is not measured concurrent throughput or full retrieval-pipeline throughput.

| Dataset | Method | Median ms | Descriptive P95 ms | Reciprocal median queries/s |
|---|---|---:|---:|---:|
| sift | Checked FP32 (TopK,Reshape) | 32.0308 | 38.0562 | 31.22 |
| sift | Checked stable INT32 | 42.8179 | 49.7030 | 23.35 |
| sift | Existing FP16 static wrapper | 33.2578 | 38.3708 | 30.07 |
| sift | Native NPU wrapper | 178.8288 | 187.9690 | 5.59 |
| sift | C++ CPU | 22.4144 | 37.3328 | 44.61 |
| sift | Stable CUDA | 7.2142 | 11.7252 | 138.61 |
| scifact | Checked FP32 (TopK,Reshape) | 1.3395 | 2.0236 | 746.55 |
| scifact | Checked stable INT32 | 1.4047 | 2.1028 | 711.90 |
| scifact | Existing FP16 static wrapper | 1.3673 | 2.1421 | 731.36 |
| scifact | Native NPU wrapper | 2.1738 | 3.6958 | 460.02 |
| scifact | C++ CPU | 0.1913 | 0.3387 | 5226.03 |
| scifact | Stable CUDA | 1.1381 | 4.2865 | 878.62 |
| fiqa | Checked FP32 (TopK,Reshape) | 5.1247 | 6.9476 | 195.13 |
| fiqa | Checked stable INT32 | 5.2450 | 7.2123 | 190.66 |
| fiqa | Existing FP16 static wrapper | 5.1055 | 7.0223 | 195.87 |
| fiqa | Native NPU wrapper | 20.5106 | 28.9230 | 48.76 |
| fiqa | C++ CPU | 1.5113 | 2.3460 | 661.66 |
| fiqa | Stable CUDA | 1.8785 | 4.5599 | 532.34 |

`table4_end_to_end_performance.json` also provides geometric mean session/query medians and session-cluster 95% intervals for absolute latency. Those differ from the median column's estimand. The paired ratios below are computed from paired log medians, **not ratios of the table's pooled medians**.

## Prespecified primary result

The geometric mean paired **INT32/FP32 ratio is 1.284668**, with 95% session-cluster bootstrap CI **[1.254773, 1.314964]**. This corresponds to **22.16% lower FP32 latency**, with transformed interval [20.30%, 23.95%]. The two-sided exact 12-session sign-flip p value is **0.000488281** (2 of 4,096 sign assignments). The lower ratio bound exceeds the frozen **1.10** threshold; all primary checked executions passed. The primary success rule is satisfied.

Crossed query/session bootstrap CI: [1.250864, 1.319822]. Consecutive-pair session-block CI: [1.257912, 1.311788], block sign-flip p=0.03125. Leave-one-query-out ratios range from 1.280668 to 1.286951; lag-1 session-effect correlation is 0.109. All twelve session ratios and all eight query-average ratios exceed one. These sensitivities support the direction within the study; they do not prove temporal independence.

The experimental unit for primary inference is one session's mean of eight paired query log median ratios. Calls and queries within a session are not treated as independent hardware units. The exact sign-flip test assumes exchangeable symmetric paired session errors under the null; balanced method order does not by itself guarantee that assumption. The percentile bootstrap resamples twelve session clusters (50,000 draws); the crossed sensitivity uses 20,000 draws. The interval is primarily conditional on this fixed query panel and device. Twelve processes ran sequentially on one date, not on twelve days or machines. Block sensitivities cannot remove all shared thermal/OS-state dependence.

## Secondary family and negative results

Ratios are **baseline/FP32**: greater than one favors FP32, less than one favors the baseline. Fourteen prespecified secondary tests form a separate Holm family. The intervals are marginal 95% session-cluster bootstrap intervals, **not multiplicity-adjusted simultaneous intervals**. Holm-adjusted p values govern secondary significance; an interval excluding one can coexist with a nonsignificant Holm result. The primary comparison is outside this secondary family.

| Dataset | Numerator baseline | Ratio | Marginal 95% CI | Raw p | Holm p |
|---|---|---:|---|---:|---:|
| sift | Native NPU wrapper | 5.5052 | [5.3444, 5.6446] | 0.000488 | 0.006836 |
| sift | Existing FP16 static wrapper | 1.0283 | [1.0222, 1.0344] | 0.000488 | 0.006836 |
| sift | C++ CPU | 0.6743 | [0.6303, 0.7177] | 0.000488 | 0.006836 |
| sift | Stable CUDA | 0.2193 | [0.2060, 0.2347] | 0.000488 | 0.006836 |
| scifact | Native NPU wrapper | 1.7299 | [1.6873, 1.7722] | 0.000488 | 0.006836 |
| scifact | Existing FP16 static wrapper | 1.0208 | [0.9996, 1.0445] | 0.100586 | 0.201172 |
| scifact | C++ CPU | 0.1338 | [0.1211, 0.1460] | 0.000488 | 0.006836 |
| scifact | Stable CUDA | 0.8222 | [0.7271, 0.8982] | 0.000977 | 0.006836 |
| fiqa | Native NPU wrapper | 3.9290 | [3.8357, 4.0469] | 0.000488 | 0.006836 |
| fiqa | Existing FP16 static wrapper | 1.0036 | [0.9896, 1.0194] | 0.668457 | 0.668457 |
| fiqa | C++ CPU | 0.2713 | [0.2528, 0.2897] | 0.000488 | 0.006836 |
| fiqa | Stable CUDA | 0.3626 | [0.3380, 0.3856] | 0.000488 | 0.006836 |
| scifact | Checked stable INT32 | 1.0483 | [1.0296, 1.0677] | 0.000977 | 0.006836 |
| fiqa | Checked stable INT32 | 1.0276 | [1.0050, 1.0509] | 0.043457 | 0.130371 |

Native wrapper/FP32 paired ratios are 5.505 on SIFT, 1.730 on SciFact and 3.929 on FiQA. They describe these implementations, including native wrapper validation, rather than pure kernel speedups. FP32 and native both submit one NPU request per query: this comparison does not isolate a synchronization-count advantage.

SciFact's INT32/FP32 ratio is 1.048, significant after Holm but below a 10% practical difference; the SIFT practical rule is not transferred to it. FiQA's ratio is 1.028 with Holm p=0.1304, so FP32 superiority is **not established** there. FP32 versus FP16-static shows only a small SIFT median difference, and no adjusted significant difference on the BEIR datasets. FP16 is not universally faster than preserved FP32.

C++ and CUDA have lower paired median latency than FP32 on all three base datasets after Holm adjustment. Their CPU/FP32 ratios are 0.674, 0.134 and 0.271; CUDA/FP32 ratios are 0.219, 0.822 and 0.363. This does not mean every tail statistic favors CUDA: SciFact's CUDA pooled P95 is 4.2865 ms versus FP32's 2.0236 ms. Tail summaries are descriptive. No NPU-over-CPU/CUDA claim is supported.

## Targeted sensitivity (descriptive)

Two fresh queries per setting, twelve sessions, six calls per method; no additional significance tests or favorable configuration selection.

| Dataset | N | K | W | Paired INT32/FP32 ratio |
|---|---:|---:|---:|---:|
| fiqa | 57638 | 1 | 128 | 0.9881 |
| fiqa | 57638 | 100 | 512 | 1.0414 |
| scifact | 5183 | 1 | 128 | 1.0489 |
| scifact | 5183 | 100 | 512 | 1.0356 |
| sift | 16384 | 10 | 256 | 1.0454 |
| sift | 131072 | 10 | 256 | 1.1070 |
| sift | 1000000 | 1 | 128 | 1.3905 |
| sift | 1000000 | 1 | 256 | 1.3964 |
| sift | 1000000 | 10 | 128 | 1.2100 |
| sift | 1000000 | 10 | 512 | 1.3566 |
| sift | 1000000 | 100 | 256 | 1.0558 |
| sift | 1000000 | 100 | 512 | 1.1052 |

At SIFT K=10/W=256 the ratio grows from about 1.045 at N=16,384 to 1.107 at N=131,072; the primary full-size eight-query result is 1.285. At N=1M, K=1/W=256 has ratio 1.396, whereas K=100/W=256 has 1.056. The larger candidate budget changes the comparison substantially. FiQA K=1/W=128 has ratio 0.988, slightly favoring INT32 descriptively. These observations reject a universal FP32 advantage and do not identify an optimal tile width or a causal compiler-kernel cost model. Configurations that change K and W together cannot isolate their individual effects.

## Measured components, ties and domain exposure

`results/phase13/supplemental/performance_integrity_and_components.json` retains pooled component summaries. Checked FP32 median preparation / synchronous inference / output-and-certificate times (ms) are approximately **1.675 / 23.792 / 6.740** on SIFT, **0.022 / 1.121 / 0.190** on SciFact, and **0.121 / 4.029 / 0.971** on FiQA. The synchronous region includes runtime staging/wait; it is not device-only kernel time. Marginal component medians need not sum to a total median, and the output region includes copying and certification together. INT32 encoding and certification are fully timed but not individually instrumented by its existing runner, so their separate causal contributions are not inferred by subtraction.

All native/CUDA base calls record zero invocations of the expensive CPU tie-repair fallback. Final order checking/canonicalization still runs and remains timed. No historical Phase 7 tie-repair-inflated speedup is used. This does not prove equal-score absence: it describes the recorded fallback counter. Raw counters for sensitivity cases are retained too.

Fresh SIFT vectors contain no subnormals. Across the eight SciFact vectors there are five subnormal scores, and across FiQA there are 234; **none occurs among any query's top 100**. All methods passing these retrieval vectors is therefore compatible with the separately demonstrated adversarial subnormal failures. No real-query ranking corruption or improved retrieval relevance is claimed. This empirical separation is a central limit of the correctness/performance comparison.

## Evidence and conclusion

The six tables, per-session case medians, raw IDs and input hashes are under `results/phase13/analysis/` and the original session directories. The frozen analysis rechecks every accepted timed ID. The explicitly post-freeze supplementary analysis also rechecks warm-ups, balanced positions, placement and raw numerical outputs; it adds no hypothesis or significance test. All failed cases remain represented in the numerical tables.

**C3 is supported within this input-residency and single-stack scope.** Preserved FP32 offers a meaningful checked-NPU improvement on the prespecified large SIFT workload, with smaller, null or reversed differences elsewhere. The stronger practical conclusion remains that optimized CPU/CUDA controls are preferable for median latency on these tested host-resident base workloads. These are experimental trade-offs among established methods, not a new selection algorithm or a device-wide performance guarantee.
