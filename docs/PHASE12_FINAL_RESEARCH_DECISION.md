# Phase 12 fast-track findings and decision

**2026-10-29: working numerical repair implemented and physically tested; NO-GO for claiming a novel research mechanism.** The user selected the fastest empirical track after brainstorming. This study isolates the existing FP16-subnormal failure and retains a checked FP32 baseline. It does not claim a new counterexample generator or modify the NPU compiler binary.

**New evidence in this repository.**

1. Direct NPU TopK on dynamic input `[0, 2^-24]`, shape `[1,2]`, K=1, stable=true, value sorting, returns index 0 instead of 1. The value agrees with the incorrectly chosen index. This is a smaller reproduction of the known failure class, not a new bug or generation method.
2. Separate identity/conversion controls preserve the inputs. Normal FP16 neighbors with equally small gaps pass. The controls contradict a subtraction-underflow-only explanation; they do not establish an instruction-level cause or correctness of every transport path.
3. Explicit promotion to FP32 alone does not fix the default compiled selector. The existing `compute-layers-with-higher-precision=TopK` option fixes the tested direct promoted selector.
4. TopK-only preservation is insufficient for the hierarchy. On the original ramp its IDs are correct but output values become zero. A focused two-tile construction makes the loss affect membership: expected IDs `[256,257,258,259]`, observed `[3,2,1,0]`. TopK+Reshape preservation fixes both contracts in the two saved process runs; Reshape-only also fails. This is a configuration-sensitive composition counterexample, not a new precision-analysis theorem.
5. A `[63487,2]`, K=1 batch covers every adjacent pair of finite FP16 representations, including numerical-zero equality. Default execution has **2,046 membership failures and 2,047 failures of the complete output contract**. Explicit promotion plus TopK preservation has zero failures. This is exhaustive for those concrete batch rows, not arbitrary tensors, K values, shapes or executables.
6. The checked promoted hierarchy passes **ten distinct correctness workloads in two processes each**: 20 saved raw preflights and 60 timed repetitions. Nine cases are in `repair_session01/02`; the tenth is in `composition_session01/02`. The three real development vectors below also pass.

**Implemented baseline.** [PrecisionNPUTopK](../src/cats_npu/precision_topk.py) accepts immutable finite FP16 host scores, promotes inside the graph, and uses the existing stable local/global hierarchy with precision preservation for `TopK,Reshape`. It requires execution devices exactly `['NPU']`. Full host rank/value/order checking supplies exact-or-reject behavior. It never selects a replacement CPU answer or silently switches methods. Historical implementations remain unchanged. A separate conventional dyadic-scaling control repairs the recorded ramp but explicitly rejects magnitude spans that cannot fit a common normal, finite FP16 scale.

**Performance.** Five interleaved repetitions after preflight, existing development query 0, K=10, W=256. Timings include host validation/preparation, submission/wait, copying and graph certificates. Compilation, first inference and independent reference construction are excluded. Median milliseconds from the final fully prepared comparisons:

| Method | SIFT1M N=1,000,000 | SciFact N=5,183 | FiQA N=57,638 |
|---|---:|---:|---:|
| Existing native NPU control | 130.8272 | 1.8704 | 11.0080 |
| Existing FP16 static graph | 27.8495 | 1.0694 | 3.4754 |
| Checked stable INT32 hierarchy | 30.4467 | 1.0643 | 3.4248 |
| **Checked promoted FP32 hierarchy** | **26.6541** | **1.1378** | **3.7769** |
| Optimized C++ CPU | 15.8404 | 0.2073 | 0.5459 |
| Stable CUDA | 2.5182 | 0.3671 | 0.6565 |

The promoted path has 12.5% lower latency than checked INT32 in that SIFT run, but 6.9% and 10.3% higher latency on SciFact and FiQA. Its descriptive five-sample P95 values are 28.5837, 1.57684 and 5.04922 ms; INT32's are 30.8697, 1.1969 and 4.63526 ms. These are weak tail estimates. Reciprocal-median sequential query rates are in the analysis; concurrent throughput was not measured. Both graph paths make one synchronous NPU request per query. CPU/CUDA remain faster. Passing one development query does not make the known-unsafe FP16 graph an equally safe full-domain alternative.

Earlier SIFT process runs have INT32/FP32 median ratios 1.097 and 1.166, versus 1.142 in the final comparison. Their evolving harness/control sets prevent pooling them as a frozen confirmation. There are no P values, confidence intervals, significance claims, held-out generalization estimates or new confirmatory results. Repetitions within a process are not independent sessions, and 2,046 failing rows are not 2,046 independent defects.

**Novelty assessment.** No original algorithm/compiler mechanism is established. The graph is the existing hierarchy with exact promotion and an already implemented layer-type precision option. Intel's pinned public source declares the option and implements operation exemptions. Layer precision/output-type control also predates this project in TensorRT. The [originality report](PHASE12_NOVELTY_ASSESSMENT.md) establishes this constructive equivalence. The method adds no automated placement analysis, new synthesis rule, optimality result, compiler proof or stronger guarantee than the existing host certificate. The numerical/composition observations are useful new repository evidence; that does not establish the requested method novelty.

**Attribution limit.** Changing precision preservation causally changes the recorded configuration's output. Public lowering source is consistent with that observation. The packaged binary's exact pass sequence and SHAVE instruction behavior were not traced. The public kernel is prebuilt. NPU `get_runtime_model()` snapshots here contain placeholder constants and are not treated as actual executable precision traces. No unique silicon root cause or newly confirmed upstream defect is claimed. No external bug report was submitted.

| Gate | Decision | Evidence and limit |
|---|---|---|
| A — Originality | **FAIL for a new mechanism** | Known promotion, hierarchy, precision controls and instance checking. |
| B — Mathematical validity | **PASS for conditional graph/checker arguments** | [Proof and assumptions](PHASE12_MATHEMATICAL_FOUNDATION.md); no universal device proof. |
| C — Implementation | **PASS for a conventional repair** | Source and tests; no novel generator implemented. |
| D — Physical execution | **PASS within recorded scope** | Strict NPU placement, raw outputs, adjacent and composition controls. |
| E — Scientific advantage | **NOT ESTABLISHED over equivalent methods** | Useful alternative to INT32 with mixed latency outcomes; existing controls explain it. |
| F — Generalization | **LIMITED development evidence** | Numeric regimes and three reused development queries; no untouched evaluation study. |
| G — Reproducibility | **PASS for this bounded artifact** | Inputs, outputs, graphs, source snapshots/hashes and reproduction scripts. |

**Integrity.** Local base `05e74cd030bf1ca380ac4404777c0aa5f474ed20`, branch `codex/phase12-boundary-counterexamples`; remote HEAD/master checked at `345c24cebd717b57e167ca6e696bbcd1e819b2d4`. Current discovery records Windows build 26300, OpenVINO 2026.4.1, AI Boost 3720, driver 1004512/compiler 458778. The older doctor record is preserved. All 49 historical Phase 9 raw hashes still verify. [analysis_final.json](../results/phase12/fast_track/analysis_final.json) recomputes the saved checks and timings.

Initial harness failures are retained and excluded from device claims: mixed Node/Output objects in the first hierarchy constructor, and a missing `prepare()` call for a newly added native benchmark control. Both were corrected and affected comparisons rerun into new directories. Their failed native timings are invalid, not fast TopK results. See [the execution record](PHASE12_EXPERIMENTAL_PROTOCOL.md).

**Final disposition.** Retain the checked FP32 baseline and stronger composition diagnostics. **Novelty has not been achieved at the requested method standard: NO-GO for a manuscript on this repair.** The separate layout/candidate-budget idea remains unimplemented and unassessed by this track. No manuscript, publication figures, broad fuzzing campaign, journal selection or confirmation-data access was undertaken.
