# Phase 13 evidence gap audit

**Prospective audit, 2026-11-03, starting revision `342b3726f531143ab5aac1666f2ec79a9baea839`.** The research question is now empirical characterization, not algorithmic novelty. The Phase 12 mechanism NO-GO remains unchanged; it does not automatically prohibit an appropriately scoped empirical study.

The working tree initially was clean on `codex/phase12-boundary-counterexamples`. The requested final/Phase 12 reports, mathematical assumptions, Phase 7 statistical record, real-data preparation/evaluation and benchmark fairness records were reviewed against current code and manifests. `results/phase13/audit/audit.json` independently verifies all 1,641 files bound by the Phase 12 manifest and the source archive hashes. No mismatch was found. Historical raw observations and statistical interpretations remain intact.

| Topic | Classification before confirmation | Missing evidence/action |
|---|---|---|
| Physical execution and recorded subnormal counterexamples | Established evidence | Replicate on the current stack with independent input construction; do not count repeated witnesses as new defects. |
| Adjacent finite-FP16 batch | Established bounded exhaustive evidence | Retain its exact domain/count. One independent rerun can test reproducibility; no tensor-universal claim follows. |
| Two-tile precision composition | Established exploratory evidence | Extend widths/K, padding and distribution; compare each precision setting independently. |
| Checked INT32/FP32 host safety arguments | Established conditional/checker facts | No new proof required. Test the harness/oracle and collect independent physical coverage. |
| Phase 12 performance | Confirmatory evidence needed | One development query per dataset and changing harness/control sets do not supply final uncertainty. |
| Phase 7 1.666x estimate | Historical confirmatory evidence for its locked code | Preserve it; expensive historical tie repair prevents carrying its magnitude into the present comparison. |
| Phase 8 dispatcher result | Historical separate frozen study | No new dispatcher experiment is needed. Its holdouts are already consumed. |
| New query generalization | Confirmatory evidence needed | Exclude all previously prepared IDs, select before scores, regenerate with identical metrics/model revision. |
| CPU/CUDA controls | Established implementations, current validation needed | Keep corrected tie handling; precompile native NPU outside timing; validate every measured output. |
| Second NPU or compiler version | Optional strengthening | Not available/required; retain the single-device/version limitation. |
| Universal hardware exactness, unique silicon cause, new selection/precision algorithm | Unsupported claims | Exclude from the empirical contribution and readiness assessment. |

**Exposure.** All original SIFT IDs 0–15 and all 16 original test-query IDs in each BEIR manifest are excluded, including development, descriptive and consumed confirmation queries. Inventory of historical `.npy` filenames found no prepared query positions beyond 15. The Phase 11 audit generated no fresh score arrays. Phase 12 repeatedly reused query 0. Freshness is relative to recorded project preparation/execution, not an unknowable external access history.

The new selection takes the eight smallest SHA256 values of `CATS-NPU-Phase13-20261103|dataset|queryID` among eligible IDs: SIFT has 9,984 eligible queries, SciFact 284 and FiQA 632. Exact IDs are in `results/phase13/query_selection.json` and the frozen protocol. BEIR uses the existing test-query population; only query IDs determine selection, with no relevance-quality metric. No new score array had been generated or inspected during this audit.

**Pilot and harness.** Three new pilot processes use only the old SIFT query 0. Their INT32/FP32 ratios are approximately 1.250, 1.205 and 1.257. Historical Phase 12 log-ratio SD is 0.0311. Use a conservative planning SD at least 0.07, rather than treating three pilots as a precise variance estimate. Final harness smoke uses old query 0/prefixes at all 15 planned shape/K/width settings; all six methods pass. This is engineering validation, not confirmation or a performance claim.

**Independence and comparisons.** Repeated calls are subsamples. A process session supplies one paired mean-of-query-log-ratios effect. Queries share a corpus/device/session; do not treat query rows or thousands of failures as independent hardware trials. Use session inference plus crossed query/session and consecutive-session-block sensitivities. Native/static FP16 are unsafe controls under a broader numerical domain; successful timings do not make them correctness-equivalent. Compilation, warm-up and ground-truth preparation can be excluded, but encoding, copies, verification and existing tie repair must remain timed.

**Closeout fields.** Whether these gaps were resolved is determined by the separately frozen runs and the Phase 13 numerical/performance/readiness reports. This audit is not rewritten to fit their outcomes.
