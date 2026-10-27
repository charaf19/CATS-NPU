# Research progress and evidence ledger

Last updated: 2026-10-27

This ledger records completed work, evidence strength, limitations, and the next gate. Research outputs and figures are stored under `results/`. Manuscript and TeX work is outside the repository scope for this execution.

| Phase | Status | Evidence | Gate |
|---|---|---|---|
| Phase 2: feasibility and batching | Complete | Physical NPU feasibility, strict correctness guards, C++ and CUDA controls, 1,920-observation strict pilot | Engineering gain established; NPU performance advantage not established |
| Phase 3: profiling and optimization | Complete for development | 20-config/1,920-observation NPU sweep; T256/B32 recommendation; vectorized extraction before/after; CPU/GPU gap | Engineering characterization passed; competitiveness failed |
| Phase 4: contribution and novelty | Prototype and critical assessment complete | Exact submission-aware implementation, proof, tests, 1,600-observation pilot, related-work comparison | Systems mechanism partially supported; algorithmic novelty not established |
| Phase 5: real data | Descriptive evaluation complete | SIFT1M, BEIR SciFact, and BEIR FiQA; fixed splits, hashes, precision analysis; 3,360 observations | Real-score feasibility passed; holdouts were untouched at this checkpoint |
| Phase 6: final comparison | Descriptive comparison complete | C++, stable CUDA, direct NPU, fixed NPU, geometric and submission-aware NPU through 1M scores | NPU-internal crossover supported descriptively; CPU/GPU competitiveness failed |
| Phase 7: confirmatory statistics | Complete for the locked SIFT1M question | Six sessions, 2,880 valid observations; direct/geometric NPU speedup 1.666×, CI 1.646–1.685×, exact p=0.03125; 1.370× in a post-hoc no-tie sensitivity | Narrow within-NPU advantage passed; tie effect and one-machine scope limit interpretation |
| Phase 8: predictive contribution and go/no-go | Complete for tested predictor | 48 real and 16 synthetic development cases; frozen six-session BEIR test with 4,608 correct observations; three integrated sessions with 1,728 correct observations | **NO-GO for the tested predictive contribution and a Q2 manuscript now:** zero-overhead method selection gain does not survive decision overhead and is reproducible by a static shape map |

## Verified Phase 2 baseline

- Repository evidence was checked in `results/phase2_audit/final_validation.json`, `results/phase2_pilot/key_metrics_strict.json`, `results/phase2_pilot/analysis.json`, and `results/phase2_audit/source_change_summary.json`.
- The validated source baseline was committed as `49c5a37`.
- Default tests at that point: 123 passed, three physical tests skipped.
- Hardware suite: 124 passed, two documented bitonic expected failures.
- Physical pilot: 1,920 observations, no skips, no wrong results.
- The optimized C++ baseline remained much faster than the NPU implementation.

## Phase 3 record

Changes:

- added bounded ablation automation and per-configuration process isolation;
- added raw aggregation, validity checks, recommendations, CSV tables, and SVG figures;
- profiled host staging, synchronous backend interval, output, validation, extraction, merge, certification, submissions, and bytes;
- vectorized candidate extraction across equal-budget tile rows.

Measurements:

- all 20 sweep configurations valid; 1,920 observations;
- recommendation: tile width 256, tile batch 32, query batch 1;
- median T256/B32 improvement over T16/B32: 8.87x fixed and 4.37x adaptive;
- vectorization median total speedup: 1.078x fixed and 1.026x adaptive;
- optimized fixed NPU median slowdown: 20.30x versus C++ and 6.60x versus stable CUDA;
- optimized geometric NPU median slowdown: 34.50x versus C++ and 11.89x versus stable CUDA.

Limitations:

- query batching is not implemented;
- the OpenVINO synchronous interval cannot be split reliably into transfer, execution, and waiting;
- tuning and measurement used development cases rather than held-out sessions.

Details: `docs/PHASE3_PROFILING.md`.

## Phase 4 record

Changes:

- added `numpy_costadaptive` and `ov_<device>_<kernel>_*_costadaptive`;
- added a measured submission-versus-candidate jump rule without changing the exact certificate;
- added policy telemetry, regression coverage, a pilot script, and comparative analysis;
- fixed PowerShell analysis parameters that conflicted with automatic `$input`.

Measurements:

- 1,600 Phase 4 pilot observations, no skips or wrong results;
- submission-aware beat geometric in 13/16 cases;
- geometric/submission-aware latency ratio: 0.928x minimum, 1.272x median, 2.263x maximum;
- benefits coincided with avoided submissions on hard distributions;
- easy cases sometimes lost because the jump transferred more candidates without saving a submission;
- C++ remained fastest.

Interpretation:

- exactness and termination are proved under a correct local Top-K primitive;
- the policy shows a falsifiable and measured systems effect;
- related work contains close precedents for certified refinement, threshold filtering, hardware-aware adaptation, and cost-based selection;
- no broad novelty claim is supported.

Details: `docs/ALGORITHMIC_CONTRIBUTION.md` and `docs/NOVELTY_ASSESSMENT.md`.

## Phase 5 and 6 record

- Acquired the original SIFT1M archive and verified all computed FP32 Top-100 sets against supplied ground truth.
- Acquired BEIR SciFact and FiQA archives with published MD5 verification.
- Pinned semantic embeddings to `all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Preserved eight confirmatory queries per dataset.
- Collected 2,240 SIFT comparison observations and 1,120 BEIR observations with no skips or wrong results.
- Observed a within-NPU crossover: direct native TopK is best at small N; certified geometric refinement is faster on larger SIFT/FiQA arrays.
- Observed no NPU crossover against optimized C++ or stable CUDA.
- Found submission-aware refinement inconsistent on real data: median geometric/submission-aware ratio 1.014 on SIFT and 0.961 on BEIR.

Details: `docs/REAL_DATA_EVALUATION.md` and `docs/FINAL_BENCHMARKS.md`.

## Phase 7 record

- Locked revision `7fd0a10`; all six sessions passed exact correctness and device-placement checks.
- The prespecified primary direct-versus-geometric NPU comparison passed its statistical and 1.10× practical criteria.
- FP16 boundary tie on SIFT query 15 triggered roughly 0.46–0.48 seconds of measured host repair in direct NPU and CUDA at locked revision `7fd0a10`. A post-hoc analysis excluding it still favors geometric NPU by 1.370×.
- A subsequent audit identified an avoidable known-unique-ID uniqueness sort in that CPU repair. Revision `fdb5e1c` removed it while preserving stable output. A same-process exploratory microbenchmark fell from 305.79 to 4.70 ms median on query 15. The historical Phase 7 estimate remains unchanged but its magnitude does not carry over to the corrected code.
- C++ and stable CUDA remain substantially faster than every NPU method under host-input, host-output timing.

Details: `docs/STATISTICAL_VALIDATION.md`.

## Phase 8 decision record

- A shape-only leave-one-query-out rule had 1.048× regret to the per-case oracle over previously used real-score evaluation cases, assuming zero lookup cost. The label medians came from separate Phase 5/6 sessions, so this is a development feasibility estimate, not confirmatory evidence.
- A cheap score-sketch decision tree lost to that shape rule on grouped real-data development checks, though it helped synthetic cases. A 128-tree forest had roughly 5.1 ms measured decision cost.
- The tree's frozen BEIR holdout choices beat the predeclared shape rule by 1.097× with six-session interval [1.094, 1.100] and exact p=0.03125 **before decision overhead**. Every differing choice was the same fixed-hierarchy switch for SciFact `N=5183,K=4`; the model made no query-dependent choice within any held-out shape.
- Three integrated exploratory sessions measured 0.461 ms median decision cost and found original-shape/dynamic-tree latency ratio 0.924×. A static `N,K` map reproducing every frozen tree choice achieved static/dynamic-tree ratio 0.836×. All 1,728 observations were exact on the requested NPU.
- Prior exact Top-K work already includes threshold-guided selection and hardware-aware algorithm dispatch. The current system has a valid narrow systems observation but lacks a differentiated predictive mechanism, measured net benefit, broad hardware generalization, and universal exactness proof over opaque NPU numerical behavior.

The publication decision is **NO-GO at this milestone**. Do not prepare a manuscript or run another confirmatory study of this decision tree. A new line of work would first need to articulate and development-test a mechanism that cannot be replaced by a zero-cost shape map, preserves the exactness contract on its declared numerical domain, and earns enough benefit to pay its measured dispatch cost. Preserve both Phase 7 and Phase 8 raw records unchanged. Details: `docs/PREDICTIVE_FEASIBILITY.md` and `docs/PUBLICATION_GO_NO_GO.md`.
