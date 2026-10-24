# Research progress and evidence ledger

Last updated: 2026-10-24

This ledger records completed work, evidence strength, limitations, and the next gate. Research outputs and figures are stored under `results/`. Manuscript and TeX work is outside the repository scope for this execution.

| Phase | Status | Evidence | Gate |
|---|---|---|---|
| Phase 2: feasibility and batching | Complete | Physical NPU feasibility, strict correctness guards, C++ and CUDA controls, 1,920-observation strict pilot | Engineering gain established; NPU performance advantage not established |
| Phase 3: profiling and optimization | Complete for development | 20-config/1,920-observation NPU sweep; T256/B32 recommendation; vectorized extraction before/after; CPU/GPU gap | Engineering characterization passed; competitiveness failed |
| Phase 4: contribution and novelty | Prototype and critical assessment complete | Exact submission-aware implementation, proof, tests, 1,600-observation pilot, related-work comparison | Systems mechanism partially supported; algorithmic novelty not established |
| Phase 5: real data | Descriptive evaluation complete | SIFT1M, BEIR SciFact, and BEIR FiQA; fixed splits, hashes, precision analysis; 3,360 observations | Real-score feasibility passed; confirmatory holdouts untouched |
| Phase 6: final comparison | Descriptive comparison complete | C++, stable CUDA, direct NPU, fixed NPU, geometric and submission-aware NPU through 1M scores | NPU-internal crossover supported descriptively; CPU/GPU competitiveness failed |
| Phase 7: confirmatory statistics | Not yet evaluated | Analysis supports session-level inference; current Phase 3/4 evidence is development-only | Open |
| Phase 8: go/no-go | Premature | Novelty, real data, scaling, and independent sessions missing | Current trajectory: conditional/no-go unless later evidence adds a defensible systems contribution |

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

## Next justified work

1. Lock primary hypotheses around the NPU-internal direct/geometric crossover and the negative NPU-versus-CUDA comparison.
2. Use only the preserved confirmatory queries and frozen T256/B32 configuration.
3. Collect at least six separately initiated, counterbalanced sessions.
4. Apply session-clustered paired inference and practical effect thresholds.
5. Reassess novelty and the publication decision after confirmatory evidence.

No confirmatory or publication decision should be made before those items are evaluated.
