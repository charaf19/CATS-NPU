# Research progress and evidence ledger

Last updated: 2026-10-24

This ledger records completed work, evidence strength, limitations, and the next gate. Research outputs and figures are stored under `results/`. Manuscript and TeX work is outside the repository scope for this execution.

| Phase | Status | Evidence | Gate |
|---|---|---|---|
| Phase 2: feasibility and batching | Complete | Physical NPU feasibility, strict correctness guards, C++ and CUDA controls, 1,920-observation strict pilot | Engineering gain established; NPU performance advantage not established |
| Phase 3: profiling and optimization | Complete for development | 20-config/1,920-observation NPU sweep; T256/B32 recommendation; vectorized extraction before/after; CPU/GPU gap | Engineering characterization passed; competitiveness failed |
| Phase 4: contribution and novelty | Prototype and critical assessment complete | Exact submission-aware implementation, proof, tests, 1,600-observation pilot, related-work comparison | Systems mechanism partially supported; algorithmic novelty not established |
| Phase 5: real data | Not yet evaluated | Existing generic embedding-to-score command only | Open |
| Phase 6: final comparison | Not yet evaluated | Development comparisons at N=4K/16K only | Open |
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

## Next justified work

1. Acquire legitimate real retrieval data with recorded source, checksum, and license metadata.
2. Generate development and held-out FP16 score vectors without including embedding inference time.
3. Test the frozen T256/B32 methods on real score distributions and examine near-boundary ties.
4. Define a bounded scaling matrix through 1M scores and include C++, stable CUDA, fixed NPU, geometric NPU, and submission-aware NPU.
5. Lock primary hypotheses only after real-data feasibility, then collect at least six independent counterbalanced sessions.

No confirmatory or publication decision should be made before those items are evaluated.
