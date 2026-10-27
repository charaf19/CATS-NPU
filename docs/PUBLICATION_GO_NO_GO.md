# CATS-NPU publication decision, 2026-10-27

**Decision: NO-GO for a Q2 journal manuscript on the current predictive execution strategy.** This is a decision about the evidence and implementation now in the repository, not a proof that exact NPU Top-K can never yield a publishable contribution. No manuscript, TeX file, or paper figure has been added to the project. Research figures and raw results remain under `results/`.

| Criterion | Assessment | Evidence and limiting fact |
|---|---|---|
| Scientific originality | **Not established** | Exact hierarchical filtering, threshold-guided refinement, and hardware-aware Top-K dispatch have close prior art. The tested tree's unseen BEIR choices collapse to a static `N,K` mapping. See `docs/NOVELTY_ASSESSMENT.md` and `docs/PREDICTIVE_FEASIBILITY.md`. |
| Correctness | **Empirically supported for measured workloads; universal claim unavailable** | Phase 7 and Phase 8 runs have zero incorrect outputs; strict models report `NPU`. The certificate proof assumes a correct local Top-K primitive. Known FP16-subnormal NPU counterexamples are detected and rejected by host validation, not corrected by silent CPU substitution. See `docs/MATHEMATICAL_SPECIFICATION.md` and the physical NPU tests. |
| Predictive validity | **Failed for the tested model** | Frozen BEIR choice comparison yielded a favorable zero-overhead 1.097× ratio, but every differing choice was the same SciFact `K=4` shape switch. No held-out per-query adaptation was demonstrated. |
| Practical performance | **Failed for the tested model** | In three integrated exploratory sessions, original shape-rule latency / dynamic-tree latency was 0.924×; a static map making identical choices / dynamic-tree latency was 0.836×. Median measured decision cost was 0.461 ms. Phase 7 C++ and CUDA baselines remained substantially faster than NPU selection. |
| Experimental reliability | **Strong for narrow comparisons, insufficient for broad claims** | Phase 7 has a locked six-session SIFT1M comparison; Phase 8 has a separately frozen six-session BEIR choice screen with 4,608 valid observations, followed by a clearly exploratory integrated test with 1,728 valid observations. All runs share one laptop and NPU/compiler stack. The old Phase 7 effect size contains an avoidable CPU tie-repair cost fixed in revision `fdb5e1c`. |
| Journal suitability | **Not ready** | [The Journal of Supercomputing](https://link.springer.com/journal/11227) has a plausible systems/performance scope, but its scope does not remedy an unproven original contribution or a negative net-latency result. Current Q2 classification was not independently verified and must be checked from the applicable year's recognized ranking source before any future venue decision. |

## Strongest defensible findings

1. Exact, stable host-input/host-output Top-K can be executed through explicit Intel consumer-NPU TopK calls with host certification and strict result checks on the tested FP16 workloads. It remains a hybrid CPU/NPU system because staging, candidate merge, certification, and some tie repairs run on the host.
2. The locked Phase 7 implementation's geometric certified method beat native direct NPU TopK by 1.666× on eight SIFT1M queries at `N=1,000,000,K=10` (six-session 95% interval [1.646,1.685], exact p=0.03125). This is a historically valid result for source revision `7fd0a10`. The post-hoc no-tie sensitivity was 1.370×. The stronger original magnitude is partly a software artifact: a known-unique-ID sort during CPU tie repair. The corrected helper retained exact output and reduced the same-process repair microbenchmark from 305.79 to 4.70 ms median. Fresh confirmatory results for the corrected implementation do not exist.
3. A fixed 512+512-score sketch and shallow decision tree sometimes chose a faster NPU method than a peer-query shape rule, but the favorable frozen BEIR result did not represent score-sensitive dispatch. Its actual decision overhead erased the apparent benefit. This negative result is traceable to frozen predictions and raw paired observations, not to an inferred future workload claim.
4. Strong C++ and CUDA controls retain a large practical advantage under the same host-input/host-output contract. The corrected tie repair also removes a misleading CUDA tail on the audited SIFT query 15: an exploratory 20-observation-per-method run measured CUDA at 9.30 ms, C++ at 10.90 ms, geometric NPU at 109.89 ms, and direct NPU at 147.37 ms. This run is separate from Phase 7 and is not confirmatory.

## Mathematical and systems interpretation

For a fixed finite FP16 input `x` and `K`, let `T(x,K)` denote stable exact Top-K. If every candidate implementation `A_j` returns `T(x,K)` on a specified domain, a dispatcher that chooses any `j=f(x,N,K)` also returns `T(x,K)`. This is a closure property, not a new exactness certificate. The real latency is

`L_dispatch(x) = L_score_inspection(x) + L_prediction(x) + L_chosen_method(x)`.

The BEIR zero-overhead gate measured only the last term. On the held-out shapes, the selected `j` was constant within each `N,K` group, so `L_score_inspection + L_prediction` paid a positive cost without changing any per-query method choice. A static map therefore has the same selected-method work with lower decision cost. The observed 0.836× static/dynamic latency ratio is consistent with this mechanism. It does not establish an asymptotic lower bound or imply that all possible predictors fail.

The local-K union and strict omitted-score certificates remain mathematically valid conditional on exact local selection over the supplied precision. OpenVINO's opaque NPU numerical path, known subnormal behavior, and tie handling prevent a proof of universal hardware exactness over all finite FP16 inputs. Future work must state a narrower tested domain or introduce a validation/implementation mechanism that closes this gap without trivializing the NPU computation.

## Evidence and reproducibility

- Locked historical study: `configs/phase7_confirmatory.json`, `results/phase7/confirmatory/20261024T212141Z/`, `results/phase7/confirmatory_analysis.json`, and `docs/STATISTICAL_VALIDATION.md`.
- Corrected tie audit: `results/phase8/tie_repair_reference_audit.json` and `results/phase8/tie_repair_exploratory/`.
- Development feasibility: `results/phase8/predictive_headroom.json`, `results/phase8/synthetic_exploratory_fixed/`, and the tree/forest cross-validation JSON reports.
- Frozen BEIR choices and protocol: `configs/phase8_beir_validation.json` and `configs/phase8_beir_predictions.json`, committed at `fdb5e1c` before new holdout timing.
- BEIR raw validation and paired analysis: `results/phase8/beir_validation/20261027T111433Z/`, `results/phase8/beir_validation_analysis.json`, and its CSV.
- Integrated raw timing and analysis: `results/phase8/integrated_dispatch_exploratory/`, `results/phase8/integrated_dispatch_analysis.json`, and its CSV.
- Reproduce the analyses without rerunning hardware via `./.venv/Scripts/python.exe scripts/analyze_phase8_beir_validation.py results/phase8/beir_validation/20261027T111433Z` and `./.venv/Scripts/python.exe scripts/analyze_phase8_integrated_dispatch.py` from the repository root.

The primary paths above are intentionally separate; no Phase 7 raw observation was overwritten or reused as fresh confirmatory evidence for Phase 8. Existing dataset manifests record source archive hashes, embedding model revision for BEIR, score-array hashes, query splits, and the Top-K precision contract. The `results/` tree is ignored by Git but preserved in this workspace; it must accompany the source when sharing or reviewing numerical claims. `results.zip` is a pre-existing user file and was not modified.

## Conditions to reopen the publication decision

A new contribution would need a mechanism that makes different justified choices for different unseen score vectors at the same `N,K`, or a different static-NPU technique with a distinct mathematical/systems property. It should first beat a zero-cost shape-only comparator on development workloads after all decision overhead, maintain exactness over a clearly declared numerical domain, and survive adversarial ties. Only then should a separate protocol be frozen for genuinely untouched SIFT1M/BEIR queries and, ideally, a second consumer-NPU compiler/hardware stack. A future venue choice should follow that evidence rather than precede it.
