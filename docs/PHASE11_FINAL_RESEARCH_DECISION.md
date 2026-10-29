# Phase 11 final research decision

**NO-GO for the investigated rank-stable embedding-calibration contribution. Date: 2026-10-29.** The originality gate failed before substantial implementation. No sufficiently distinct method was found beyond ordinary learned rotation/ranking distillation and established numerical preprocessing. This is a completed early-stop assessment under the mission's rules; it is not a completed numerical retrieval evaluation or a claim that every possible method in this field is unoriginal.

**Original scientific contribution.** None established at manuscript standard. The concrete candidate is a frozen-encoder orthogonal transformation trained to match pairwise reference rankings or margins. It is expressible as the required simple rank-aware rotation baseline. Error-covariance language, different weights on small-margin pairs, or wrapping an actual NPU scorer does not by itself supply a new capability. The [novelty assessment](PHASE11_NOVELTY_ASSESSMENT.md) gives the formal reduction, comparison with the closest applicable methods, primary references through October 2026, and explicit limits on source access. No identical FP16 Intel implementation is claimed to exist in the literature.

**Mathematical findings.** The [foundation](PHASE11_MATHEMATICAL_FOUNDATION.md) specifies the exact-reference contract, geometry defect, stage-wise error identity, known pairwise stability conditions, and the absence of a rotation-dependent improvement in the standard norm envelope. Two restricted negative properties sharpen the feasibility boundary: a geometry-preserving rotation cannot remove output-only rounding ties, and reciprocal powers-of-two scaling leaves normal-range quantized coordinate products unchanged under stated assumptions. These are elementary numerical facts applied to this hypothesis, not a new calibration theory or universal hardware theorem.

The executed CPU audit demonstrates the output-tie witness and checks 1,480,650 safe-region scalar rounding cases with zero dyadic-homogeneity mismatches. This is bounded analytical support, not an improvement experiment or a test of actual NPU accumulation. No synthetic embeddings were substituted for the real datasets.

**Implementation and physical evidence.** No new calibration method, optimizer, scoring graph or encoder regeneration was implemented. A small read-only inventory/analytical audit is archived under `results/phase11/20261029/`. Device discovery found Intel AI Boost 3720 with the previously recorded driver/compiler versions; **no NPU inference was run in Phase 11**. Historical physical TopK records remain historical. NPU availability was not the blocker: the originality stopping rule applied first.

**Real numerical problem.** The [feasibility-status report](PHASE11_NUMERICAL_FEASIBILITY.md) verifies source availability and explains why old score-conversion/selection results cannot establish NPU similarity-scoring damage. Original embedding matrices were not retained by the old preparation script; the exact pinned model cache and source corpora are available for possible regeneration. Damage frequency, affected margins, arithmetic-stage attribution, retrieval relevance, and feasibility/cost of a wider-precision control remain **unmeasured**, not negligible or disproved.

**Retrieval quality, performance, statistics and generalization.** No Phase 11 Top-K overlap, rank-inversion rate, nDCG, Recall, end-to-end latency, query throughput or calibration time was measured. There is no empirical advantage claim, confidence interval, significance result or held-out generalization result. No final confirmation queries or relevance judgments were consumed. [The experimental stopping record](PHASE11_EXPERIMENTAL_PROTOCOL.md) explicitly documents the absence of a frozen or activated confirmatory protocol. Repeated scalar arithmetic checks are not independent query or hardware-session replications.

**Independent gates.**

| Gate | Status | Evidence and limit |
|---|---|---|
| A — Originality | **FAIL** | Concrete rotation/loss reduces to the closest equivalent baseline; alternative scaling supplies no distinct capability. No new mechanism or stronger useful guarantee established. |
| B — Mathematical validity | **PASS for the stated analytical facts; no new method guarantee** | Explicit proofs and bounded host checks. They do not model the actual NPU scorer. |
| C — Real numerical problem | **NOT ESTABLISHED** | No physical similarity-scoring experiment; historical stored-score evidence cannot answer it. |
| D — New method implementation | **NOT SATISFIED** | No justified new mechanism to implement; audit code is not represented as a calibration method. |
| E — Meaningful empirical advantage | **NOT ESTABLISHED** | No method-versus-baseline ranking or cost observations. |
| F — Generalization | **NOT ESTABLISHED** | No new query experiment or final holdout evaluation. |
| G — Reproducibility | **PASS for the limited assessment/audit artifact** | Source-base revision, inspected software revisions/blob IDs, source hashes, dataset archive checks, analytical code and raw audit output. No unperformed experiment is presented as reproducible evidence. |

**Scientific risks and remaining limits.** The literature review is targeted, not exhaustive, and two requested sources had limited full-text access. The actual scorer may have a meaningful problem; no conclusion about its size follows. A genuinely new hardware-error-aware representation remains conceivable. Neither generic optimization against a device oracle nor a larger benchmark would resolve the present missing distinction. One bounded timing experiment cannot turn the current candidate into a different mechanism, so CONDITIONAL GO is not justified.

**Reproducibility.** Starting revision: `345c24cebd717b57e167ca6e696bbcd1e819b2d4`. The Phase 11 closeout commit contains these reports and the evidence manifest. From the repository root:

```powershell
.\.venv\Scripts\python.exe results/phase11/20261029/audit_phase11_basis.py --out results/phase11/reproduced/basis_audit.json
.\.venv\Scripts\python.exe -m ruff check results/phase11/20261029/audit_phase11_basis.py
```

The output path must be new. The audit reads local archives, manifests, model-cache presence and package/device discovery; it does not download, train, run inference, inspect relevance labels or compute real scores. Its four analytical assertions and the archive SHA-256/MD5 checks passed; Ruff passed. The manifest binds report/source/evidence hashes. No production code, previous report, historical result or locked protocol was overwritten; only the progress ledger gains this phase's status.

**Final disposition.** Stop development of these candidates. No manuscript, journal selection, paper figure, model training or confirmatory campaign is justified by this assessment. The useful outcome is a precise novelty reduction and numerical feasibility boundary, with the hardware-scoring question explicitly left unanswered. Reopening requires an independently defensible new mechanism/property before that question becomes a justified experiment for this research mission.
