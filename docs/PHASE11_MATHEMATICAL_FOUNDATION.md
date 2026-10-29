# Phase 11 mathematical foundation and limits

**2026-10-29 — Established numerical facts and rejected candidate properties; no new method theorem claimed.** This document formalizes the problem after the [novelty review](PHASE11_NOVELTY_ASSESSMENT.md). It does not supply an executable Intel NPU arithmetic model or claim that its assumptions were measured on that device.

**Reference contract.** Freeze the original normalized FP32 query/document embedding values. Treat those stored binary values as exact real numbers and define `s_i = sum_l q_l*d_il`. This real sum is the mathematical reference. A conventional FP32 reduction is a separate implementation; an FP64 reduction is more accurate but is not automatically an exact real sum. FP32 products can be represented exactly in FP64, while their sum can still round. A rigorous numerical comparison must use an exact accumulator or a justified reference-error bound; neither was applied to real embedding arrays in this phase.

Rank score descending, then ascending original corpus position. Signed zeros are numerically equal; NaNs/infinities are unsupported. Document-ID strings need not sort in corpus order, so the position-to-ID map must be preserved. Numerical rank fidelity concerns this fixed reference. nDCG/Recall against relevance labels is a different quantity: better agreement with the reference does not imply better semantic retrieval.

**Geometric invariance.** For an exact common orthogonal R, `(Rq)^T(Rd)=q^T d`, norms and Euclidean distances are unchanged, and cosine is unchanged when defined on the same vectors. This is standard linear algebra. If the stored transform is A instead, its geometric defect is `q^T(A^T A-I)d`, bounded by `||A^T A-I||_2 ||q||_2 ||d||_2`. Rounded transformation application adds further error. Orthogonality of an ideal parameterization does not eliminate those finite-precision terms. Re-normalizing transformed/rounded vectors is an additional operation that changes the numerical path and must be analyzed explicitly.

**Error decomposition.** Let u and v be the actual operands supplied to the low-precision scorer, and write `u=Rq+delta_q`, `v=Rd+delta_d`, where the deltas include transform-application and input-rounding errors. Define the internal aggregate error alpha by `m=u^T v+alpha`, and output conversion error beta by `h=m+beta`. Then exactly

    e_R(q,d) = h - q^T d
             = delta_q^T Rd + (Rq)^T delta_d
               + delta_q^T delta_d + alpha + beta.

This is an identity, not an error bound for the NPU. If deltas are instead defined relative to A, add the geometric-defect term and replace R by A in the cross terms. Mixing those two definitions would double-count error. Alpha may include compiler conversions, multiplication/reduction error, and execution-specific effects. No assumed FP32 accumulation bound is silently assigned to it.

For a query and two documents, shared query error contributes `delta_q^T R(di-dj)`. Pairwise errors can therefore cancel even when absolute score errors do not. However, optimizing the residual difference is exactly optimizing a teacher/student score-margin discrepancy; that observation does not establish a new calibration mechanism.

**Pairwise and boundary conditions.** For `s_i>s_j`, define `Delta_ij=s_i-s_j`. Strict ordering survives if `e_j-e_i<Delta_ij`; the stronger condition `|e_i-e_j|<Delta_ij` is sufficient. If `|e_i|<=epsilon_i`, then `Delta_ij>epsilon_i+epsilon_j` is sufficient. Applying this to all original Top-K/outside pairs preserves membership; the familiar uniform condition is boundary margin `>2*epsilon`. Internal order requires the relevant internal pair conditions as well. Zero reference margin has no strict separation guarantee; exact equalities and the index tie rule must be handled separately. These are known perturbation facts, including in the 2026 retrieval-margin literature, not new theorems.

**Why a standard bound supplies no new rotation guarantee.** With round-to-nearest normal FP16 inputs and no underflow/overflow, `||delta_q||<=u16||q||` and similarly for d when transformation is exact, with `u16=2^-11`. The input term is bounded by `(2*u16+u16^2)||q||||d||`. Orthogonal transforms preserve the norms, so this worst-case bound is independent of R. Actual errors depend on coordinates and can improve or worsen. Subnormals require absolute-error terms and flush-to-zero is a different model. The actual NPU's conversions and accumulation remain unmeasured; GPU mixed-precision analysis [R15 in the novelty report] cannot establish their bounds by analogy.

For an explicitly assumed scalar sequential arithmetic model with unit roundoff ua, a standard reduction bound involves `gamma_D = D*ua/(1-D*ua)` when `D*ua<1` and a sum of absolute products. A blocked/fused accelerator may follow another model. Replacing that sum by `||q||||d||` via Cauchy–Schwarz again yields an invariant envelope, not a proof that a learned rotation reduces ranking risk. Neither training loss nor a lower reconstruction error guarantees unseen-query ordering.

**Output-rounding obstruction.** Consider an ideal scorer with exact transformed inner products followed only by FP16 round-to-nearest, `h_R=RN16((Rq)^T(Rd))`. Invariance gives `h_R=RN16(q^T d)` for every orthogonal R. Thus an output tie produced solely by that final quantizer cannot be removed by an otherwise exact geometry-preserving rotation.

For an explicit witness, let `a=1/2+2^-14` and `b=1/2+2^-13`. They are distinct, exactly representable FP32 numbers, but `RN16(a)=RN16(b)=1/2`. Put the smaller score at corpus position 0. The reference Top-1 is position 1; stable rounded Top-1 is position 0. These scores can occur for unit vectors: take `q=(1,0)`, `d0=(a,sqrt(1-a^2))`, `d1=(b,sqrt(1-b^2))` in exact real arithmetic. This existence construction is not a substitute for the intended real embeddings.

This proves only the stated output-only obstruction. Operand errors or a deliberately geometry-changing calibration could move one score into another bin, and a rotation may reduce errors from other stages. It does not prove that rotations never help retrieval. It prevents claiming universal recovery of pure output-rounding losses from orthogonal invariance alone. Higher-precision scoring or correctly defined tie controls remain essential comparators.

**A distinct alternative checked: reciprocal dyadic scaling.** Let `S=diag(2^t_l)`. Exact inner products satisfy `(Sq)^T(S^-1 d)=q^T d`. In a normal-range binary floating-point region where original/scaled rounding neighborhoods remain normal and finite, round-to-nearest commutes with exact exponent shifts:

    RN16(2^t*x) = 2^t*RN16(x).

Proof: multiplying by a power of two shifts the exponent of both adjacent representable values and their midpoint without changing their significands or tie-to-even parity. Distances to the neighboring rounding candidates scale equally, so the chosen candidate scales identically. The premise excludes underflow, overflow and neighborhoods crossing the exceptional boundaries. It follows coordinatewise that

    RN16(2^t*q_l) * RN16(2^-t*d_l)
      = RN16(q_l) * RN16(d_l)

as exact real products. A fixed sum of those products is unchanged. Actual backend code generation could introduce additional behavior; the identity is not an installed-executable equivalence proof. The CPU audit tested midpoint/predecessor/successor cases for both signs and t in −4..4 within a deliberately narrower safe normal region: 1,480,650 eligible cases, zero mismatches. This bounded check supports the ordinary rounding argument; it is not exhaustive over all FP32 values or hardware states.

Consequently the error-free-exponent-scaling idea has no normal-range operand-error improvement under this model. Exceptional-range mitigation could still matter but supplies neither an established new mechanism nor real-query evidence here. General non-dyadic scaling changes mantissas and returns to the established precision-balancing family; its usefulness would need separate analysis.

**Candidate optimization, cost, and limitations.** The candidate is the ordinary constrained ranking objective specified in the novelty report. Orthogonal optimization is nonconvex; a straight-through quantization gradient is a surrogate rather than a proof of descent for the device's discrete ranking objective. A zero calibration loss supplies no distribution-free guarantee for unseen queries, another corpus, or another compiler path. No optimizer, convergence result or robustness theorem was developed in this phase.

A dense D-by-D query transformation costs O(D^2); full corpus transformation costs O(ND^2) offline and produces another representation. D=384 is not a power of two: structured transforms need an explicit block/Kronecker construction or isometric padding, with a possibly different scored dimension and computational cost. It is invalid to assume a free D-dimensional Walsh–Hadamard transform. Calibration scoring, pair selection, corpus preprocessing, online transformation, transfers, scoring and stable selection all have distinct costs. No measured complexity or latency of a proposed method is claimed.

**Mathematical decision.** Problem definition, identities, standard sufficient conditions, and restricted negative properties are valid under their explicit premises. They do not meet the requirement for a genuinely new useful bound or mechanism. Full real-query references, physical accumulation semantics, optimized calibration, generalization and numerical-risk guarantees remain unestablished. The [final research decision](PHASE11_FINAL_RESEARCH_DECISION.md) is NO-GO on originality, not a universal impossibility claim.
