# Phase 13 final publication-readiness decision

**READY FOR MANUSCRIPT — for a bounded empirical numerical-fidelity and performance study on the recorded Intel NPU stack.** The essential Phase 13 experiments and analyses are complete. This decision concerns whether writing can begin; it does not establish a new algorithm, guarantee journal acceptance or authorize manuscript preparation in this phase. The earlier algorithmic novelty NO-GO remains unchanged.

The defensible framing is: **Exact Top-K output semantics on a consumer NPU depend on the numerical path through the compiled graph; controlled precision-composition probes reveal failures that successful compilation, sorted outputs and ordinary retrieval tests can miss, while correctness-aware execution has workload-dependent costs.**

The original empirical value lies in the reproducible combination of contract-level failure mapping, isolated composition witnesses, and prospectively frozen latency trade-offs. Neither generic accelerator numerical characterization nor mixed precision is claimed as a first discovery.

## Completed evidence and contributions

| Contribution | What this project establishes | Boundary |
|---|---|---|
| **C1 — Numerical fidelity** | Three fresh processes, 54 independently generated numerical cases, six configurations and saved raw CPU/NPU outputs replicate membership, value and order failures. The bounded adjacent batch independently reproduces 2,046 membership failures and 2,047 complete failures in default execution. | Counts describe these executed domains; repeated witnesses are not independent defects or population error-rate estimates. |
| **C2 — Precision-sensitive composition** | All nine W/K cross-tile settings fail with TopK-only or Reshape-only preservation; combined preservation and stable INT32 pass. The result extends the original witness across widths, K and irregular padding. | Existing precision controls are used; no new compiler analysis, universal minimal preservation set or instruction-level cause is proved. |
| **C3 — Correctness/performance** | Twelve frozen process sessions on 24 fresh query IDs, six methods and 48 workloads quantify full host-input/host-output costs. FP32 has a meaningful paired SIFT advantage over INT32; small/uncertain/reversed effects elsewhere and faster CPU/CUDA controls are retained. | Inference is conditional on one device/version, the fixed query panel and the session design; no general NPU supremacy or retrieval-quality benefit. |

The decisive new evidence beyond Phase 12 is independent input construction and process replication, a nine-setting composition matrix, fresh query IDs selected before scores, balanced method order, prespecified session-level statistics, targeted N/K/W sensitivities and complete raw-output rechecking. This closes the exploratory-performance and narrow-composition gaps without changing the algorithms.

Historical evidence was audited before collection: all **1,641 Phase 12 manifest-bound files** and source archive checksums verified. All previously prepared query IDs, including consumed Phase 7/8 confirmation queries, were excluded. The original evidence-gap audit and protocol remain prospective records; this report supplies their outcome, rather than rewriting them after observing results.

## Numerical result and exactness scope

The primary numerical panel contains **972 case/configuration/session rows, 1,944 physical NPU calls and 972 CPU graph controls**. There were no construction/runtime errors, and every CPU control passed. Every NPU executable reported strict `["NPU"]` placement. All six NPU outputs per unique case/configuration were byte-identical across processes/repeats.

Default FP16, unprotected FP32 promotion, TopK-only preservation and Reshape-only preservation each failed **29/54 unique inputs**, or **174/324 calls per setting**. Their error categories differ: default/promotion/Reshape-only each have 168 membership and 54 value failures; TopK-only has 144 membership and 174 value failures. Categories overlap. All configurations pass the normal-neighbor and signed-zero families; both combined preservation and INT32 pass every case in the panel.

All nine explicit cross-tile W/K settings show TopK-only failure: **54/54 membership and value failures**, versus zero with combined preservation or INT32. This experimentally demonstrates that protecting the selector alone can be insufficient for the tested graph's complete ranking/value contract. Abstract reshape remains an exact routing operation; the observation concerns the compiled value path under different precision settings.

Passing the adjacent direct promoted selector with TopK preservation does not predict passing the hierarchical graph. Likewise, all ordinary retrieval calls passing does not imply safety near subnormal cutoffs. Five SciFact and 234 FiQA scores in the new vectors are subnormal, but none lies in a top-100 result; SIFT has none. No real-query selection error was observed.

The inherited promotion/hierarchy/checker arguments are retained in [the mathematical foundation](PHASE12_MATHEMATICAL_FOUNDATION.md). The supported input contract is immutable **finite FP16**, descending numerical score and ascending original index; signed zeros are equal. NaNs/infinities and invalid or uncompiled sizes are explicitly unsupported. Integer routing/dimensions and device resources impose further implementation limits. The host certificate proves **exact-or-reject** under trusted host execution; sampled hardware agreement does not prove total device correctness. No wrong NPU proposal is silently replaced with a CPU-selected answer in the checked FP32/INT32 methods.

Details, disjoint error partitions and raw links: [numerical report](PHASE13_NUMERICAL_RESULTS.md), `results/phase13/analysis/table2_numerical_correctness.json`, `table3_composition_ablation.json` and `results/phase13/supplemental/`.

## Performance, significance and negative findings

The preregistered primary workload is SIFT1M, N=1,000,000, K=10, W=256, eight fresh queries, twelve separately launched sessions, six timed repetitions per query/method/session. The protocol, generators, method code and primary analysis were frozen at **`d97544ac4514bafbb81b0cf38044982f43fb1379`**, before fresh score generation. Three old-query pilots justified twelve sessions by precision planning; the confirmatory sample was not extended after results.

**Paired INT32/FP32 geometric mean ratio: 1.284668; 95% session-cluster bootstrap CI [1.254773, 1.314964]; two-sided exact sign-flip p=0.000488281.** This is about **22.16% lower FP32 latency**, and clears the frozen lower-CI >1.10 practical rule. There are no rejected or incorrect primary calls. Crossed query/session CI [1.250864, 1.319822], six consecutive-pair block CI [1.257912, 1.311788] and block p=0.03125 support the direction within the study. The primary success does not depend on choosing a favorable dataset after observation.

Across the full matrix, **20,736 timed and 6,912 warm-up calls** returned the independently expected stable IDs, with zero rejections, accepted errors, construction failures or invalid environment sessions. All methods occupied all six timing positions once per case/session. Latencies include validation/encoding, staging, submission/wait, copying, verification and existing tie logic. Compilation, warm-up and the independent oracle are excluded consistently.

Median query/session latencies (ms), with K=10/W=256:

| Method | SIFT1M | SciFact | FiQA |
|---|---:|---:|---:|
| Checked FP32 | 32.0308 | 1.3395 | 5.1247 |
| Checked INT32 | 42.8180 | 1.4047 | 5.2450 |
| Existing FP16 static wrapper | 33.2578 | 1.3673 | 5.1055 |
| Native NPU wrapper | 178.8289 | 2.1738 | 20.5106 |
| C++ CPU | 22.4144 | 0.1914 | 1.5114 |
| Stable CUDA | 7.2143 | 1.1382 | 1.8785 |

These absolute medians have a different aggregation from the paired ratio. Session uncertainty, pooled descriptive P95, sequential reciprocal-latency rates and all fourteen Holm-adjusted secondary comparisons appear in [the performance report](PHASE13_PERFORMANCE_RESULTS.md) and Tables 4–6.

Negative and limiting results are substantive: SciFact's INT32/FP32 difference is only about 4.8% in ratio terms despite adjusted significance; FiQA's 1.028 ratio has Holm p=0.1304 and does not establish superiority. A two-query FiQA K=1/W=128 sensitivity slightly favors INT32. SIFT's ratio falls to 1.056 at K=100/W=256. CPU and CUDA have lower paired median latency on every base dataset after Holm correction. SciFact CUDA has a worse pooled P95 than FP32, so median comparisons must not be generalized to every tail measure. No concurrent throughput experiment was performed.

The native wrapper comparison includes its existing validation; it is not pure device kernel speedup. NPU graph methods already use one request, so these comparisons do not show a submission-count reduction. All native/CUDA base calls recorded zero expensive tie-repair fallback invocations; the historical inflated Phase 7 effect is not reused.

## Closest prior work and empirical differentiation

References were verified against primary papers, author repositories and official documentation on 2026-11-03. The structured record is `results/phase13/prior_art_review.json`. This is a targeted comparison, not a systematic proof of absence of prior reports.

| Prior work | Established contribution relevant here | What remains specific to this evidence |
|---|---|---|
| Fasi et al., 2021 [R1] | Designed probes characterize tensor-core arithmetic, including subnormal and precision behavior. | Exact Top-K membership/value/stability contracts and NPU graph-composition ablations. Characterization as a methodology is inherited. |
| Xie et al., FPRev, ATC 2025 [R2] | Numerical probes infer software/hardware accumulation order. | Output-level ranking evidence with explicit attribution limits; no recovered instruction tree or new inference method. |
| Kumaresan, Orion, 2026 preprint [R3] | Apple NPU compiler/runtime restrictions and numerical characterization. | Intel Top-K contract and precision-composition evidence; no claim to introduce NPU characterization. |
| onw author technical notes [R4] | Reports Intel NPU subnormal-related degradation and scaling remedies in inference. | Bounded exact-selection output counts and controlled composed-graph failures. Broad subnormal loss is not exclusive to CATS-NPU. |
| npunlock author software [R5] | Exposes custom Intel SHAVE kernels and validated FP32 execution paths. | Stock graph precision ablations and ranking contracts; no new NPU programmability or custom-kernel mechanism. |
| TensorRT/OpenVINO documentation and compiler source [R6–R8] | Precision controls, FP32 mitigation and stable ordering semantics already exist. | Independent physical validation of which settings pass these particular graphs, and their complete measured costs. |

The previous CUB, DeepSelect, RadiK, hierarchical selection and deterministic-order comparisons remain in [the original contribution audit](FINAL_RESEARCH_CONTRIBUTION.md). The algorithms, integer order embedding, stable tie rule, hierarchy and full-source rank certificate remain conventional. An empirical paper must explain how its observed counterexamples, composition controls and reproducible cost evidence add useful knowledge; renaming the baseline supplies no novelty.

The specific practical lesson is experimentally supported: **a successful direct-selector precision test and successful ordinary retrieval tests are insufficient validation for a composed exact-selection graph**. Independent membership, value and stable-order checks are necessary to reveal the recorded differences. This lesson is narrower than a general compiler theorem and more informative than reporting another Top-K speedup.

## Limitations and excluded claims

One Core Ultra 7 255H / NPU 3720 laptop, Windows build 26300, OpenVINO 2026.4.1, driver 1004512 and compiler 458778 define the external-validity scope. No second generation/version was available or measured. Twelve sequential processes on one date do not establish independent thermal/OS states or cross-day reproducibility; bootstrap/sign-flip assumptions remain explicit. Eight queries per dataset and two per sensitivity setting limit query extrapolation. Source scoring is FP16-rounded; original FP32 retrieval order and retrieval relevance were not evaluated.

The adversarial panel is not a random production distribution. Exhausting adjacent pairs in one fixed batch does not exhaust arbitrary graphs. Native/FP16 wrappers passing retrieval inputs does not erase their raw graph counterexamples. Internal compiler IR and instruction behavior are opaque; public source consistency does not identify a unique cause in the installed binary. No energy, peak hardware memory, concurrency or end-to-end RAG claims are made.

Statistical significance is not the basis of readiness by itself. The main evidence is replicated numerical counterexamples, passing controls, intervention-based graph composition tests, fresh-query full-cost comparisons, transparent null/negative results and reproducible artifacts. The study establishes no new algorithm, no automatic compiler repair, no universal hardware proof, no first-ever accelerator subnormal discovery and no promised Q2 acceptance.

## Integrity, completion and disposition

All scheduled attempts completed, no performance outlier was removed, and all six required tables were regenerated from raw records. Tables 1–6 are JSON under `results/phase13/analysis/`; `supplemental/table6_scope_and_limitations.json` adds the explicit scope/failure mapping to Table 6 without changing frozen inference. Existing source algorithms and historical Phase 7–12 records are preserved. The final file manifest, source digests, test logs, native binary snapshot, commands and [reproduction instructions](PHASE13_REPRODUCIBILITY.md) support independent rechecking.

The software suite reports **219 passed, 3 skipped**; those three tests require an opt-in physical-regression flag. They are not counted as executed hardware checks. Actual Phase 13 device evidence comes from the saved independent numerical/performance processes. Source Ruff checks pass. The post-freeze supplementary scripts perform descriptive integrity checks and output routing only; the frozen experiment/analysis required no amendment.

**No essential Phase 13 evidence gap remains for starting a carefully scoped empirical manuscript.** A second device/version and multi-day replication would strengthen later external validation, but are not represented as completed or silently required to reinterpret these results. Writing is a separate phase. Historical method-novelty NO-GO decisions stand; this READY decision applies only to the revised empirical C1–C3 objective.

## Verified references

1. **[R1]** Massimiliano Fasi, Nicholas J. Higham, Mantas Mikaitis, Srikara Pranesh. *Numerical Behavior of NVIDIA Tensor Cores*. PeerJ Computer Science 7:e330, 2021. DOI: [10.7717/peerj-cs.330](https://doi.org/10.7717/peerj-cs.330). [Author paper](https://eprints.maths.manchester.ac.uk/2791/1/fhmp20.pdf).
2. **[R2]** Peichen Xie, Yanjie Gao, Yang Wang, Jilong Xue. *Revealing Floating-Point Accumulation Orders in Software/Hardware Implementations*. USENIX ATC 2025, pp. 1425–1440. [Primary publication](https://www.usenix.org/conference/atc25/presentation/xie).
3. **[R3]** Ramchand Kumaresan. *Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference*. arXiv:2603.06728v1, 2026, preprint. [Primary text](https://arxiv.org/html/2603.06728v1).
4. **[R4]** ryugyosoft. *onw technical notes*. Primary author software report, revision `7ed25189ba7f10aa25a281859ed9492b53b2164e`; not peer-reviewed. [Pinned text](https://huggingface.co/ryugyosoft/onw/blob/7ed25189ba7f10aa25a281859ed9492b53b2164e/TECHNICAL_en.md).
5. **[R5]** hsfzxjy and contributors. *npunlock: Write and run custom C kernels for Intel Core Ultra NPUs*. Primary software repository and [compatibility limits](https://github.com/hsfzxjy/npunlock/blob/master/wiki/LIMITATIONS.md), accessed 2026-11-03. [Repository](https://github.com/hsfzxjy/npunlock).
6. **[R6]** NVIDIA. *TensorRT Accuracy Considerations*. [Official documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html), accessed 2026-11-03.
7. **[R7]** Intel/OpenVINO. *TopK-11 operation specification*. [2026 documentation](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/sort/top-k-11.html), accessed 2026-11-03.
8. **[R8]** OpenVINO NPU compiler contributors. *ConvertPrecisionToFP16* and public precision option, revision `c74b1265f28dc5a8f08dd50babd2a063cb59ac6e`. [Transformation source](https://github.com/openvinotoolkit/npu_compiler/blob/c74b1265f28dc5a8f08dd50babd2a063cb59ac6e/src/vpux_compiler/src/dialect/IE/transforms/passes/convert_precision_to_fp16.cpp), [option source](https://github.com/openvinotoolkit/npu_compiler/blob/c74b1265f28dc5a8f08dd50babd2a063cb59ac6e/src/vpux_compiler/include/vpux/compiler/core/public_options.hpp). Pinned Phase 12 source audit retained; exact installed-binary equivalence is unestablished.
