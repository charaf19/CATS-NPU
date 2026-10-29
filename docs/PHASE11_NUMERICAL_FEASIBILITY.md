# Phase 11 numerical-feasibility status

**2026-10-29 — Real NPU scoring damage NOT ESTABLISHED in this phase.** The investigation stopped at the [originality gate](PHASE11_NOVELTY_ASSESSMENT.md). This is neither a finding of negligible damage nor a hardware-failure finding. No new MatMul graph, embedding regeneration, physical scoring call, retrieval metric, or timing was executed. Device discovery and CPU analytical checks are explicitly labeled in [basis_audit.json](../results/phase11/20261029/basis_audit.json).

**What the existing evidence actually measures.** `scripts/prepare_beir_scores.py` generated normalized FP32 embeddings, computed corpus/query similarities outside the selection benchmark, converted the resulting scores to FP16, and saved one-dimensional score arrays. Consequently, its conversion-induced ranking changes combine neither NPU operand quantization nor NPU accumulation. The physical FP16-subnormal TopK counterexample concerns selection of supplied scores. Neither record establishes an NPU dot-product problem. Phase 2–10 timing cannot be reused as Phase 11 scoring latency.

**Completed source and environment inventory.** The CPU audit checked both existing BEIR archives against the recorded SHA-256 and MD5 values; all checks passed. It located the extracted corpus/query files and pinned model cache. No array of original query/document embeddings was found in `data/`, and the workspace filename search for embedding-named NPY/NPZ files found none. This is a bounded workspace inventory, not a claim about other machines or arbitrary archives. The preparation code does not save its embedding matrices; Phase 5's stored arrays are scores.

| Item | Preserved source / discovered state |
|---|---|
| SciFact | 5,183 documents; dimension 384; SHA-256 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` |
| FiQA | 57,638 documents; dimension 384; SHA-256 `32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02` |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; cached snapshot present |
| Historical document preprocessing | `(title + " " + text).strip()`; original corpus order; `normalize_embeddings=True`; batch size default 256 |
| Installed packages | NumPy 2.4.6, OpenVINO 2026.4.1, Torch 2.11.0+cu128, SentenceTransformers 5.1.2 |
| Discovered devices | CPU, GPU.0, GPU.1, NPU |
| NPU discovery properties | Intel AI Boost, architecture 3720, driver 1004512, compiler 458778 |
| Current physical scoring execution | **None**; discovery does not establish execution placement or arithmetic precision |

Regeneration is feasible in principle from these sources, but was not justified after Gate A failed. The historical preparation script resolves the model's current remote revision and writes historical output paths. Re-running it unchanged would not be a safe Phase 11 regeneration workflow. Any later permitted regeneration must explicitly pin the recorded model, preserve preprocessing and document-ID order, save original FP32 embeddings and hashes in a new directory, and distinguish a reproducible configuration from bit-identical GPU embedding generation. No encoder substitution or synthetic dataset was made here.

**Stage separation and missing measurements.**

| Stage | Required meaning | Phase 11 evidence |
|---|---|---|
| Original representation | Frozen normalized FP32 embeddings, original document IDs and query IDs | Provenance/preparation code inspected; matrices not regenerated |
| Operand conversion | Rounding original or calibrated embeddings to FP16 | Analytical rounding checks only; no real-query loss estimate |
| Multiplication/accumulation | Actual installed NPU executable, with checked device placement and numeric output | Not measured; accumulation format/reduction tree unestablished |
| Output rounding | Actual returned score dtype and any internal conversions | Not measured for a scoring graph; an FP32 interface alone would not prove FP32 internal arithmetic |
| Selection/ties | Stable descending numerical score, ascending original document position | Contract specified; no new retrieval rankings produced |

CPU scoring of original embeddings, CPU scoring of FP16-rounded embeddings in wider arithmetic, NPU scoring, and host output rounding would be different controls. Their differences cannot automatically be assigned one-to-one to internal stages; hidden conversions and algorithm changes may confound them. In particular, upcasting already rounded operands cannot restore information lost in the embedding conversion. The higher-precision-scoring control described by the ACL 2026 study [R9 in the novelty report] must be defined in terms of the operands and scorer actually used, rather than its name.

**Executed analytical checks.** The audit constructs two exact FP32-representable scores, `0.50006103515625` and `0.5001220703125`, that both round to FP16 `0.5`, changing stable Top-1 for the input order used. It also checks dyadic scaling homogeneity near normal FP16 bin midpoints: 1,480,650 eligible scalar cases across exponents −4 through +4, zero mismatches. These are host NumPy checks supporting the [restricted mathematical arguments](PHASE11_MATHEMATICAL_FOUNDATION.md). They measure neither real retrieval damage nor NPU behavior, and are not a calibration benchmark.

**Numerical-feasibility decision.** Frequency of changes, margin dependence, relevance impact, higher-precision cost, NPU scoring latency and calibration benefit remain **unmeasured**. No claims of statistical significance, hardware feasibility of a particular scoring graph, or generalization follow from this inventory. Hardware unavailability was not the stopping reason. The explicit novelty stopping rule made a new scoring campaign unjustified for the mechanisms actually identified.
