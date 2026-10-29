# CATS-NPU research charter

Authoritative project scope and evidence ledger. Updated 2026-10-23.
Detailed findings: [repository and hardware audit](AUDIT_2026-10-23.md).
Latest engineering milestone: [Windows setup repair](SETUP_FIX_2026-10-23.md).

## Fixed objective and exclusions

Working title: **Certified Adaptive Top-K Selection for Static-Shape Neural
Processing Units**. Investigate whether an exact hardware-aware Top-K selector
over already-computed similarity scores offers measurable benefits on consumer
NPUs against competitive exact CPU, GPU, and hierarchical baselines.

The contribution is one selection operator. Embedding models, RAG pipelines,
vector databases/ANN indexes, LLM inference, and general heterogeneous schedulers
are outside scope. No energy-consumption or peak hardware-memory claims are
planned. Publication suitability and venue rank are contingent on evidence;
manuscript drafting is deferred.

## Contract and mathematical assumptions

For a nonempty 1D score array, return K original zero-based indices ordered by
descending score then ascending original index. K must be an integer in [1,N];
booleans/nonintegers, K=0, K>N, NaNs, infinities, and unsupported dtypes are
rejected. Signed zeros tie by index. Padding is internal `-inf`, never a valid
winner. The reference supports FP16/FP32/FP64; tile backends support FP16/FP32.
Exactness is relative to the **supplied numeric values**, not agreement between
FP16 and FP32 rankings of an original unquantized vector.

Partition into fixed-width tiles. The fixed exact baseline retains local
min(K,tile length), with stable boundary handling, and globally selects K.
The adaptive prototype retains r candidates and the next-candidate score bound,
merges on the host, and doubles budgets for incomplete uncertified tiles.
Terminate when every tile is complete or its omitted-score bound is strictly
below the current global Kth retained score. Tied bounds conservatively refine.

The local-K lemma and omission certificate are conditional on **correct local
selection and valid bounds on the same values**. Budget doubling terminates in
O(log K) rounds. Actual costs include full tile staging, repeated launches,
bound outputs, host merges, copies, validation, and tie repair. Candidate-count
reduction alone implies neither less device work nor lower latency.

The present native NPU path violates the local-selection assumption on finite
FP16 subnormals and FP32 near ties. It cannot be called a general exact certified
NPU selector until that is resolved. No silent precision conversion, flushing,
CPU substitution, or post-hoc contract narrowing is permitted.

## Questions and hypotheses

| Hypothesis | Test | Current evidence/state |
|---|---|---|
| H1: feasible exact NPU mechanism | Explicit-device compile, inference, numerical/ranking checks, placement evidence over the declared domain | Native TopK executes on NPU for small cases. Full-domain exactness fails. Bitonic construction is repaired and CPU-validated, but NPU compilation fails on boolean Select. H1 unresolved. |
| H2: adaptive allocation saves selection work | Fixed-vs-adaptive counters including cumulative candidates, bounds, calls, bytes, and repeated work | Basic adaptive mechanics exist; final retained count alone is insufficient. Benefit unestablished. |
| H3: savings outweigh systems costs | Paired end-to-end and component measurements with compilation reuse and transfers | No controlled performance evidence. |
| H4: competitive performance in identifiable regimes | Strong deterministic CPU/GPU/NPU comparisons with held-out sessions and uncertainty | C++ DLL absent; PyTorch absent; GPU execution untested; no comparative study. |

These hypotheses may fail. Negative and unsupported outcomes are retained.

## Mechanism and prior-art distinction

Implemented: deterministic CPU reference; fixed hierarchical selection; doubling
adaptive tile budgets; static OpenVINO native TopK and experimental bitonic graph
builders; CPU routing/merging/tie repair; benchmark/statistical scaffolding.
All current OpenVINO application-facing selectors have host work, including the
single-graph `native_direct` wrapper. Raw graph and hybrid wrapper claims must
be separated.

Not established: a distinct cost-aware NPU allocation policy, fused exact device
selector, reduced data movement, or competitive performance. Partitioning,
thresholds, hierarchical merge, and certification are not novelty claims.

Preliminary primary sources and distinctions are recorded in the audit:
[DeepSelect](https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.md),
[RadiK](https://arxiv.org/abs/2501.14336),
[RTop-K](https://arxiv.org/abs/2409.00822),
[Ascend-RaBitQ](https://arxiv.org/abs/2605.16007),
[TPU-KNN](https://arxiv.org/abs/2206.14286),
[generalized two-stage approximate selection](https://arxiv.org/abs/2506.04165),
and [classical threshold aggregation](https://arxiv.org/abs/cs/0204046).
Full code/algorithm comparisons with pinned versions remain outstanding.
Approximate recall guarantees are distinct from per-input exactness; Ascend
execution engines and broader retrieval scope must be distinguished explicitly.

## Hardware/software and provenance

Observed on Windows 11 Pro build 26200: Core Ultra 7 255H, Intel AI Boost
architecture 3720, Arc 140T, RTX 5060 Laptop GPU, approximately 32 GB installed
RAM. Python 3.11.0, NumPy 2.4.6, OpenVINO 2026.4.1; NPU Windows driver
32.0.100.4512 and compiler property 458778. Detailed properties and GPU driver
versions are in the audit evidence. Device discovery alone proves no operation
support. Compiled-model placement raw value was `NPU`; no physical trace was
collected. No implicit AUTO/HETERO execution is allowed as NPU evidence.

The workspace is an extracted snapshot without Git history. The original
41-file source and hashes are preserved in `results/audit_2026-10-23/`.
Future milestones require a real commit or immutable source artifact hash,
resolved dependencies, commands, environment and driver identity, and raw data.
Archive ignored results alongside source; Git alone would not preserve them.

## Experimental protocol and gates

Follow feasibility, mathematical correctness, NPU-aware implementation, controlled
benchmarking, scientific evaluation, statistical validation, then ablations.
The existing [methodology](METHODOLOGY.md), [specification](MATHEMATICAL_SPECIFICATION.md),
and [gates](RESEARCH_GATES.md) provide background; this charter and dated evidence
qualify their historical claims.

1. Repair and regression-test known blockers. Run full Windows tests and
   exact device tests separately. Test finite edges, ties, padding, K=1/N,
   K>tile, partial tile batches, nondefault initial budgets, randomized and
   structured inputs. Preserve every failure and raw placement property.
2. Determine a mathematically valid numeric representation or explicit measured
   fallback for unsupported inputs. Validate omission bounds, not only selected
   value/index consistency. Label native device work and CPU steps accurately.
3. Prepare all reachable static shapes outside steady-state timing. Log cold
   compilation/cache state independently; prohibit compilation during refinement.
4. Establish competitive exact baselines: C++ nth_element, deterministic GPU
   selection, native NPU where correct, fixed hierarchical, simple adaptive,
   cooperative CPU/NPU, and the proposed policy. Use identical score bytes and
   precision with SHA-256 and explicit input/output location. GPU tie handling
   must satisfy the same index-order contract.
5. Begin with a small supported-shape pilot. Ultimately investigate N in
   {1K,16K,128K,1M}, K in {1,10,50,100}, tile widths {256,1024,4096}, and query
   batches {1,8,32} as feasible. Current CLI batch means **tile batch**, not
   query batch. Bitonic currently limits width to powers of two through 256.
   Document these gaps; do not silently replace the intended matrix.
6. Use uniform, Gaussian, clustered, tied, nearly tied and adversarial inputs
   plus numeric edge cases; include held-out real score arrays with provenance.
   Separate tuning from evaluation and freeze parameters before held-out runs.
7. Record exact agreement, median/P95 host-to-host latency, throughput under a
   defined query protocol, cumulative/retained candidates, refinement rounds,
   launches, transfer/synchronization and host/device components. Device-resident
   timing needs a separate residency protocol; backend wall time is not a
   hardware kernel timer. No energy or peak-memory study is authorized.
8. Use repeated time-separated independent sessions, warm-up and power/load
   controls, and recorded randomized/counterbalanced method orders independent
   of workload seed. Pair session outcomes on identical inputs; report effect
   sizes, 95% session-cluster CIs, appropriate multiplicity correction, practical
   significance, and all null/negative results. Iterations/blocks are dependent.
   Six sessions is a minimum resolution condition for the current test, not a
   guarantee of power. Predeclare comparison families and analysis decisions.
9. Ablate fixed/adaptive budgets, initial budget, refinement policy, tile size,
   CPU/NPU merge, certification, score distributions, and transfer/launch costs.
   Establish the cause of an actual latency improvement before a novelty claim.

## Completed milestones and evidence ledger

| Milestone | Evidence | Status |
|---|---|---|
| Historical authoring validation | `docs/VALIDATION.md`: claimed Linux 106 passed / 2 skipped; no original raw logs in this workspace | Historical record, not independently reconstructed |
| Windows audit of unchanged source | `pytest_initial.log`: 105 passed / 2 failed / 1 skipped | Completed; suite not green |
| Device inventory | `doctor.json`, `windows_hardware.json`, dependency freeze | Completed for this session |
| Original strict FP16 NPU operation probe | Six primitive cases pass; bitonic graph construction fails; raw device string recovered separately | Partial feasibility only |
| Targeted mathematical/hardware checks | 320 cases: 143 exact, 76 unprepared-shape exceptions, 96 selected-value exceptions, 5 wrong returned results; eight raw TopK and two Reshape checks | Completed diagnostic matrix; exactness gate fails |
| Minimal blockers and harness audit | `minimal_counterexamples.json`, `harness_reproduction.json` and scripts | Reproducible correctness and failure-propagation defects |
| Audit documentation and preserved source | `AUDIT_2026-10-23.md`, source snapshot and hashes | Completed |
| Windows bitonic construction repair | `results/setup_fix_2026-10-23/`: full suite 111 passed / 1 skipped; four added CPU regression cases; strict NPU bitonic compiler failure retained | Setup test stage repaired; NPU bitonic feasibility remains open |

## Outstanding risks and go/no-go decisions

**Critical:** a certificate can be issued over invalid device omission bounds;
short tiles miss precompiled shapes; bitonic NPU compilation rejects boolean
Select. The Windows bitonic dtype-construction blocker is repaired.
**Methodological:** analysis ignores runtime-failure logs, incomplete configuration
identity can pool policies, query batching is absent, competitive baselines are
unverified, cost counters are incomplete, and novelty is unresolved.

**Go:** focused correctness and feasibility engineering on this machine.
**No-go:** broad optimization, competitive or confirmatory performance claims,
full-domain exact native NPU claims, publication claims, and manuscript drafting.
Reassess after the regression-backed numeric/shape fixes. If an exact useful NPU
path cannot be established, explicitly revise the mechanism or recommend stopping
the direction; retain the negative evidence.

## Current milestone handoff

**Completed:** repository/specification audit, real Windows/NPU diagnostics, and
the narrow bitonic dtype repair with regression tests.
**Evidence:** latest suite 111 passed / 1 skipped; CPU bitonic graph correctness;
strict NPU bitonic compiler failure; dated audit and raw artifacts listed above.
**Limitations:** no generally exact verified device selector, performance benefit,
or distinct publishable mechanism.
**Next step:** investigate an equivalent graph avoiding the unsupported boolean
Select path, then validate on CPU and NPU; preserve the separate shape and numeric
counterexamples as required correctness work before performance evaluation.
