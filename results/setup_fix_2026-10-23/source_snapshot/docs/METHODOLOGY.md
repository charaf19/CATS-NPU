# Study protocol and methodological restrictions (v0.1)

## Scope
**One operator**: exact Top-K selection over already computed similarity
scores. Do not integrate embedding inference, vector indexing, reranking,
energy or hardware peak-memory measurements into the claimed contribution.

## Research hypotheses
- H1: a fixed-shape, correctly executing native TopK or comparison network
  can run on the Intel NPU 3720 with explicit device selection.
- H2: certified adaptation reduces retained candidates compared to exact
  fixed local-K, without sacrificing correctness.
- H3: the saved work can outweigh extra submission, routing, and copying costs.
- H4: some specified real-world operating regimes favor the NPU-assisted
  implementation over optimized CPU and GPU baselines.

H1 is a hardware feasibility gate. H2 is algorithmic. H3 and H4 are
quantitative and falsifiable, NOT promised outcomes.

## Comparators
- `cpu_exact`: deterministic NumPy partition/select, host input/output.
- `cpp_exact`: optimized C++ `std::nth_element` and stable sorting (optional
  compilation; important as a fair CPU baseline).
- `numpy_hier` and `numpy_adaptive`: Python orchestration controls.
- `ov_cpu_native_direct` (small N), `ov_cpu_native_hier`,
  `ov_cpu_native_adaptive`: OpenVINO CPU controls.
- `ov_npu_native_direct`, `ov_npu_native_hier`, `ov_npu_native_adaptive`:
  NPU native TopK variants, if supported.
- `ov_npu_bitonic_hier`, `ov_npu_bitonic_adaptive`: static comparison network
  fallback, if Gather/Select and associated primitives compile.
- `ov_gpu.0_native_hier` etc: Intel/OpenVINO GPU controls, if available.
- `torch_cpu` and `torch_cuda`: optional PyTorch controls; CUDA refers to
  NVIDIA CUDA GPU, not necessarily an OpenVINO GPU plugin. Native torch.topk
  does not promise deterministic index ordering for ties, so tied-input
  disagreement is reported as an invalid observation.

## Feasibility / execution evidence
1. `doctor`: record available devices and names, versions and drivers.
2. `probe`: compile each operator with exact requested device `NPU`.
3. Require actual inference and numerical correctness, not compile-only.
4. Record `EXECUTION_DEVICES` when the property is available. An absent
   property is unknown, not proof of a particular device.
5. Independently inspect Intel NPU performance activity and drivers if
   available. OpenVINO NPU profiling may be incomplete/experimental.
6. Store the raw JSON probe and, if supported, a screenshot / profiling log.
7. A broken kernel is a genuine FAIL; never relabel a CPU result as NPU.

## Performance protocol
- Compare all methods on the exact same score bytes and K, with an SHA-256.
- Precompile every fixed-shape OpenVINO graph and warm up outside timing.
- Record compile_ms separately; do not conflate cold start with steady-state.
- Time per iteration from host score input through host indices output,
  including backend submission and transfers; no CUDA async timing artifacts.
- Always run same process power/thermal conditions and randomized method
  ordering in blocks; reserve distinct time-separated **sessions** for stats.
- Record N, K, dtype, tile width, tile batch, distribution, seed, git hash,
  Python/OpenVINO versions, and available execution-device metadata.
- Preserve incorrect or skipped configurations in explicit files.
- Report CPU tie-repair frequency. A repaired hybrid result is *not*
  evidence of fully device-resident Top-K.

## Data / workloads
- Scores: uniform, Gaussian, clustered high scores, tied/quantized scores,
  near ties, and adversarial concentrated winners.
- Pilot N=512/4096, K=4/10, tile=16, batch=1 to expose software support.
- Larger confirmatory runs: N in {1k, 16k, 128k, 1M}, K in {1,10,50,100},
  tile={16,32,64,128,256}, tile_batch={1,8,32} **only for settings actually
  supported by the backend**. Do not run an unbounded full-factorial grid.
- Use held-out `.npy` score arrays derived from SIFT1M/text retrieval scores
  after feasibility. Report their provenance and licensing. The `.npy` files
  must be finite 1D scores; the CLI `--input` loads them.

## Outcome measures
Primary: exact sorted Top-K agreement (indices), median and P95 wall latency,
queries/sec (derived from time if desired), calls, refinement rounds,
candidate count, and host overhead. NOTE `backend_ns` includes backend call
wall time from Python, not verified NPU kernel-only time. Do not call it a
GPU/NPU hardware profile. It excludes host merge/orchestration time.

## Statistical plan
- Use separate `session_id` values from separately launched benchmark runs.
  In-process `block_id` values are not independent observations for tests.
- Pair per-session medians on identical case configuration and inputs.
- Analyze log speedup and report exp(mean paired log speedup).
- Report 95% session-cluster bootstrap confidence intervals with >=2 sessions;
  do not interpret them as confirmatory with tiny session counts.
- Starting with >=6 sessions, use exact sign-flip permutation test for <=17
  paired sessions or Monte Carlo sign-flip for >=18; apply Holm correction
  per-case across predeclared method comparisons.
- Prefer >=8-10 real independent sessions to have meaningful power; small
  samples cannot support strong conclusions.
- The pilot's large exploratory search does not generate confirmatory p-values.
  Freeze hypotheses/configuration list before a new held-out confirmatory run.
- A valid result can show CPU/GPU is fastest; negative/null findings are
  reported in full.

## Gate criteria (no predetermined speedup claims)
A. Hardware probe passes for at least one native/bitonic Top-K path.
B. Correctness of Top-K is exact across tied, near-tied and adversarial scores.
C. Algorithm improves a strong fixed local-K baseline (if it does).
D. Overall CPU/NPU/GPU comparisons remain valid with uncertainty reporting.
E. Assess novelty against DeepSelect, TPU two-stage Top-K, RadiK, RTop-K,
   and classical threshold aggregation before committing to a manuscript.

This v0.1 repository is a *research harness*. It is not itself evidence of
publication-quality novelty or superior NPU performance.
