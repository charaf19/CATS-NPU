# Iterative research checkpoints

**Current decision (2026-10-23):** native TopK execution on explicit NPU is
observed for small cases, but finite-input exactness and full test-suite gates
remain open. Bitonic fails at host graph construction; short tiles and numerical
counterexamples require repairs. See the [research charter](RESEARCH_CHARTER.md)
and [audit evidence](AUDIT_2026-10-23.md). Performance and publication gates
have not been reached.

## Gate 0 — environment
- [ ] Windows 11, Python 3.11, current OpenVINO and Intel NPU driver installed.
- [ ] `python -m cats_npu doctor` records NPU and CPU, and identifies GPU(s).
- [ ] Full `pytest` passes.

## Gate 1 — critical NPU operator feasibility
- [ ] NPU probe stores compile / inference / correctness results separately.
- [ ] Native `TopK` actually runs on `NPU`, OR the bitonic selection network runs.
- [ ] Validate tie-free and tied cases on the exact requested NPU.
- [ ] Store `EXECUTION_DEVICES` when available; verify physical NPU activity.
- [ ] Do NOT claim NPU feasibility solely from prior embedding inference tests.

**If both TopK and the required bitonic primitives fail, pause this NPU TopK
paper direction. Do not silently change to CPU-only selection.**

## Gate 2 — correctness and algorithm mechanics
- [ ] Reference matches exhaustive brute-force stable ranking.
- [ ] Hierarchical and adaptive outputs match reference on all distributions.
- [ ] Stable ties and FP16/FP32 effects are properly accounted for.
- [ ] Compile all shapes before timing; no compilation in adaptive rounds.
- [ ] Observe number of candidates and rounds in output JSONL.

## Gate 3 — CPU/GPU baseline fairness
- [ ] Compile C++ nth_element baseline.
- [ ] Include OpenVINO CPU, and GPU baselines where applicable.
- [ ] Include CUDA torch baseline if NVIDIA is relevant.
- [ ] Compare identical input precision and data residency assumptions.
- [ ] Report direct/host latency separately if device-resident implementation exists.

## Gate 4 — performance, ablations, and statistics
- [ ] Fix one development workload subset for tuning.
- [ ] Hold out test distributions and configurations.
- [ ] Analyze actual wall time vs candidate count and launch overhead.
- [ ] Run independent sessions, paired CIs, sign-flip tests + Holm correction.
- [ ] Keep all failures and negative measurements in artifacts.

## Gate 5 — scientific contribution decision
- [ ] Explain the new selection policy and derive correctness/cost properties.
- [ ] Demonstrate benefits caused by the new policy, not merely a device swap.
- [ ] Compare against 2026 DeepSelect and other strong relevant methods.
- [ ] If results are not competitive, narrow claim or stop before manuscript.

## What to send back for iteration
1. `results/doctor.json`
2. `results/probe_npu_f16.json`
3. `results/bench_npu/manifest_*.json` and `observations_*.jsonl`
4. `results/bench_npu/skips_*.jsonl` if produced
5. Any failing traceback and exact CLI arguments
