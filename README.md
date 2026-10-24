# CATS-NPU — Certified Adaptive Top-K Selection (Research Prototype v0.1)

A complete, executable **research starting repository** for studying one
component of vector retrieval: exact Top-K selection on the Intel AI Boost NPU,
with fair CPU/GPU controls and explicit hardware feasibility checks.

**Status (Windows validation, 2026-10-24):** **168 tests pass and three physical
NPU regressions are skipped by default**. Phase 3 selected tile width 256 and
tile batch 32 after a 20-configuration physical sweep. Phase 4 added an exact
submission-aware policy, but its real-data benefit is inconsistent. SIFT1M,
BEIR SciFact, and BEIR FiQA evaluation found a crossover among NPU mappings:
direct TopK is best for small arrays while certified geometric refinement is
faster at larger arrays. Optimized C++ and stable CUDA remain substantially
faster. With hardware regressions enabled, the two documented bitonic defects are strict
expected failures. The portable C++ baseline builds with
project-local CMake/Ninja/Zig and passes deterministic tie tests. No
confirmatory or algorithmic novelty claim is currently supported. See the
[research ledger](docs/RESEARCH_PROGRESS.md).

Read the [authoritative research charter](docs/RESEARCH_CHARTER.md) and
[dated audit, evidence, and prioritized repair plan](docs/AUDIT_2026-10-23.md)
before running performance comparisons. The older 106-pass authoring result is
preserved as a historical record in [Validation](docs/VALIDATION.md).

The current adaptive implementation is **HYBRID CPU/NPU**: static-shaped tile
TopK may execute on the device, while candidate budgeting, tie repair, and
final merge are still performed on CPU. Do not describe it as fully NPU-native.

## Quick start — Windows 11 PowerShell

Open **PowerShell** in the extracted repository folder. Python 3.11, pip,
Internet access for dependencies, and updated Intel NPU drivers are required.

```powershell
# Step 0: create .venv, install dependencies, run unit tests
.\scripts\00_setup.ps1

# Step 1: discover devices, driver and runtime versions
.\scripts\01_doctor.ps1

# Step 2: probe strict NPU execution, correctness, individual operations
.\scripts\02_probe_npu.ps1

# Step 3: confirm all CPU logic is correct
.\scripts\03_smoke_cpu.ps1

# Step 4: try a very small actual NPU TopK method
.\scripts\04_smoke_npu.ps1

# If native TopK is unsupported but comparison/reduction primitives work:
.\scripts\04_smoke_npu.ps1 -Kernel bitonic
```

If PowerShell refuses unsigned scripts, run once for this PowerShell process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Critical decision:** If both native TopK and the bitonic network fail on
`NPU`, do not proceed with NPU claims or the NPU benchmarking script. Inspect
`results/probe_npu_f16.json` for exact failures and revise the research plan.

## Benchmarking after feasibility

```powershell
# Optional but recommended, requires CMake / C++ Build Tools:
.\.venv\Scripts\python.exe -m pip install -e '.[native]'
.\scripts\08_build_cpp.ps1

# Record installed packages and resolved devices:
.\scripts\09_freeze_environment.ps1

# Baseline CPU benchmark:
.\scripts\05_benchmark_cpu.ps1

# NPU-assisted benchmark. First confirm NPU smoke passes.
.\scripts\06_benchmark_npu.ps1

# If only bitonic kernel works, use:
.\scripts\06_benchmark_npu.ps1 -Kernel bitonic

# Descriptive summaries; inferential tests require independent sessions:
.\scripts\07_analyze.ps1

# Phase 2 development pilots (sequential/batched NPU and stable CUDA control):
.\scripts\10_pilot_npu.ps1
.\scripts\11_pilot_gpu.ps1
.\scripts\12_analyze_phase2.ps1

# Phase 3/4 development work:
.\scripts\13_phase3_ablation.ps1
.\scripts\14_analyze_phase3.ps1 -ResultsPath results/phase3/ablation_baseline
.\scripts\15_phase4_policy_pilot.ps1
.\scripts\16_analyze_phase4.ps1 -ResultsPath results/phase4/policy_pilot

# Real-data preparation and descriptive comparison:
.\scripts\17_prepare_sift1m.ps1 -Download
.\scripts\18_benchmark_sift1m.ps1
.\scripts\20_benchmark_direct_npu.ps1
.\scripts\22_prepare_beir.ps1
.\scripts\23_benchmark_beir.ps1

# Locked six-session confirmatory experiment; run only after reading the protocol:
.\scripts\25_run_confirmatory.ps1
.\scripts\26_analyze_confirmatory.ps1 -ResultsPath results/phase7/confirmatory/<timestamp>
```

### Direct Python commands

All scripts simply wrap `python -m cats_npu`:

```powershell
.\.venv\Scripts\python.exe -m cats_npu doctor --out results/doctor.json
.\.venv\Scripts\python.exe -m cats_npu probe --device NPU --dtype f16 --ops TopK,Bitonic
.\.venv\Scripts\python.exe -m cats_npu smoke --method numpy_adaptive --n 1024 --k 10
.\.venv\Scripts\python.exe -m cats_npu smoke --method ov_npu_native_hier --n 64 --k 4 --tile 16 --batch 1
.\.venv\Scripts\python.exe -m cats_npu bench --methods cpu_exact,cpp_exact,numpy_hier,numpy_adaptive --ns 1024,16384 --ks 10 --distributions normal,tied --repeats 20 --blocks 2
.\.venv\Scripts\python.exe -m cats_npu analyze results/bench --baseline cpp_exact
```

PyTorch is **optional**; to benchmark your NVIDIA RTX GPU, install the
appropriate CUDA-enabled PyTorch wheel following https://pytorch.org/get-started/locally/
and then use `torch_cuda`. On an Intel GPU accessible to OpenVINO, try
`ov_gpu.0_native_hier` after the device appears in `doctor`. Do not assume
OpenVINO `GPU.1` means NVIDIA without checking the device description.

`--allow-skips` is explicitly opt-in. Without it, a benchmark fails overall
if a method cannot execute or violates exact correctness. Failures are
recorded in separate JSONL files.

## Experimental structure

```
CATS-NPU/
  src/cats_npu/
    reference.py      Exact deterministic (score descending, index ascending)
    backends.py       Numpy tiles, strict OpenVINO native/bitonic tile kernels
    algorithms.py     Exact local-K and certified adaptive hybrid algorithm
    native.py         Optional C++ nth_element CPU baseline loader
    workloads.py      Seeded finite score distributions
    probe.py          Device/operation compilation + execution correctness
    benchmark.py      Warmups, timed measurements, manifests and failure logs
    statistics.py     Session-level paired CIs and Holm-adjusted sign flips
    cli.py            Entry point and command-line arguments
  native/             C++ reference + CMake
  scripts/            Numbered Windows PowerShell walkthrough
  tests/              Brute force, adversarial, integration and stats tests
  configs/            Pilot experimental plan
  docs/               Proofs, protocol, research gates, limitations
  results/            Output directory (gitignored, created on first run)
```

## Method names

| Method | What actually runs |
|---|---|
| `cpu_exact` | NumPy deterministic partition and sorting |
| `cpp_exact` | Optimized C++ exact TopK, after building native DLL |
| `numpy_hier` | Exact local-K across CPU tiles + host merge |
| `numpy_adaptive` | Certified adaptive local candidate budgets (CPU) |
| `numpy_costadaptive` | Submission-aware certified candidate budgets (CPU) |
| `ov_cpu_native_hier` | OpenVINO CPU TopK tiles + host merge |
| `ov_npu_native_direct` | Single native TopK graph compiled explicitly on NPU |
| `ov_npu_native_hier` | NPU tile TopK + host merge |
| `ov_npu_native_adaptive` | NPU tile TopK + **CPU** adaptive control/merge |
| `ov_npu_native_sequential_hier` | Explicit one-tile-per-submission NPU control |
| `ov_npu_native_batched_hier` | Fixed-shape batched NPU local TopK + host exact merge |
| `ov_npu_native_batched_adaptive` | Batched NPU local TopK + CPU adaptive certificate |
| `ov_npu_native_batched_costadaptive` | Batched NPU local TopK + CPU submission-aware certificate |
| `ov_npu_bitonic_hier` | Static compare-exchange graph + host merge |
| `ov_npu_bitonic_adaptive` | Static comparison graph + CPU adaptation |
| `torch_cuda` | PyTorch CUDA TopK, host input/output; equal ties may differ |
| `torch_cuda_stable` | CUDA TopK plus measured host canonicalization/boundary-tie repair |

The OpenVINO naming convention is `ov_<device>_<native|bitonic>_<direct|hier|adaptive>`,
where `<device>` can be `cpu`, `npu`, or exact discovered `gpu.0`, etc.
All OpenVINO compiled shapes are prepared *before* benchmark timing.

## Mathematical correctness

For incomplete tiles, the K-th score `tau` among retained candidates is
compared with the next-local-candidate score `u_j`. If every unfinished tile
has `u_j < tau`, the current result is certified exact. Completed tiles retain
local Top-K and are exact by the local-K lemma. Ties at a completed local
boundary are repaired through a **measured CPU fallback**. See
[Mathematical specification](docs/MATHEMATICAL_SPECIFICATION.md).

This certificate and ordinary hierarchical TopK are **not claimed to be
novel**. The future research contribution must be demonstrated by a
hardware-aware allocation/scheduling policy that saves actual latency while
preserving exact results and comparisons with existing fast TopK methods.

## Reproducible and statistically meaningful measurements

- Each observation is a JSONL row with input SHA-256, elapsed time, N, K,
  dtype, distribution, number of backend calls, refinement rounds, compile
  time excluded, any CPU tie repairs, and available execution-device info.
- `elapsed_ns` is host-input-to-host-output wall time, including transfer,
  synchronizations, orchestration and merges. Component fields record host tile
  staging, synchronous backend execution, output copy, validation, candidate
  extraction, merge, certification, submission counts and bytes. Synchronous
  backend time can include transfers and waiting; **it is not an NPU hardware
  cycle-level timer**. Rows containing FP16 subnormals also record how many
  exact CPU row validations were performed.
- A session is a separate run with a distinct `--session-id`. 20 iterations
  or 2 blocks within one process do **not** count as independent sessions.
- Results with fewer than six independent sessions are descriptive only;
  six is a minimum for the provided sign-flip test, and more are preferred.
- Report negative/null findings and all explicit skipped/failed methods.
- No energy consumption or peak hardware-memory metrics are asserted.

Read [Methodology](docs/METHODOLOGY.md), [Research gates](docs/RESEARCH_GATES.md),
[Notes and limitations](docs/NOTES_AND_LIMITATIONS.md), and
[Validation](docs/VALIDATION.md) **before** using any
results in a manuscript.

## Troubleshooting

**`No module named openvino`:** install `.\scripts\00_setup.ps1` or
`python -m pip install -e '.[all]'` using Python 3.11.

**NPU absent from doctor:** update Intel NPU driver; confirm Windows 11 device
manager and follow https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html

**NPU TopK compiler error:** expected possibility. Read per-operation probe
results. Do not change to `AUTO` to create a fake passing NPU result. The
bitonic driver graph now compiles after converting its boolean direction data
to i32, but this driver returns incorrect indices; the plugin compiler passes
ordinary inputs but changes an FP16 subnormal value. Bitonic remains explicitly
experimental.

**TopK indices differ on ties:** exact CPU reference selects the smaller
original index. Torch TopK is not guaranteed stable; test all indices and
report mismatches. `ov_*` local-K variants repair boundary ties on CPU and
count repairs explicitly.

**OpenVINO NPU f32 mismatches:** FP32 host inputs may compute internally in
lower precision. The strict NPU selector explicitly rejects all FP32 inputs on
this measured stack. Rows containing nonzero FP16 subnormals receive measured
exact CPU validation because the device has reproduced wrong rankings/values;
a disagreement rejects the result rather than repairing it. Never reinterpret
a disagreement as exact.

**Slow adaptive algorithm:** extra device launches, host routing, and
threshold bound retrieval can dominate saved work. This is a falsifiable
research hypothesis, not an implementation bug by default.

## Licensing and reference documentation

MIT license. Foundational sources include the
[OpenVINO NPU plugin documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html),
[OpenVINO TopK API](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.opset5.topk.html),
[DeepSelect](https://github.com/deepseek-ai/DeepSelect),
and classical threshold selection / aggregation literature.

## Optional: real retrieval-score workloads from your own embeddings

If you have local `embeddings.npy` shaped `[N,D]` and `queries.npy` shaped
`[Q,D]`, produce a deterministic similarity-score vector without introducing
text-embedding inference into the benchmark:

```powershell
.\.venv\Scripts\python.exe -m cats_npu from-embeddings `
  --embeddings data/embeddings.npy --queries data/queries.npy `
  --query-index 0 --dtype f16 --out results/real_scores.npy

# Set --ns to the exact number of embeddings N:
.\.venv\Scripts\python.exe -m cats_npu bench --input results/real_scores.npy `
  --ns 100000 --ks 10 --distributions real --methods cpu_exact,numpy_adaptive `
  --tile 128 --batch 8
```

Normalization is enabled by default (cosine scores); use `--no-normalize`
for inner-product scores. Keep the dataset provenance and query index with
the generated `.json` metadata. The `--distributions real` label is only
metadata when `--input` is provided, not a synthetic generator name.
