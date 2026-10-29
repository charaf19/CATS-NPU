# Validation records — 2026-10-23

## Windows audit of unchanged source

The current machine ran the suite: **105 passed, 2 failed, 1 skipped**. Both
failures involve bitonic graph construction; the skipped test needs the native
C++ DLL. OpenVINO 2026.4.1 discovered CPU, GPU.0, GPU.1, and NPU. Six original
FP16 primitive cases compiled, inferred, and passed their numerical checks on
explicit NPU; bitonic failed before device compilation. Two separately tested
Reshape cases passed. Targeted checks exposed short-tile precompilation errors,
wrong FP16 subnormal winners and certificates, and FP32 precision disagreements.
Native TopK execution is observed; general exactness is not established.

See [the full audit](AUDIT_2026-10-23.md) and
[research charter](RESEARCH_CHARTER.md). Raw logs and unchanged source hashes
are under `results/audit_2026-10-23/`. No comparative speedups were evaluated.

## Historical authoring report (not the current Windows result)

## Validated here (Linux build/authoring environment)
- Python compilation: source and tests `compileall` passed.
- `PYTHONPATH=src pytest -q`: **106 passed, 2 skipped**.
- CPU reference exhaustive small-case validation: PASS.
- NumPy fixed hierarchical and certified adaptive selectors: PASS for ties,
  near ties, clustered scores, concentrated maxima, mixed tile sizes.
- Native C++ `std::nth_element` benchmark module: compiled with GNU C++17,
  loaded through ctypes, and matched CPU reference in unit tests.
- End-to-end pilot: all four methods `cpu_exact,cpp_exact,numpy_hier,
  numpy_adaptive` correctly returned TopK over a 65-score tied input.
- Statistics pipeline: tested correct session grouping, paired CI and
  sign-flip/Holm, invalidation after wrong results, and score SHA integrity.

## Not validated here
- OpenVINO model construction / compilation / inference (not installed in
  this environment; PyPI network lookup is unavailable).
- Intel NPU device execution (no Intel AI Boost NPU exposed here).
- OpenVINO CPU/GPU/NPU TopK or the bitonic network; the two corresponding
  tests are **skipped**, NOT passing.
- Windows 11 setup and PowerShell script execution.
- Actual hardware latency or speedup results from user's machine.
- The future cost-aware novel allocation strategy or journal acceptance.

## Safety of research interpretation
The included benchmark produced pilot latency rows *solely to test the
measurement pipeline*. None are included in this source ZIP; they are not
publication evidence. Running this repository on the user's Windows laptop
is required to resolve the hardware feasibility gate.
