# Validation record — project source snapshot, 2026-10-23

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
