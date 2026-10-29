# OpenVINO configuration and error analysis

Date: 2026-10-23  
Scope: current Windows/OpenVINO/NPU configuration and the failures reproduced in this repository.  
Repository policy: this artifact is experimental evidence under `results/`; it does not add or modify a paper, manuscript, or TeX source.

## Decision

The machine and software installation are recognized by OpenVINO, and Intel lists Arrow Lake NPU 3720 on Windows 11 as supported. The current exact-NPU Top-K gate is nevertheless **not passed**.

Two independent problems block that gate:

1. The default compiler-in-driver rejects the bitonic graph, while an explicitly forced compiler-in-plugin compiles and correctly executes an ordinary-value control case.
2. Both native TopK and the bitonic graph return incorrect rankings for representable FP16 subnormal inputs. The bitonic graph also fails an FP32 one-ULP case because this NPU executes with FP16 compute precision. Changing to `ACCURACY` mode does not change the effective FP16 precision, and this installation rejects an F32 inference-precision hint.

The compiler-in-plugin result narrows the graph compilation failure to a compiler-path issue. It does not solve the numerical exactness problem.

## Configuration actually observed

| Property | Runtime value |
|---|---|
| OpenVINO | `2026.4.1-22982-e213a147257-releases/2026/4` |
| Python used by project | 3.11 |
| NPU full name | `Intel(R) AI Boost` |
| NPU architecture | `3720` |
| NPU driver value | `1004512`, corresponding to `32.0.100.4512` |
| NPU compiler version | `458778` |
| Default compiler selected | `CompilerType.DRIVER` |
| Default inference precision | `float16` |
| Device capabilities reported here | `FP16`, `INT8`, `EXPORT_IMPORT` |
| Default execution mode | `PERFORMANCE` |
| Default performance hint | `LATENCY` |

Official OpenVINO documentation is consistent with this setup:

- Arrow Lake NPU 3720 is listed for Windows 11 in the official NPU plugin support table.
- Python 3.11 is within OpenVINO 2026's Windows range of Python 3.10 through 3.14.
- NPU hardware computation precision is documented as FP16.
- Dynamic shapes start with NPU40XX and Windows driver 32.0.100.4621, and require compiler-in-plugin. This system is NPU 3720 with driver 32.0.100.4512, so fixed shapes and precompiled shape variants are required here.

## Reproduced errors and causes

### 1. Initial setup test failures: Windows NumPy dtype identity

The original setup transcript failed two CPU OpenVINO tests while constructing the bitonic graph:

```text
Unsupported data type: int32
```

The displayed dtype was misleading. On this Windows/NumPy build, the XOR expression produced `np.intc` (`dtype.char == "i"`) even though it printed as `int32`; a normal `np.arange(..., dtype=np.int32)` used the OpenVINO-accepted identity (`dtype.char == "l"`). OpenVINO's Python constant conversion rejected the former.

The explicit `.astype(np.int32)` now present after XOR fixes graph construction. The current suite result is:

```text
111 passed, 1 skipped in 4.39s
```

This was a host-side binding/dtype issue, not evidence of NPU operator support.

### 2. NPU bitonic compilation: default driver compiler rejects boolean data operands

The current graph contains:

```python
choose_self = ops.select(mask, better, ops.logical_not(better))
```

Here the `then` and `else` operands are boolean. The default driver compiler lowers them to `i8` and reports:

```text
'IE.Select' op operand #1 must be ... i32/i64/f16/f32/f64,
but got 'tensor<1x16xi8>'
```

This matches the OpenVINO Select operation specification: the condition is boolean, while `then` and `else` are specified as a supported numeric type. CPU accepting this graph does not establish NPU compiler support.

Replacing the boolean Select with the equivalent boolean XNOR expressed as `Equal(mask, better)` did not fix the driver path; the driver compiler then rejected boolean/i8 inputs to `Equal`.

Compiler-path experiment, width 16, K=4, FP16:

| Compiler request | Compiler used / result | Ordinary-value correctness |
|---|---|---|
| default | driver; compile failed | not run |
| `PREFER_PLUGIN` | fell back to driver; compile failed | not run |
| `DRIVER` | driver; compile failed | not run |
| `PLUGIN` | plugin; compile and inference succeeded | correct |

The official NPU documentation says `PREFER_PLUGIN` falls back to compiler-in-driver when platform/compiler/driver compatibility conditions are not met, and that `PLUGIN` can force compiler-in-plugin. The observed results follow that behavior. Because forcing `PLUGIN` can bypass the compatibility choice, it needs a broader stability and correctness matrix before it becomes a project default.

### 3. Native TopK numerical correctness: representable values are mis-ranked

A focused width-16, K=4 probe used ascending FP16 subnormal values. CPU returned the expected highest indices `[15, 14, 13, 12]`. NPU native TopK returned `[0, 1, 2, 3]` with both the driver and plugin compilers.

The output values still matched the values stored at the returned indices. That exposes a validation hole in `OpenVINOTileBackend.select_rows`: it checks only that each returned value belongs to its returned index. It does not check that the indices are actually the highest-ranked elements. Consequently, an incorrect local Top-r result can enter the adaptive algorithm and produce a false `strict_omission_bound` certificate.

Observed modes did not repair it:

| Mode | Effective precision | FP16 subnormal TopK |
|---|---|---|
| default driver | FP16 | wrong |
| driver + `ACCURACY` | FP16 | wrong |
| forced plugin | FP16 | wrong |
| forced plugin + `ACCURACY` | FP16 | wrong |

This probe shows incorrect comparison/ranking behavior at the FP16 subnormal edge. It does not by itself identify whether the hardware comparator, compiler lowering, or TopK implementation is the lowest-level cause.

### 4. FP32 input cannot provide FP32-exact ranking on this NPU configuration

For 16 distinct FP32 values separated by one ULP around 1.0, CPU returned `[15, 14, 13, 12]`. NPU native TopK returned `[0, 1, 2, 3]` and output all four values as `1.0`. `ACCURACY` mode still reported effective FP16 inference precision.

An explicit F32 inference-precision hint failed at compile time with:

```text
Wrong value f32 for property key INFERENCE_PRECISION_HINT.
Supported values: f16, i8
```

The device capability query likewise reports FP16 and INT8, not FP32. Therefore a model whose input storage type is FP32 must not be treated as an FP32-exact NPU ranking implementation on this configuration.

The forced-plugin bitonic graph also failed the one-ULP case and returned a duplicate index (`[0, 1, 2, 2]`). It therefore is not a numerical workaround.

### 5. Short final tile asks for an unprepared static K

For `n=65`, `tile=16`, the final tile contains one real value. Runtime code asks for one retained value plus one omission bound, so it calls the backend with K=2. `MethodRunner.prepare()` only enumerates the schedule for full tiles and does not add the tail-specific K=2 model. Execution then raises:

```text
RuntimeError: K=2 not prepared; precompile outside timing
```

This is a project static-shape enumeration bug. Dynamic shapes are not an appropriate fix on this hardware. Preparation must enumerate the union of K values required by full tiles and the actual tail size for both hierarchical and adaptive schedules.

### 6. Execution-device evidence is parsed incorrectly

On this NPU compiled model, `EXECUTION_DEVICES` is the string `"NPU"`. The code calls `list(raw_value)`, producing `["N", "P", "U"]`. This affects both backend metadata and the probe report. It does not redirect execution, but it corrupts the evidence used to demonstrate placement.

Normalize the property as `[raw_value]` when it is a string; preserve a returned sequence otherwise. Also record the compiled model's `NPU_COMPILER_TYPE`, because this machine has materially different behavior between driver and plugin compilers.

### 7. Probe command reports process success when an operation fails

The probe JSON correctly sets `all_passed=false`, but the CLI prints a note and then returns process exit code 0. PowerShell automation therefore cannot rely on `$LASTEXITCODE`. The probe command should return 1 when `all_passed` is false, matching the smoke command's failure behavior.

## Correctness implications

The adaptive certificate is mathematically conditional on every local Top-r call being correct on the same values used by the reference ordering. The observed NPU counterexample violates that premise. The final result can therefore be wrong while the algorithm labels it certified.

The current NPU methods must remain excluded from performance claims. A benchmark's final reference comparison can mark a run incorrect, but it cannot make the underlying certificate valid. Correctness needs to be established over a declared numerical domain before timing comparisons are meaningful.

## Recommended engineering order

1. Fix static K enumeration for tail tiles and add `n % tile` cases for hierarchical and adaptive schedules.
2. Fix `EXECUTION_DEVICES` normalization and record compiler type/version per compiled model.
3. Make probe CLI failure observable through its exit code.
4. Add compiler selection as an explicit experimental configuration. Keep `DRIVER`, `PREFER_PLUGIN`, and forced `PLUGIN` results separate.
5. Expand NPU adversarial correctness tests before benchmarking: zeros, signed zeros, FP16 subnormals, smallest normal values, adjacent representable values, ties, infinities, padding, and all K boundaries.
6. Treat native TopK and bitonic NPU paths as unsupported for exact project semantics until that matrix passes. `ACCURACY` mode and forced plugin did not repair the numerical counterexamples.
7. Decide the research numerical contract explicitly. If exact ranking over all finite FP16 values remains required, the current NPU paths fail. Restricting or transforming the score domain would be a different contract and would need its own reference definition and justification.

## Evidence files

- `openvino_runtime_probe.json`: structured runtime properties, compiler variants, and numerical outputs.
- `openvino_runtime_probe.log`: driver-compiler diagnostics.
- `diagnose_openvino.py`: reproducible diagnostic program; it does not change project implementation.
- `pytest.log`: current full unit-test result.
- `../audit_2026-10-23/minimal_counterexamples.json`: earlier end-to-end counterexamples.
- `../audit_2026-10-23/runtime_diagnostics.json`: Windows dtype and raw execution-device property evidence.
- `../setup_fix_2026-10-23/probe_npu_bitonic_f16.log`: original NPU bitonic compiler diagnostic.

## Official sources consulted

- [OpenVINO 2026 NPU device documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- [Official OpenVINO Intel NPU plugin README and support table](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md)
- [OpenVINO Select-1 operation specification](https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/operation-specs/condition/select-1.html)
- [OpenVINO precision-control documentation](https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/precision-control.html)
- [OpenVINO 2026 system requirements](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html?language=en)
- [OpenVINO operation conformance reports](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-operations.html)

The conformance tables are compatibility indicators, not proof for this exact runtime, graph, dtype, and input domain. This report relies on execution on the installed NPU for its project conclusions.
