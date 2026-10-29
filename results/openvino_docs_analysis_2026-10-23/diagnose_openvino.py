"""Focused, non-benchmark OpenVINO diagnostics for the installed NPU stack.

This file records compiler selection, precision behavior, and a graph-level
counterexample.  It does not modify the CATS-NPU implementation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import openvino.opset13 as ops

from cats_npu.backends import _bitonic_model, _native_model


HERE = Path(__file__).resolve().parent


def text(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [text(item) for item in value]
    if isinstance(value, dict):
        return {str(key): text(item) for key, item in value.items()}
    return str(value)


def error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def get_property(owner: Any, name: str) -> Any:
    try:
        return text(owner.get_property(name))
    except Exception as exc:  # diagnostics must retain unsupported-property evidence
        return {"error": error_text(exc)}


def get_core_property(core: ov.Core, device: str, name: str) -> Any:
    try:
        return text(core.get_property(device, name))
    except Exception as exc:  # retain exact unsupported-property evidence
        return {"error": error_text(exc)}


def compiled_properties(compiled: Any) -> dict[str, Any]:
    raw_execution_devices = get_property(compiled, "EXECUTION_DEVICES")
    normalized_execution_devices = (
        [raw_execution_devices]
        if isinstance(raw_execution_devices, str)
        else raw_execution_devices
    )
    return {
        "execution_devices_raw": raw_execution_devices,
        "execution_devices_normalized": normalized_execution_devices,
        "npu_compiler_type": get_property(compiled, "NPU_COMPILER_TYPE"),
        "npu_compiler_version": get_property(compiled, "NPU_COMPILER_VERSION"),
        "inference_precision_hint": get_property(compiled, "INFERENCE_PRECISION_HINT"),
        "execution_mode_hint": get_property(compiled, "EXECUTION_MODE_HINT"),
    }


def expected_order(values: np.ndarray, k: int) -> list[int]:
    indices = np.arange(values.size, dtype=np.int64)
    return np.lexsort((indices, -values.astype(np.float64)))[:k].tolist()


def probe_topk(
    core: ov.Core,
    *,
    device: str,
    dtype: str,
    values: np.ndarray,
    config: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    k = 4
    model = _native_model(ov, ops, 1, values.size, k, dtype)
    record: dict[str, Any] = {
        "label": label,
        "device": device,
        "model_dtype": dtype,
        "config": text(config),
        "input": values.tolist(),
        "expected_indices": expected_order(values, k),
    }
    try:
        compiled = core.compile_model(model, device, config)
        record["compiled"] = True
        record["properties"] = compiled_properties(compiled)
        outputs = compiled({0: values.reshape(1, -1)})
        out_values = np.asarray(outputs[compiled.output(0)])[0]
        out_indices = np.asarray(outputs[compiled.output(1)])[0].astype(np.int64)
        record["actual_values"] = out_values.tolist()
        record["actual_indices"] = out_indices.tolist()
        record["correct_indices"] = out_indices.tolist() == record["expected_indices"]
        record["values_match_returned_indices"] = bool(
            np.array_equal(values[out_indices], out_values, equal_nan=False)
        )
    except Exception as exc:
        record["compiled"] = False
        record["error"] = error_text(exc)
    return record


def bitonic_equal_model(
    batch: int, width: int, k: int, dtype: str, *, numeric_flags: bool = False
) -> ov.Model:
    """Diagnostic XNOR variants using boolean or explicitly numeric flags."""
    np_dtype = np.dtype("float16" if dtype == "f16" else "float32")
    inp = ops.parameter([batch, width], dtype=np_dtype, name="scores")
    values = inp
    idx = ops.broadcast(
        ops.constant(np.arange(width, dtype=np.int32).reshape(1, width)),
        ops.constant(np.array([batch, width], dtype=np.int64)),
    )
    size = 2
    while size <= width:
        stride = size // 2
        while stride >= 1:
            perm = (np.arange(width, dtype=np.int32) ^ stride).astype(np.int32)
            partner_v = ops.gather(values, ops.constant(perm), ops.constant(1, dtype=np.int32))
            partner_i = ops.gather(idx, ops.constant(perm), ops.constant(1, dtype=np.int32))
            better = ops.logical_or(
                ops.greater(values, partner_v),
                ops.logical_and(ops.equal(values, partner_v), ops.less(idx, partner_i)),
            )
            positions = np.arange(width)
            choose_better = ((positions & size) == 0) == ((positions & stride) == 0)
            mask = ops.constant(choose_better.reshape(1, width).astype(bool))
            # select(mask, better, not(better)) is boolean XNOR(mask, better).
            if numeric_flags:
                choose_self = ops.equal(
                    ops.convert(mask, np.int32), ops.convert(better, np.int32)
                )
            else:
                choose_self = ops.equal(mask, better)
            values = ops.select(choose_self, values, partner_v)
            idx = ops.select(choose_self, idx, partner_i)
            stride //= 2
        size *= 2
    cut = ops.constant(np.arange(k, dtype=np.int32))
    out_v = ops.gather(values, cut, ops.constant(1, dtype=np.int32))
    out_i = ops.gather(idx, cut, ops.constant(1, dtype=np.int32))
    suffix = "i32_equal" if numeric_flags else "bool_equal"
    return ov.Model([out_v, out_i], [inp], f"bitonic_{suffix}_B{batch}_W{width}_K{k}")


def probe_bitonic(
    core: ov.Core,
    *,
    variant: str,
    device: str,
    config: dict[str, Any],
    values: np.ndarray | None = None,
    dtype: str = "f16",
    label: str = "regular_values",
) -> dict[str, Any]:
    width, k = 16, 4
    if values is None:
        values = np.array(
            [0.5, -1.0, 7.0, 7.0, 3.0, 2.0, 9.0, 4.0,
             8.0, -2.0, 6.0, 1.0, 5.0, 0.0, 7.0, 10.0],
            dtype=np.float16 if dtype == "f16" else np.float32,
        )
    if variant == "current_boolean_select":
        model = _bitonic_model(ov, ops, 1, width, k, dtype)
    elif variant == "equal_xnor":
        model = bitonic_equal_model(1, width, k, dtype)
    elif variant == "i32_equal_xnor":
        model = bitonic_equal_model(1, width, k, dtype, numeric_flags=True)
    else:
        raise ValueError(f"Unknown bitonic variant: {variant}")
    record: dict[str, Any] = {
        "variant": variant,
        "label": label,
        "device": device,
        "model_dtype": dtype,
        "config": text(config),
        "expected_indices": expected_order(values, k),
    }
    try:
        compiled = core.compile_model(model, device, config)
        record["compiled"] = True
        record["properties"] = compiled_properties(compiled)
        outputs = compiled({0: values.reshape(1, -1)})
        out_indices = np.asarray(outputs[compiled.output(1)])[0].astype(np.int64)
        out_values = np.asarray(outputs[compiled.output(0)])[0]
        record["actual_indices"] = out_indices.tolist()
        record["actual_values"] = out_values.tolist()
        record["correct_indices"] = out_indices.tolist() == record["expected_indices"]
    except Exception as exc:
        record["compiled"] = False
        record["error"] = error_text(exc)
    return record


def main() -> None:
    core = ov.Core()
    npu_properties = [
        "FULL_DEVICE_NAME",
        "DEVICE_ARCHITECTURE",
        "NPU_DRIVER_VERSION",
        "NPU_COMPILER_TYPE",
        "NPU_COMPILER_VERSION",
        "OPTIMIZATION_CAPABILITIES",
        "INFERENCE_PRECISION_HINT",
        "EXECUTION_MODE_HINT",
        "PERFORMANCE_HINT",
    ]
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "openvino_version": ov.__version__,
        "available_devices": list(core.available_devices),
        "npu_properties": {
            name: get_core_property(core, "NPU", name) for name in npu_properties
        },
        "topk": [],
        "bitonic": [],
    }

    smallest_f16 = float(np.nextafter(np.float16(0), np.float16(1)))
    f16_subnormals = (
        np.arange(16, dtype=np.float32) * smallest_f16
    ).astype(np.float16)
    f32_one_ulp = (
        np.float32(1.0)
        + np.arange(16, dtype=np.float32) * np.spacing(np.float32(1.0))
    ).astype(np.float32)

    report["topk"].append(probe_topk(
        core, device="CPU", dtype="f16", values=f16_subnormals,
        config={}, label="cpu_f16_subnormals_control",
    ))
    report["topk"].append(probe_topk(
        core, device="CPU", dtype="f32", values=f32_one_ulp,
        config={}, label="cpu_f32_one_ulp_control",
    ))
    npu_scenarios = [
        ("default", {}),
        ("prefer_plugin", {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"}),
        ("accuracy", {"EXECUTION_MODE_HINT": "ACCURACY"}),
        ("plugin", {"NPU_COMPILER_TYPE": "PLUGIN"}),
        ("plugin_accuracy", {
            "NPU_COMPILER_TYPE": "PLUGIN",
            "EXECUTION_MODE_HINT": "ACCURACY",
        }),
        ("f32_hint", {"INFERENCE_PRECISION_HINT": ov.Type.f32}),
        ("plugin_f32_hint", {
            "NPU_COMPILER_TYPE": "PLUGIN",
            "INFERENCE_PRECISION_HINT": ov.Type.f32,
        }),
        ("accuracy_f32_hint", {
            "EXECUTION_MODE_HINT": "ACCURACY",
            "INFERENCE_PRECISION_HINT": ov.Type.f32,
        }),
    ]
    for scenario, config in npu_scenarios:
        report["topk"].append(probe_topk(
            core, device="NPU", dtype="f16", values=f16_subnormals,
            config=config, label=f"npu_f16_subnormals_{scenario}",
        ))
        report["topk"].append(probe_topk(
            core, device="NPU", dtype="f32", values=f32_one_ulp,
            config=config, label=f"npu_f32_one_ulp_{scenario}",
        ))

    report["bitonic"].append(probe_bitonic(
        core, variant="current_boolean_select", device="CPU", config={},
    ))
    report["bitonic"].append(probe_bitonic(
        core, variant="equal_xnor", device="CPU", config={},
    ))
    compiler_scenarios = [
        ("default", {}),
        ("prefer_plugin", {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"}),
        ("plugin", {"NPU_COMPILER_TYPE": "PLUGIN"}),
        ("driver", {"NPU_COMPILER_TYPE": "DRIVER"}),
    ]
    for scenario, config in compiler_scenarios:
        for variant in ("current_boolean_select", "equal_xnor", "i32_equal_xnor"):
            item = probe_bitonic(core, variant=variant, device="NPU", config=config)
            item["scenario"] = scenario
            report["bitonic"].append(item)
    report["bitonic"].append(probe_bitonic(
        core,
        variant="current_boolean_select",
        device="NPU",
        config={"NPU_COMPILER_TYPE": "PLUGIN"},
        values=f16_subnormals,
        dtype="f16",
        label="f16_subnormals",
    ))
    report["bitonic"].append(probe_bitonic(
        core,
        variant="current_boolean_select",
        device="NPU",
        config={"NPU_COMPILER_TYPE": "PLUGIN"},
        values=f32_one_ulp,
        dtype="f32",
        label="f32_one_ulp",
    ))

    output = HERE / "openvino_runtime_probe.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
