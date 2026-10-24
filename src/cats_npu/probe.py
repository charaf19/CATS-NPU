"""Strict device discovery and hardware operator probing (never AUTO/HETERO)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
import json
import numpy as np

from .backends import (NPU_COMPILER_TYPES, _ov_import, _bitonic_model, _native_model,
                       BackendUnavailable)
from .reference import exact_topk


OPS = ("TopK", "ReduceMax", "GreaterSelect", "ReduceSum", "Gather", "Concat", "Bitonic")
def _execution_devices(value) -> list[str]:
    """Normalize OpenVINO's scalar NPU and sequence CPU property forms."""
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def doctor() -> dict:
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "openvino": None,
        "openvino_devices": [],
        "properties": {},
        "notes": ["Strict 'NPU' runs will never silently fall back through AUTO/HETERO."],
    }
    try:
        ov, _ = _ov_import()
        core = ov.Core()
        data["openvino"] = ov.__version__
        data["openvino_devices"] = list(core.available_devices)
        for dev in core.available_devices:
            item = {}
            for prop in ("FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE", "NPU_DRIVER_VERSION",
                         "NPU_COMPILER_VERSION", "EXECUTION_DEVICES"):
                try:
                    item[prop] = str(core.get_property(dev, prop))
                except Exception:
                    pass
            data["properties"][dev] = item
    except Exception as e:
        data["error"] = f"{type(e).__name__}: {e}"
    try:
        import torch  # type: ignore
        data["torch_version"] = torch.__version__
        data["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            data["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        data["torch_version"] = None
    return data


def _model_for_operation(ov, ops, name: str, dtype: str):
    width = 16
    if name == "TopK":
        return _native_model(ov, ops, 1, width, 4, dtype)
    if name == "Bitonic":
        return _bitonic_model(ov, ops, 1, width, 4, dtype)
    x = ops.parameter([1, width], dtype=np.float16 if dtype == "f16" else np.float32)
    axis = ops.constant(np.array([1], dtype=np.int64))
    if name == "ReduceMax":
        outputs = [ops.reduce_max(x, axis, keep_dims=True)]
    elif name == "ReduceSum":
        outputs = [ops.reduce_sum(x, axis, keep_dims=True)]
    elif name == "GreaterSelect":
        zero = ops.constant(np.array(0, dtype=np.float16 if dtype == "f16" else np.float32))
        outputs = [ops.select(ops.greater(x, zero), x, zero)]
    elif name == "Gather":
        outputs = [ops.gather(x, ops.constant(np.array([3, 1, 0], np.int32)),
                              ops.constant(np.array(1, np.int32)))]
    elif name == "Concat":
        outputs = [ops.concat([x, x], axis=1)]
    else:
        raise ValueError(f"Unknown probe operation: {name}")
    return ov.Model(outputs, [x], f"probe_{name}")


def probe(device: str = "NPU", dtype: str = "f16", operations: tuple[str, ...] = OPS,
          compiler_type: str = "DEFAULT") -> dict:
    from time import perf_counter_ns
    ov, ops = _ov_import()
    device = device.upper()
    if device.startswith(("AUTO", "HETERO")):
        raise ValueError("No AUTO/HETERO: requested device must be explicit")
    compiler_type = compiler_type.upper()
    if compiler_type not in NPU_COMPILER_TYPES:
        raise ValueError(f"compiler_type must be one of {NPU_COMPILER_TYPES}")
    if device != "NPU" and compiler_type != "DEFAULT":
        raise ValueError("NPU compiler type can only be selected for device NPU")
    compile_config = (
        {} if compiler_type == "DEFAULT" else {"NPU_COMPILER_TYPE": compiler_type}
    )
    core = ov.Core()
    devices = list(core.available_devices)
    if device not in [d.upper() for d in devices]:
        raise BackendUnavailable(f"Device {device} unavailable; found {devices}")
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_requested": device,
        "available_devices": devices,
        "openvino": ov.__version__,
        "dtype": dtype,
        "compiler_type_requested": compiler_type,
        "operations": {},
        "all_passed": False,
        "warning": (
            "Every row includes actual inference and a deterministic smoke-input check. "
            "Passing this probe is not adversarial numerical-exactness proof."
        ),
    }
    x = np.array([[float(v) / 16 for v in range(16)]],
                 dtype=np.float16 if dtype == "f16" else np.float32)
    for name in operations:
        entry = {"compiled": False, "inferred": False, "correct": False,
                 "execution_devices": [], "compile_ms": None, "inference_ms": None}
        try:
            model = _model_for_operation(ov, ops, name, dtype)
            t0 = perf_counter_ns()
            cm = core.compile_model(model, device, compile_config)
            entry["compile_ms"] = (perf_counter_ns() - t0) / 1e6
            entry["compiled"] = True
            try:
                entry["execution_devices"] = _execution_devices(
                    cm.get_property("EXECUTION_DEVICES")
                )
            except Exception:
                entry["execution_devices"] = []
            if device == "NPU":
                try:
                    entry["compiler_type_used"] = str(
                        cm.get_property("NPU_COMPILER_TYPE")
                    )
                except Exception:
                    entry["compiler_type_used"] = None
            request = cm.create_infer_request()
            t0 = perf_counter_ns()
            request.infer({0: x})
            entry["inference_ms"] = (perf_counter_ns() - t0) / 1e6
            entry["inferred"] = True
            output = np.array(request.get_output_tensor(0).data, copy=True)
            if name in ("TopK", "Bitonic"):
                idx = np.array(request.get_output_tensor(1).data, copy=True).ravel()
                expected = exact_topk(x.ravel(), 4)
                entry["correct"] = (np.array_equal(idx, expected) and
                                    np.array_equal(output.ravel(), x.ravel()[idx]))
                entry["indices"] = idx.tolist()
            else:
                expect = {
                    "ReduceMax": lambda a: a.max(axis=1, keepdims=True),
                    "ReduceSum": lambda a: a.sum(axis=1, keepdims=True),
                    "GreaterSelect": lambda a: np.where(a > 0, a, 0),
                    "Gather": lambda a: a[:, [3, 1, 0]],
                    "Concat": lambda a: np.concatenate([a, a], axis=1),
                }[name](x)
                entry["correct"] = bool(np.allclose(output, expect, atol=0.01, rtol=0.01))
                entry["output_shape"] = list(output.shape)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            # Do not fail all probes if one operator is unavailable.
        result["operations"][name] = entry
    result["all_passed"] = all(x["correct"] for x in result["operations"].values())
    return result


def save_json(data: dict, file: str | Path) -> None:
    out = Path(file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
