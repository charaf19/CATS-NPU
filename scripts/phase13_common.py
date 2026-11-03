"""Shared Phase 13 recording utilities; selection algorithms are unchanged."""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from functools import partial
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phase13"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Used only for an in-progress session's own record, never an old session.
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def power():
    class Status(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte),
                    ("percent", ctypes.c_ubyte), ("saver", ctypes.c_ubyte),
                    ("remaining", ctypes.c_uint32), ("full", ctypes.c_uint32)]

    s = Status()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    return {"ac_connected": s.ac == 1, "battery_percent": int(s.percent),
            "battery_saver": bool(s.saver),
            "power_plan": subprocess.check_output(["powercfg", "/getactivescheme"], text=True).strip()}


def environment():
    import openvino as ov
    import torch

    core = ov.Core()
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "source_revision": git("rev-parse", "HEAD"), "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "openvino", "torch", "sentence-transformers", "scipy")},
            "devices": core.available_devices,
            "cpu_name": str(core.get_property("CPU", "FULL_DEVICE_NAME")),
            "npu": {p: str(core.get_property("NPU", p)) for p in
                    ("FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE", "NPU_DRIVER_VERSION", "NPU_COMPILER_VERSION")},
            "cuda_available": torch.cuda.is_available(),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_threads": torch.get_num_threads(), "power": power()}


def reference(a, k):
    """Independent full stable sort, outside the timed region."""
    return np.lexsort((np.arange(len(a), dtype=np.int64), -a.astype(np.float64)))[:k]


def runners(n, k, width):
    from cats_npu.algorithms import MethodRunner
    from cats_npu.precision_topk import PrecisionNPUTopK
    from cats_npu.static_graph import StaticNPUTopK
    from scripts.evaluate_precision_repair import IntegerRunner

    start = time.perf_counter_ns()
    constructors = {
        "fp32": partial(PrecisionNPUTopK, n, k, width),
        "int32": partial(IntegerRunner, n, k, width),
        "fp16_static": partial(StaticNPUTopK, n, k, width),
        "native_npu": partial(MethodRunner, "ov_npu_native_batched_direct", n, k, width, 8, "f16", 1,
                              compiler_type="PLUGIN"),
        "cpp": partial(MethodRunner, "cpp_exact", n, k),
        "cuda": partial(MethodRunner, "torch_cuda_stable", n, k),
    }
    out, errors = {}, {}
    for name, factory in constructors.items():
        try:
            r = factory()
            if hasattr(r, "prepare"):
                r.prepare()
            out[name] = r
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    preparation_ms = (time.perf_counter_ns() - start) / 1e6
    placement = {}
    for name, r in out.items():
        info = getattr(r, "execution_devices", None) or getattr(r, "device_info", None)
        placement[name] = info
        if name in ("fp32", "int32", "fp16_static") and info != ["NPU"]:
            raise RuntimeError(f"Strict NPU placement failed for {name}: {info}")
        if name == "native_npu":
            reports = info.get("reported_execution_devices", {})
            if not reports or any(v != ["NPU"] for v in reports.values()):
                raise RuntimeError("Native NPU placement not established")
    return out, placement, preparation_ms, errors


def measure(runner, a, expected):
    start = time.perf_counter_ns()
    try:
        result = runner.run(a)
        stop = time.perf_counter_ns()
        ids = result if isinstance(result, np.ndarray) else result.indices
        valid = bool(np.array_equal(ids, expected))
        record = {"elapsed_ns": stop - start, "accepted": True, "correct": valid,
                  "error": None, "indices": ids.tolist()}
        for attr in ("input_preparation_ns", "infer_ns", "output_and_certificate_ns",
                     "certificate_ns", "cpu_tie_repairs", "device_submissions",
                     "backend_execute_sync_ns", "backend_validation_ns"):
            if hasattr(result, attr):
                record[attr] = int(getattr(result, attr))
        if not valid:
            record["error"] = "Accepted output differs from independent exact reference"
        return record
    except Exception as exc:
        return {"elapsed_ns": time.perf_counter_ns() - start, "accepted": False,
                "correct": False, "error": f"{type(exc).__name__}: {exc}"}
