"""Bounded physical TopK diagnosis. Raw device outputs are never repaired.

This is a diagnostic experiment, not a new selection or fuzzing algorithm.
Every invocation requires a new output directory and preserves input/model/output.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
import openvino as ov
import openvino.opset13 as ops

from cats_npu.order_encoding import score_codes
from cats_npu.dyadic_guard import normal_scale


ROOT = Path(__file__).resolve().parents[1]


def reference(row: np.ndarray, k: int) -> list[int]:
    """Python scalar ordering; keeps integer comparisons exact."""
    return sorted(range(len(row)), key=lambda i: (-row[i].item(), i))[:k]


def check_output(data, values, ids, k, stable):
    """Separate membership, ordering, and value contracts, including loose ties."""
    rows = np.asarray(data).reshape(-1, data.shape[-1])
    vals = np.asarray(values).reshape(-1, k)
    chosen = np.asarray(ids).reshape(-1, k)
    results = []
    if len(rows) != len(vals) or len(rows) != len(chosen):
        raise ValueError("Wrong output row count")
    for row, v, ii in zip(rows, vals, chosen, strict=True):
        valid = (np.issubdtype(ii.dtype, np.integer) and len(set(ii.tolist())) == k
                 and bool(np.all((ii >= 0) & (ii < len(row)))))
        expected = reference(row, k)
        cutoff = row[expected[-1]]
        membership = bool(valid and np.all(row[ii] >= cutoff)
                          and set(np.flatnonzero(row > cutoff)).issubset(ii.tolist()))
        value_match = bool(valid and np.array_equal(v, row[ii]))
        ordered = bool(valid and all(row[ii[j]] >= row[ii[j + 1]] for j in range(k - 1)))
        stable_match = bool(valid and ii.tolist() == expected)
        results.append({"valid_ids": bool(valid), "membership": membership,
                        "value_match": value_match, "source_value_sorted": ordered,
                        "stable_ids_match": stable_match, "expected_ids": expected,
                        "passed": membership and value_match and ordered
                        and (stable_match if stable else True)})
    return results


def build_model(shape, k, dtype, kind="direct", stable=True):
    inp = ops.parameter(shape, dtype=dtype, name="scores")
    if kind == "identity":
        return ov.Model([inp], [inp], "identity_transport")
    if kind == "convert":
        return ov.Model([ops.convert(inp, "f32")], [inp], "fp16_to_fp32_transport")
    if kind in ("greater", "greater_promote"):
        source = ops.convert(inp, "f32") if kind == "greater_promote" else inp
        axis = ops.constant(np.array(1, dtype=np.int32))
        left = ops.gather(source, ops.constant(np.array([0], np.int32)), axis)
        right = ops.gather(source, ops.constant(np.array([1], np.int32)), axis)
        return ov.Model([ops.greater(right, left)], [inp], kind)
    selected_input = ops.convert(inp, "f32") if kind == "promote" else inp
    local = ops.topk(selected_input, k, -1, "max", "value",
                     index_element_type="i32", stable=stable)
    if kind != "hierarchy":
        return ov.Model([local.output(0), local.output(1)], [inp], f"{kind}_topk")
    m, width = shape
    bases = ops.constant((np.arange(m, dtype=np.int32) * width).reshape(m, 1))
    ids = ops.add(local.output(1), bases)
    dims = ops.constant(np.array([1, m * k], dtype=np.int64))
    flat_values = ops.reshape(local.output(0), dims, False)
    flat_ids = ops.reshape(ids, dims, False)
    final = ops.topk(flat_values, k, -1, "max", "value",
                     index_element_type="i32", stable=stable)
    selected_ids = ops.gather(flat_ids, final.output(1),
                              ops.constant(np.array(1, dtype=np.int32)))
    return ov.Model([final.output(0), selected_ids.output(0)], [inp], "hierarchical_topk")


def cases(suite):
    tiny = np.float16(2**-24)
    if suite == "isolate":
        for exponent in (-24, -14, -12, -10, -8, -6, -4, 0):
            if exponent == -24:
                a = (np.arange(16) - 8).astype(np.float16) * tiny
            else:
                base = np.float16(2.0**exponent)
                a = base + np.arange(16).astype(np.float16) * np.spacing(base)
            for sign in (1, -1):
                x = np.ascontiguousarray((a * sign).reshape(1, -1))
                yield f"exp{exponent}_sign{sign}", x, 4, "direct", True
        a = np.full(512, -1, np.float16)
        a[:64] = (np.arange(64) - 32).astype(np.float16) * tiny
        for stable in (False, True):
            for kind, shape in (("direct", (1, 512)), ("direct", (2, 256)),
                                ("hierarchy", (2, 256)), ("promote", (1, 512)),
                                ("identity", (2, 256)), ("convert", (2, 256))):
                if kind in ("identity", "convert") and not stable:
                    continue
                yield f"historical_{kind}_{shape[0]}rows_stable{stable}", a.reshape(shape), 4, kind, stable
        yield "historical_int32_direct", score_codes(a).reshape(1, -1), 4, "direct", True
        yield "historical_int32_hierarchy", score_codes(a).reshape(2, 256), 4, "hierarchy", True
    elif suite == "pairs":
        pairs = [[0, 1], [1, 2], [-1, 0], [-2, -1], [-1, 1], [0, -0.0]]
        for i, pair in enumerate(pairs):
            x = (np.array(pair, np.float16) * tiny).reshape(1, 2)
            for kind in ("direct", "promote", "identity", "convert"):
                yield f"pair{i}_{kind}", x.copy(), 1, kind, True
        for exponent in (-14, -10, -6, -4, 0):
            base = np.float16(2.0**exponent)
            x = np.array([[base, np.nextafter(base, np.float16(np.inf))]], np.float16)
            yield f"normal_pair_exp{exponent}", x, 1, "direct", True
    elif suite == "controls":
        for pair_name, pair in (("positive", [0, 1]), ("negative", [-2, -1]),
                                ("normal_gap", [1024, 1025])):
            x = (np.array(pair, np.float16) * tiny).reshape(1, 2)
            for dtype in (np.float16, np.float32):
                data = x.astype(dtype)
                for kind in ("direct", "greater", "greater_promote"):
                    yield f"{pair_name}_{np.dtype(dtype).name}_{kind}", data, 1, kind, True
            scaled, _ = normal_scale(x.ravel())
            yield f"{pair_name}_scaled_f16", scaled.reshape(1, 2), 1, "direct", True
        a = np.full(512, -1, np.float16)
        a[:64] = (np.arange(64) - 32).astype(np.float16) * tiny
        scaled, _ = normal_scale(a)
        for name, data in (("fp32", a.astype(np.float32)), ("scaled", scaled)):
            yield f"historical_{name}_direct", data.reshape(1, 512), 4, "direct", True
            yield f"historical_{name}_hierarchy", data.reshape(2, 256), 4, "hierarchy", True


def main():
    p = ArgumentParser()
    p.add_argument("--suite", choices=("isolate", "pairs", "controls"), default="isolate")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "source.py").write_bytes(Path(__file__).read_bytes())
    core = ov.Core()
    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "exploratory numerical-path isolation; no novel algorithm claimed",
            "suite": args.suite, "source_base": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "openvino": ov.__version__, "available_devices": core.available_devices,
            "npu": {}, "repeats": args.repeats,
            "independent_hardware_sessions": 1, "cases": []}
    for prop in ("FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE", "NPU_DRIVER_VERSION", "NPU_COMPILER_VERSION"):
        meta["npu"][prop] = str(core.get_property("NPU", prop))
    cache = {}
    for name, data, k, kind, stable in cases(args.suite):
        folder = args.out / name
        folder.mkdir()
        np.save(folder / "input.npy", data, allow_pickle=False)
        item = {"name": name, "shape": list(data.shape), "dtype": str(data.dtype),
                "k": k, "kind": kind, "stable": stable, "sort": "value",
                "input_sha256": hashlib.sha256(data.tobytes()).hexdigest(), "runs": []}
        meta["cases"].append(item)
        signature = (data.shape, k, data.dtype.str, kind, stable)
        try:
            if signature not in cache:
                model = build_model(data.shape, k, data.dtype, kind, stable)
                ov.serialize(model, folder / "model.xml")
                t = perf_counter_ns()
                npu = core.compile_model(model, "NPU", {"NPU_COMPILER_TYPE": "PLUGIN"})
                compile_ms = (perf_counter_ns() - t) / 1e6
                devices = npu.get_property("EXECUTION_DEVICES")
                devices = [devices] if isinstance(devices, str) else list(devices)
                if devices != ["NPU"]:
                    raise RuntimeError(f"Strict NPU placement failed: {devices}")
                runtime_error = None
                try:
                    ov.serialize(npu.get_runtime_model(), folder / "runtime.xml")
                except Exception as exc:
                    runtime_error = str(exc)
                cpu = core.compile_model(model, "CPU")
                cache[signature] = (npu, cpu, name, compile_ms, devices, runtime_error)
            npu, cpu, graph_case, compile_ms, devices, runtime_error = cache[signature]
            item.update({"graph_case": graph_case, "compile_ms": compile_ms,
                         "execution_devices": devices, "runtime_export_error": runtime_error})
            for backend, executable in (("CPU", cpu), ("NPU", npu)):
                request = executable.create_infer_request()
                for repetition in range(1 if backend == "CPU" else args.repeats):
                    t = perf_counter_ns()
                    request.infer({0: data})
                    infer_ms = (perf_counter_ns() - t) / 1e6
                    outputs = [np.array(request.get_output_tensor(i).data, copy=True)
                               for i in range(len(executable.outputs))]
                    np.savez(folder / f"{backend}_{repetition}.npz",
                             **{f"output{i}": a for i, a in enumerate(outputs)})
                    run = {"backend": backend, "repetition": repetition,
                           "infer_sync_ms": infer_ms, "outputs": [a.tolist() for a in outputs]}
                    if kind in ("identity", "convert"):
                        run["transport_exact"] = bool(np.array_equal(outputs[0], data))
                    elif kind in ("greater", "greater_promote"):
                        expected = data[:, 1:] > data[:, :1]
                        run["transport_exact"] = bool(np.array_equal(outputs[0], expected))
                        run["expected_comparison"] = expected.tolist()
                    else:
                        source = data.reshape(1, -1) if kind == "hierarchy" else data
                        run["checks"] = check_output(source, *outputs, k, stable)
                    item["runs"].append(run)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        (args.out / "results.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        npu_runs = [r for r in item["runs"] if r["backend"] == "NPU"]
        passed = all(r.get("transport_exact", all(c["passed"] for c in r.get("checks", [])))
                     for r in npu_runs) if npu_runs else False
        print(json.dumps({"case": name, "npu_passed": passed, "error": item.get("error")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
