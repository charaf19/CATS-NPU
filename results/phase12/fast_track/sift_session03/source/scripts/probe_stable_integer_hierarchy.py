"""Ablation: documented stable TopK on integer score codes, two NPU levels."""
from __future__ import annotations

from argparse import ArgumentParser
import json
import math
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import openvino as ov
import openvino.opset13 as ops

from cats_npu.reference import exact_topk
from cats_npu.order_encoding import score_codes
from cats_npu.static_graph import certify_rank
from cats_npu.workloads import load_input


def graph(n: int, k: int, width: int, dtype=np.int32):
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           for v in (n, k, width)):
        raise TypeError("N, K, and width must be integers")
    if not 1 <= k <= width < n:
        raise ValueError("Require N > width >= K >= 1")
    m = math.ceil(n / width)
    if m * width > np.iinfo(np.int32).max:
        raise ValueError("Padded size exceeds the graph's signed INT32 index/shape domain")
    if np.dtype(dtype) not in (np.dtype(np.int16), np.dtype(np.int32)):
        raise ValueError("This probe supports INT32 keys and the historical INT16 ablation only")
    inp = ops.parameter([m, width], dtype=dtype)
    local = ops.topk(inp, k, 1, "max", "value", index_element_type="i32", stable=True)
    bases = ops.constant((np.arange(m, dtype=np.int32) * width).reshape(m, 1))
    global_ids = ops.add(local.output(1), bases)
    shape = ops.constant(np.array([1, m * k], dtype=np.int64))
    values = ops.reshape(local.output(0), shape, False)
    ids = ops.reshape(global_ids, shape, False)
    final = ops.topk(values, k, 1, "max", "value", index_element_type="i32", stable=True)
    selected_ids = ops.gather(ids, final.output(1), ops.constant(np.array(1, dtype=np.int32)))
    return ov.Model([final.output(0), selected_ids.output(0)], [inp], "stable_integer_hierarchy")


def main():
    p = ArgumentParser()
    p.add_argument("--input", type=Path)
    p.add_argument("--counterexample", action="store_true")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--key-dtype", choices=["i16", "i32"], default="i32")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.counterexample:
        tiny = np.nextafter(np.float16(0), np.float16(1))
        scores = np.full(512, -1, dtype=np.float16)
        scores[:64] = (np.arange(64) - 32).astype(np.float16) * tiny
    elif args.input:
        scores = load_input(args.input, "f16")
    else:
        raise ValueError("Supply --input or --counterexample")
    n, k, width = len(scores), args.k, args.width
    expected = exact_topk(scores, k)
    key_dtype = np.int16 if args.key_dtype == "i16" else np.int32

    def encode(a):
        codes = score_codes(a)
        return (codes - 32768).astype(np.int16) if args.key_dtype == "i16" else codes

    model = graph(n, k, width, key_dtype)
    m = math.ceil(n / width)
    data = np.full((m * width,), np.iinfo(key_dtype).min, dtype=key_dtype).reshape(m, width)
    data.ravel()[:n] = encode(scores)
    result = {"purpose": "stable integer TopK two-level ablation", "input": str(args.input),
              "n": n, "k": k, "width": width, "key_dtype": args.key_dtype,
              "correct": False, "samples": []}
    try:
        cpu = ov.Core().compile_model(model, "CPU")
        cpu_values, cpu_ids = list(cpu([data]).values())
        ci = np.asarray(cpu_ids).reshape(-1)
        result["cpu_correct"] = bool(np.array_equal(
            ci[np.lexsort((ci, -scores[ci].astype(np.float64)))], expected))
        if not result["cpu_correct"]:
            raise RuntimeError("CPU graph preflight failed; NPU execution cancelled")
        compiled = ov.Core().compile_model(model, "NPU", {"NPU_COMPILER_TYPE": "PLUGIN"})
        d = compiled.get_property("EXECUTION_DEVICES")
        result["execution_devices"] = [d] if isinstance(d, str) else list(d)
        if result["execution_devices"] != ["NPU"]:
            raise RuntimeError("Strict NPU placement not established")
        request = compiled.create_infer_request()
        for _ in range(args.repeats):
            start = perf_counter_ns()
            codes = encode(scores)
            encoded = perf_counter_ns()
            data.ravel()[:n] = codes
            request.infer({0: data})
            inferred = perf_counter_ns()
            values = np.array(request.get_output_tensor(0).data, copy=True).reshape(-1)
            ids = np.array(request.get_output_tensor(1).data, copy=True).reshape(-1)
            if np.any(ids < 0) or np.any(ids >= n) or not np.array_equal(values, codes[ids]):
                raise RuntimeError("NPU output values or IDs disagree with input")
            chosen = certify_rank(scores, ids, scores[ids], k)
            checked = perf_counter_ns()
            correct = bool(np.array_equal(chosen, expected))
            result["samples"].append({"encoding_ms": (encoded - start) / 1e6,
                                      "infer_and_staging_ms": (inferred - encoded) / 1e6,
                                      "verification_ms": (checked - inferred) / 1e6,
                                      "end_to_end_ms": (checked - start) / 1e6,
                                      "correct": correct})
            if not correct:
                raise RuntimeError("Stable TopK result differs from reference")
        result["correct"] = True
        result["selected_ids"] = chosen.tolist()
        result["expected_ids"] = expected.tolist()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("samples", "selected_ids", "expected_ids")}, indent=2))
    return 0 if result["correct"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
