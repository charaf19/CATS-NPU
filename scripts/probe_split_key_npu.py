"""Exploratory all-finite-FP16 split-key Top-K graph for the physical NPU.

Local Top-K uses an exact int32 score/local-index key. Two static global Top-K
stages find the score threshold and resolve its boundary ties by global ID.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from cats_npu.reference import exact_topk, validate_scores
from cats_npu.order_encoding import score_codes
from cats_npu.static_graph import certify_rank
from cats_npu.workloads import load_input


def graph(n: int, k: int, width: int):
    import openvino as ov
    import openvino.opset13 as ops

    # These limits keep every comparison key exactly representable even if a
    # compiler widens an i32 primitive to f32. They do not prove device behavior.
    if n <= width or k < 1 or k > width or width > 256 or n > 5_592_404:
        raise ValueError("Require width < N, 1 <= K <= width <= 256, N <= 5,592,404")
    m = math.ceil(n / width)
    inp = ops.parameter([m, width], dtype=np.int32, name="ordered_fp16_score_codes")
    local_priority = ops.constant((width - 1 - np.arange(width, dtype=np.int32)).reshape(1, width))
    local_key = ops.add(ops.multiply(inp, ops.constant(np.array(width + 1, dtype=np.int32))),
                        local_priority)
    local = ops.topk(local_key, k, 1, "max", "value", index_element_type="i32")
    bases = ops.constant((np.arange(m, dtype=np.int32) * width).reshape(m, 1))
    global_ids = ops.add(local.output(1), bases)
    flat_ids = ops.reshape(global_ids, ops.constant(np.array([1, m * k], dtype=np.int64)), False)
    flat_codes = ops.reshape(inp, ops.constant(np.array([m * width], dtype=np.int64)), False)
    candidates = ops.gather(flat_codes, flat_ids, ops.constant(np.array(0, dtype=np.int32)))
    preliminary = ops.topk(candidates, k, 1, "max", "value", index_element_type="i32")
    kth = ops.gather(preliminary.output(0),
                     ops.constant(np.array([k - 1], dtype=np.int32)),
                     ops.constant(np.array(1, dtype=np.int32)))
    greater = ops.convert(ops.greater(candidates, kth), "i32")
    equal = ops.convert(ops.equal(candidates, kth), "i32")
    group = ops.add(ops.multiply(greater, ops.constant(np.array(2, dtype=np.int32))), equal)
    priority = ops.add(ops.multiply(group, ops.constant(np.array(n + 1, dtype=np.int32))),
                       ops.subtract(ops.constant(np.array(n, dtype=np.int32)), flat_ids))
    final = ops.topk(priority, k, 1, "max", "value", index_element_type="i32")
    axis = ops.constant(np.array(1, dtype=np.int32))
    selected_ids = ops.gather(flat_ids, final.output(1), axis)
    selected_codes = ops.gather(candidates, final.output(1), axis)
    return ov.Model([selected_codes.output(0), selected_ids.output(0)], [inp],
                    "split_key_exact_topk")


def main() -> int:
    p = ArgumentParser()
    p.add_argument("--input", type=Path)
    p.add_argument("--counterexample", action="store_true")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--width", type=int, default=256)
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
    validate_scores(scores, args.k)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "exploratory physical NPU split-key graph probe",
              "input": str(args.input) if args.input else "known_subnormal_counterexample",
              "input_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
              "n": len(scores), "k": args.k, "width": args.width,
              "correct": False, "samples": [], "independent_sessions": False}
    try:
        import openvino as ov

        model = graph(len(scores), args.k, args.width)
        codes = score_codes(scores)
        m = math.ceil(len(scores) / args.width)
        data = np.zeros((m * args.width,), dtype=np.int32)
        data[:len(scores)] = codes
        data = data.reshape(m, args.width)
        cpu = ov.Core().compile_model(model, "CPU")
        cpu_result = cpu([data])
        cpu_ids = np.asarray(list(cpu_result.values())[1]).reshape(-1)
        cpu_sorted = cpu_ids[np.lexsort((cpu_ids, -scores[cpu_ids].astype(np.float64)))]
        expected = exact_topk(scores, args.k)
        report["cpu_graph_correct"] = bool(np.array_equal(cpu_sorted, expected))
        if not report["cpu_graph_correct"]:
            raise RuntimeError("Graph fails CPU reference; NPU test cancelled")
        start = perf_counter_ns()
        compiled = ov.Core().compile_model(model, "NPU", {"NPU_COMPILER_TYPE": "PLUGIN"})
        report["compile_ms"] = (perf_counter_ns() - start) / 1e6
        devices = compiled.get_property("EXECUTION_DEVICES")
        report["execution_devices"] = [devices] if isinstance(devices, str) else list(devices)
        if report["execution_devices"] != ["NPU"]:
            raise RuntimeError("Explicit NPU placement was not established")
        request = compiled.create_infer_request()
        for _ in range(args.repeats):
            start = perf_counter_ns()
            codes = score_codes(scores)
            encoded = perf_counter_ns()
            data.ravel()[:len(scores)] = codes
            staged = perf_counter_ns()
            request.infer({0: data})
            inferred = perf_counter_ns()
            returned_codes = np.array(request.get_output_tensor(0).data, copy=True).reshape(-1)
            ids = np.array(request.get_output_tensor(1).data, copy=True).reshape(-1)
            copied = perf_counter_ns()
            if np.any(ids < 0) or np.any(ids >= len(scores)):
                raise RuntimeError("NPU returned out-of-range ID")
            if not np.array_equal(returned_codes, codes[ids]):
                raise RuntimeError("NPU output code differs from input code")
            chosen = certify_rank(scores, ids, scores[ids], args.k)
            checked = perf_counter_ns()
            correct = bool(np.array_equal(chosen, expected))
            report["samples"].append({"encoding_ms": (encoded - start) / 1e6,
                                      "staging_ms": (staged - encoded) / 1e6,
                                      "infer_sync_ms": (inferred - staged) / 1e6,
                                      "copy_ms": (copied - inferred) / 1e6,
                                      "verification_ms": (checked - copied) / 1e6,
                                      "end_to_end_ms": (checked - start) / 1e6,
                                      "correct": correct})
            if not correct:
                raise RuntimeError("NPU result differs from independent reference")
        report["correct"] = True
        report["submissions_per_query"] = 1
        report["staged_input_bytes"] = data.nbytes
        report["retrieved_output_bytes"] = returned_codes.nbytes + ids.nbytes
        report["selected_ids"] = chosen.tolist()
        report["expected_ids"] = expected.tolist()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("samples", "selected_ids", "expected_ids")}, indent=2))
    return 0 if report["correct"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
