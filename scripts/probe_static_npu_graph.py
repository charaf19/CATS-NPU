"""Minimal physical probe for a one-submission, two-level static Top-K graph.

Exploratory Phase 9 only. A successful compilation is not a correctness claim.
This early graph tests a minimum local gap; the rank-certified prototype in
cats_npu.static_graph supersedes this deliberately conservative diagnostic.
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
from cats_npu.workloads import load_input, scores_for


def graph(ov, ops, n: int, k: int, width: int):
    tiles = math.ceil(n / width)
    if tiles < 2 or k + 1 > width or tiles * k < k + 1:
        raise ValueError("Require at least two tiles, K+1 <= width, and enough candidates")
    inp = ops.parameter([tiles, width], dtype=np.float16, name="scores")
    local = ops.topk(inp, k + 1, axis=1, mode="max", sort="value",
                     index_element_type="i32")
    axis = ops.constant(np.array(1, dtype=np.int32))
    keep = ops.constant(np.arange(k, dtype=np.int32))
    local_values = ops.gather(local.output(0), keep, axis)
    local_indices = ops.gather(local.output(1), keep, axis)
    bases = ops.constant((np.arange(tiles, dtype=np.int32) * width).reshape(tiles, 1))
    global_ids = ops.add(local_indices, bases)
    flat_shape = ops.constant(np.array([1, tiles * k], dtype=np.int64))
    candidate_values = ops.reshape(local_values, flat_shape, False)
    candidate_ids = ops.reshape(global_ids, flat_shape, False)
    final = ops.topk(candidate_values, k + 1, axis=1, mode="max", sort="value",
                     index_element_type="i32")
    selected_ids = ops.gather(candidate_ids, final.output(1), axis)
    kth = ops.gather(local.output(0), ops.constant(np.array([k - 1], dtype=np.int32)), axis)
    next_value = ops.gather(local.output(0), ops.constant(np.array([k], dtype=np.int32)), axis)
    local_gaps = ops.subtract(kth, next_value)
    minimum_gap = ops.reduce_min(local_gaps,
                                 ops.constant(np.array([0, 1], dtype=np.int64)), False)
    return ov.Model([final.output(0), selected_ids.output(0), minimum_gap.output(0)], [inp],
                    f"static_twolevel_N{n}_K{k}_W{width}")


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--distribution", default="normal")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path,
                        default=Path("results/phase9/static_graph_probe.json"))
    args = parser.parse_args()
    import openvino as ov
    import openvino.opset13 as ops

    scores = (load_input(args.input, "f16") if args.input else
              scores_for(args.n, args.k, args.distribution, 42027, "f16"))
    if len(scores) != args.n:
        raise ValueError("Input N mismatch")
    validate_scores(scores, args.k)
    tiles = math.ceil(args.n / args.width)
    padded = np.full((tiles * args.width,), -np.inf, dtype=np.float16)
    padded[:args.n] = scores
    data = padded.reshape(tiles, args.width)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "exploratory physical feasibility probe",
              "n": args.n, "k": args.k, "width": args.width, "tiles": tiles,
              "distribution": args.distribution if args.input is None else str(args.input),
              "input_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
              "contains_subnormal": bool(np.any(
                  (np.abs(scores) > 0) & (np.abs(scores) < np.finfo(np.float16).tiny))),
              "npu_compiler_type_requested": "PLUGIN",
              "cpu_graph_correct": False, "npu_compiled": False,
              "npu_executed": False, "npu_exact": False,
              "one_infer_request_per_topk": True,
              "returned_values_per_query": 2 * (args.k + 1) + 1}
    try:
        model = graph(ov, ops, args.n, args.k, args.width)
        core = ov.Core()
        cpu = core.compile_model(model, "CPU")
        cpu_values, cpu_ids, cpu_gap = [np.asarray(x) for x in cpu([data]).values()]
        expected = exact_topk(scores, args.k)
        cpu_chosen = cpu_ids.reshape(-1)[:args.k].astype(np.int64)
        cpu_order = np.lexsort((cpu_chosen, -scores[cpu_chosen].astype(np.float64)))
        report["cpu_graph_correct"] = bool(np.array_equal(cpu_chosen[cpu_order], expected))
        report["cpu_min_local_gap"] = float(cpu_gap)
        report["cpu_final_gap"] = float(cpu_values.reshape(-1)[args.k - 1] -
                                          cpu_values.reshape(-1)[args.k])
        if not report["cpu_graph_correct"]:
            raise RuntimeError("Graph fails CPU correctness; NPU test cancelled")
        start = perf_counter_ns()
        compiled = core.compile_model(model, "NPU", {"NPU_COMPILER_TYPE": "PLUGIN"})
        report["compile_ms"] = (perf_counter_ns() - start) / 1e6
        report["npu_compiled"] = True
        raw_devices = compiled.get_property("EXECUTION_DEVICES")
        report["execution_devices"] = ([raw_devices] if isinstance(raw_devices, str)
                                       else list(raw_devices))
        if report["execution_devices"] != ["NPU"]:
            raise RuntimeError("Compiled graph does not report strict NPU execution")
        request = compiled.create_infer_request()
        times = []
        for _ in range(args.repeats):
            start = perf_counter_ns()
            request.infer({0: data})
            times.append((perf_counter_ns() - start) / 1e6)
        report["inference_ms"] = times
        npu_values = np.array(request.get_output_tensor(0).data, copy=True).reshape(-1)
        npu_ids = np.array(request.get_output_tensor(1).data, copy=True).reshape(-1)
        npu_gap = float(np.array(request.get_output_tensor(2).data, copy=True))
        report["npu_executed"] = True
        report["npu_min_local_gap"] = npu_gap
        report["npu_final_gap"] = float(npu_values[args.k - 1] - npu_values[args.k])
        chosen = npu_ids[:args.k].astype(np.int64)
        valid_ids = bool(np.all((chosen >= 0) & (chosen < args.n)))
        valid_values = bool(valid_ids and np.array_equal(scores[chosen], npu_values[:args.k]))
        if valid_ids:
            order = np.lexsort((chosen, -scores[chosen].astype(np.float64)))
            chosen = chosen[order]
        report["npu_exact"] = bool(valid_values and np.array_equal(chosen, expected))
        report["tie_free_boundaries"] = bool(npu_gap > 0 and report["npu_final_gap"] > 0)
        report["selected_ids"] = chosen.tolist()
        report["expected_ids"] = expected.tolist()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["npu_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
