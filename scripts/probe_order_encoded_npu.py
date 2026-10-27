"""Bounded Phase 9 novelty probe: stable FP16 ordering as exact int32 keys.

The encoding precondition is checked on each input. Physical NPU results are
verified against the source scores and never repaired by CPU selection.
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
from cats_npu.static_graph import certify_rank
from cats_npu.workloads import load_input


def encode(a: np.ndarray) -> tuple[np.ndarray, dict]:
    """Order embed finite FP16 value/index pairs in signed int32 when span fits.

    The binary16 bit transform is monotone including subnormals. Both zeros
    map to the same score; the global index supplies their stable tie order.
    """
    if a.dtype != np.float16:
        raise ValueError("Only finite FP16 scores are supported")
    bits = a.view(np.uint16).copy()
    bits[bits == 0x8000] = 0
    rank = np.where((bits & 0x8000) != 0, (~bits) & 0xffff, bits ^ 0x8000).astype(np.int64)
    minimum = int(rank.min())
    span = int(rank.max()) - minimum + 1
    n = len(a)
    maximum_unsigned = (span - 1) * (n + 1) + n - 1
    if maximum_unsigned > 2**32 - 2:
        raise ValueError(f"FP16 order span {span} exceeds exact int32 key domain at N={n}")
    unsigned = (rank - minimum) * (n + 1) + (n - 1 - np.arange(n, dtype=np.int64))
    keys = (unsigned - 2**31 + 1).astype(np.int32)
    return keys, {"ordered_fp16_span": span, "max_unsigned_key": maximum_unsigned,
                  "int32_capacity_span": (2**32 - 1) // (n + 1)}


def build_graph(n: int, k: int, width: int):
    import openvino as ov
    import openvino.opset13 as ops

    m = math.ceil(n / width)
    inp = ops.parameter([m, width], dtype=np.int32, name="stable_integer_keys")
    local = ops.topk(inp, k, 1, "max", "value", index_element_type="i32")
    base = ops.constant((np.arange(m, dtype=np.int32) * width).reshape(m, 1))
    ids = ops.add(local.output(1), base)
    shape = ops.constant(np.array([1, m * k], dtype=np.int64))
    values_flat = ops.reshape(local.output(0), shape, False)
    ids_flat = ops.reshape(ids, shape, False)
    final = ops.topk(values_flat, k, 1, "max", "value", index_element_type="i32")
    final_ids = ops.gather(ids_flat, final.output(1),
                           ops.constant(np.array(1, dtype=np.int32)))
    return ov.Model([final.output(0), final_ids.output(0)], [inp], "integer_two_level_topk")


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
              "status": "exploratory physical NPU order-encoding probe",
              "input": str(args.input) if args.input else "known_subnormal_counterexample",
              "n": len(scores), "k": args.k, "width": args.width,
              "input_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
              "correct": False, "samples": [],
              "host_input_to_host_indices": True,
              "independent_sessions": False}
    try:
        import openvino as ov

        start = perf_counter_ns()
        keys, info = encode(scores)
        report.update(info)
        report["initial_encoding_ms"] = (perf_counter_ns() - start) / 1e6
        tiles = math.ceil(len(scores) / args.width)
        padded = np.full(tiles * args.width, np.iinfo(np.int32).min, dtype=np.int32)
        padded[:len(keys)] = keys
        data = padded.reshape(tiles, args.width)
        start = perf_counter_ns()
        compiled = ov.Core().compile_model(build_graph(len(scores), args.k, args.width),
                                           "NPU", {"NPU_COMPILER_TYPE": "PLUGIN"})
        report["compile_ms"] = (perf_counter_ns() - start) / 1e6
        dev = compiled.get_property("EXECUTION_DEVICES")
        report["execution_devices"] = [dev] if isinstance(dev, str) else list(dev)
        if report["execution_devices"] != ["NPU"]:
            raise RuntimeError("Explicit NPU execution was not established")
        request = compiled.create_infer_request()
        expected = exact_topk(scores, args.k)
        for _ in range(args.repeats):
            start = perf_counter_ns()
            encoded, _ = encode(scores)
            prepared = perf_counter_ns()
            data.ravel()[:len(encoded)] = encoded
            staged = perf_counter_ns()
            request.infer({0: data})
            inferred = perf_counter_ns()
            got_keys = np.array(request.get_output_tensor(0).data, copy=True).reshape(-1)
            ids = np.array(request.get_output_tensor(1).data, copy=True).reshape(-1)
            copied = perf_counter_ns()
            if np.any(ids < 0) or np.any(ids >= len(scores)):
                raise RuntimeError("NPU returned an out-of-range index")
            if not np.array_equal(got_keys, encoded[ids]):
                raise RuntimeError("NPU output key differs from original encoded key")
            certified = certify_rank(scores, ids, scores[ids], args.k)
            checked = perf_counter_ns()
            correct = bool(np.array_equal(certified, expected))
            report["samples"].append({"encoding_ms": (prepared - start) / 1e6,
                                      "staging_ms": (staged - prepared) / 1e6,
                                      "infer_sync_ms": (inferred - staged) / 1e6,
                                      "copy_ms": (copied - inferred) / 1e6,
                                      "verification_ms": (checked - copied) / 1e6,
                                      "end_to_end_ms": (checked - start) / 1e6,
                                      "correct": correct})
            if not correct:
                raise RuntimeError("Certified result differs from independent reference")
        report["correct"] = True
        report["selected_ids"] = certified.tolist()
        report["expected_ids"] = expected.tolist()
        report["submissions_per_query"] = 1
        report["staged_input_bytes"] = data.nbytes
        report["retrieved_output_bytes"] = got_keys.nbytes + ids.nbytes
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "selected_ids", "expected_ids")}, indent=2))
    return 0 if report["correct"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
