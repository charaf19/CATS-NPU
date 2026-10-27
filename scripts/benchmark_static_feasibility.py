"""Small exploratory Phase 9 comparison; never a confirmatory experiment."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter_ns

import numpy as np

from cats_npu.algorithms import MethodRunner
from cats_npu.reference import exact_topk
from cats_npu.static_graph import StaticNPUTopK
from cats_npu.workloads import load_input, scores_for


METHODS = (
    "static_npu_rank_certified", "ov_npu_native_batched_direct",
    "ov_npu_native_batched_hier", "ov_npu_native_batched_adaptive",
    "ov_npu_native_batched_costadaptive", "cpp_exact", "torch_cuda_stable",
)


def main() -> int:
    p = ArgumentParser()
    p.add_argument("--input", type=Path)
    p.add_argument("--n", type=int, default=16384)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--distribution", default="normal")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    scores = (load_input(args.input, "f16") if args.input else
              scores_for(args.n, args.k, args.distribution, 42027, "f16"))
    reference = exact_topk(scores, args.k)
    methods = args.methods.split(",")
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory; one process, no inferential statistics",
        "source": str(args.input) if args.input else args.distribution,
        "input_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
        "n": int(scores.size), "k": args.k, "width": args.width,
        "batch": args.batch, "input_residency": "host FP16 scores",
        "output_residency": "host stable Top-K indices",
        "compilation_and_warmup_excluded": True,
        "samples_within_one_process_are_not_independent_sessions": True,
        "methods": {},
    }
    runners = {}
    for method in methods:
        item = {"samples": []}
        record["methods"][method] = item
        try:
            if method == "static_npu_rank_certified":
                runner = StaticNPUTopK(scores.size, args.k, args.width)
                item["execution_devices"] = runner.execution_devices
            else:
                runner = MethodRunner(method, scores.size, args.k, args.width,
                                      args.batch, "f16", 1, compiler_type="PLUGIN")
                runner.prepare()
                item["device_info"] = runner.device_info
            item["compile_ms"] = runner.compile_ms
            smoke = runner.run(scores)
            if not np.array_equal(smoke.indices, reference):
                raise RuntimeError("Preflight differs from exact reference")
            runners[method] = runner
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
    if runners:
        order = list(runners)
        random.Random(42027).shuffle(order)
        for _ in range(args.repeats):
            for method in order:
                runner = runners[method]
                item = record["methods"][method]
                if "error" in item:
                    continue
                try:
                    start = perf_counter_ns()
                    result = runner.run(scores)
                    elapsed = perf_counter_ns() - start
                    correct = bool(np.array_equal(result.indices, reference))
                    sample = {"elapsed_ms": elapsed / 1e6, "correct": correct}
                    if method == "static_npu_rank_certified":
                        sample.update({"device_submissions": 1,
                                       "sync_boundaries": 1,
                                       "input_preparation_ms": result.input_preparation_ns / 1e6,
                                       "infer_sync_ms": result.infer_ns / 1e6,
                                       "output_copy_ms": result.output_copy_ns / 1e6,
                                       "certificate_ms": result.certificate_ns / 1e6,
                                       "staged_input_bytes": result.staged_input_bytes,
                                       "retrieved_output_bytes": result.retrieved_output_bytes})
                    else:
                        sample.update(result.metadata())
                    item["samples"].append(sample)
                    if not correct:
                        item["error"] = "Timed result differs from exact reference"
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite historical artifact: {args.out}")
    args.out.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.out), "n": scores.size,
                      "methods": {key: {"samples": len(value["samples"]),
                                        "error": value.get("error"),
                                        "median_ms": (float(np.median([s["elapsed_ms"] for s in value["samples"]]))
                                                      if value["samples"] else None)}
                                  for key, value in record["methods"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
