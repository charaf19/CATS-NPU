"""One-process, interleaved pilot for the split-key novelty hypothesis."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter_ns

import numpy as np
import openvino as ov

from cats_npu.algorithms import MethodRunner
from cats_npu.order_encoding import score_codes
from cats_npu.reference import exact_topk
from cats_npu.static_graph import StaticNPUTopK, certify_rank
from cats_npu.workloads import load_input
from probe_split_key_npu import graph
from probe_stable_integer_hierarchy import graph as stable_integer_graph


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    scores = load_input(args.input, "f16")
    expected = exact_topk(scores, args.k)
    n, k, width = len(scores), args.k, args.width
    m = math.ceil(n / width)
    split_data = np.zeros((m * width,), dtype=np.int32).reshape(m, width)
    stable_data = np.zeros((m * width,), dtype=np.int32).reshape(m, width)
    split = ov.Core().compile_model(graph(n, k, width), "NPU",
                                    {"NPU_COMPILER_TYPE": "PLUGIN"})
    raw_devices = split.get_property("EXECUTION_DEVICES")
    devices = [raw_devices] if isinstance(raw_devices, str) else list(raw_devices)
    if devices != ["NPU"]:
        raise RuntimeError(f"Split graph did not establish strict NPU: {devices}")
    request = split.create_infer_request()
    stable = ov.Core().compile_model(stable_integer_graph(n, k, width), "NPU",
                                     {"NPU_COMPILER_TYPE": "PLUGIN"})
    stable_raw_devices = stable.get_property("EXECUTION_DEVICES")
    stable_devices = ([stable_raw_devices] if isinstance(stable_raw_devices, str)
                      else list(stable_raw_devices))
    if stable_devices != ["NPU"]:
        raise RuntimeError(f"Stable integer graph did not establish strict NPU: {stable_devices}")
    stable_request = stable.create_infer_request()
    fp = StaticNPUTopK(n, k, width)
    direct = MethodRunner("ov_npu_native_batched_direct", n, k, width, 8, "f16", 1,
                          compiler_type="PLUGIN")
    direct.prepare()
    cpp = MethodRunner("cpp_exact", n, k)
    cuda = MethodRunner("torch_cuda_stable", n, k)
    runners = {"split_key_npu": None, "stable_i32_npu": None,
               "fp16_static_npu": fp, "direct_npu": direct,
               "cpp_exact": cpp, "torch_cuda_stable": cuda}
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "exploratory interleaved one-process pilot; not independent sessions",
              "input": str(args.input), "input_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
              "n": n, "k": k, "width": width, "repeats": args.repeats,
              "input_residency": "host FP16 scores", "output_residency": "host stable IDs",
              "compilation_and_first_inference_excluded": True,
              "execution_devices": {"split_key_npu": devices,
                                    "stable_i32_npu": stable_devices,
                                    "fp16_static_npu": fp.execution_devices,
                                    "direct_npu": direct.device_info},
              "samples": {name: [] for name in runners}}

    def run(name: str) -> dict:
        start = perf_counter_ns()
        if name == "split_key_npu":
            codes = score_codes(scores)
            encoded = perf_counter_ns()
            split_data.ravel()[:n] = codes
            request.infer({0: split_data})
            inferred = perf_counter_ns()
            values = np.array(request.get_output_tensor(0).data, copy=True).reshape(-1)
            ids = np.array(request.get_output_tensor(1).data, copy=True).reshape(-1)
            if np.any(ids < 0) or np.any(ids >= n) or not np.array_equal(values, codes[ids]):
                raise RuntimeError("Split graph returned invalid code or ID")
            got = certify_rank(scores, ids, scores[ids], k)
            end = perf_counter_ns()
            sample = {"elapsed_ms": (end - start) / 1e6,
                      "encoding_ms": (encoded - start) / 1e6,
                      "staging_and_infer_ms": (inferred - encoded) / 1e6,
                      "verification_and_copy_ms": (end - inferred) / 1e6,
                      "device_submissions": 1,
                      "staged_input_bytes": split_data.nbytes,
                      "retrieved_output_bytes": values.nbytes + ids.nbytes}
        elif name == "stable_i32_npu":
            codes = score_codes(scores)
            encoded = perf_counter_ns()
            stable_data.ravel()[:n] = codes
            stable_request.infer({0: stable_data})
            inferred = perf_counter_ns()
            values = np.array(stable_request.get_output_tensor(0).data, copy=True).reshape(-1)
            ids = np.array(stable_request.get_output_tensor(1).data, copy=True).reshape(-1)
            if np.any(ids < 0) or np.any(ids >= n) or not np.array_equal(values, codes[ids]):
                raise RuntimeError("Stable integer graph returned invalid code or ID")
            got = certify_rank(scores, ids, scores[ids], k)
            end = perf_counter_ns()
            sample = {"elapsed_ms": (end - start) / 1e6,
                      "encoding_ms": (encoded - start) / 1e6,
                      "staging_and_infer_ms": (inferred - encoded) / 1e6,
                      "verification_and_copy_ms": (end - inferred) / 1e6,
                      "device_submissions": 1,
                      "staged_input_bytes": stable_data.nbytes,
                      "retrieved_output_bytes": values.nbytes + ids.nbytes}
        else:
            got = runners[name].run(scores).indices
            end = perf_counter_ns()
            sample = {"elapsed_ms": (end - start) / 1e6}
        sample["correct"] = bool(np.array_equal(got, expected))
        if not sample["correct"]:
            raise RuntimeError(f"{name} differed from independent exact reference")
        return sample

    for name in runners:
        run(name)  # preflight and first inference outside timing
    order = list(runners)
    rng = random.Random(42027)
    for _ in range(args.repeats):
        rng.shuffle(order)
        for name in order:
            report["samples"][name].append(run(name))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({name: round(float(np.median([x["elapsed_ms"] for x in samples])), 3)
                      for name, samples in report["samples"].items()}, indent=2))


if __name__ == "__main__":
    main()
