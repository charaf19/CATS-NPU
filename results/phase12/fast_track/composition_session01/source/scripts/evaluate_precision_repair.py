"""Small checked repair evaluation; separate process invocations are sessions.

Timings include host validation/preparation, execution, copying and certificate.
Compilation, first inference and independent reference construction are excluded.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import subprocess
from time import perf_counter_ns

import numpy as np
import openvino as ov

from cats_npu.algorithms import MethodRunner
from cats_npu.order_encoding import score_codes
from cats_npu.precision_topk import PrecisionNPUTopK
from cats_npu.reference import validate_scores
from cats_npu.static_graph import StaticNPUTopK, certify_rank
from scripts.probe_stable_integer_hierarchy import graph as integer_graph


ROOT = Path(__file__).resolve().parents[1]


class IntegerRunner:
    def __init__(self, n, k, width):
        self.n, self.k = n, k
        self.compiled = ov.Core().compile_model(integer_graph(n, k, width), "NPU",
                                                {"NPU_COMPILER_TYPE": "PLUGIN"})
        d = self.compiled.get_property("EXECUTION_DEVICES")
        self.execution_devices = [d] if isinstance(d, str) else list(d)
        if self.execution_devices != ["NPU"]:
            raise RuntimeError("INT32 baseline placement failed")
        self.request = self.compiled.create_infer_request()
        self.data = np.full(((n + width - 1) // width, width), np.iinfo(np.int32).min, np.int32)

    def run(self, scores):
        codes = score_codes(scores)
        self.data.ravel()[:self.n] = codes
        self.request.infer({0: self.data})
        values = np.array(self.request.get_output_tensor(0).data, copy=True).reshape(-1)
        ids = np.array(self.request.get_output_tensor(1).data, copy=True).reshape(-1)
        if (np.any(ids < 0) or np.any(ids >= self.n)
                or not np.array_equal(values, codes[ids])):
            raise RuntimeError("INT32 output codes/IDs do not match the supplied input")
        checked = certify_rank(scores, ids, scores[ids], self.k)
        if not np.array_equal(ids, checked):
            raise RuntimeError("INT32 device output order is not stable")
        return ids


def workloads():
    tiny = np.float16(2**-24)
    known = np.full(512, -1, np.float16)
    known[:64] = (np.arange(64) - 32).astype(np.float16) * tiny
    yield "known_subnormal", known, 4, 256
    mixed = known.copy()
    mixed[-2:] = [65504, -65504]
    yield "subnormal_boundary_and_extremes", mixed, 4, 256
    across = np.full(512, -1, np.float16)
    across[:4] = np.array([1, 2, 3, 4], np.float16) * tiny
    across[256:260] = np.array([8, 7, 6, 5], np.float16) * tiny
    yield "subnormal_across_tiles", across, 4, 256
    zero = np.zeros(513, np.float16)
    zero[::2] = -0.0
    yield "signed_zero_irregular", zero, 10, 256
    tie = np.full(5183, -1, np.float16)
    tie[::100] = 1
    yield "boundary_ties_across_tiles", tie, 10, 256
    yield "irregular_normal", np.arange(257, dtype=np.float16), 10, 256
    domain = np.arange(65536, dtype=np.uint16).view(np.float16)
    domain = domain[np.isfinite(domain)]
    rng = np.random.default_rng(291026)
    rng.shuffle(domain)
    yield "all_finite_words_shuffled", domain, 10, 256
    for seed in (101, 202, 303):
        rng = np.random.default_rng(seed)
        a = rng.choice(domain, size=513).astype(np.float16)
        # A distinct subnormal boundary is deliberately made observable.
        a[a > 0] *= -1
        a[:32] = np.arange(32).astype(np.float16) * tiny
        yield f"subnormal_boundary_seed{seed}", a, 10, 256


def main():
    p = ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--input", type=Path)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=291026)
    p.add_argument("--case", help="Run one named generated correctness case")
    p.add_argument("--controls", action="store_true", help="Add existing C++ and stable CUDA controls")
    args = p.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).resolve(), ROOT / "src/cats_npu/precision_topk.py",
                    ROOT / "scripts/probe_stable_integer_hierarchy.py",
                    ROOT / "src/cats_npu/static_graph.py", ROOT / "src/cats_npu/order_encoding.py"]
    sources = {}
    for path in source_paths:
        relative = path.relative_to(ROOT).as_posix()
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        target = args.out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    core = ov.Core()
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "exploratory interleaved checked baseline evaluation; no novelty claim",
              "source_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": sources, "openvino": ov.__version__, "numpy": np.__version__,
              "npu_driver": str(core.get_property("NPU", "NPU_DRIVER_VERSION")),
              "npu_compiler": str(core.get_property("NPU", "NPU_COMPILER_VERSION")),
              "independent_process_sessions_in_this_file": 1,
              "input_contract": "immutable finite FP16 host vector",
              "output_contract": "host stable IDs; graph methods use full host certificates; "
                                 "historical native/CUDA controls retain documented host tie repair",
              "excluded_from_timing": ["compilation", "first inference", "independent Python reference"],
              "seed": args.seed, "repeats": args.repeats, "case_filter": args.case, "cases": []}
    selected = ([('real_development_input', np.load(args.input, allow_pickle=False), args.k, 256)]
                if args.input else workloads())
    selected = [case for case in selected if not args.case or case[0] == args.case]
    if not selected:
        raise ValueError("No matching workload")
    for name, scores, k, width in selected:
        validate_scores(scores, k)
        n = len(scores)
        folder = args.out / name
        folder.mkdir()
        np.save(folder / "input.npy", scores, allow_pickle=False)
        # A separate, scalar Python sort is outside every timed run.
        expected = np.asarray(sorted(range(n), key=lambda i: (-float(scores[i]), i))[:k], np.int64)
        case = {"name": name, "input_path": str(args.input) if args.input else None,
                "input_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
                "n": n, "k": k, "width": width, "expected_ids": expected.tolist(), "methods": {}}
        report["cases"].append(case)
        runners = {}
        constructors = {"fp16_static": partial(StaticNPUTopK, n, k, width),
                        "stable_i32": partial(IntegerRunner, n, k, width),
                        "fp32_default": partial(PrecisionNPUTopK, n, k, width, ""),
                        "fp32_topk_only": partial(PrecisionNPUTopK, n, k, width, "TopK"),
                        "fp32_reshape_only": partial(PrecisionNPUTopK, n, k, width, "Reshape"),
                        "fp32_island": partial(PrecisionNPUTopK, n, k, width)}
        if args.controls:
            constructors.update({"cpp_exact": partial(MethodRunner, "cpp_exact", n, k),
                                 "cuda_stable": partial(MethodRunner, "torch_cuda_stable", n, k),
                                 "native_npu": partial(MethodRunner, "ov_npu_native_batched_direct",
                                                       n, k, width, 8, "f16", 1,
                                                       compiler_type="PLUGIN")})
        for method, factory in constructors.items():
            result = {"samples": []}
            case["methods"][method] = result
            try:
                runner = factory()
                if hasattr(runner, "prepare"):
                    runner.prepare()
                runners[method] = runner
                result["execution_devices"] = getattr(runner, "execution_devices", None)
                result["device_info"] = getattr(runner, "device_info", None)
                result["compile_config"] = getattr(runner, "config", None)
                if hasattr(runner, "model"):
                    ov.serialize(runner.model, folder / f"{method}.xml")
            except Exception as exc:
                result["construction_error"] = f"{type(exc).__name__}: {exc}"

        def run(method, warmup=False, *, case=case, runners=runners, scores=scores,
                expected=expected, folder=folder):
            result, runner = case["methods"][method], runners[method]
            start = perf_counter_ns()
            try:
                obj = runner.run(scores)
                elapsed = (perf_counter_ns() - start) / 1e6
                ids = obj if isinstance(obj, np.ndarray) else obj.indices
                sample = {"elapsed_ms": elapsed, "correct": bool(np.array_equal(ids, expected))}
                if hasattr(obj, "infer_ns"):
                    sample["infer_sync_ms"] = obj.infer_ns / 1e6
                if hasattr(obj, "cpu_tie_repairs"):
                    sample["cpu_tie_repairs"] = obj.cpu_tie_repairs
                    sample["device_submissions"] = obj.device_submissions
                if not sample["correct"]:
                    sample["incorrect_ids"] = ids.tolist()
            except Exception as exc:
                sample = {"elapsed_ms": (perf_counter_ns() - start) / 1e6,
                          "correct": False, "error": f"{type(exc).__name__}: {exc}"}
            if warmup:
                result["preflight"] = sample
                if hasattr(runner, "request"):
                    np.savez(folder / f"{method}_raw.npz", **{
                        f"output{i}": np.array(runner.request.get_output_tensor(i).data, copy=True)
                        for i in range(2)})
            else:
                result["samples"].append(sample)

        for method in runners:
            run(method, warmup=True)
        rng = random.Random(args.seed)
        order = list(runners)
        for _ in range(args.repeats):
            rng.shuffle(order)
            for method in order:
                run(method)
        for result in case["methods"].values():
            samples = result["samples"]
            result["all_correct"] = bool(samples and all(s["correct"] for s in samples)
                                         and result["preflight"]["correct"])
            if samples:
                result["median_ms"] = float(np.median([s["elapsed_ms"] for s in samples]))
                result["descriptive_p95_ms"] = float(np.percentile(
                    [s["elapsed_ms"] for s in samples], 95))
        (args.out / "results.json").write_text(json.dumps(report, indent=2, default=str) + "\n",
                                               encoding="utf-8")
        print(json.dumps({"case": name, "methods": {m: {"correct": r["all_correct"],
                         "median_ms": r.get("median_ms"), "error": r.get("construction_error")}
                         for m, r in case["methods"].items()}}), flush=True)
    return 0 if all(c["methods"]["fp32_island"]["all_correct"] for c in report["cases"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
