"""Reproducible, host-input/host-output end-to-end Top-K benchmarking.

Every method is checked against the same deterministic CPU reference. Device
compilation is excluded from steady-state timing; warmups are also excluded.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import numpy as np

from .algorithms import MethodRunner
from .reference import exact_topk
from .workloads import scores_for, load_input


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, timeout=3, text=True).strip()
    except Exception:
        return None


def _append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, default=str, sort_keys=True) + "\n")


def _save_mismatch(path: Path, *, scores: np.ndarray, expected: np.ndarray,
                   obtained: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scores=scores, expected=expected, obtained=obtained)


def bench(*, methods: list[str], ns: list[int], ks: list[int], distributions: list[str],
          dtype: str, seed: int, tile: int, batch: int, initial_budget: int,
          warmup: int, repeats: int, blocks: int, out: Path,
          session_id: str | None = None, allow_skips: bool = False,
          input_path: str | None = None, compiler_type: str = "DEFAULT") -> dict:
    if not methods or not ns or not ks or not distributions:
        raise ValueError("At least one method, N, K and distribution are required")
    if tile <= 0 or batch <= 0 or repeats <= 0 or blocks <= 0 or warmup < 0:
        raise ValueError("Invalid tile, batch, repeats, blocks or warmup")
    session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + \
        f"_{socket.gethostname()}_{os.getpid()}"
    run_dir = Path(out)
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / f"observations_{session_id}.jsonl"
    skips_file = run_dir / f"skips_{session_id}.jsonl"
    errors_dir = run_dir / "mismatches"
    rng = random.Random(seed + 2026)
    manifest = {
        "session_id": session_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "numpy": np.__version__, "git_commit": _git_commit(),
        "methods": methods, "ns": ns, "ks": ks, "distributions": distributions,
        "dtype": dtype, "seed": seed, "tile": tile, "batch": batch,
        "initial_budget": initial_budget, "warmup": warmup, "repeats": repeats,
        "compiler_type": compiler_type,
        "blocks": blocks, "input_path": input_path, "scope": "Top-K operator (scores provided)",
        "measurement": "host input to host indices; compilation and warmup excluded",
        "individual_blocks_independent_sessions": False,
    }
    try:
        import openvino as ov
        manifest["openvino"] = ov.__version__
    except ImportError:
        manifest["openvino"] = None
    (run_dir / f"manifest_{session_id}.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    run_stats = {"session_id": session_id, "observations": 0, "skipped": 0,
                 "wrong_results": 0, "completed": False, "valid_for_inference": False,
                 "results_file": str(output), "skips_file": str(skips_file)}
    for n in ns:
        for k in ks:
            if not 1 <= k <= n:
                continue
            for distribution in distributions:
                scores = (load_input(input_path, dtype) if input_path else
                          scores_for(n, k, distribution, seed, dtype))
                if input_path and scores.size != n:
                    raise ValueError(f"Loaded N={scores.size}, command requested N={n}")
                reference = exact_topk(scores, k)
                digest = hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest()
                case_id = f"N{n}_K{k}_{distribution}_{dtype}_T{tile}_B{batch}_S{seed}"
                runners: dict[str, MethodRunner] = {}
                print(f"CASE {case_id}: preparing {len(methods)} methods", flush=True)
                for method in methods:
                    try:
                        print(f"  PREP {case_id} {method}", flush=True)
                        runner = MethodRunner(
                            method, n, k, tile, batch, dtype, initial_budget,
                            compiler_type=compiler_type,
                        )
                        runner.prepare()
                        # Smoke-infer before any timed observations: missing kernels
                        # are explicit failures, never masked with a CPU result.
                        smoke = runner.run(scores)
                        if not np.array_equal(smoke.indices, reference):
                            _save_mismatch(
                                errors_dir / f"{case_id}_{method}_preflight.npz",
                                scores=scores, expected=reference, obtained=smoke.indices,
                            )
                            raise RuntimeError("Preflight result DOES NOT match exact reference")
                        runners[method] = runner
                    except Exception as e:
                        run_stats["skipped"] += 1
                        why = f"{type(e).__name__}: {e}"
                        _append_jsonl(skips_file, {"session_id": session_id, "case_id": case_id,
                                                   "method": method, "error": why})
                        print(f"  SKIP/FAIL {case_id} {method}: {why}", flush=True)
                for block_id in range(blocks):
                    order = list(runners)
                    rng.shuffle(order)
                    for method in order:
                        runner = runners[method]
                        try:
                            for _warmup_iteration in range(warmup):
                                runner.run(scores)
                        except Exception as e:
                            run_stats["skipped"] += 1
                            _append_jsonl(skips_file, {
                                "session_id": session_id, "case_id": case_id,
                                "block_id": block_id, "method": method,
                                "warmup_iteration": _warmup_iteration,
                                "error": f"WARMUP: {type(e).__name__}: {e}",
                            })
                            print(f"  WARMUP FAIL {method}: {type(e).__name__}: {e}",
                                  flush=True)
                            runners.pop(method, None)
                            continue
                        runtime_failed = False
                        for iteration in range(repeats):
                            t0 = perf_counter_ns()
                            try:
                                result = runner.run(scores)
                                elapsed_ns = perf_counter_ns() - t0
                                correct = bool(np.array_equal(result.indices, reference))
                            except Exception as e:
                                run_stats["skipped"] += 1
                                _append_jsonl(skips_file, {
                                    "session_id": session_id, "case_id": case_id,
                                    "block_id": block_id, "method": method,
                                    "iteration": iteration, "error": f"INFERENCE: {type(e).__name__}: {e}",
                                })
                                print(f"  RUNTIME FAIL {method}: {type(e).__name__}: {e}",
                                      flush=True)
                                runners.pop(method, None)
                                runtime_failed = True
                                break
                            if not correct:
                                run_stats["wrong_results"] += 1
                                errors_dir.mkdir(exist_ok=True)
                                filename = f"{case_id}_{method}_block{block_id}_iter{iteration}.npz"
                                _save_mismatch(errors_dir / filename, scores=scores,
                                               expected=reference, obtained=result.indices)
                            record = {
                                "session_id": session_id, "case_id": case_id,
                                "block_id": block_id, "iteration": iteration,
                                "n": n, "k": k, "distribution": distribution, "dtype": dtype,
                                "tile": tile, "batch": batch, "seed": seed,
                                "input_sha256": digest, "method": method,
                                "elapsed_ns": elapsed_ns, "elapsed_ms": elapsed_ns / 1e6,
                                "correct": correct, "compile_ms_excluded": runner.compile_ms,
                                "device_info": runner.device_info,
                                **result.metadata(),
                            }
                            _append_jsonl(output, record)
                            run_stats["observations"] += 1
                        if not runtime_failed:
                            print(f"  {case_id} block={block_id+1}/{blocks} {method}: measured",
                                  flush=True)
    summary_file = run_dir / f"summary_{session_id}.json"
    run_stats["completed"] = True
    run_stats["valid_for_inference"] = not (
        run_stats["skipped"] or run_stats["wrong_results"]
    )
    summary_file.write_text(json.dumps(run_stats, indent=2) + "\n", encoding="utf-8")
    if (run_stats["skipped"] or run_stats["wrong_results"]) and not allow_skips:
        raise RuntimeError(f"Experiment contained failures; see {skips_file} and {errors_dir}. "
                           f"Stats: {run_stats}")
    return run_stats
