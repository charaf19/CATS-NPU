"""Exploratory end-to-end score-sketch dispatch against frozen NPU choices.

Each invocation is one separately initiated session. Compilation and score
loading are excluded; sketch, prediction, and method execution are timed.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import subprocess
from time import perf_counter_ns

import numpy as np

from cats_npu.algorithms import MethodRunner
from cats_npu.reference import exact_topk
from explore_phase8_predictor import METHODS, load_cases, score_sketch, train


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def execute(arm: str, *, model, scores: np.ndarray, case: dict, arms: dict,
            runners: dict, session_id: str, case_id: str,
            reference: np.ndarray) -> dict:
    started = perf_counter_ns()
    dispatch_ns = 0
    if arm == "tree_dynamic":
        dispatch_start = perf_counter_ns()
        selected = METHODS[int(np.argmin(model.predict(
            score_sketch(scores, case["k"])[None, :])[0]))]
        dispatch_ns = perf_counter_ns() - dispatch_start
    else:
        selected = arms[arm]
    if selected != arms[arm]:
        raise AssertionError("Runtime method differs from frozen choice")
    result = runners[selected].run(scores)
    elapsed_ns = perf_counter_ns() - started
    if not np.array_equal(result.indices, reference):
        raise AssertionError(f"Incorrect result for {case_id}/{arm}")
    return {"session_id": session_id, "case_id": case_id, "arm": arm,
            "method": selected, "dataset": case["dataset"],
            "query_position": case["query_position"], "n": case["n"],
            "k": case["k"], "score_sha256": case["score_sha256"],
            "elapsed_ms": elapsed_ns / 1e6, "dispatch_ms": dispatch_ns / 1e6,
            "correct": True, "device_info": runners[selected].device_info,
            "compile_ms_excluded": runners[selected].compile_ms,
            **result.metadata()}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--session-index", type=int, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("results/phase8/integrated_dispatch_exploratory"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.session_index < 0 or min(args.blocks, args.repeats) < 1:
        raise ValueError("Invalid session or timing parameters")
    config_path = Path("configs/phase8_beir_validation.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen_path = Path(config["predictions_path"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    cases = load_cases(
        Path("results/phase8/predictive_headroom.json"),
        Path("results/phase8/synthetic_exploratory_fixed/observations_phase8_exploratory_fixed_20261027.jsonl"),
    )
    training_hashes = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in frozen["training_sources_sha256"]
    }
    if training_hashes != frozen["training_sources_sha256"]:
        raise ValueError("Development data changed after predictions were frozen")
    model = train(cases, "tree")
    static_by_shape = defaultdict(set)
    for row in frozen["predictions"]:
        static_by_shape[row["n"], row["k"]].add(row["tree_choice"])
    if any(len(choices) != 1 for choices in static_by_shape.values()):
        raise ValueError("Tree choices are not constant within an N/K group")
    static_by_shape = {shape: next(iter(choices))
                       for shape, choices in static_by_shape.items()}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_id = f"phase8_integrated_{timestamp}_{args.session_index:02d}"
    root = args.out / session_id
    root.mkdir(parents=True, exist_ok=False)
    output = root / "observations.jsonl"
    manifest = {
        "session_id": session_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    text=True).strip(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "training_sources_sha256": training_hashes,
        "session_index": args.session_index,
        "warmup": args.warmup, "blocks": args.blocks, "repeats": args.repeats,
        "arms": ["tree_dynamic", "shape_rule", "tree_static_by_shape"],
        "scope": "host score vector to host exact indices; compilation and loading excluded",
        "runs": [], "all_valid": False,
    }
    predictions = list(frozen["predictions"])
    random.Random(47000 + args.session_index).shuffle(predictions)
    rng = random.Random(47027 + args.session_index)
    for position, case in enumerate(predictions, start=1):
        path = Path(case["score_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["score_sha256"]:
            raise ValueError(f"Input changed: {path}")
        scores = np.load(path, allow_pickle=False)
        reference = exact_topk(scores, case["k"])
        prediction = METHODS[int(np.argmin(model.predict(
            score_sketch(scores, case["k"])[None, :])[0]))]
        if prediction != case["tree_choice"]:
            raise ValueError("Recomputed model choice differs from frozen prediction")
        arms = {"tree_dynamic": case["tree_choice"],
                "shape_rule": case["shape_choice"],
                "tree_static_by_shape": static_by_shape[case["n"], case["k"]]}
        runners = {}
        for method in sorted(set(arms.values())):
            runner = MethodRunner(method, case["n"], case["k"],
                                  config["tile_width"], config["tile_batch"],
                                  config["dtype"], config["initial_candidate_budget"],
                                  compiler_type=config["npu_compiler_type"])
            runner.prepare()
            device_map = runner.device_info.get("reported_execution_devices", {})
            if not device_map or any(devices != ["NPU"] for devices in device_map.values()):
                raise ValueError(f"Unexpected execution device: {method}")
            if not np.array_equal(runner.run(scores).indices, reference):
                raise AssertionError(f"Preflight mismatch: {method}")
            runners[method] = runner
        tag = case["dataset"].split("/")[-1]
        case_id = f"{tag}_q{case['query_position']:04d}_k{case['k']}"

        execute_case = partial(
            execute, model=model, scores=scores, case=case, arms=arms,
            runners=runners, session_id=session_id, case_id=case_id,
            reference=reference)
        for _ in range(args.warmup):
            for arm in arms:
                execute_case(arm)
        for block in range(args.blocks):
            for repetition in range(args.repeats):
                order = list(arms)
                rng.shuffle(order)
                for arm in order:
                    row = execute_case(arm)
                    row["block"] = block
                    row["repetition"] = repetition
                    append(output, row)
        manifest["runs"].append({"case_id": case_id, "score_sha256": case["score_sha256"],
                                 "arms": arms, "compiled_methods": sorted(runners)})
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                             encoding="utf-8")
        print(f"[{position}/{len(predictions)}] {case_id}: valid", flush=True)
    manifest["all_valid"] = True
    manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"output": str(root), "observations": len(predictions) *
                      len(manifest["arms"]) * args.repeats * args.blocks}, indent=2))


if __name__ == "__main__":
    main()
