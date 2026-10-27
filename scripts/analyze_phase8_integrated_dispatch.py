"""Validate and summarize exploratory integrated Phase 8 dispatcher runs."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


ARMS = ("tree_dynamic", "shape_rule", "tree_static_by_shape")


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("results/phase8/integrated_dispatch_exploratory"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/phase8/integrated_dispatch_analysis.json"))
    args = parser.parse_args()
    sessions = sorted(args.input.glob("phase8_integrated_*"))
    if len(sessions) != 3:
        raise ValueError(f"Expected three separately initiated sessions, got {len(sessions)}")
    config = json.loads(Path("configs/phase8_beir_validation.json").read_text(encoding="utf-8"))
    frozen_path = Path(config["predictions_path"])
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    grouped = defaultdict(list)
    methods = defaultdict(set)
    dispatch_ms = []
    all_rows = []
    for directory in sessions:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not manifest["all_valid"] or manifest["predictions_sha256"] != frozen_sha:
            raise ValueError(f"Incomplete or changed session: {directory}")
        rows = [json.loads(line) for line in (directory / "observations.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_count = 32 * len(ARMS) * manifest["blocks"] * manifest["repeats"]
        if len(rows) != expected_count:
            raise ValueError(f"Expected {expected_count} rows in {directory}, got {len(rows)}")
        for row in rows:
            if not row["correct"] or row["arm"] not in ARMS:
                raise ValueError("Wrong output or unexpected arm")
            device_map = row["device_info"].get("reported_execution_devices", {})
            if not device_map or any(devices != ["NPU"] for devices in device_map.values()):
                raise ValueError("Execution device is not strict NPU")
            key = (row["session_id"], row["case_id"])
            grouped[key, row["arm"]].append(row["elapsed_ms"])
            methods[key, row["arm"]].add(row["method"])
            if row["arm"] == "tree_dynamic":
                dispatch_ms.append(row["dispatch_ms"])
        all_rows.extend(rows)
    session_ids = sorted({row["session_id"] for row in all_rows})
    case_ids = sorted({row["case_id"] for row in all_rows})
    if len(session_ids) != 3 or len(case_ids) != 32:
        raise ValueError("Incomplete session or case set")
    pair_rows = []
    effects = defaultdict(list)
    for session in session_ids:
        for case in case_ids:
            key = (session, case)
            m = {}
            for arm in ARMS:
                if len(grouped[key, arm]) != 6 or len(methods[key, arm]) != 1:
                    raise ValueError(f"Missing replications: {key}, {arm}")
                m[arm] = median(grouped[key, arm])
            if methods[key, "tree_dynamic"] != methods[key, "tree_static_by_shape"]:
                raise ValueError("Static map did not reproduce tree choice")
            pair_rows.append({"session_id": session, "case_id": case,
                              **{f"{arm}_median_ms": m[arm] for arm in ARMS},
                              "shape_over_tree": m["shape_rule"] / m["tree_dynamic"],
                              "static_over_tree": m["tree_static_by_shape"] /
                                                  m["tree_dynamic"]})
            effects[session].append(pair_rows[-1])
    session_summaries = []
    for session in session_ids:
        rows = effects[session]
        session_summaries.append({"session_id": session,
                                  "shape_over_tree": geomean(
                                      [row["shape_over_tree"] for row in rows]),
                                  "static_over_tree": geomean(
                                      [row["static_over_tree"] for row in rows])})
    result = {"status": "exploratory integrated dispatcher, not a locked confirmatory test",
              "sessions": len(session_ids), "cases_per_session": len(case_ids),
              "observation_count": len(all_rows), "all_correct": True,
              "all_reported_npu": True,
              "tree_choice_constant_within_shape": True,
              "dispatch_ms": {"median": median(dispatch_ms),
                              "p95": float(np.quantile(dispatch_ms, 0.95))},
              "comparisons": {
                  arm: {"geomean_baseline_over_tree": geomean(
                      [row[f"{arm}_over_tree"] for row in pair_rows]),
                        "session_speedups": [row[f"{arm}_over_tree"]
                                             for row in session_summaries]}
                  for arm in ("shape", "static")},
              "session_summaries": session_summaries,
              "limitations": ["Three separately initiated sessions on one laptop, exploratory.",
                              "No inferential p-value or confirmatory performance claim.",
                              "The static comparator is derived from frozen pre-timing tree choices; "
                              "it uses only N and K on these datasets."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
