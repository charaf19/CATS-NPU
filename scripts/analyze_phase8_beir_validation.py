"""Strict paired analysis of frozen BEIR predictor feasibility choices."""
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

from analyze_phase7_confirmatory import bootstrap_ci, exact_sign_flip


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/phase8_beir_validation.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/phase8/beir_validation_analysis.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen_path = Path(config["predictions_path"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    manifest = json.loads((args.input / "validation_manifest.json").read_text(encoding="utf-8"))
    if not manifest["all_valid"]:
        raise ValueError("Incomplete physical experiment")
    if manifest["config_sha256"] != hashlib.sha256(args.config.read_bytes()).hexdigest():
        raise ValueError("Protocol changed after collection")
    if manifest["predictions_sha256"] != hashlib.sha256(frozen_path.read_bytes()).hexdigest():
        raise ValueError("Predictions changed after collection")
    predictions = {
        (row["dataset"].split("/")[-1], row["query_position"], row["k"]): row
        for row in frozen["predictions"]
    }
    files = sorted(args.input.rglob("observations_*.jsonl"))
    records = [json.loads(line) for path in files
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    session_ids = sorted({row["session_id"] for row in records})
    queries = sorted({(key[0], key[1]) for key in predictions})
    reps = config["repeats"] * config["blocks"]
    expected = len(session_ids) * len(predictions) * len(config["methods"]) * reps
    if len(session_ids) != config["independent_sessions"] or len(records) != expected:
        raise ValueError(f"Expected {config['independent_sessions']} sessions and "
                         f"{expected} observations, got {len(session_ids)} and {len(records)}")
    if len(predictions) != len(queries) * len(config["ks"]):
        raise ValueError("Incomplete predictions")
    grouped = defaultdict(list)
    score_digests = defaultdict(set)
    for row in records:
        if not row["correct"] or row["dtype"] != config["dtype"]:
            raise ValueError("Wrong result or input precision")
        if row["tile"] != config["tile_width"] or row["batch"] != config["tile_batch"]:
            raise ValueError("Tile configuration changed")
        device_map = row["device_info"].get("reported_execution_devices", {})
        if not device_map or any(devices != ["NPU"] for devices in device_map.values()):
            raise ValueError("NPU device placement was not proven")
        parts = row["distribution"].split("_")
        dataset, query_position = parts[1], int(parts[2][1:])
        key = (dataset, query_position, row["k"])
        prediction = predictions.get(key)
        if prediction is None or row["n"] != prediction["n"]:
            raise ValueError("Unexpected case")
        if row["method"] not in config["methods"]:
            raise ValueError("Unexpected method")
        grouped[row["session_id"], key, row["method"]].append(row["elapsed_ms"])
        score_digests[key].add(row["input_sha256"])
    if any(len(digests) != 1 for digests in score_digests.values()):
        raise ValueError("Input changed across sessions")
    for prediction in predictions.values():
        if hashlib.sha256(Path(prediction["score_path"]).read_bytes()).hexdigest() != \
                prediction["score_sha256"]:
            raise ValueError("Saved holdout input changed")
    rows = []
    session_logs = []
    for session in session_ids:
        log_ratios = []
        for key, prediction in sorted(predictions.items()):
            medians = {}
            for method in config["methods"]:
                samples = grouped[session, key, method]
                if len(samples) != reps:
                    raise ValueError(f"Missing observations: {session}, {key}, {method}")
                medians[method] = median(samples)
            tree = prediction["tree_choice"]
            shape = prediction["shape_choice"]
            speedup = medians[shape] / medians[tree]
            log_ratios.append(math.log(speedup))
            rows.append({"session_id": session, "dataset": key[0],
                         "query_position": key[1], "n": prediction["n"], "k": key[2],
                         "tree_choice": tree, "shape_choice": shape,
                         "tree_median_ms": medians[tree],
                         "shape_median_ms": medians[shape],
                         "shape_over_tree": speedup,
                         "oracle_method": min(medians, key=medians.get),
                         "oracle_ms": min(medians.values())})
        session_logs.append(sum(log_ratios) / len(log_ratios))
    effects = np.asarray(session_logs)
    estimate = math.exp(float(effects.mean()))
    ci = bootstrap_ci(effects)
    p = exact_sign_flip(effects)
    primary = {"comparison": "shape-rule / tree-choice latency, zero overhead",
               "case_count_per_session": len(predictions),
               "session_count": len(session_ids),
               "geomean_speedup": estimate,
               "ci95_session_cluster_bootstrap": ci,
               "two_sided_exact_sign_flip_p": p,
               "session_speedups": [math.exp(value) for value in session_logs],
               "disagreement_count_per_session": sum(
                   p["tree_choice"] != p["shape_choice"] for p in predictions.values()),
               "practical_threshold": config["practical_speedup_threshold"],
               "passes_zero_overhead_gate": (
                   estimate >= config["practical_speedup_threshold"]
                   and ci[0] > 1 and p < config["alpha"]),
               "warning": "This excludes the measured score-sketch and model decision overhead; "
                          "passing is necessary but not sufficient for actual end-to-end benefit."}
    subgroups = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        for k in config["ks"]:
            ratios = [row["shape_over_tree"] for row in rows
                      if row["dataset"] == dataset and row["k"] == k]
            subgroups[f"{dataset}_K{k}"] = {"geomean_speedup": geomean(ratios),
                                             "paired_rows": len(ratios)}
    report = {"status": "separate Phase 8 holdout feasibility test",
              "protocol": config["name"], "observation_count": len(records),
              "all_correct": True, "all_reported_npu": True,
              "primary": primary, "subgroups_descriptive": subgroups,
              "limitations": ["One laptop and one OpenVINO/NPU compiler stack.",
                              "Session blocks are sequential on one host.",
                              "Tree and shape models were selected after development data inspection; "
                              "the BEIR holdout predictions were frozen before timing.",
                              "This is a zero-overhead feasibility screen, not an integrated dispatcher benchmark."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["primary"], indent=2))


if __name__ == "__main__":
    main()
