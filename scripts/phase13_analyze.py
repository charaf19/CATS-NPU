"""Recompute Phase 13 tables and prespecified session-level statistics."""
from __future__ import annotations

from argparse import ArgumentParser
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.phase13_common import BASE, ROOT, reference, sha, write_json
from scripts.phase13_design import frozen


def sign_flip(effects):
    x = np.asarray(effects, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty effects required")
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(x))), dtype=np.float64)
    null = np.abs(signs @ x / len(x))
    return float(np.mean(null >= abs(float(np.mean(x))) - 1e-14))


def bootstrap_log_ci(effects, seed, repetitions=50000):
    x = np.asarray(effects, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = x[rng.integers(0, len(x), size=(repetitions, len(x)))].mean(axis=1)
    return np.exp(np.quantile(values, [0.025, 0.975])).tolist()


def crossed_ci(matrix, seed, repetitions=20000):
    a = np.asarray(matrix, dtype=np.float64)
    rng = np.random.default_rng(seed)
    s = rng.integers(0, a.shape[0], (repetitions, a.shape[0]))
    q = rng.integers(0, a.shape[1], (repetitions, a.shape[1]))
    values = a[s[:, :, None], q[:, None, :]].mean(axis=(1, 2))
    return np.exp(np.quantile(values, [0.025, 0.975])).tolist()


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def comparison(matrix, protocol, seed):
    matrix = np.asarray(matrix, dtype=float)
    session = matrix.mean(axis=1)
    blocks = session.reshape(-1, 2).mean(axis=1)
    return {"ratio": float(np.exp(session.mean())),
            "session_ci95": bootstrap_log_ci(session, seed, protocol["bootstrap_resamples"]),
            "exact_signflip_p": sign_flip(session),
            "sessions": len(session), "queries": matrix.shape[1],
            "within_session_repetitions": protocol["timed_repetitions"],
            "session_ratios": np.exp(session).tolist(), "query_ratios": np.exp(matrix.mean(axis=0)).tolist(),
            "crossed_session_query_ci95": crossed_ci(matrix, seed + 1, protocol["crossed_bootstrap_resamples"]),
            "consecutive_pair_block_ci95": bootstrap_log_ci(blocks, seed + 2, protocol["bootstrap_resamples"]),
            "consecutive_pair_block_signflip_p": sign_flip(blocks),
            "leave_one_query_out_ratio_range": [float(np.exp(np.delete(matrix, i, axis=1).mean()))
                                                  for i in range(matrix.shape[1])],
            "lag1_session_correlation": float(np.corrcoef(session[:-1], session[1:])[0, 1])
            if np.std(session[:-1]) and np.std(session[1:]) else None}


def analyze(root, out):
    protocol, lock = frozen()
    out.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((root / "inputs/manifest.json").read_text())
    info = {(x["dataset"], x["query_id"]): x for x in inputs["scores"]}
    cases = {x["id"]: x for x in protocol["performance_cases"]}
    targets = {}
    for case in cases.values():
        source = info[case["dataset"], case["query_id"]]
        assert sha(ROOT / source["path"]) == source["sha256"]
        a = np.load(ROOT / source["path"], allow_pickle=False)[:case["n"]]
        targets[case["id"]] = reference(a, case["k"]).tolist()
    medians, rows, environments = {}, [], []
    raw_times = {}
    failed = []
    for session in range(1, protocol["performance_sessions"] + 1):
        path = root / "performance" / f"session{session:02d}" / "results.json"
        r = json.loads(path.read_text())
        assert r["status"] == "complete", f"Incomplete session {session}"
        environments.append(r["environment_start"])
        condition_valid = r["condition_valid_start"] and r["condition_valid_end"]
        grouped = {}
        for obs in r["observations"]:
            if obs["accepted"]:
                actual = obs["indices"] == targets[obs["case_id"]]
                assert actual == obs["correct"], "Recorded correctness flag disagrees with raw IDs"
            grouped.setdefault((obs["case_id"], obs["method"]), []).append(obs)
        warm = {}
        for obs in r["warmups"]:
            warm.setdefault((obs["case_id"], obs["method"]), []).append(obs)
        for case_id, case in cases.items():
            for method in protocol["methods"]:
                observations = grouped[case_id, method]
                assert len(observations) == protocol["timed_repetitions"]
                valid = bool(condition_valid and all(x["correct"] and x["accepted"] for x in observations)
                             and len(warm.get((case_id, method), [])) == protocol["warmups"]
                             and all(x["correct"] for x in warm.get((case_id, method), [])))
                elapsed = [x["elapsed_ns"] / 1e6 for x in observations if x["elapsed_ns"] is not None]
                raw_times.setdefault((case_id, method), []).extend(elapsed)
                median = float(np.median(elapsed)) if elapsed else None
                medians[session, case_id, method] = median if valid else None
                row = {**case, "session": session, "method": method, "valid": valid,
                       "median_attempt_ms": median,
                       "median_success_ms": median if valid else None,
                       "p95_attempt_ms": float(np.percentile(elapsed, 95)) if elapsed else None,
                       "timed_calls": len(observations),
                       "rejections": sum(not x["accepted"] for x in observations),
                       "accepted_errors": sum(x["accepted"] and not x["correct"] for x in observations)}
                rows.append(row)
                if not valid:
                    failed.append(row)
    write_json(out / "session_case_metrics.json", rows)

    def matrix_for(dataset, numerator, denominator, case_filter=None):
        selected = [x for x in cases.values() if x["dataset"] == dataset and
                    (case_filter(x) if case_filter else x["k"] == 10 and x["w"] == 256
                     and x["n"] == {"sift": 1000000, "scifact": 5183, "fiqa": 57638}[dataset])]
        values = []
        for session in range(1, protocol["performance_sessions"] + 1):
            paired = [(medians[session, x["id"], numerator], medians[session, x["id"], denominator])
                      for x in selected]
            if not paired or any(a is None or b is None for a, b in paired):
                return None
            values.append([np.log(a / b) for a, b in paired])
        return np.asarray(values)

    stats = []
    primary_matrix = matrix_for("sift", "int32", "fp32")
    if primary_matrix is not None:
        primary = {"role": "primary", "dataset": "sift", "numerator": "int32", "denominator": "fp32",
                   **comparison(primary_matrix, protocol, protocol["analysis_seed"])}
        primary["statistical_and_practical_advantage"] = (
            primary["exact_signflip_p"] < 0.05 and primary["session_ci95"][0] > 1.10)
        stats.append(primary)
    else:
        primary = {"role": "primary", "status": "not estimable without excluding failures"}
        stats.append(primary)
    secondary = []
    for i, spec in enumerate(protocol["secondary_holm_family"]):
        matrix = matrix_for(**spec)
        row = {"role": "secondary", **spec}
        if matrix is None:
            row.update({"status": "not estimable without excluding failures", "exact_signflip_p": 1.0})
        else:
            row.update(comparison(matrix, protocol, protocol["analysis_seed"] + 10 + i))
        secondary.append(row)
    adjusted = holm([x["exact_signflip_p"] for x in secondary])
    for row, p_adjusted in zip(secondary, adjusted, strict=True):
        row["holm_adjusted_p"] = p_adjusted
    stats.extend(secondary)
    write_json(out / "table5_statistical_comparison.json", stats)

    performance_table = []
    for dataset in ("sift", "scifact", "fiqa"):
        for method in protocol["methods"]:
            subset = [x for x in rows if x["dataset"] == dataset and x["method"] == method
                      and x["role"] in ("primary", "dataset")]
            valid = [x["median_success_ms"] for x in subset if x["valid"]]
            n = sum(x["timed_calls"] for x in subset)
            absolute = None
            if len(valid) == len(subset):
                session_log = [float(np.mean([np.log(x["median_success_ms"]) for x in subset if x["session"] == s]))
                               for s in range(1, protocol["performance_sessions"] + 1)]
                absolute = {"geometric_mean_session_query_median_ms": float(np.exp(np.mean(session_log))),
                            "session_cluster_ci95_ms": bootstrap_log_ci(session_log, protocol["analysis_seed"] + 90)}
            pooled = [v for x in cases.values() if x["dataset"] == dataset and x["role"] in ("primary", "dataset")
                      for v in raw_times[x["id"], method]]
            performance_table.append({"dataset": dataset, "method": method,
                                      "session_query_rows": len(subset), "valid_rows": len(valid),
                                      "median_session_query_latency_ms": float(np.median(valid)) if len(valid) == len(subset) else None,
                                      "sequential_queries_per_second_from_median": float(1000 / np.median(valid))
                                      if len(valid) == len(subset) else None,
                                      "calls": n, "rejections": sum(x["rejections"] for x in subset),
                                      "accepted_errors": sum(x["accepted_errors"] for x in subset),
                                      "absolute_session_uncertainty": absolute,
                                      "pooled_descriptive_p95_attempt_ms": float(np.percentile(pooled, 95)) if pooled else None})
    write_json(out / "table4_end_to_end_performance.json", performance_table)
    sensitivity = []
    configs = sorted({(x["dataset"], x["n"], x["k"], x["w"]) for x in cases.values()
                      if x["role"] == "sensitivity"})
    for dataset, n, k, w in configs:
        matrix = matrix_for(dataset, "int32", "fp32", lambda x, n=n, k=k, w=w:
                            (x["n"], x["k"], x["w"]) == (n, k, w))
        sensitivity.append({"dataset": dataset, "n": n, "k": k, "w": w,
                            "interpretation": "prespecified sensitivity, descriptive; no extra significance claim",
                            "int32_over_fp32_ratio": float(np.exp(matrix.mean())) if matrix is not None else None,
                            "session_ratios": np.exp(matrix.mean(axis=1)).tolist() if matrix is not None else None})
    write_json(out / "table6_sensitivity.json", sensitivity)

    numerical_rows = []
    for session in range(1, protocol["numerical_sessions"] + 1):
        r = json.loads((root / "numerical" / f"session{session:02d}" / "results.json").read_text())
        assert r["status"] == "complete" and r["environment_valid"] and r["condition_valid_end"]
        for case in r["cases"]:
            for name, c in case["configurations"].items():
                npu = [x for x in c["runs"] if x["backend"] == "NPU"]
                cpu = [x for x in c["runs"] if x["backend"] == "CPU"]
                from scripts.probe_topk_numerical_path import check_output
                from cats_npu.order_encoding import score_codes
                a = np.load(ROOT / case["path"], allow_pickle=False)
                source = score_codes(a) if name == "int32" else a
                for obs in c["runs"]:
                    with np.load(ROOT / obs["raw"], allow_pickle=False) as raw:
                        recomputed = check_output(source.reshape(1, -1), raw["values"], raw["ids"], case["k"], True)[0]
                    assert recomputed == obs["checks"], "Raw numerical check mismatch"
                numerical_rows.append({"session": session, "case_id": case["id"], "family": case["family"],
                                       "n": case["n"], "k": case["k"], "w": case["w"], "configuration": name,
                                       "construction_or_runtime_error": c["error"],
                                       "cpu_reference_passed": bool(cpu and all(x["checks"]["passed"] for x in cpu)),
                                       "npu_calls": len(npu),
                                       **{f"{field}_failures": sum(not x["checks"][field] for x in npu)
                                          for field in ("valid_ids", "membership", "value_match", "source_value_sorted", "stable_ids_match", "passed")}})
    write_json(out / "numerical_case_metrics.json", numerical_rows)
    numerical_table = []
    for name in ("fp16", "promotion", "topk", "reshape", "combined", "int32"):
        subset = [x for x in numerical_rows if x["configuration"] == name]
        numerical_table.append({"configuration": name, "unique_cases": len(protocol["numerical_cases"]),
                                "sessions": protocol["numerical_sessions"],
                                "npu_calls": sum(x["npu_calls"] for x in subset),
                                "membership_failures": sum(x["membership_failures"] for x in subset),
                                "value_failures": sum(x["value_match_failures"] for x in subset),
                                "stable_sequence_failures": sum(x["stable_ids_match_failures"] for x in subset),
                                "complete_contract_failures": sum(x["passed_failures"] for x in subset),
                                "errors": sum(x["construction_or_runtime_error"] is not None for x in subset),
                                "cpu_reference_failures": sum(not x["cpu_reference_passed"] for x in subset)})
    write_json(out / "table2_numerical_correctness.json", numerical_table)
    composition = []
    for case in protocol["numerical_cases"]:
        if case["family"] != "cross_tile_boundary":
            continue
        composition.append({**case, "outcomes": {
            name: {"calls": sum(x["npu_calls"] for x in numerical_rows if x["case_id"] == case["id"] and x["configuration"] == name),
                   "membership_failures": sum(x["membership_failures"] for x in numerical_rows if x["case_id"] == case["id"] and x["configuration"] == name),
                   "value_failures": sum(x["value_match_failures"] for x in numerical_rows if x["case_id"] == case["id"] and x["configuration"] == name)}
            for name in ("fp16", "promotion", "topk", "reshape", "combined", "int32")}})
    write_json(out / "table3_composition_ablation.json", composition)
    write_json(out / "table1_environment.json", {"environments": environments, "input_manifest": inputs,
                                                "frozen_protocol": protocol, "freeze": lock})
    summary = {"primary": primary, "numerical": numerical_table,
               "performance": performance_table, "sensitivity": sensitivity,
               "performance_timed_calls": len(rows) * protocol["timed_repetitions"],
               "performance_case_rows": len(rows), "invalid_performance_case_rows": failed,
               "unique_queries": sum(len(x["selected_ids"]) for x in protocol["query_selection"].values()),
               "performance_sessions": protocol["performance_sessions"],
               "raw_result_sha256": {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.glob("*/session*/results.json"))}}
    write_json(out / "summary.json", summary)
    print(json.dumps({"primary": primary, "numerical": numerical_table,
                      "invalid_performance_rows": len(failed), "timed_calls": summary["performance_timed_calls"]}, indent=2))


def main():
    p = ArgumentParser()
    p.add_argument("--root", type=Path, default=BASE)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    analyze(args.root, args.out)


if __name__ == "__main__":
    main()
