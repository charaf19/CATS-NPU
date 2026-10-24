"""Session-level paired inference; in-process blocks are NOT independent sessions.

Statistics:
  - per-case/method median and P95 latency, excluding incorrect runs;
  - paired session median log speedups against one baseline;
  - session-cluster bootstrap 95% CI;
  - two-sided sign-flip permutation p value + Holm adjustment by case.

This is exploratory until >= 6 independent session IDs exist, and results
still require controls for ambient thermal/load confounders.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from pathlib import Path
import json
import math
import numpy as np


def read_observations(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for item in paths:
        if item.is_dir():
            files = sorted(item.glob("observations_*.jsonl"))
        elif "*" in str(item):
            files = sorted(item.parent.glob(item.name))
        else:
            files = [item]
        for p in files:
            session_id = p.stem.removeprefix("observations_")
            summary_path = p.with_name(f"summary_{session_id}.json")
            session_valid = True
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                session_valid = bool(summary.get("completed", True)) and not (
                    int(summary.get("skipped", 0)) or int(summary.get("wrong_results", 0))
                )
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        row["_session_valid"] = session_valid
                        records.append(row)
    if not records:
        raise ValueError("No observation records found")
    return records


def _sign_flip_p(d: np.ndarray, rng: np.random.Generator) -> float:
    n = d.size
    observed = abs(float(d.mean()))
    if n <= 17:
        count = 0
        total = 0
        for signs in product((-1.0, 1.0), repeat=n):
            x = abs(float(np.mean(d * np.asarray(signs))))
            count += x >= observed - 1e-12
            total += 1
        return count / total
    draws = 50000
    flips = rng.choice(np.array([-1.0, 1.0]), size=(draws, n))
    shuffled = np.abs((flips * d[None, :]).mean(axis=1))
    return (1 + int(np.count_nonzero(shuffled >= observed - 1e-12))) / (draws + 1)


def _bootstrap_ci(d: np.ndarray, rng: np.random.Generator, rounds: int = 10000) -> list[float]:
    idx = rng.integers(0, d.size, size=(rounds, d.size))
    v = np.exp(d[idx].mean(axis=1))
    return [float(x) for x in np.quantile(v, [0.025, 0.975])]


def analyze(records: list[dict], baseline: str = "cpu_exact", min_sessions: int = 6) -> dict:
    obs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    case_digests: dict[str, set[str]] = defaultdict(set)
    incorrect: dict[tuple[str, str], int] = defaultdict(int)
    for row in records:
        if row.get("input_sha256"):
            case_digests[row["case_id"]].add(row["input_sha256"])
        key = (row["case_id"], row["method"])
        if not row.get("correct", False):
            incorrect[key] += 1
        else:
            obs[key].append(row)
    invalid_cases = [case for case, digests in case_digests.items() if len(digests) > 1]
    if invalid_cases:
        raise ValueError(f"Score bytes differ for the same case_id: {invalid_cases}")
    report: dict = {
        "baseline": baseline,
        "unit_of_inference": "independent session_id (not iterations or blocks)",
        "minimum_sessions_for_tests": min_sessions,
        "cases": {},
        "warnings": ["Timing distributions are not proof of hardware superiority without comparable devices/configurations.",
                     "Incorrect Top-K runs are excluded and counted explicitly."],
    }
    rng = np.random.default_rng(42026)
    for case in sorted(set(k[0] for k in obs) | set(k[0] for k in incorrect)):
        methods = sorted({key[1] for key in obs if key[0] == case} |
                         {key[1] for key in incorrect if key[0] == case})
        results = {}
        session_medians: dict[str, dict[str, float]] = defaultdict(dict)
        for method in methods:
            item = obs.get((case, method), [])
            valid_session_item = [row for row in item if row.get("_session_valid", True)]
            by_session: dict[str, list[float]] = defaultdict(list)
            for row in valid_session_item:
                by_session[row["session_id"]].append(float(row["elapsed_ms"]))
            # Never infer speedups from a partially incorrect implementation.
            if incorrect.get((case, method), 0) == 0:
                for sess, values in by_session.items():
                    session_medians[method][sess] = float(np.median(values))
            if item:
                vals = np.array([float(row["elapsed_ms"]) for row in item])
                median_ms = float(np.median(vals))
                results[method] = {
                    "n_observations": len(item),
                    "n_independent_sessions": len(by_session),
                    "median_ms": median_ms,
                    "p95_ms": float(np.quantile(vals, 0.95)),
                    "median_throughput_per_second": 1000.0 / median_ms,
                    "incorrect_runs": incorrect.get((case, method), 0),
                    "valid_session_observations": len(valid_session_item),
                    "excluded_failed_session_observations": len(item) - len(valid_session_item),
                    "valid_for_inference": (
                        incorrect.get((case, method), 0) == 0 and bool(by_session)
                    ),
                }
                for source, target in (
                    ("backend_ns", "median_backend_ms"),
                    ("host_input_preparation_ns", "median_host_input_preparation_ms"),
                    ("backend_execute_sync_ns", "median_backend_execute_sync_ms"),
                    ("backend_output_copy_ns", "median_backend_output_copy_ms"),
                    ("backend_validation_ns", "median_backend_validation_ms"),
                    ("cpu_exact_validation_rows", "median_cpu_exact_validation_rows"),
                    ("candidate_extraction_ns", "median_candidate_extraction_ms"),
                    ("merge_ns", "median_merge_ms"),
                    ("certification_ns", "median_certification_ms"),
                    ("device_submissions", "median_device_submissions"),
                    ("staged_input_bytes", "median_staged_input_bytes"),
                    ("retrieved_output_bytes", "median_retrieved_output_bytes"),
                ):
                    available = [float(row[source]) for row in item if source in row]
                    if available:
                        value = float(np.median(available))
                        results[method][target] = value / 1e6 if source.endswith("_ns") else value
            else:
                results[method] = {"n_observations": 0,
                                   "incorrect_runs": incorrect.get((case, method), 0),
                                   "status": "no valid observations"}
        comparisons = {}
        raw_pvalues = {}
        if baseline in session_medians:
            for method in methods:
                if method == baseline or method not in session_medians:
                    continue
                shared = sorted(set(session_medians[baseline]) & set(session_medians[method]))
                if not shared:
                    continue
                d = np.array([
                    math.log(session_medians[baseline][s] / session_medians[method][s])
                    for s in shared
                ], dtype=np.float64)
                item = {"paired_sessions": len(shared),
                        "paired_session_ids": shared,
                        "geometric_speedup": float(np.exp(d.mean())),
                        "significance_status": ("ready" if len(shared) >= min_sessions
                                                else "exploratory_insufficient_sessions")}
                if len(shared) >= 2:
                    item["bootstrap_95ci_speedup"] = _bootstrap_ci(d, rng)
                if len(shared) >= min_sessions:
                    item["two_sided_signflip_p"] = _sign_flip_p(d, rng)
                    raw_pvalues[method] = item["two_sided_signflip_p"]
                comparisons[method] = item
        # Holm-Bonferroni across tested method comparisons within each case.
        order = sorted(raw_pvalues, key=lambda m: raw_pvalues[m])
        lower = 0.0
        for idx, method in enumerate(order):
            corrected = min(1.0, (len(order) - idx) * raw_pvalues[method])
            lower = max(lower, corrected)
            comparisons[method]["holm_adjusted_p"] = lower
        report["cases"][case] = {"methods": results, "comparisons_vs_baseline": comparisons}
    return report


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
