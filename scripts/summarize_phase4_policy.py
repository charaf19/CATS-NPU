"""Summarize fixed, geometric, and submission-aware Phase 4 methods."""
from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from statistics import median

from cats_npu.statistics import read_observations


METHODS = {
    "cpp": "cpp_exact",
    "fixed": "ov_npu_native_batched_hier",
    "geometric": "ov_npu_native_batched_adaptive",
    "submission_aware": "ov_npu_native_batched_costadaptive",
}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("input", type=Path, default=Path("results/phase4/policy_pilot"),
                        nargs="?")
    parser.add_argument("--out", type=Path,
                        default=Path("results/phase4/policy_comparison.csv"))
    args = parser.parse_args()
    records = [row for row in read_observations([args.input])
               if row.get("correct") and row.get("_session_valid", True)]
    rows = []
    for case_id in sorted({row["case_id"] for row in records}):
        case = [row for row in records if row["case_id"] == case_id]
        grouped = {label: [row for row in case if row["method"] == method]
                   for label, method in METHODS.items()}
        if any(not value for value in grouped.values()):
            continue
        first = case[0]
        item = {
            "case_id": case_id,
            "n": first["n"],
            "k": first["k"],
            "distribution": first["distribution"],
        }
        for label, values in grouped.items():
            item[f"{label}_median_ms"] = median(float(row["elapsed_ms"]) for row in values)
            item[f"{label}_median_submissions"] = median(
                float(row.get("device_submissions", 0)) for row in values
            )
            item[f"{label}_median_candidates"] = median(
                float(row.get("candidate_count", 0)) for row in values
            )
            item[f"{label}_median_rounds"] = median(
                float(row.get("rounds", 0)) for row in values
            )
        cost_values = grouped["submission_aware"]
        item["submission_aware_median_jumps"] = median(
            float(row.get("cost_aware_jumps", 0)) for row in cost_values
        )
        item["cost_over_geometric_speedup"] = (
            item["geometric_median_ms"] / item["submission_aware_median_ms"]
        )
        item["cost_over_fixed_speedup"] = (
            item["fixed_median_ms"] / item["submission_aware_median_ms"]
        )
        rows.append(item)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    ratios = [row["cost_over_geometric_speedup"] for row in rows]
    summary = {
        "case_count": len(rows),
        "submission_aware_faster_than_geometric": sum(value > 1 for value in ratios),
        "submission_aware_slower_than_geometric": sum(value < 1 for value in ratios),
        "cost_over_geometric_speedup_min_median_max": [
            min(ratios), median(ratios), max(ratios)
        ],
        "median_jumps": median(row["submission_aware_median_jumps"] for row in rows),
        "interpretation": "development pilot only; one process is not an independent-session study",
    }
    args.out.with_name("policy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
