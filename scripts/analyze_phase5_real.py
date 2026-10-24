"""Aggregate strict SIFT1M observations without treating repeats as sessions."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/phase5/sift1m_summary"))
    args = parser.parse_args()
    records = []
    for path in args.input.rglob("observations_*.jsonl"):
        session = path.stem.removeprefix("observations_")
        summary_path = path.with_name(f"summary_{session}.json")
        summary = json.loads(summary_path.read_text())
        if not summary.get("valid_for_inference"):
            raise ValueError(f"Invalid session cannot enter analysis: {summary_path}")
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not records:
        raise ValueError("No observations found")
    if any(not row["correct"] for row in records):
        raise ValueError("Incorrect result present")
    grouped: dict[tuple[int, int, str, str], list[float]] = defaultdict(list)
    profiles: dict[tuple[int, int, str, str], list[dict]] = defaultdict(list)
    for row in records:
        key = (int(row["n"]), int(row["k"]), row["distribution"], row["method"])
        grouped[key].append(float(row["elapsed_ms"]))
        profiles[key].append(row)
    rows = []
    for key, values in sorted(grouped.items()):
        n, k, query, method = key
        source = profiles[key]
        rows.append({
            "n": n, "k": k, "query": query, "method": method,
            "observations": len(values),
            "median_ms": median(values),
            "p95_ms": sorted(values)[int(0.95 * (len(values) - 1))],
            "median_submissions": median(float(x.get("device_submissions", 0)) for x in source),
            "median_rounds": median(float(x.get("rounds", 0)) for x in source),
            "median_candidates": median(float(x.get("candidate_count", 0)) for x in source),
            "median_sync_ms": median(float(x.get("backend_execute_sync_ns", 0)) / 1e6
                                      for x in source),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    method_by_case = defaultdict(dict)
    for row in rows:
        method_by_case[(row["n"], row["k"], row["query"])][row["method"]] = row
    comparisons = []
    for case, methods in sorted(method_by_case.items()):
        required = ("cpp_exact", "torch_cuda_stable", "ov_npu_native_batched_hier",
                    "ov_npu_native_batched_adaptive", "ov_npu_native_batched_costadaptive")
        if not all(method in methods for method in required):
            raise ValueError(f"Missing required method for {case}")
        cpp = methods["cpp_exact"]["median_ms"]
        cuda = methods["torch_cuda_stable"]["median_ms"]
        fixed = methods["ov_npu_native_batched_hier"]["median_ms"]
        geometric = methods["ov_npu_native_batched_adaptive"]["median_ms"]
        cost = methods["ov_npu_native_batched_costadaptive"]["median_ms"]
        comparisons.append({
            "n": case[0], "k": case[1], "query": case[2],
            "cost_over_geometric_speedup": geometric / cost,
            "cost_over_fixed_speedup": fixed / cost,
            "fixed_vs_cpp_slowdown": fixed / cpp,
            "cost_vs_cpp_slowdown": cost / cpp,
            "fixed_vs_cuda_slowdown": fixed / cuda,
            "cost_vs_cuda_slowdown": cost / cuda,
        })
    args.out.with_name(args.out.name + "_comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    summary = {
        "observation_count": len(records),
        "case_count": len(method_by_case),
        "all_correct": True,
        "query_count": len({row["query"] for row in rows}),
        "session_interpretation": "descriptive; query subprocesses are not time-separated sessions",
    }
    for field in ("cost_over_geometric_speedup", "cost_over_fixed_speedup",
                  "fixed_vs_cpp_slowdown", "cost_vs_cpp_slowdown",
                  "fixed_vs_cuda_slowdown", "cost_vs_cuda_slowdown"):
        values = [item[field] for item in comparisons]
        summary[field + "_min_median_max"] = [min(values), median(values), max(values)]
    args.out.with_name(args.out.name + "_key_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
