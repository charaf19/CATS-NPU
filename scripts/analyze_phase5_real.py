"""Aggregate strict SIFT1M observations without treating repeats as sessions."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median


COLORS = {
    "cpp_exact": "#2563eb",
    "torch_cuda_stable": "#16a34a",
    "ov_npu_native_direct": "#0891b2",
    "ov_npu_native_batched_hier": "#dc2626",
    "ov_npu_native_batched_adaptive": "#9333ea",
    "ov_npu_native_batched_costadaptive": "#ea580c",
}


def write_scaling_svg(rows: list[dict], path: Path, k: int) -> None:
    import math

    methods = [method for method in COLORS
               if any(row["method"] == method and row["k"] == k for row in rows)]
    ns = sorted({row["n"] for row in rows if row["k"] == k})
    aggregate = {(n, method): median(row["median_ms"] for row in rows
                                     if row["n"] == n and row["k"] == k
                                     and row["method"] == method)
                 for n in ns for method in methods}
    width, height = 920, 560
    left, right, top, bottom = 90, 250, 55, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    lx = [math.log10(n) for n in ns]
    ly = [math.log10(value) for value in aggregate.values()]
    xmin, xmax = min(lx), max(lx)
    ymin, ymax = math.floor(min(ly)), math.ceil(max(ly))
    def x(n: int) -> float:
        return left + (math.log10(n) - xmin) / (xmax - xmin) * plot_w

    def y(value: float) -> float:
        return top + (ymax - math.log10(value)) / (ymax - ymin) * plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="28" font-family="sans-serif" font-size="20">SIFT1M exact Top-K latency scaling (K={k})</text>']
    for power in range(ymin, ymax + 1):
        yy = y(10 ** power)
        parts += [f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#ddd"/>',
                  f'<text x="{left - 10}" y="{yy + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{10 ** power:g}</text>']
    for n in ns:
        xx = x(n)
        parts += [f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + plot_h}" stroke="#eee"/>',
                  f'<text x="{xx:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{n:,}</text>']
    parts += [f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">score-vector length N (log scale)</text>',
              f'<text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="13">median latency (ms, log scale)</text>']
    labels = {
        "cpp_exact": "C++ exact", "torch_cuda_stable": "CUDA stable",
        "ov_npu_native_direct": "NPU native direct",
        "ov_npu_native_batched_hier": "NPU fixed",
        "ov_npu_native_batched_adaptive": "NPU geometric",
        "ov_npu_native_batched_costadaptive": "NPU submission-aware",
    }
    for row_no, method in enumerate(methods):
        points = " ".join(f"{x(n):.1f},{y(aggregate[n, method]):.1f}" for n in ns)
        color = COLORS[method]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for n in ns:
            parts.append(f'<circle cx="{x(n):.1f}" cy="{y(aggregate[n, method]):.1f}" r="4" fill="{color}"/>')
        yy = top + 25 + row_no * 28
        parts += [f'<line x1="{left + plot_w + 25}" y1="{yy}" x2="{left + plot_w + 55}" y2="{yy}" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{left + plot_w + 64}" y="{yy + 5}" font-family="sans-serif" font-size="13">{labels[method]}</text>']
    parts += [f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>',
              '<text x="90" y="545" font-family="sans-serif" font-size="11" fill="#555">Four evaluation queries; per-query medians aggregated by median. Descriptive, one experimental launch.</text>',
              '</svg>']
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/phase5/sift1m_summary"))
    args = parser.parse_args()
    records = []
    for root in args.input:
        for path in root.rglob("observations_*.jsonl"):
            session = path.stem.removeprefix("observations_")
            summary_path = path.with_name(f"summary_{session}.json")
            summary = json.loads(summary_path.read_text())
            if not summary.get("valid_for_inference"):
                raise ValueError(f"Invalid session cannot enter analysis: {summary_path}")
            records.extend(json.loads(line) for line in path.read_text().splitlines()
                           if line.strip())
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
    direct_present = any(row["method"] == "ov_npu_native_direct" for row in rows)
    for case, methods in sorted(method_by_case.items()):
        required = ("cpp_exact", "torch_cuda_stable", "ov_npu_native_batched_hier",
                    "ov_npu_native_batched_adaptive", "ov_npu_native_batched_costadaptive")
        if direct_present:
            required += ("ov_npu_native_direct",)
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
            "geometric_over_fixed_speedup": fixed / geometric,
            "fixed_vs_cpp_slowdown": fixed / cpp,
            "cost_vs_cpp_slowdown": cost / cpp,
            "fixed_vs_cuda_slowdown": fixed / cuda,
            "cost_vs_cuda_slowdown": cost / cuda,
        })
        if direct_present:
            direct = methods["ov_npu_native_direct"]["median_ms"]
            comparisons[-1].update({
                "direct_vs_cpp_slowdown": direct / cpp,
                "direct_vs_cuda_slowdown": direct / cuda,
                "geometric_over_direct_speedup": direct / geometric,
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
                  "geometric_over_fixed_speedup",
                  "fixed_vs_cpp_slowdown", "cost_vs_cpp_slowdown",
                  "fixed_vs_cuda_slowdown", "cost_vs_cuda_slowdown"):
        values = [item[field] for item in comparisons]
        summary[field + "_min_median_max"] = [min(values), median(values), max(values)]
    if direct_present:
        for field in ("direct_vs_cpp_slowdown", "direct_vs_cuda_slowdown",
                      "geometric_over_direct_speedup"):
            values = [item[field] for item in comparisons]
            summary[field + "_min_median_max"] = [min(values), median(values), max(values)]
    summary["submission_aware_faster_than_geometric_cases"] = sum(
        item["cost_over_geometric_speedup"] > 1 for item in comparisons
    )
    summary["geometric_faster_than_fixed_cases"] = sum(
        item["geometric_over_fixed_speedup"] > 1 for item in comparisons
    )
    args.out.with_name(args.out.name + "_key_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_scaling_svg(rows, args.out.with_name("sift1m_scaling_k4.svg"), 4)
    write_scaling_svg(rows, args.out.with_name("sift1m_scaling_k10.svg"), 10)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
