"""Session-clustered paired analysis for the locked Phase 7 protocol."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import csv
from itertools import product
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


METHOD_LABELS = {
    "cpp_exact": "C++",
    "torch_cuda_stable": "CUDA",
    "ov_npu_native_direct": "NPU direct",
    "ov_npu_native_batched_hier": "NPU fixed",
    "ov_npu_native_batched_adaptive": "NPU geometric",
    "ov_npu_native_batched_costadaptive": "NPU submission-aware",
}


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    outcomes = [abs(float(np.mean(values * np.asarray(signs))))
                for signs in product((-1.0, 1.0), repeat=values.size)]
    return sum(value >= observed - 1e-15 for value in outcomes) / len(outcomes)


def bootstrap_ci(values: np.ndarray, seed: int = 42027) -> list[float]:
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, values.size, size=(50000, values.size))
    speedups = np.exp(values[positions].mean(axis=1))
    return [float(value) for value in np.quantile(speedups, [0.025, 0.975])]


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def write_primary_figure(path: Path, primary: dict) -> None:
    """Write a dependency-free SVG of session effects and the clustered CI."""
    width, height = 920, 500
    left, right, top, bottom = 110, 55, 55, 80
    session_values = primary["session_speedups"]
    ci_low, ci_high = primary["ci95_session_cluster_bootstrap"]
    candidates = session_values + [ci_low, ci_high, 1.0, 1.1]
    low = max(0.5, min(candidates) * 0.92)
    high = max(candidates) * 1.08

    def x(value: float) -> float:
        return left + (value - low) / (high - low) * (width - left - right)

    rows = []
    rows.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    rows.append('<style>text{font-family:Arial,sans-serif;fill:#172033}'
                '.axis{stroke:#59657a;stroke-width:1}.guide{stroke:#aab2c0;stroke-dasharray:5 5}'
                '.point{fill:#2563eb}.estimate{stroke:#7c3aed;stroke-width:5}</style>')
    rows.append('<text x="30" y="32" font-size="20" font-weight="700">'
                'Confirmatory SIFT1M direct / geometric NPU speedup</text>')
    axis_y = height - bottom
    rows.append(f'<line class="axis" x1="{left}" y1="{axis_y}" '
                f'x2="{width-right}" y2="{axis_y}"/>')
    for value, label in [(1.0, "no difference"), (1.1, "practical threshold")]:
        rows.append(f'<line class="guide" x1="{x(value):.1f}" y1="{top}" '
                    f'x2="{x(value):.1f}" y2="{axis_y}"/>')
        rows.append(f'<text x="{x(value):.1f}" y="{axis_y+45}" text-anchor="middle" '
                    f'font-size="13">{label} ({value:.1f}x)</text>')
    for index, value in enumerate(session_values, start=1):
        y = top + index * 42
        rows.append(f'<text x="{left-14}" y="{y+5}" text-anchor="end" '
                    f'font-size="14">Session {index}</text>')
        rows.append(f'<circle class="point" cx="{x(value):.1f}" cy="{y}" r="7"/>')
        rows.append(f'<text x="{x(value)+12:.1f}" y="{y+5}" font-size="13">{value:.3f}x</text>')
    estimate_y = top + 7 * 42
    rows.append(f'<text x="{left-14}" y="{estimate_y+5}" text-anchor="end" '
                'font-size="14" font-weight="700">Aggregate</text>')
    rows.append(f'<line class="estimate" x1="{x(ci_low):.1f}" y1="{estimate_y}" '
                f'x2="{x(ci_high):.1f}" y2="{estimate_y}"/>')
    estimate = primary["geomean_speedup"]
    rows.append(f'<circle cx="{x(estimate):.1f}" cy="{estimate_y}" r="9" fill="#7c3aed"/>')
    rows.append(f'<text x="{x(estimate)+13:.1f}" y="{estimate_y+5}" font-size="13">'
                f'{estimate:.3f}x [{ci_low:.3f}, {ci_high:.3f}]</text>')
    rows.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-15}" '
                'text-anchor="middle" font-size="14">Speedup (direct latency / geometric latency)</text>')
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
                    + ''.join(rows) + '</svg>\n', encoding="utf-8")


def write_latency_figure(path: Path, method_summaries: list[dict]) -> None:
    """Write an SVG with median and P95 latency for every locked method."""
    width, height = 1000, 560
    left, right, top, bottom = 245, 65, 55, 70
    values = [row[key] for row in method_summaries for key in ("median_ms", "p95_ms")]
    low = max(0.01, min(values) * 0.75)
    high = max(values) * 1.25
    log_low, log_high = math.log10(low), math.log10(high)

    def x(value: float) -> float:
        return left + (math.log10(value) - log_low) / (log_high - log_low) * (width-left-right)

    rows = [f'<rect width="{width}" height="{height}" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#172033}'
            '.axis{stroke:#59657a;stroke-width:1}.range{stroke:#94a3b8;stroke-width:5}'
            '.median{fill:#2563eb}.p95{fill:#f97316}</style>',
            '<text x="30" y="32" font-size="20" font-weight="700">'
            'Confirmatory SIFT1M latency across raw observations</text>']
    axis_y = height - bottom
    rows.append(f'<line class="axis" x1="{left}" y1="{axis_y}" x2="{width-right}" y2="{axis_y}"/>')
    power = math.floor(log_low)
    while power <= math.ceil(log_high):
        tick = 10 ** power
        if low <= tick <= high:
            rows.append(f'<line class="axis" x1="{x(tick):.1f}" y1="{axis_y}" '
                        f'x2="{x(tick):.1f}" y2="{axis_y+7}"/>')
            rows.append(f'<text x="{x(tick):.1f}" y="{axis_y+28}" text-anchor="middle" '
                        f'font-size="13">{tick:g}</text>')
        power += 1
    for index, row in enumerate(method_summaries):
        y = top + 55 + index * 58
        label = METHOD_LABELS[row["method"]]
        rows.append(f'<text x="{left-14}" y="{y+5}" text-anchor="end" font-size="14">{label}</text>')
        rows.append(f'<line class="range" x1="{x(row["median_ms"]):.1f}" y1="{y}" '
                    f'x2="{x(row["p95_ms"]):.1f}" y2="{y}"/>')
        rows.append(f'<circle class="median" cx="{x(row["median_ms"]):.1f}" cy="{y}" r="7"/>')
        rows.append(f'<rect class="p95" x="{x(row["p95_ms"])-6:.1f}" y="{y-6}" width="12" height="12"/>')
        rows.append(f'<text x="{x(row["p95_ms"])+12:.1f}" y="{y+5}" font-size="12">'
                    f'{row["median_ms"]:.3f} / {row["p95_ms"]:.3f} ms</text>')
    rows.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-15}" text-anchor="middle" '
                'font-size="14">Latency in ms (log scale); circle = median, square = P95</text>')
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
                    + ''.join(rows) + '</svg>\n', encoding="utf-8")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/phase7_confirmatory.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/phase7/confirmatory_analysis.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = []
    for path in args.input.rglob("observations_*.jsonl"):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                       if line.strip())
    if not records or any(not row["correct"] for row in records):
        raise ValueError("Missing or incorrect confirmatory observations")
    sessions = sorted({row["session_id"] for row in records})
    if len(sessions) != config["independent_sessions"]:
        raise ValueError(f"Expected {config['independent_sessions']} sessions, got {len(sessions)}")
    digests = defaultdict(set)
    values = defaultdict(list)
    for row in records:
        digests[row["distribution"]].add(row["input_sha256"])
        values[(row["session_id"], row["distribution"], row["method"])].append(
            float(row["elapsed_ms"])
        )
    if any(len(group) != 1 for group in digests.values()):
        raise ValueError("Input digest changed across sessions")
    query_names = sorted(digests)
    methods = config["methods"]
    expected_queries = [f"sift1m_q{query_id:04d}" for query_id in config["query_ids"]]
    if query_names != sorted(expected_queries):
        raise ValueError(f"Expected locked query set {expected_queries}, got {query_names}")
    expected_replicates = config["repeats"] * config["blocks"]
    expected_observations = (config["independent_sessions"] * len(query_names) *
                             len(methods) * expected_replicates)
    if len(records) != expected_observations:
        raise ValueError(f"Expected {expected_observations} observations, got {len(records)}")
    for session in sessions:
        for query in query_names:
            for method in methods:
                count = len(values[session, query, method])
                if count != expected_replicates:
                    raise ValueError(f"Expected {expected_replicates} rows for "
                                     f"{session}/{query}/{method}, got {count}")
    for row in records:
        locked = (row["n"] == config["n"] and row["k"] == config["k"] and
                  row["dtype"] == config["dtype"] and row["tile"] == config["tile_width"] and
                  row["batch"] == config["tile_batch"])
        if not locked:
            raise ValueError("Observation does not match the locked configuration")
    medians = {(session, query, method): median(values[session, query, method])
               for session in sessions for query in query_names for method in methods}

    def comparison(baseline: str, method: str, *, inferential: bool) -> dict:
        by_session = []
        query_speedups = []
        for session in sessions:
            log_ratios = [math.log(medians[session, query, baseline] /
                                   medians[session, query, method]) for query in query_names]
            by_session.append(float(np.mean(log_ratios)))
            query_speedups.extend(math.exp(value) for value in log_ratios)
        array = np.asarray(by_session)
        result = {
            "baseline": baseline, "method": method,
            "unit": "session mean log speedup across eight paired queries",
            "session_count": len(sessions),
            "geomean_speedup": math.exp(float(array.mean())),
            "median_session_speedup": median(math.exp(value) for value in array),
            "mean_log_speedup": float(array.mean()),
            "ci95_session_cluster_bootstrap": bootstrap_ci(array),
            "query_pair_speedup_min_median_max": [min(query_speedups),
                                                    median(query_speedups), max(query_speedups)],
            "session_speedups": [math.exp(value) for value in array],
            "paired_query_win_fraction": sum(value > 1 for value in query_speedups) /
                                         len(query_speedups),
            "inferential_status": "primary" if inferential else "secondary_exploratory",
        }
        if inferential:
            result["two_sided_exact_sign_flip_p"] = exact_sign_flip(array)
            supported = (result["two_sided_exact_sign_flip_p"] < config["alpha"]
                         and result["ci95_session_cluster_bootstrap"][0] > 1)
            practical = result["geomean_speedup"] >= config["practical_speedup_threshold"]
            result["statistically_supported"] = supported
            result["practically_meaningful"] = supported and practical
            result["decision"] = (
                "supported_and_practically_meaningful" if supported and practical
                else "supported_but_below_practical_threshold" if supported
                else "insufficient_evidence"
            )
        return result

    primary = comparison(config["primary_baseline"], config["primary_method"], inferential=True)
    secondary = [
        comparison("ov_npu_native_batched_hier", "ov_npu_native_batched_adaptive",
                   inferential=False),
        comparison("ov_npu_native_batched_adaptive", "ov_npu_native_batched_costadaptive",
                   inferential=False),
        comparison("torch_cuda_stable", "ov_npu_native_batched_adaptive", inferential=False),
        comparison("cpp_exact", "ov_npu_native_batched_adaptive", inferential=False),
    ]
    session_rows = []
    for session in sessions:
        for query in query_names:
            item = {"session_id": session, "query": query}
            item.update({method: medians[session, query, method] for method in methods})
            session_rows.append(item)
    method_summaries = []
    for method in methods:
        raw = [float(row["elapsed_ms"]) for row in records if row["method"] == method]
        method_summaries.append({
            "method": method,
            "observations": len(raw),
            "median_ms": median(raw),
            "p95_ms": percentile(raw, 0.95),
            "median_million_scores_per_second": config["n"] / median(raw) / 1000,
            "p05_million_scores_per_second": config["n"] / percentile(raw, 0.95) / 1000,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(session_rows[0]))
        writer.writeheader()
        writer.writerows(session_rows)
    summary_csv = args.out.with_name(args.out.stem + "_method_summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_summaries[0]))
        writer.writeheader()
        writer.writerows(method_summaries)
    primary_figure = args.out.with_name(args.out.stem + "_primary_speedup.svg")
    latency_figure = args.out.with_name(args.out.stem + "_latency.svg")
    write_primary_figure(primary_figure, primary)
    write_latency_figure(latency_figure, method_summaries)
    report = {"protocol": config["name"], "observation_count": len(records),
              "sessions": sessions, "queries": query_names, "all_correct": True,
              "method_summaries": method_summaries,
              "primary": primary, "secondary": secondary,
              "artifacts": {"session_query_medians_csv": str(args.out.with_suffix(".csv")),
                            "method_summary_csv": str(summary_csv),
                            "primary_speedup_figure": str(primary_figure),
                            "latency_figure": str(latency_figure)},
              "limitations": [
                  "Six sessions give coarse exact-test resolution.",
                  "Sessions share one machine and may retain slow thermal/OS correlations.",
                  "Secondary comparisons are exploratory and have no multiplicity-adjusted tests.",
              ]}
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
