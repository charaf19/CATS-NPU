"""Summarize a Phase 3 sweep and generate dependency-free SVG figures."""
from __future__ import annotations

from argparse import ArgumentParser
import csv
from collections import defaultdict
from datetime import datetime, timezone
from math import exp, log
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


PROFILE_FIELDS = (
    "host_input_preparation_ns",
    "backend_input_cast_ns",
    "backend_execute_sync_ns",
    "backend_output_copy_ns",
    "backend_validation_ns",
    "candidate_extraction_ns",
    "merge_ns",
    "certification_ns",
)


def _latest_sweep(root: Path) -> Path:
    if (root / "sweep_manifest.json").exists():
        return root
    manifests = list(root.rglob("sweep_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No sweep_manifest.json under {root}")
    return max(manifests, key=lambda item: item.stat().st_mtime_ns).parent


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _algorithm(method: str) -> str:
    if method.endswith("_hier"):
        return "fixed_hierarchical"
    if method.endswith("_adaptive"):
        return "geometric_adaptive"
    return method


def _gmean(values: list[float]) -> float:
    return exp(sum(log(value) for value in values) / len(values))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _color(ratio: float | None) -> str:
    if ratio is None:
        return "#d1d5db"
    value = min(max((ratio - 1.0) / 2.0, 0.0), 1.0)
    red = int(80 + 160 * value)
    green = int(190 - 100 * value)
    return f"#{red:02x}{green:02x}78"


def _heatmap(path: Path, rows: list[dict[str, Any]], tiles: list[int], batches: list[int]) -> None:
    cell_w, cell_h, left, top = 112, 54, 110, 76
    panels = ("fixed_hierarchical", "geometric_adaptive")
    width = left + len(batches) * cell_w + 40
    panel_h = top + len(tiles) * cell_h + 60
    height = panel_h * len(panels)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#111827}'
        '.small{font-size:12px}.label{font-size:13px}.title{font-size:18px;font-weight:600}</style>',
    ]
    lookup = {(row["algorithm"], row["tile_width"], row["tile_batch"]): row for row in rows}
    for panel_no, algorithm in enumerate(panels):
        y0 = panel_no * panel_h
        title = algorithm.replace("_", " ").title()
        pieces.append(f'<text class="title" x="20" y="{y0 + 28}">{title}</text>')
        pieces.append(
            f'<text class="small" x="20" y="{y0 + 48}">Geometric mean slowdown vs '
            'the fastest valid configuration in each workload (lower is better)</text>'
        )
        for col, batch in enumerate(batches):
            x = left + col * cell_w
            pieces.append(f'<text class="label" text-anchor="middle" x="{x + cell_w / 2}" '
                          f'y="{y0 + top - 12}">tile batch {batch}</text>')
        for row_no, tile in enumerate(tiles):
            y = y0 + top + row_no * cell_h
            pieces.append(f'<text class="label" text-anchor="end" x="{left - 10}" '
                          f'y="{y + 31}">width {tile}</text>')
            for col, batch in enumerate(batches):
                x = left + col * cell_w
                item = lookup.get((algorithm, tile, batch))
                ratio = item.get("geomean_case_slowdown") if item else None
                label = "unsupported" if ratio is None else f'{ratio:.2f}x'
                pieces.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" '
                              f'height="{cell_h - 4}" rx="5" fill="{_color(ratio)}"/>')
                pieces.append(f'<text class="label" text-anchor="middle" x="{x + cell_w / 2}" '
                              f'y="{y + 31}">{label}</text>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def _breakdown(path: Path, config_rows: list[dict[str, Any]], recommendation: dict) -> None:
    fields = [
        ("median_backend_execute_sync_ms", "sync inference", "#2563eb"),
        ("median_candidate_extraction_ms", "candidate extraction", "#059669"),
        ("median_backend_validation_ms", "validation", "#d97706"),
        ("median_merge_ms", "merge", "#7c3aed"),
        ("median_host_input_preparation_ms", "input preparation", "#db2777"),
        ("median_other_measured_ms", "other measured", "#64748b"),
        ("median_unattributed_ms", "unattributed", "#cbd5e1"),
    ]
    width, height = 960, 300
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#111827}'
        '.small{font-size:12px}.label{font-size:14px}.title{font-size:18px;font-weight:600}</style>',
        '<text class="title" x="20" y="30">Recommended configuration latency breakdown</text>',
        '<text class="small" x="20" y="50">Host wall-clock components; sync inference includes '
        'submission, transfers, execution, synchronization, and waiting.</text>',
    ]
    recs = recommendation["by_algorithm"]
    selected = []
    for algorithm, rec in recs.items():
        selected.append(next(row for row in config_rows if row["algorithm"] == algorithm
                             and row["config_id"] == rec["config_id"]))
    maximum = max(row["median_case_latency_ms"] for row in selected)
    for row_no, row in enumerate(selected):
        y = 90 + row_no * 76
        pieces.append(f'<text class="label" x="20" y="{y + 19}">'
                      f'{row["algorithm"].replace("_", " ")} — {row["config_id"]}</text>')
        x = 300
        for field, _label, color in fields:
            value = max(float(row.get(field, 0.0)), 0.0)
            bar_w = 600 * value / maximum if maximum else 0
            pieces.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="28" '
                          f'fill="{color}"/>')
            if bar_w > 42:
                pieces.append(f'<text class="small" text-anchor="middle" x="{x + bar_w / 2:.1f}" '
                              f'y="{y + 19}" fill="white">{value:.2f}</text>')
            x += bar_w
        pieces.append(f'<text class="small" x="{x + 8:.1f}" y="{y + 19}">'
                      f'{row["median_case_latency_ms"]:.2f} ms median-of-cases</text>')
    legend_x, legend_y = 20, 254
    for index, (_, label, color) in enumerate(fields):
        x = legend_x + (index % 4) * 230
        y = legend_y + (index // 4) * 24
        pieces.append(f'<rect x="{x}" y="{y}" width="14" height="14" fill="{color}"/>')
        pieces.append(f'<text class="small" x="{x + 20}" y="{y + 12}">{label}</text>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path("results/phase3/ablation_baseline"))
    args = parser.parse_args()
    sweep = _latest_sweep(args.input)
    manifest = json.loads((sweep / "sweep_manifest.json").read_text(encoding="utf-8"))
    detailed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for config in manifest["configurations"]:
        config_dir = Path(config["output_directory"])
        observations: list[dict[str, Any]] = []
        for path in config_dir.glob("observations_*.jsonl"):
            observations.extend(_read_jsonl(path))
        for path in config_dir.glob("skips_*.jsonl"):
            for failure in _read_jsonl(path):
                failures.append({"config_id": config["config_id"], **failure})
        grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for row in observations:
            grouped[(row["n"], row["k"], row["distribution"], _algorithm(row["method"]))].append(row)
        for (n, k, distribution, algorithm), rows in sorted(grouped.items()):
            correct = [row for row in rows if row.get("correct")]
            elapsed = [float(row["elapsed_ms"]) for row in correct]
            item: dict[str, Any] = {
                "config_id": config["config_id"],
                "tile_width": config["tile_width"],
                "tile_batch": config["tile_batch"],
                "query_batch": 1,
                "algorithm": algorithm,
                "n": n,
                "k": k,
                "distribution": distribution,
                "observations": len(rows),
                "correct_observations": len(correct),
                "incorrect_observations": len(rows) - len(correct),
                "median_ms": float(median(elapsed)) if elapsed else None,
                "p95_ms": float(np.quantile(elapsed, 0.95)) if elapsed else None,
                "throughput_per_second": 1000.0 / median(elapsed) if elapsed else None,
            }
            for field in PROFILE_FIELDS:
                values = [float(row.get(field, 0)) / 1e6 for row in correct]
                item[f"median_{field.removesuffix('_ns')}_ms"] = (
                    float(median(values)) if values else None
                )
            for field in ("device_submissions", "backend_calls", "candidate_count",
                          "cpu_exact_validation_rows", "staged_input_bytes",
                          "retrieved_output_bytes", "rounds"):
                values = [float(row.get(field, 0)) for row in correct]
                item[f"median_{field}"] = float(median(values)) if values else None
            detailed.append(item)

    case_keys = sorted({(row["algorithm"], row["n"], row["k"], row["distribution"])
                        for row in detailed if row["median_ms"] is not None})
    best = {}
    for key in case_keys:
        values = [row["median_ms"] for row in detailed
                  if (row["algorithm"], row["n"], row["k"], row["distribution"]) == key
                  and row["median_ms"] is not None and not row["incorrect_observations"]]
        if values:
            best[key] = min(values)

    by_config: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        by_config[(row["config_id"], row["algorithm"])].append(row)
    expected_cases = len(manifest["ns"]) * len(manifest["ks"]) * len(manifest["distributions"])
    configurations = []
    for (config_id, algorithm), rows in sorted(by_config.items()):
        valid_rows = [row for row in rows if row["median_ms"] is not None
                      and not row["incorrect_observations"]]
        ratios = []
        for row in valid_rows:
            key = (algorithm, row["n"], row["k"], row["distribution"])
            ratios.append(row["median_ms"] / best[key])
        profile_names = [field.removesuffix("_ns") for field in PROFILE_FIELDS]
        item = {
            "config_id": config_id,
            "tile_width": rows[0]["tile_width"],
            "tile_batch": rows[0]["tile_batch"],
            "query_batch": 1,
            "algorithm": algorithm,
            "valid_cases": len(valid_rows),
            "expected_cases": expected_cases,
            "fully_valid": len(valid_rows) == expected_cases,
            "geomean_case_slowdown": _gmean(ratios) if len(ratios) == expected_cases else None,
            "median_case_latency_ms": float(median(row["median_ms"] for row in valid_rows))
            if valid_rows else None,
            "median_device_submissions": float(median(row["median_device_submissions"]
                                                       for row in valid_rows))
            if valid_rows else None,
        }
        for name in profile_names:
            values = [row[f"median_{name}_ms"] for row in valid_rows]
            item[f"median_{name}_ms"] = float(median(values)) if values else None
        measured_sum = sum(item[f"median_{name}_ms"] or 0.0 for name in profile_names)
        item["median_other_measured_ms"] = (
            (item["median_backend_input_cast_ms"] or 0.0)
            + (item["median_backend_output_copy_ms"] or 0.0)
            + (item["median_certification_ms"] or 0.0)
        )
        visible = (
            (item["median_backend_execute_sync_ms"] or 0.0)
            + (item["median_candidate_extraction_ms"] or 0.0)
            + (item["median_backend_validation_ms"] or 0.0)
            + (item["median_merge_ms"] or 0.0)
            + (item["median_host_input_preparation_ms"] or 0.0)
            + item["median_other_measured_ms"]
        )
        item["median_unattributed_ms"] = max(
            (item["median_case_latency_ms"] or 0.0) - visible, 0.0
        )
        item["median_all_profile_components_ms"] = measured_sum
        configurations.append(item)

    recommendations = {}
    for algorithm in ("fixed_hierarchical", "geometric_adaptive"):
        candidates = [row for row in configurations if row["algorithm"] == algorithm
                      and row["fully_valid"] and row["geomean_case_slowdown"] is not None]
        winner = min(candidates, key=lambda row: row["geomean_case_slowdown"])
        recommendations[algorithm] = {
            "config_id": winner["config_id"],
            "tile_width": winner["tile_width"],
            "tile_batch": winner["tile_batch"],
            "query_batch": 1,
            "geomean_case_slowdown": winner["geomean_case_slowdown"],
            "median_case_latency_ms": winner["median_case_latency_ms"],
        }
    recommendation = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "basis": "lowest geometric mean per-workload slowdown among fully valid development configurations",
        "confirmatory_status": "tuned on development sweep; must be frozen before held-out sessions",
        "query_batch": 1,
        "query_batch_supported": False,
        "by_algorithm": recommendations,
    }
    output = {
        "sweep_directory": str(sweep),
        "manifest": manifest,
        "detailed_rows": detailed,
        "configuration_rows": configurations,
        "failures": failures,
        "recommendation": recommendation,
    }
    (sweep / "ablation_summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    (sweep / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(sweep / "ablation_detailed.csv", detailed)
    _write_csv(sweep / "configuration_summary.csv", configurations)
    _write_csv(sweep / "failures.csv", failures)
    _heatmap(sweep / "latency_heatmap.svg", configurations,
             manifest["tile_widths"], manifest["tile_batches"])
    _breakdown(sweep / "recommended_profile_breakdown.svg", configurations, recommendation)
    print(json.dumps({
        "sweep": str(sweep),
        "detailed_rows": len(detailed),
        "failures": len(failures),
        "recommendation": recommendation,
    }, indent=2))


if __name__ == "__main__":
    main()
