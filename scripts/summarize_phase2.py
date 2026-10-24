"""Create compact descriptive sequential/batched comparison tables."""
from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from statistics import median

from cats_npu.statistics import read_observations


METHODS = {
    "seq_hier": "ov_npu_native_sequential_hier",
    "batch_hier": "ov_npu_native_batched_hier",
    "seq_adaptive": "ov_npu_native_sequential_adaptive",
    "batch_adaptive": "ov_npu_native_batched_adaptive",
}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path("results/phase2_pilot/batching_comparison.csv"))
    args = parser.parse_args()
    records = [row for row in read_observations([args.input])
               if row.get("correct") and row.get("_session_valid", True)]
    cases = sorted({row["case_id"] for row in records})
    output = []
    for case_id in cases:
        case_rows = [row for row in records if row["case_id"] == case_id]
        by_method = {
            label: [row for row in case_rows if row["method"] == method]
            for label, method in METHODS.items()
        }
        if any(not values for values in by_method.values()):
            continue
        first = case_rows[0]
        item = {
            "case_id": case_id,
            "n": first["n"],
            "k": first["k"],
            "distribution": first["distribution"],
        }
        for label, values in by_method.items():
            item[f"{label}_median_ms"] = median(float(row["elapsed_ms"]) for row in values)
            item[f"{label}_submissions"] = median(
                float(row["device_submissions"]) for row in values
            )
        item["hier_batching_speedup"] = (
            item["seq_hier_median_ms"] / item["batch_hier_median_ms"]
        )
        item["adaptive_batching_speedup"] = (
            item["seq_adaptive_median_ms"] / item["batch_adaptive_median_ms"]
        )
        item["batched_adaptive_over_hier_speedup"] = (
            item["batch_hier_median_ms"] / item["batch_adaptive_median_ms"]
        )
        output.append(item)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    args.out.with_suffix(".json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(output)} case comparisons to {args.out}")


if __name__ == "__main__":
    main()
