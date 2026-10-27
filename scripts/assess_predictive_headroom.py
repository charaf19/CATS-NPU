"""Exploratory NPU dispatcher headroom from existing real-score evaluation runs.

The script never reads the locked Phase 7 collection or the BEIR confirmatory
holdout. This is a feasibility analysis, not a confirmatory experiment.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median


METHODS = (
    "ov_npu_native_direct",
    "ov_npu_native_batched_hier",
    "ov_npu_native_batched_adaptive",
    "ov_npu_native_batched_costadaptive",
)


def read_cases(roots: list[Path]) -> list[dict]:
    grouped: dict[tuple[int, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    digests: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for root in roots:
        for path in root.rglob("observations_*.jsonl"):
            summary = path.with_name(path.name.replace("observations_", "summary_")
                                     .replace(".jsonl", ".json"))
            if not json.loads(summary.read_text(encoding="utf-8"))["valid_for_inference"]:
                raise ValueError(f"Invalid descriptive run: {summary}")
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row["correct"]:
                    raise ValueError(f"Incorrect observation: {path}")
                key = (int(row["n"]), int(row["k"]), row["distribution"])
                digests[key].add(row["input_sha256"])
                grouped[key][row["method"]].append(float(row["elapsed_ms"]))
    cases = []
    for (n, k, query), methods in sorted(grouped.items()):
        if not all(method in methods for method in METHODS):
            raise ValueError(f"Incomplete comparator set: {n}, {k}, {query}")
        if len(digests[n, k, query]) != 1:
            raise ValueError(f"Input scores differ for {n}, {k}, {query}")
        times = {method: median(methods[method]) for method in METHODS}
        cases.append({"n": n, "k": k, "query": query,
                      "dataset": "SIFT1M" if query.startswith("sift1m") else
                                 "SciFact" if "scifact" in query else "FiQA",
                      "times_ms": times, "oracle_method": min(times, key=times.get),
                      "oracle_ms": min(times.values())})
    return cases


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize(cases: list[dict], choices: list[str]) -> dict:
    ratios = [case["times_ms"][choice] / case["oracle_ms"]
              for case, choice in zip(cases, choices, strict=True)]
    return {"case_count": len(cases), "oracle_hit_rate":
            sum(choice == case["oracle_method"]
                for case, choice in zip(cases, choices, strict=True)) / len(cases),
            "geomean_regret": geomean(ratios), "mean_regret": sum(ratios) / len(ratios),
            "median_regret": median(ratios), "max_regret": max(ratios),
            "geomean_headroom_to_oracle": geomean(ratios),
            "chosen_methods": dict(sorted((method, choices.count(method))
                                   for method in set(choices)))}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--sift", type=Path,
                        default=Path("results/phase5/sift1m_evaluation"))
    parser.add_argument("--direct", type=Path,
                        default=Path("results/phase6/direct_npu_sift1m"))
    parser.add_argument("--beir", type=Path,
                        default=Path("results/phase5/beir_evaluation"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/phase8/predictive_headroom.json"))
    args = parser.parse_args()
    cases = read_cases([args.sift, args.direct, args.beir])
    if len(cases) != 48:
        raise ValueError(f"Expected 48 descriptive real-score cases, got {len(cases)}")
    # Leave the target query out when choosing a method for its N/K shape.
    # This is an optimistic shape-only comparator, since other queries of
    # the same dataset are available at every N in this collection.
    shape_choices = []
    for case in cases:
        peers = [other for other in cases if other is not case and
                 (other["n"], other["k"]) == (case["n"], case["k"])]
        if not peers:
            raise ValueError("Every N/K shape needs at least one peer")
        shape_choices.append(min(METHODS, key=lambda method:
                                 median(peer["times_ms"][method] for peer in peers)))
    strategies = {method: summarize(cases, [method] * len(cases)) for method in METHODS}
    strategies["leave_one_query_out_shape_only"] = summarize(cases, shape_choices)
    report = {
        "status": "exploratory predictive feasibility; no held-out Phase 7 data read",
        "case_count": len(cases),
        "strategy_definitions": {
            "oracle": "Minimum of all four measured NPU method medians for each query; impossible at runtime",
            "leave_one_query_out_shape_only":
                "Best peer-query median for the same N/K shape; zero runtime overhead assumed",
        },
        "strategies": strategies,
        "cases": [{**case, "shape_choice": choice,
                   "shape_regret": case["times_ms"][choice] / case["oracle_ms"]}
                  for case, choice in zip(cases, shape_choices, strict=True)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": report["case_count"],
                      "strategies": strategies}, indent=2))


if __name__ == "__main__":
    main()
