"""Exploratory score-sketch dispatcher test on previously used development data.

This does not read the new BEIR holdout or any Phase 7 observations. Its
cross-validation keeps whole queries/distributions together, but all timing
labels are descriptive and cannot establish a confirmatory claim.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from cats_npu.workloads import scores_for


METHODS = (
    "ov_npu_native_direct",
    "ov_npu_native_batched_hier",
    "ov_npu_native_batched_adaptive",
    "ov_npu_native_batched_costadaptive",
)
HIER = METHODS.index("ov_npu_native_batched_hier")


def score_sketch(scores: np.ndarray, k: int) -> np.ndarray:
    """Inspect at most 1,024 entries without computing the requested Top-K."""
    n = len(scores)
    count = min(n, 512)
    spaced = scores[np.linspace(0, n - 1, count, dtype=np.int64)].astype(np.float32)
    head = scores[:count].astype(np.float32)
    q05, q50, q95, q99 = np.quantile(spaced, (0.05, 0.5, 0.95, 0.99))
    spread = max(float(q95 - q05), 1e-7)
    top = np.partition(spaced, -min(k + 1, count))[-min(k + 1, count):]
    top.sort()
    return np.array((
        math.log2(n), math.log2(k), n / 256 / 32,
        float(np.std(spaced)) / spread,
        (float(q99) - float(q50)) / spread,
        (float(np.max(spaced)) - float(q99)) / spread,
        (float(np.max(head)) - float(np.max(spaced))) / spread,
        (float(np.mean(head)) - float(np.mean(spaced))) / spread,
        float(1.0 - np.unique(spaced).size / count),
        float((top[-1] - top[-min(k + 1, len(top))]) / spread),
        float(np.count_nonzero(spaced == top[-1]) / count),
    ), dtype=np.float64)


def real_paths() -> dict[tuple[str, int], Path]:
    paths = {}
    manifests = (
        Path("results/phase5/sift1m/dataset_manifest.json"),
        Path("results/phase5/beir/scifact/dataset_manifest.json"),
        Path("results/phase5/beir/fiqa/dataset_manifest.json"),
    )
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["items"]:
            if item["split"] != "evaluation":
                continue
            if "query_position" in item:
                dataset = item["dataset"].split("/")[-1]
                name = f"beir_{dataset}_q{item['query_position']:04d}"
            else:
                name = f"sift1m_q{item['query_id']:04d}"
            paths[name, item["n"]] = Path(item["path"])
    return paths


def load_cases(real_report: Path, synthetic_jsonl: Path) -> list[dict]:
    paths = real_paths()
    real = json.loads(real_report.read_text(encoding="utf-8"))
    cases = []
    for row in real["cases"]:
        scores = np.load(paths[row["query"], row["n"]], allow_pickle=False)
        group = (row["dataset"], row["query"])
        cases.append({"source": "real", "group": group,
                      "id": f"{row['query']}_n{row['n']}_k{row['k']}",
                      "n": row["n"], "k": row["k"],
                      "features": score_sketch(scores, row["k"]),
                      "times": row["times_ms"]})
    by_case: dict[tuple[int, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for line in synthetic_jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row["correct"]:
            raise ValueError("Incorrect synthetic observation")
        by_case[row["n"], row["k"], row["distribution"]][row["method"]].append(
            row["elapsed_ms"]
        )
    for (n, k, distribution), times in sorted(by_case.items()):
        if set(times) != set(METHODS):
            raise ValueError(f"Missing method: {(n, k, distribution)}")
        scores = scores_for(n, k, distribution, 42, "f16")
        cases.append({"source": "synthetic", "group": ("synthetic", distribution),
                      "id": f"synthetic_{distribution}_n{n}_k{k}",
                      "n": n, "k": k,
                      "features": score_sketch(scores, k),
                      "times": {method: median(times[method]) for method in METHODS}})
    return cases


def make_model(kind: str) -> RandomForestRegressor | DecisionTreeRegressor:
    if kind == "tree":
        return DecisionTreeRegressor(max_depth=4, min_samples_leaf=3,
                                     random_state=42)
    return RandomForestRegressor(n_estimators=128, max_depth=4, min_samples_leaf=3,
                                 random_state=42, n_jobs=1)


def train(cases: list[dict], kind: str) -> RandomForestRegressor | DecisionTreeRegressor:
    model = make_model(kind)
    x = np.stack([case["features"] for case in cases])
    y = np.array([[math.log(case["times"][method] /
                            case["times"][METHODS[HIER]]) for method in METHODS]
                  for case in cases])
    model.fit(x, y)
    return model


def summarize(rows: list[dict], field: str) -> dict:
    ratios = [row[f"{field}_regret"] for row in rows]
    return {"cases": len(rows), "geomean_regret":
            math.exp(sum(math.log(value) for value in ratios) / len(ratios)),
            "oracle_hit_rate": sum(row[f"{field}_choice"] == row["oracle"]
                                   for row in rows) / len(rows),
            "max_regret": max(ratios)}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--real", type=Path,
                        default=Path("results/phase8/predictive_headroom.json"))
    parser.add_argument("--synthetic", type=Path, default=Path(
        "results/phase8/synthetic_exploratory_fixed/observations_phase8_exploratory_fixed_20261027.jsonl"
    ))
    parser.add_argument("--out", type=Path,
                        default=Path("results/phase8/score_sketch_crossvalidation.json"))
    parser.add_argument("--model", choices=("forest", "tree"), default="forest")
    args = parser.parse_args()
    cases = load_cases(args.real, args.synthetic)
    if len(cases) != 64:
        raise ValueError(f"Expected 48 real and 16 synthetic cases, found {len(cases)}")
    rows = []
    for case in cases:
        peers = [other for other in cases if other["group"] != case["group"]]
        model = train(peers, args.model)
        prediction = model.predict(case["features"][None, :])[0]
        model_choice = METHODS[int(np.argmin(prediction))]
        shape_peers = [peer for peer in peers
                       if (peer["n"], peer["k"]) == (case["n"], case["k"])]
        shape_choice = min(METHODS, key=lambda method:
                           median(peer["times"][method] for peer in shape_peers))
        oracle = min(METHODS, key=lambda method: case["times"][method])
        rows.append({"id": case["id"], "source": case["source"],
                     "group": case["group"], "oracle": oracle,
                     "model_choice": model_choice, "shape_choice": shape_choice,
                     "model_regret": case["times"][model_choice] / case["times"][oracle],
                     "shape_regret": case["times"][shape_choice] / case["times"][oracle],
                     "times_ms": case["times"]})
    full_model = train(cases, args.model)
    for source in ("real", "synthetic"):
        selected = [row for row in rows if row["source"] == source]
        print(source, {field: summarize(selected, field) for field in ("model", "shape")})
    # Timed CPU-only decision cost. Input loading and NPU compilation are
    # excluded, matching the steady-state benchmark contract.
    probe = np.load(real_paths()["sift1m_q0004", 1_000_000], allow_pickle=False)
    sketch_ns = []
    predict_ns = []
    for _ in range(100):
        t0 = perf_counter_ns()
        feature = score_sketch(probe, 10)
        sketch_ns.append(perf_counter_ns() - t0)
        t0 = perf_counter_ns()
        full_model.predict(feature[None, :])
        predict_ns.append(perf_counter_ns() - t0)
    report = {"status": "exploratory only; group-held-out CV on prior evaluation data",
              "training_cases": len(cases),
              "model": ("128-tree multi-output random forest" if args.model == "forest"
                        else "single multi-output decision tree") + "; depth 4; leaf >=3; seed 42",
              "sketch_entries": "512 spaced and 512 leading scores",
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (args.real, args.synthetic)},
              "crossvalidation": {
                  source: {field: summarize([row for row in rows if row["source"] == source],
                                            field) for field in ("model", "shape")}
                  for source in ("real", "synthetic")},
              "decision_overhead_ms": {
                  "sketch_median": median(sketch_ns) / 1e6,
                  "model_median": median(predict_ns) / 1e6,
              },
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
