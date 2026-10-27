"""Freeze Phase 8 NPU method choices before BEIR holdout timing is collected."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

from explore_phase8_predictor import METHODS, load_cases, score_sketch, train


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/phase8_beir_validation.json"))
    parser.add_argument("--training-real", type=Path,
                        default=Path("results/phase8/predictive_headroom.json"))
    parser.add_argument("--training-synthetic", type=Path, default=Path(
        "results/phase8/synthetic_exploratory_fixed/observations_phase8_exploratory_fixed_20261027.jsonl"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(config["predictions_path"])
    if output.exists():
        raise FileExistsError(f"Frozen predictions already exist: {output}")
    cases = load_cases(args.training_real, args.training_synthetic)
    if len(cases) != 64:
        raise ValueError("Training set changed")
    model = train(cases, "tree")
    predictions = []
    for manifest_path in map(Path, config["manifest_paths"]):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["items"]:
            if item["split"] != config["split"]:
                continue
            if item["query_position"] not in config["query_positions"]:
                raise ValueError("Unexpected holdout query")
            path = Path(item["path"])
            actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_digest != item["score_sha256"]:
                raise ValueError(f"Score digest mismatch: {path}")
            scores = np.load(path, allow_pickle=False)
            for k in config["ks"]:
                feature = score_sketch(scores, k)
                relative_log_costs = model.predict(feature[None, :])[0]
                tree_choice = METHODS[int(np.argmin(relative_log_costs))]
                shape_peers = [peer for peer in cases
                               if (peer["n"], peer["k"]) == (item["n"], k)]
                if not shape_peers:
                    raise ValueError(f"No shape baseline for N={item['n']}, K={k}")
                shape_choice = min(METHODS, key=lambda method:
                                   median(peer["times"][method] for peer in shape_peers))
                predictions.append({
                    "dataset": manifest["dataset"],
                    "query_position": item["query_position"],
                    "query_id": item["query_id"],
                    "n": item["n"], "k": k,
                    "score_path": str(path), "score_sha256": actual_digest,
                    "tree_choice": tree_choice, "shape_choice": shape_choice,
                    "relative_log_cost_prediction": relative_log_costs.tolist(),
                })
    expected_count = len(config["manifest_paths"]) * len(config["query_positions"]) * len(config["ks"])
    if len(predictions) != expected_count:
        raise ValueError(f"Expected {expected_count} predictions, got {len(predictions)}")
    frozen = {
        "status": "frozen before holdout timing",
        "training_sources_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.training_real, args.training_synthetic)
        },
        "model": "depth-4 multi-output decision tree; min leaf 3; random seed 42",
        "method_order": METHODS,
        "zero_overhead_only": True,
        "predictions": predictions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"predictions": len(predictions),
                      "tree_vs_shape_disagreements": sum(p["tree_choice"] != p["shape_choice"]
                                                       for p in predictions),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
