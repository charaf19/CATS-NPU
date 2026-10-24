"""Run the frozen methods over the predeclared SIFT1M evaluation queries."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase5_sift1m.json"))
    parser.add_argument("--out", type=Path, default=Path("results/phase5/sift1m_evaluation"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = json.loads(Path(config["dataset_manifest"]).read_text(encoding="utf-8"))
    expected_queries = set(config["query_ids"])
    if expected_queries & set(config["confirmatory_holdout_query_ids"]):
        raise ValueError("Evaluation and confirmatory holdout query IDs overlap")
    items = [item for item in dataset["items"]
             if item["split"] == config["split"]
             and item["query_id"] in expected_queries
             and item["n"] in config["n"]]
    expected_count = len(expected_queries) * len(config["n"])
    if len(items) != expected_count:
        raise ValueError(f"Expected {expected_count} score arrays, found {len(items)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.out / stamp
    root.mkdir(parents=True, exist_ok=False)
    runs = []
    for position, item in enumerate(sorted(items, key=lambda row: (row["query_id"], row["n"])), 1):
        run_id = f"sift_q{item['query_id']:04d}_n{item['n']}"
        run_dir = root / run_id
        command = [
            sys.executable, "-u", "-m", "cats_npu", "bench",
            "--methods", ",".join(config["methods"]),
            "--ns", str(item["n"]),
            "--ks", ",".join(str(value) for value in config["k"]),
            "--distributions", f"sift1m_q{item['query_id']:04d}",
            "--dtype", config["dtype"],
            "--tile", str(config["tile_width"]),
            "--batch", str(config["tile_batch"]),
            "--initial-budget", str(config["initial_candidate_budget"]),
            "--warmup", str(config["warmup"]),
            "--repeats", str(config["repeats"]),
            "--blocks", str(config["blocks"]),
            "--compiler-type", config["npu_compiler_type"],
            "--input", item["path"],
            "--session-id", f"phase5_{stamp}_{run_id}",
            "--out", str(run_dir),
        ]
        print(f"[{position}/{len(items)}] {run_id}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        log = root / f"{run_id}.log"
        log.write_text(result.stdout, encoding="utf-8")
        print(result.stdout, end="", flush=True)
        summaries = sorted(run_dir.glob("summary_*.json"))
        summary = json.loads(summaries[-1].read_text()) if summaries else None
        runs.append({
            "run_id": run_id,
            "query_id": item["query_id"],
            "n": item["n"],
            "score_sha256": item["score_sha256"],
            "score_path": item["path"],
            "command": command,
            "returncode": result.returncode,
            "wall_seconds": time.perf_counter() - started,
            "summary": summary,
            "log": str(log),
        })
        if result.returncode:
            (root / "evaluation_manifest.json").write_text(
                json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "config": config, "runs": runs}, indent=2) + "\n"
            )
            raise SystemExit(f"Strict SIFT1M run failed: {run_id}; see {log}")
    valid = all(run["summary"] and run["summary"].get("valid_for_inference") for run in runs)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config": config,
        "dataset_archive_sha256": dataset["archive_sha256"],
        "run_count": len(runs),
        "all_runs_valid": valid,
        "runs": runs,
    }
    (root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(root), "runs": len(runs), "all_valid": valid}, indent=2))
    if not valid:
        raise SystemExit("At least one strict SIFT1M run was invalid")


if __name__ == "__main__":
    main()
