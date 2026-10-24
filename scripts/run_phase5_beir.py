"""Run strict Top-K benchmarks over fixed BEIR semantic-score evaluation queries."""
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
    parser.add_argument("--config", type=Path, default=Path("configs/phase5_beir.json"))
    parser.add_argument("--out", type=Path, default=Path("results/phase5/beir_evaluation"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = []
    archive_hashes = {}
    model_revisions = set()
    for manifest_name in config["dataset_manifests"]:
        manifest = json.loads(Path(manifest_name).read_text(encoding="utf-8"))
        dataset_tag = manifest["dataset"].split("/")[-1]
        archive_hashes[dataset_tag] = manifest["archive_sha256"]
        model_revisions.add(manifest["model_revision"])
        selected.extend((dataset_tag, item) for item in manifest["items"]
                        if item["split"] == config["split"])
    if len(model_revisions) != 1:
        raise ValueError(f"Mixed embedding model revisions: {model_revisions}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.out / stamp
    root.mkdir(parents=True, exist_ok=False)
    runs = []
    for position, (dataset_tag, item) in enumerate(selected, 1):
        run_id = f"beir_{dataset_tag}_q{item['query_position']:04d}_n{item['n']}"
        run_dir = root / run_id
        command = [
            sys.executable, "-u", "-m", "cats_npu", "bench",
            "--methods", ",".join(config["methods"]), "--ns", str(item["n"]),
            "--ks", ",".join(str(value) for value in config["k"]),
            "--distributions", f"beir_{dataset_tag}_q{item['query_position']:04d}",
            "--dtype", config["dtype"], "--tile", str(config["tile_width"]),
            "--batch", str(config["tile_batch"]),
            "--initial-budget", str(config["initial_candidate_budget"]),
            "--warmup", str(config["warmup"]), "--repeats", str(config["repeats"]),
            "--blocks", str(config["blocks"]),
            "--compiler-type", config["npu_compiler_type"], "--input", item["path"],
            "--session-id", f"phase5_{stamp}_{run_id}", "--out", str(run_dir),
        ]
        print(f"[{position}/{len(selected)}] {run_id}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        log = root / f"{run_id}.log"
        log.write_text(result.stdout, encoding="utf-8")
        print(result.stdout, end="", flush=True)
        summaries = sorted(run_dir.glob("summary_*.json"))
        summary = json.loads(summaries[-1].read_text()) if summaries else None
        runs.append({"run_id": run_id, "dataset": dataset_tag,
                     "query_position": item["query_position"], "query_id": item["query_id"],
                     "n": item["n"], "score_sha256": item["score_sha256"],
                     "returncode": result.returncode, "wall_seconds": time.perf_counter() - started,
                     "summary": summary, "command": command, "log": str(log)})
        if result.returncode:
            (root / "evaluation_manifest.json").write_text(
                json.dumps({"config": config, "runs": runs}, indent=2) + "\n"
            )
            raise SystemExit(f"Strict BEIR run failed: {run_id}; see {log}")
    valid = all(run["summary"] and run["summary"].get("valid_for_inference") for run in runs)
    manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": str(args.config), "config": config,
                "archive_sha256": archive_hashes,
                "model_revision": next(iter(model_revisions)),
                "run_count": len(runs), "all_runs_valid": valid, "runs": runs}
    (root / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(root), "runs": len(runs), "all_valid": valid}, indent=2))
    if not valid:
        raise SystemExit("At least one strict BEIR run was invalid")


if __name__ == "__main__":
    main()
