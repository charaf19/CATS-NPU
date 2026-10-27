"""Collect paired physical-NPU latency for frozen Phase 8 BEIR predictions."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/phase8_beir_validation.json"))
    parser.add_argument("--out", type=Path, default=Path("results/phase8/beir_validation"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    predictions_path = Path(config["predictions_path"])
    frozen = json.loads(predictions_path.read_text(encoding="utf-8"))
    if frozen["status"] != "frozen before holdout timing":
        raise ValueError("Method choices were not frozen")
    predictions = frozen["predictions"]
    grouped = {}
    for prediction in predictions:
        key = (prediction["dataset"], prediction["query_position"])
        grouped.setdefault(key, []).append(prediction)
    if len(grouped) != 2 * len(config["query_positions"]):
        raise ValueError("Incomplete frozen query set")
    for key, rows in grouped.items():
        if sorted(row["k"] for row in rows) != config["ks"]:
            raise ValueError(f"Incomplete K values: {key}")
        if len({row["score_sha256"] for row in rows}) != 1:
            raise ValueError(f"Inconsistent input digest: {key}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.out / stamp
    root.mkdir(parents=True, exist_ok=False)
    doctor = subprocess.run([sys.executable, "-m", "cats_npu", "doctor",
                             "--out", str(root / "doctor.json")],
                            text=True, capture_output=True, check=False)
    if doctor.returncode:
        raise RuntimeError(doctor.stdout + doctor.stderr)
    manifest = {"started_utc": now(), "config": config,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                "sessions": [], "all_valid": False}
    manifest_path = root / "validation_manifest.json"
    for session_index in range(config["independent_sessions"]):
        session_id = f"phase8_{stamp}_session{session_index + 1:02d}"
        session_dir = root / session_id
        session_dir.mkdir()
        query_keys = sorted(grouped)
        random.Random(42027 + session_index).shuffle(query_keys)
        session = {"session_id": session_id, "started_utc": now(),
                   "query_order": query_keys, "runs": [], "completed": False}
        print(f"SESSION {session_index+1}/{config['independent_sessions']}", flush=True)
        for dataset, query_position in query_keys:
            rows = grouped[dataset, query_position]
            score_path = Path(rows[0]["score_path"])
            digest = hashlib.sha256(score_path.read_bytes()).hexdigest()
            if digest != rows[0]["score_sha256"]:
                raise ValueError(f"Holdout score file changed: {score_path}")
            methods = config["methods"]
            offset = (session_index + query_position) % len(methods)
            methods = methods[offset:] + methods[:offset]
            tag = dataset.split("/")[-1]
            run_id = f"{tag}_q{query_position:04d}"
            run_dir = session_dir / run_id
            command = [sys.executable, "-u", "-m", "cats_npu", "bench",
                       "--methods", ",".join(methods), "--ns", str(rows[0]["n"]),
                       "--ks", ",".join(map(str, config["ks"])),
                       "--distributions", f"beir_{tag}_q{query_position:04d}",
                       "--dtype", config["dtype"],
                       "--tile", str(config["tile_width"]),
                       "--batch", str(config["tile_batch"]),
                       "--initial-budget", str(config["initial_candidate_budget"]),
                       "--warmup", str(config["warmup"]),
                       "--repeats", str(config["repeats"]),
                       "--blocks", str(config["blocks"]),
                       "--compiler-type", config["npu_compiler_type"],
                       "--input", str(score_path), "--session-id", session_id,
                       "--out", str(run_dir)]
            start = perf_counter()
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            log = session_dir / f"{run_id}.log"
            log.write_text(result.stdout + result.stderr, encoding="utf-8")
            summaries = list(run_dir.glob("summary_*.json"))
            summary = json.loads(summaries[0].read_text(encoding="utf-8")) if summaries else None
            run = {"run_id": run_id, "command": command, "returncode": result.returncode,
                   "wall_seconds": perf_counter() - start, "summary": summary,
                   "score_sha256": digest, "log": str(log)}
            session["runs"].append(run)
            save(manifest_path, manifest | {"in_progress_session": session})
            if result.returncode or not summary or not summary["valid_for_inference"]:
                raise RuntimeError(f"Invalid run {session_id}/{run_id}; see {log}")
            print(f"  {run_id}: valid ({run['wall_seconds']:.1f}s)", flush=True)
        session["completed"] = True
        session["ended_utc"] = now()
        manifest["sessions"].append(session)
        save(manifest_path, manifest)
    manifest["all_valid"] = True
    manifest["ended_utc"] = now()
    save(manifest_path, manifest)
    print(json.dumps({"output": str(root), "sessions": len(manifest["sessions"]),
                      "all_valid": True}, indent=2))


if __name__ == "__main__":
    main()
