"""Execute the locked SIFT1M confirmatory protocol in fresh subprocesses."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time


def rotate(values: list, amount: int) -> list:
    amount %= len(values)
    return values[amount:] + values[:amount]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase7_confirmatory.json"))
    parser.add_argument("--out", type=Path, default=Path("results/phase7/confirmatory"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config.get("locked_before_data_collection"):
        raise ValueError("Confirmatory config is not locked")
    dataset = json.loads(Path(config["dataset_manifest"]).read_text(encoding="utf-8"))
    items = [item for item in dataset["items"]
             if item["split"] == config["split"] and item["n"] == config["n"]]
    if sorted(item["query_id"] for item in items) != sorted(config["query_ids"]):
        raise ValueError("Dataset holdout does not match locked query IDs")
    by_query = {item["query_id"]: item for item in items}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.out / stamp
    root.mkdir(parents=True, exist_ok=False)
    doctor_path = root / "doctor.json"
    doctor = subprocess.run(
        [sys.executable, "-u", "-m", "cats_npu", "doctor", "--out", str(doctor_path)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    print(doctor.stdout, end="", flush=True)
    if doctor.returncode or not doctor_path.exists():
        raise SystemExit("Could not record the confirmatory hardware/runtime snapshot")
    config_bytes = args.config.read_bytes()
    protocol_manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                         "config_path": str(args.config), "config": config,
                         "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                         "dataset_archive_sha256": dataset["archive_sha256"],
                         "python_executable": sys.executable,
                         "platform": platform.platform(),
                         "doctor_path": str(doctor_path), "sessions": []}
    for session_index in range(config["independent_sessions"]):
        session_id = f"phase7_{stamp}_session{session_index + 1:02d}"
        session_root = root / session_id
        session_root.mkdir()
        queries = rotate(config["query_ids"], session_index)
        session_record = {"session_id": session_id,
                          "started_utc": datetime.now(timezone.utc).isoformat(),
                          "query_order": queries, "runs": [], "completed": False}
        print(f"SESSION {session_index + 1}/{config['independent_sessions']} {session_id}",
              flush=True)
        for query_position, query_id in enumerate(queries):
            item = by_query[query_id]
            methods = rotate(config["methods"], session_index + query_position)
            run_dir = session_root / f"q{query_id:04d}"
            command = [
                sys.executable, "-u", "-m", "cats_npu", "bench",
                "--methods", ",".join(methods), "--ns", str(config["n"]),
                "--ks", str(config["k"]), "--distributions", f"sift1m_q{query_id:04d}",
                "--dtype", config["dtype"], "--seed", "42",
                "--tile", str(config["tile_width"]), "--batch", str(config["tile_batch"]),
                "--initial-budget", str(config["initial_candidate_budget"]),
                "--warmup", str(config["warmup"]), "--repeats", str(config["repeats"]),
                "--blocks", str(config["blocks"]),
                "--compiler-type", config["npu_compiler_type"], "--input", item["path"],
                "--session-id", session_id, "--out", str(run_dir),
            ]
            started = time.perf_counter()
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, check=False)
            log = session_root / f"q{query_id:04d}.log"
            log.write_text(result.stdout, encoding="utf-8")
            print(result.stdout, end="", flush=True)
            summaries = sorted(run_dir.glob("summary_*.json"))
            summary = json.loads(summaries[-1].read_text()) if summaries else None
            session_record["runs"].append({
                "query_id": query_id, "method_order": methods,
                "input_sha256": item["score_sha256"], "command": command,
                "returncode": result.returncode, "wall_seconds": time.perf_counter() - started,
                "summary": summary, "log": str(log),
            })
            if result.returncode or not summary or not summary.get("valid_for_inference"):
                session_record["ended_utc"] = datetime.now(timezone.utc).isoformat()
                protocol_manifest["sessions"].append(session_record)
                (root / "confirmatory_manifest.json").write_text(
                    json.dumps(protocol_manifest, indent=2) + "\n", encoding="utf-8"
                )
                raise SystemExit(f"Invalid confirmatory run: {session_id} query {query_id}")
        session_record["completed"] = True
        session_record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        protocol_manifest["sessions"].append(session_record)
        (root / "confirmatory_manifest.json").write_text(
            json.dumps(protocol_manifest, indent=2) + "\n", encoding="utf-8"
        )
    protocol_manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
    protocol_manifest["all_sessions_valid"] = True
    (root / "confirmatory_manifest.json").write_text(
        json.dumps(protocol_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(root), "sessions": len(protocol_manifest["sessions"]),
                      "all_valid": True}, indent=2))


if __name__ == "__main__":
    main()
