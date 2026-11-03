"""Replay frozen Phase 13 methods into a new directory, never the locked evidence.

Added after confirmation was frozen; only output routing and process launch differ.
The original input arrays, methods, ordering, sample sizes and analysis stay fixed.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.phase13_common import BASE, ROOT, write_json
from scripts.phase13_design import frozen


def run(out: Path, kind: str, analyze: bool) -> None:
    protocol, lock = frozen()
    out = out.resolve()
    # Graph/raw paths are repository-relative in the frozen implementation.
    out.relative_to(ROOT.resolve())
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    (out / "logs").mkdir()
    shutil.copyfile(BASE / "inputs/manifest.json", out / "inputs/manifest.json")
    record = {"freeze": lock, "note": "post-freeze replay; retained original inputs; output routing only",
              "started_utc": datetime.now(timezone.utc).isoformat(), "commands": []}
    # A fresh interpreter per session; original frozen() still checks BASE.
    child = ("import sys; from pathlib import Path; import scripts.phase13_run as r; "
             "r.BASE=Path(sys.argv[1]); sys.argv=['phase13_run',sys.argv[2],'--session',sys.argv[3]]; "
             "raise SystemExit(r.main())")
    kinds = ("numerical", "performance") if kind == "all" else (kind,)
    for current in kinds:
        for session in range(1, protocol[current + "_sessions"] + 1):
            if current == "performance":
                time.sleep(protocol["session_idle_seconds"])
            argv = [sys.executable, "-c", child, str(out), current, str(session)]
            log = out / "logs" / f"{current}{session}.log"
            print(f"START {current} {session}", flush=True)
            started = time.perf_counter()
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
            record["commands"].append({"kind": current, "session": session, "argv": argv,
                                       "exit_code": completed.returncode,
                                       "wall_seconds": time.perf_counter() - started,
                                       "log": str(log.relative_to(ROOT))})
            write_json(out / "execution_log.json", record)
            print(f"DONE {current} {session}: {completed.returncode}", flush=True)
            # Keep failed evidence and stop; never replace a failed scheduled attempt.
            if completed.returncode:
                raise RuntimeError(f"Session failed; retained output at {out}")
    if analyze:
        if kind != "all":
            raise ValueError("Analysis requires both numerical and performance sessions (--kind all)")
        from scripts.phase13_analyze import analyze as analyze_frozen
        analyze_frozen(out, out / "analysis")
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "execution_log.json", record)
    print(json.dumps({"replay": str(out), "analyzed": analyze}))


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory inside repository")
    parser.add_argument("--kind", choices=("all", "numerical", "performance"), default="all")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze and args.kind != "all":
        parser.error("--analyze requires --kind all")
    run(args.out, args.kind, args.analyze)


if __name__ == "__main__":
    main()
