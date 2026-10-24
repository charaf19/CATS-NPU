"""Run isolated Phase 3 NPU tile-width/tile-batch configurations."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _latest(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern), key=lambda item: item.stat().st_mtime_ns)
    return matches[-1] if matches else None


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--tiles", default="16,32,64,128,256")
    parser.add_argument("--tile-batches", default="1,8,32,64")
    parser.add_argument("--ns", default="4096,16384")
    parser.add_argument("--ks", default="4,10")
    parser.add_argument("--distributions", default="normal,tied,clustered,adversarial")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--compiler-type", default="PLUGIN")
    parser.add_argument("--out", type=Path, default=Path("results/phase3"))
    args = parser.parse_args()

    tiles = _csv_ints(args.tiles)
    batches = _csv_ints(args.tile_batches)
    if any(tile <= 0 for tile in tiles) or any(batch <= 0 for batch in batches):
        raise ValueError("Tile widths and tile batches must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = args.out / f"ablation_{args.label}" / stamp
    sweep_dir.mkdir(parents=True, exist_ok=False)
    configurations: list[dict] = []
    for tile in tiles:
        for batch in batches:
            mode = "sequential" if batch == 1 else "batched"
            methods = (
                f"ov_npu_native_{mode}_hier",
                f"ov_npu_native_{mode}_adaptive",
            )
            config_id = f"T{tile}_TB{batch}_QB1"
            config_dir = sweep_dir / config_id
            session_id = f"phase3_{args.label}_{stamp}_{config_id}"
            command = [
                sys.executable, "-u", "-m", "cats_npu", "bench",
                "--methods", ",".join(methods),
                "--ns", args.ns,
                "--ks", args.ks,
                "--distributions", args.distributions,
                "--dtype", "f16",
                "--tile", str(tile),
                "--batch", str(batch),
                "--initial-budget", "1",
                "--warmup", str(args.warmup),
                "--repeats", str(args.repeats),
                "--blocks", str(args.blocks),
                "--compiler-type", args.compiler_type,
                "--session-id", session_id,
                "--out", str(config_dir),
                "--allow-skips",
            ]
            print(f"[{len(configurations) + 1}/{len(tiles) * len(batches)}] {config_id}",
                  flush=True)
            started = perf_counter()
            process = subprocess.run(command, text=True, capture_output=True, check=False)
            elapsed_seconds = perf_counter() - started
            log_path = sweep_dir / f"{config_id}.log"
            log_path.write_text(
                process.stdout + ("\nSTDERR\n" + process.stderr if process.stderr else ""),
                encoding="utf-8",
            )
            summary_path = _latest(config_dir, "summary_*.json")
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path is not None
                else None
            )
            record = {
                "config_id": config_id,
                "tile_width": tile,
                "tile_batch": batch,
                "query_batch": 1,
                "methods": list(methods),
                "command": command,
                "returncode": process.returncode,
                "wall_seconds": elapsed_seconds,
                "output_directory": str(config_dir),
                "log": str(log_path),
                "summary_file": str(summary_path) if summary_path else None,
                "summary": summary,
            }
            configurations.append(record)
            state = "complete"
            if process.returncode or summary is None:
                state = "process-failed"
            elif not summary.get("valid_for_inference"):
                state = "measured-with-failures"
            print(f"  {state}: {elapsed_seconds:.1f}s", flush=True)
            (sweep_dir / "sweep_manifest.partial.json").write_text(
                json.dumps({"configurations": configurations}, indent=2) + "\n",
                encoding="utf-8",
            )

    manifest = {
        "sweep_id": f"phase3_{args.label}_{stamp}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "tile_widths": tiles,
        "tile_batches": batches,
        "query_batch": 1,
        "query_batch_supported": False,
        "ns": _csv_ints(args.ns),
        "ks": _csv_ints(args.ks),
        "distributions": [x.strip() for x in args.distributions.split(",") if x.strip()],
        "compiler_type": args.compiler_type,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "blocks": args.blocks,
        "configurations": configurations,
    }
    manifest_path = sweep_dir / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    partial = sweep_dir / "sweep_manifest.partial.json"
    if partial.exists():
        partial.unlink()
    valid = sum(
        bool(item.get("summary", {}).get("valid_for_inference"))
        for item in configurations
    )
    print(f"Sweep complete: {valid}/{len(configurations)} fully valid configurations")
    print(manifest_path)
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
