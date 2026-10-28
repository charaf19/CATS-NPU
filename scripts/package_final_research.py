"""Package the final NO-GO record with source history and preserved evidence."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = Path("results/final_research/20261028")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/CATS-NPU-final-research-20261028.zip"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to replace {args.out}")
    if subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT, check=False).returncode:
        raise RuntimeError("Commit tracked changes before packaging the source revision")
    audit = json.loads((ROOT / AUDIT_DIR / "audit.json").read_text(encoding="utf-8"))
    if not audit["all_audit_checks_passed"]:
        raise RuntimeError("Final evidence audit has not passed")
    commit = git("rev-parse", "HEAD").decode().strip()
    bundle = ROOT / AUDIT_DIR / "source_history.bundle"
    if bundle.exists():
        raise FileExistsError(bundle)
    git("bundle", "create", str(bundle), "HEAD")
    git("bundle", "verify", str(bundle))
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(git("archive", "--format=zip", "HEAD"))) as source:
        for name in source.namelist():
            if not name.endswith("/"):
                entries["source/" + name] = source.read(name)
    result_dirs = ("results/phase7", "results/phase8", "results/phase9",
                   "results/phase9_novelty", str(AUDIT_DIR))
    for relative in result_dirs:
        for path in sorted((ROOT / relative).rglob("*")):
            if path.is_file():
                if not path.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError(f"Result file resolves outside repository: {path}")
                entries["source/" + path.relative_to(ROOT).as_posix()] = path.read_bytes()
    inputs = (
        "results/phase5/sift1m/q0000_n1000000_neg_l2_f16.npy",
        "results/phase5/sift1m/q0000_n16384_neg_l2_f16.npy",
        "results/phase5/beir/scifact/q0000_1_n5183_cosine_f16.npy",
        "results/phase5/beir/fiqa/q0000_8_n57638_cosine_f16.npy",
        "results/phase5/sift1m/dataset_manifest.json",
        "results/phase5/beir/scifact/dataset_manifest.json",
        "results/phase5/beir/fiqa/dataset_manifest.json",
        "results/doctor.json",
    )
    for relative in inputs:
        entries["source/" + relative] = (ROOT / relative).read_bytes()
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "package inventory; no new inference measurement",
        "python": sys.version, "platform": platform.platform(),
        "packages": sorted({(d.metadata.get("Name", "unknown"), d.version)
                            for d in importlib.metadata.distributions()}),
    }
    entries["environment.json"] = (json.dumps(environment, indent=2) + "\n").encode()
    readme = """CATS-NPU final research record, 2026-10-28

Decision: NO-GO for manuscript preparation on the investigated mechanisms.
Start with source/docs/FINAL_RESEARCH_CONTRIBUTION.md.

source/ contains the committed repository, Phase 7/8 historical results,
Phase 9 exploratory results, final audit, pinned compiler source evidence,
and the four development score arrays needed for the decisive pilot.
The bundle under source/results/final_research/20261028/ preserves source
history through the package commit, including the historical probe revisions.
Other dataset arrays can be regenerated using their provenance manifests.

Verify every payload file against manifest.json before using the evidence.
From source/, with the repository dependencies installed, run:
python scripts/audit_final_research.py --out results/reproduced/final_audit.json
python -m pytest -q

The audit is CPU-only. It does not run a new NPU experiment. Hardware pilot
reproduction requires the documented NPU/compiler stack, optional C++ build,
and CUDA PyTorch baseline. The existing benchmark can be repeated with:
python scripts/compare_split_key_pilot.py --input results/phase5/sift1m/q0000_n1000000_neg_l2_f16.npy --k 10 --repeats 5 --out results/reproduced/paired_pilot.json

Every output path must be new. Pilot repetitions are not independent sessions
or a confirmatory protocol. No new confirmatory study was launched after the
novelty gate failed. Phase 7/8 records retain their original scope and status.
"""
    entries["README.txt"] = readme.encode()
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "NO-GO", "final_source_commit": commit,
        "historical_probe_commits": ["0d86a8c59ec679b18146b1d6894b225e70e90760",
                                     "fe7efd912fbcafa50691f65cb1ca1eea25e7cb8f"],
        "new_confirmatory_experiment": False,
        "payload_sha256": {name: digest(data) for name, data in sorted(entries.items())},
    }
    entries["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(args.out) as archive:
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise RuntimeError(f"Archive CRC failure: {bad_crc}")
        for name, expected in manifest["payload_sha256"].items():
            if digest(archive.read(name)) != expected:
                raise RuntimeError(f"Archive hash mismatch: {name}")
    archive_sha = digest(args.out.read_bytes())
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        f"{archive_sha}  {args.out.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(args.out), "source_commit": commit,
                      "files": len(entries), "sha256": archive_sha,
                      "bytes": args.out.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
