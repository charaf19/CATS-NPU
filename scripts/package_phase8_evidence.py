"""Create a portable, hashed Phase 8 evidence bundle without changing raw files."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase8"
MANIFEST = RESULTS / "evidence_manifest.json"
ARCHIVE = ROOT / "results" / "phase8_evidence_20261027.zip"
SOURCES = (
    "configs/phase8_beir_validation.json",
    "configs/phase8_beir_predictions.json",
    "docs/PREDICTIVE_FEASIBILITY.md",
    "docs/PUBLICATION_GO_NO_GO.md",
    "docs/STATISTICAL_VALIDATION.md",
    "scripts/assess_predictive_headroom.py",
    "scripts/explore_phase8_predictor.py",
    "scripts/audit_tie_repair.py",
    "scripts/prepare_phase8_beir_predictions.py",
    "scripts/run_phase8_beir_validation.py",
    "scripts/analyze_phase8_beir_validation.py",
    "scripts/bench_phase8_integrated_dispatch.py",
    "scripts/analyze_phase8_integrated_dispatch.py",
    "src/cats_npu/reference.py",
)


def main() -> None:
    if ARCHIVE.exists() or MANIFEST.exists():
        raise FileExistsError("Evidence package already exists; preserve the first archive")
    paths = sorted(RESULTS.rglob("*"))
    paths = [path for path in paths if path.is_file()]
    paths.extend(ROOT / name for name in SOURCES)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Evidence source file missing")
    entries = []
    for path in paths:
        payload = path.read_bytes()
        entries.append({"path": path.relative_to(ROOT).as_posix(),
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Phase 8 raw results, analyses, frozen protocol, and relevant source",
        "file_count_excluding_manifest": len(entries),
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with ZipFile(ARCHIVE, "x", compression=ZIP_DEFLATED, compresslevel=6) as output:
        for path in paths + [MANIFEST]:
            output.write(path, arcname=path.relative_to(ROOT).as_posix())
    print(json.dumps({"archive": str(ARCHIVE.relative_to(ROOT)),
                      "files": len(entries) + 1,
                      "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                      "archive_bytes": ARCHIVE.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
