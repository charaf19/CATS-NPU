"""Seal or verify Phase 13 files and verify preservation of Phase 12 evidence."""
from __future__ import annotations

from argparse import ArgumentParser
import json

from scripts.phase13_common import BASE, ROOT, sha, write_json
from scripts.phase13_design import frozen


def mismatches(entries):
    return [p for p, digest in entries.items() if not (ROOT / p).is_file() or sha(ROOT / p) != digest]


def check(create=False):
    protocol, lock = frozen()
    assert sha(ROOT / "docs/PHASE13_CONFIRMATORY_PROTOCOL.md") == lock["protocol_document_sha256"]
    old_path = ROOT / "results/phase12/fast_track/manifest.json"
    old = json.loads(old_path.read_text())
    assert not mismatches(old["file_sha256"]), "Historical Phase 12 file changed"
    library = json.loads((BASE / "provenance/native_library.json").read_text())
    assert sha(ROOT / library["original_path"]) == library["sha256"]
    assert sha(ROOT / library["snapshot"]) == library["sha256"]
    for kind in ("numerical", "performance"):
        for session in range(1, protocol[kind + "_sessions"] + 1):
            result = json.loads((BASE / kind / f"session{session:02d}/results.json").read_text())
            assert result["status"] == "complete", f"{kind} session {session} incomplete"
    manifest = BASE / "artifact_manifest.json"
    if create:
        if manifest.exists():
            raise FileExistsError(manifest)
        paths = {p for p in BASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        paths.update((ROOT / "docs").glob("PHASE13_*.md"))
        paths.update((ROOT / "scripts").glob("phase13_*.py"))
        paths.update(ROOT / p for p in ("scripts/check_phase13_artifacts.py", "scripts/replay_phase13.py",
                                       "scripts/summarize_phase13.py", "tests/test_phase13.py",
                                       "scripts/.gitattributes", "docs/.gitattributes", "tests/.gitattributes"))
        write_json(manifest, {"freeze": lock, "phase12_manifest_sha256": sha(old_path),
                              "file_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}})
    saved = json.loads(manifest.read_text())
    assert sha(old_path) == saved["phase12_manifest_sha256"]
    bad = mismatches(saved["file_sha256"])
    assert not bad, f"Phase 13 mismatches: {bad}"
    print(json.dumps({"phase13_files_verified": len(saved["file_sha256"]),
                      "phase12_files_preserved": len(old["file_sha256"]),
                      "protocol_and_sources_unchanged": True, "native_binary_unchanged": True,
                      "artifact_manifest_sha256": sha(manifest)}))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    check(args.create)
