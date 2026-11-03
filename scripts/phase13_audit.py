"""Read-only historical audit and outcome-independent fresh query selection."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from scripts.phase13_common import BASE, ROOT, environment, git, sha, write_json
from scripts.prepare_beir_scores import test_query_ids


def main():
    p = ArgumentParser()
    p.add_argument("--out", type=Path, default=BASE / "audit")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    original = {}
    selections = {}
    for dataset, location in (("sift", "sift1m"), ("scifact", "beir/scifact"), ("fiqa", "beir/fiqa")):
        manifest = ROOT / "results/phase5" / location / "dataset_manifest.json"
        d = json.loads(manifest.read_text())
        used = {str(i["query_id"]) for i in d["items"]}
        if dataset == "sift":
            population = [str(i) for i in range(10000)]
        else:
            # Only test-query IDs determine population membership; no relevance metric is computed.
            population = test_query_ids(ROOT / "data/extracted/beir" / dataset)
        eligible = set(population) - used
        ranked = sorted(eligible, key=lambda x: hashlib.sha256(
            f"CATS-NPU-Phase13-20261103|{dataset}|{x}".encode()).hexdigest())
        chosen = ranked[:8]
        selections[dataset] = {"excluded_prepared_query_ids": sorted(used),
                               "population": len(population), "eligible": len(eligible),
                               "selected_ids": chosen,
                               "selection": "8 smallest SHA256(seed|dataset|queryID) among unused IDs"}
        archive = ROOT / d["archive"]
        original[dataset] = {"manifest_sha256": sha(manifest), "archive": str(archive.relative_to(ROOT)),
                             "archive_sha256": sha(archive),
                             "archive_matches": sha(archive) == d["archive_sha256"],
                             "model_revision": d.get("model_revision"),
                             "previous_prepared_ids": sorted(used)}
    phase12 = json.loads((ROOT / "results/phase12/fast_track/manifest.json").read_text())
    mismatches = [x for x, h in phase12["file_sha256"].items() if sha(ROOT / x) != h]
    # Inventory filenames, not score payloads, to detect an exposure outside the original manifests.
    old_paths = [x for x in (ROOT / "results").rglob("*.npy") if "phase13" not in x.parts]
    query_positions = sorted({int(m.group(1)) for x in old_paths
                              if (m := re.match(r"q(\d+)_", x.name))})
    if query_positions and max(query_positions) > 15:
        raise RuntimeError(f"Additional historical prepared positions need review: {query_positions}")
    pilot = {}
    for session in ("sift_session01", "sift_session02", "sift_session03"):
        r = json.loads((ROOT / "results/phase12/fast_track" / session / "results.json").read_text())
        methods = r["cases"][0]["methods"]
        pilot[session] = float(np.log(methods["stable_i32"]["median_ms"] /
                                      methods["fp32_island"]["median_ms"]))
    report = {"base_revision": git("rev-parse", "HEAD"), "environment": environment(),
              "source_archives": original, "query_selection": selections,
              "historical_npy_files_inventoried": len(old_paths),
              "historical_prepared_positions": query_positions,
              "phase12_manifest_files_verified": len(phase12["file_sha256"]),
              "phase12_hash_mismatches": mismatches,
              "phase12_sift_exploratory_log_ratios": pilot,
              "phase12_log_ratio_sample_sd": float(np.std(list(pilot.values()), ddof=1)),
              "fresh_score_arrays_read_or_created": False,
              "scope": "Freshness is relative to recorded preparation/execution history, not an unknowable external access history."}
    assert not mismatches and all(v["archive_matches"] for v in original.values())
    write_json(args.out / "audit.json", report)
    write_json(BASE / "query_selection.json", selections)
    print(json.dumps({"phase12_hash_mismatches": mismatches, "selections": selections,
                      "old_sift_log_sd": report["phase12_log_ratio_sample_sd"]}, indent=2))


if __name__ == "__main__":
    main()
