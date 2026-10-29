"""Read-only research inventory and CPU checks of Phase 11 analytical claims.

No embeddings are generated, no retrieval scores are measured, and no model or
NPU inference is run. This is not a calibration-method implementation.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def inventory() -> dict:
    datasets = {}
    for name in ("scifact", "fiqa"):
        p = ROOT / f"results/phase5/beir/{name}/dataset_manifest.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        archive = ROOT / m["archive"]
        extracted = ROOT / f"data/extracted/beir/{name}"
        datasets[name] = {
            "manifest": str(p.relative_to(ROOT)), "manifest_sha256": digest(p),
            "model": m["model"], "model_revision": m["model_revision"],
            "corpus_size": m["corpus_size"], "dimension": m["embedding_dimension"],
            "historical_development_query_ids": m["splits_fixed_before_embedding_inspection"][
                "development"],
            "historical_confirmation_label_does_not_mean_unused": True,
            "archive_sha256_matches": archive.is_file() and digest(archive) == m["archive_sha256"],
            "archive_md5_matches": archive.is_file() and digest(archive, "md5") == m[
                "archive_md5_expected_and_verified"],
            "extracted_corpus_files": [str(f.relative_to(ROOT))
                                       for f in sorted(extracted.rglob("corpus.jsonl"))],
            "extracted_query_files": [str(f.relative_to(ROOT))
                                      for f in sorted(extracted.rglob("queries.jsonl"))],
            "stored_phase5_arrays_are_score_vectors": True,
            "score_array_contents_read_in_this_audit": False,
            "relevance_judgments_read": False,
        }
    versions = {}
    for package in ("numpy", "openvino", "torch", "sentence-transformers"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    placement = {"enumeration_only": True, "inference_executed": False}
    try:
        import openvino as ov
        core = ov.Core()
        placement["available_devices"] = core.available_devices
        if "NPU" in core.available_devices:
            for prop in ("FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE", "NPU_DRIVER_VERSION",
                         "NPU_COMPILER_VERSION"):
                try:
                    placement[prop] = str(core.get_property("NPU", prop))
                except Exception as exc:
                    placement[prop] = f"Unavailable: {type(exc).__name__}"
    except Exception as exc:
        placement["error"] = f"{type(exc).__name__}: {exc}"
    pinned_model = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    cached = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    data_arrays = list((ROOT / "data").rglob("*.npy")) + list((ROOT / "data").rglob("*.npz"))
    files = ["scripts/prepare_beir_scores.py", "src/cats_npu/data_import.py",
             "docs/FINAL_RESEARCH_CONTRIBUTION.md", "docs/CONTRACT_LOWERING_FEASIBILITY.md",
             "results/phase9/manifest.json", "results/phase9_novelty/manifest.json",
             "results/phase10/20261029/manifest.json"]
    return {"datasets": datasets, "package_versions": versions,
            "device_inventory": placement,
            "pinned_model_cache_directory_present": (cached / "snapshots" / pinned_model).is_dir(),
            "data_npy_npz_files": [str(f.relative_to(ROOT)) for f in sorted(data_arrays)],
            "source_inventory_sha256": {f: digest(ROOT / f) for f in files},
            "regeneration_performed": False, "dataset_scope": "SciFact/FiQA source inventory only"}


def analytical_checks() -> dict:
    # Exact dyadic scores that share an FP16 output bin. No real embeddings.
    a, b = Fraction(1, 2) + Fraction(1, 2**14), Fraction(1, 2) + Fraction(1, 2**13)
    pair = np.array([float(a), float(b)], dtype=np.float32)
    rounded = pair.astype(np.float16)
    source_ids = sorted(range(2), key=lambda i: (-Fraction(float(pair[i])), i))
    target_ids = sorted(range(2), key=lambda i: (-float(rounded[i]), i))
    witness = {"source_scores": pair.tolist(), "fp16_output_scores": rounded.tolist(),
               "reference_k1_id": source_ids[0], "rounded_k1_id": target_ids[0],
               "different_winner": source_ids[0] != target_ids[0],
               "scope": "Analytical scalar counterexample, not a real retrieval/NPU observation"}

    # Probe rounding homogeneity near normal FP16 bin midpoints, in host NumPy.
    half = np.arange(0x0400, 0x7c00, dtype=np.uint16).view(np.float16).astype(np.float64)
    mid = ((half[1:] + half[:-1]) / 2).astype(np.float32)
    positive = np.concatenate([np.nextafter(mid, np.float32(-np.inf)), mid,
                               np.nextafter(mid, np.float32(np.inf))]).astype(np.float64)
    values = np.concatenate([positive, -positive])
    original = values.astype(np.float16).astype(np.float64)
    tiny, maximum = float(np.finfo(np.float16).tiny), float(np.finfo(np.float16).max)
    probes = []
    for exponent in range(-4, 5):
        scaled = np.ldexp(values, exponent)
        expected = np.ldexp(original, exponent)
        valid = ((np.abs(values) >= 2*tiny) & (np.abs(scaled) >= 2*tiny)
                 & (np.abs(scaled) <= maximum) & (np.abs(expected) <= maximum))
        got = scaled[valid].astype(np.float16).astype(np.float64)
        probes.append({"power_of_two_exponent": exponent, "eligible_scalar_cases": int(valid.sum()),
                       "mismatches": int(np.count_nonzero(got != expected[valid]))})
    checks = {"exact_dyadic_scores_are_distinct": a < b,
              "fp32_stores_both_exactly": Fraction(float(pair[0])) == a and Fraction(float(pair[1])) == b,
              "output_rounding_changes_stable_top1": witness["different_winner"],
              "normal_range_homogeneity_probes_pass": all(p["mismatches"] == 0 for p in probes)}
    return {"output_rounding_witness": witness, "dyadic_scaling_probes": probes,
            "checks": {k: bool(v) for k, v in checks.items()},
            "scope": "CPU analytical checks; not a universal hardware proof or a calibration benchmark"}


def main() -> int:
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/phase11/20261029/basis_audit.json"))
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    inv, analysis = inventory(), analytical_checks()
    passed = all(analysis["checks"].values()) and all(
        d["archive_sha256_matches"] and d["archive_md5_matches"] for d in inv["datasets"].values())
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                             cwd=ROOT, text=True).strip(),
              "audit_script_sha256": digest(Path(__file__).resolve()),
              "hardware_inference_experiment": False, "real_retrieval_experiment": False,
              "new_method_implemented": False, "inventory": inv, "analytical_checks": analysis,
              "all_audit_checks_passed": bool(passed)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "all_audit_checks_passed": bool(passed),
                      "scaling_cases": sum(p["eligible_scalar_cases"] for p in analysis[
                          "dyadic_scaling_probes"]), "hardware_inference_experiment": False}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
