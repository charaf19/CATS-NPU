"""CPU-only Phase 10 rank-contract analysis; not a compiler or NPU experiment.

The conversion functions are explicit mathematical models. They do not inspect
the installed executable, infer a hardware contract, or establish novelty.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from cats_npu.order_encoding import score_codes
from scripts.audit_final_research import audit_manifest


ROOT = Path(__file__).resolve().parents[1]


def stable_order(values: np.ndarray) -> list[int]:
    """Independent Python reference for the small counterexample witnesses."""
    return sorted(range(len(values)), key=lambda i: (-float(values[i]), i))


def rank_profile(scores: np.ndarray, mapped: np.ndarray) -> dict:
    """Check an enumerated scalar map on a numerically sorted finite domain.

    Adjacent strict inequalities and equalities suffice by transitivity.
    A passing result is conditional on the given elementwise model and an exact
    stable selector. It is not validation of a compiled tensor computation.
    """
    scores, mapped = np.asarray(scores), np.asarray(mapped)
    if scores.ndim != 1 or mapped.shape != scores.shape or scores.size < 2:
        raise ValueError("Expected equally sized one-dimensional domains of length >= 2")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(mapped)):
        raise ValueError("This diagnostic requires finite numeric values")
    if np.any(scores[1:] < scores[:-1]):
        raise ValueError("Reference domain must be sorted numerically")
    strict = scores[1:] > scores[:-1]
    equal = scores[1:] == scores[:-1]
    rises = mapped[1:] > mapped[:-1]
    ties = mapped[1:] == mapped[:-1]
    fails = (strict & ~rises) | (equal & ~ties)
    result = {
        "enumerated_words": int(scores.size),
        "distinct_source_values": int(np.unique(scores).size),
        "distinct_target_values": int(np.unique(mapped).size),
        "adjacent_distinct_score_collisions": int(np.count_nonzero(strict & ties)),
        "adjacent_order_inversions": int(np.count_nonzero(strict & (mapped[1:] < mapped[:-1]))),
        "adjacent_equal_score_splits": int(np.count_nonzero(equal & ~ties)),
        "preserves_enumerated_rank_contract": bool(not np.any(fails)),
        "counterexample": None,
    }
    if np.any(fails):
        i = int(np.flatnonzero(fails)[0])
        indices = [i, i + 1]
        if equal[i] and mapped[i] > mapped[i + 1]:
            indices.reverse()
        source_pair, target_pair = scores[indices], mapped[indices]
        witness = {"source_scores": source_pair.astype(np.float64).tolist(),
                   "mapped_scores": target_pair.astype(np.float64).tolist(),
                   "expected_ids_k1": stable_order(source_pair)[:1],
                   "model_ids_k1": stable_order(target_pair)[:1]}
        if source_pair.dtype == np.float16:
            witness["source_binary16_words"] = [f"0x{int(v):04x}"
                                                 for v in source_pair.view(np.uint16)]
        if witness["expected_ids_k1"] == witness["model_ids_k1"]:
            raise AssertionError("Rank violation did not produce a counterexample")
        result["counterexample"] = witness
    return result


def analyze_models() -> dict:
    words = np.arange(65536, dtype=np.uint16)
    domain = words.view(np.float16)
    domain = domain[np.isfinite(domain)]
    domain = domain[np.argsort(domain.astype(np.float64), kind="stable")]
    codes = score_codes(domain)
    centered = (codes - 32768).astype(np.int16)
    raw = domain.view(np.uint16)
    unnormalized = np.where(raw & 0x8000, (~raw) & 0xffff, raw ^ 0x8000).astype(np.int32)
    ftz = domain.copy()
    ftz[np.abs(ftz) < np.finfo(np.float16).tiny] = 0
    integer_domain = np.array([0, 2**31 - 1, 2**31, 2**32], dtype=np.int64)
    profiles = {
        "manual_int32_identity": rank_profile(domain, codes),
        "int32_codes_to_fp32_model": rank_profile(domain, codes.astype(np.float32)),
        "centered_int16_identity": rank_profile(domain, centered),
        "centered_int16_via_fp16_model": rank_profile(domain, centered.astype(np.float16)),
        "unnormalized_zero_order_codes": rank_profile(domain, unnormalized),
        "hypothetical_fp16_flush_to_zero_model": rank_profile(domain, ftz),
        "int64_to_signed32_bounded_witness_domain": rank_profile(
            integer_domain, integer_domain.astype(np.int32)),
    }
    expected_safe = {"manual_int32_identity", "int32_codes_to_fp32_model",
                     "centered_int16_identity"}
    checks = {
        "finite_domain_size": domain.size == 63488,
        "numeric_classes": np.unique(domain).size == 63487,
        "expected_models_only_are_safe": all(
            p["preserves_enumerated_rank_contract"] == (name in expected_safe)
            for name, p in profiles.items()),
        "int16_via_fp16_target_classes": profiles[
            "centered_int16_via_fp16_model"]["distinct_target_values"] == 12160,
        "int16_via_fp16_adjacent_merges": profiles[
            "centered_int16_via_fp16_model"]["adjacent_distinct_score_collisions"] == 51327,
    }
    return {"models": profiles, "internal_checks": {k: bool(v) for k, v in checks.items()}}


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("results/phase10/20261029/contract_audit.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    evidence = [audit_manifest("results/phase9/manifest.json"),
                audit_manifest("results/phase9_novelty/manifest.json")]
    analysis = analyze_models()
    passed = all(analysis["internal_checks"].values()) and all(
        not e["raw_hash_failures"] and e["source_revision_available"] for e in evidence)
    files = ["scripts/audit_contract_lowering.py", "src/cats_npu/order_encoding.py",
             "scripts/probe_stable_integer_hierarchy.py", "src/cats_npu/static_graph.py",
             "results/final_research/20261028/upstream/adjust_software_ops_precision.cpp",
             "results/final_research/20261028/upstream/convert_precision_to_i32.cpp"]
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "source_base_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                                for f in files},
              "python": sys.version, "numpy": np.__version__,
              "hardware_experiment": False, "historical_evidence": evidence,
              **analysis, "all_audit_checks_passed": bool(passed),
              "limitations": ["Scalar models are supplied, not extracted from the executable.",
                              "Flush-to-zero is hypothetical; no FP16 root cause is inferred.",
                              "INT64 is a bounded witness domain, not exhaustive enumeration.",
                              "No physical inference, timing, synthesis or compiler validation.",
                              "No scientific originality decision is computed by this script."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "all_audit_checks_passed": bool(passed),
                      "historical_raw_files_verified": sum(e["raw_files_checked"] for e in evidence),
                      "hardware_experiment": False}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
