"""Recompute the final research audit from preserved evidence, without hardware.

This audits finite-domain encoding and recorded observations. It does not run
NPU experiments, infer statistical significance, or decide scientific novelty.
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


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_manifest(relative: str) -> dict:
    path = ROOT / relative
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest["raw_sha256"].items():
        candidate = path.parent / name
        if not candidate.is_file() or sha256(candidate) != expected:
            failures.append(name)
    return {"manifest": relative, "manifest_sha256": sha256(path),
            "source_commit": manifest["source_commit"],
            "raw_files_checked": len(manifest["raw_sha256"]),
            "raw_hash_failures": failures,
            "source_revision_available": subprocess.run(
                ["git", "cat-file", "-e", manifest["source_commit"] + "^{commit}"],
                cwd=ROOT, capture_output=True, check=False).returncode == 0}


def exhaustive_encoding() -> dict:
    words = np.arange(65536, dtype=np.uint16)
    represented = words.view(np.float16)
    scores = represented[np.isfinite(represented)]
    codes = score_codes(scores)
    ids = np.arange(len(scores), dtype=np.int64)
    numeric_order = np.lexsort((ids, -scores.astype(np.float64)))
    code_order = np.lexsort((ids, -codes.astype(np.int64)))
    zeros = codes[scores == 0]
    subnormal = (np.abs(scores) > 0) & (np.abs(scores) < np.finfo(np.float16).tiny)
    return {"domain": "all 65,536 binary16 words, excluding NaNs and infinities",
            "finite_bit_patterns": int(len(scores)),
            "distinct_numeric_codes": int(np.unique(codes).size),
            "nonzero_subnormal_bit_patterns": int(np.count_nonzero(subnormal)),
            "minimum_code": int(codes.min()), "maximum_code": int(codes.max()),
            "all_63488_sorted_positions_match": bool(np.array_equal(numeric_order, code_order)),
            "signed_zeros_share_code": bool(len(zeros) == 2 and zeros[0] == zeros[1]),
            "numeric_order_sha256": hashlib.sha256(numeric_order.tobytes()).hexdigest(),
            "scope_limit": "exhaustive encoding order; not exhaustive NPU input vectors"}


def conversion_predictions() -> dict:
    i16_path = "results/phase9_novelty/stable_i16_n1m_failure_diagnostic.json"
    i16 = read_json(i16_path)
    original = np.asarray(i16["expected_values_at_got_ids"], dtype=np.int16)
    predicted = original.astype(np.float16).astype(np.int16)
    i64_path = "results/phase9_novelty/i64_wrap_probe.json"
    i64 = read_json(i64_path)
    n, k = 512, 4
    inputs = {
        "unit_delta_high_base": np.int64(1 << 40) + np.arange(n, dtype=np.int64),
        "across_32bit_boundary": np.concatenate((
            np.array([(1 << 32) - 2, (1 << 32) - 1, 1 << 32, (1 << 32) + 1], dtype=np.int64),
            np.arange(n - 4, dtype=np.int64))),
        "negative_64bit": -np.int64(1 << 40) + np.arange(n, dtype=np.int64),
    }
    cases = {}
    for name, original64 in inputs.items():
        observed = i64["cases"][name]
        original_ids = np.argsort(-original64, kind="stable")[:k]
        narrowed = original64.astype(np.int32)
        narrowed_ids = np.argsort(-narrowed.astype(np.int64), kind="stable")[:k]
        cases[name] = {
            "case_reconstruction_matches_recorded_reference": bool(
                np.array_equal(original_ids, observed["expected_indices"]) and
                np.array_equal(original64[original_ids], observed["expected_values"])),
            "observed_ids_match_signed32_prediction": bool(
                np.array_equal(narrowed_ids, observed["got_indices"])),
            "observed_values_match_signed32_prediction": bool(
                np.array_equal(narrowed[narrowed_ids], observed["got_values"])),
        }
    return {"kind": "CPU model predictions compared with preserved physical observations",
            "int16_source": i16_path,
            "int16_observed_values_match_fp16_roundtrip": bool(
                np.array_equal(predicted, i16["got_values"])),
            "int16_predicted_values": predicted.tolist(),
            "int64_source": i64_path, "int64_cases": cases,
            "causal_limit": "consistent with pinned upstream passes; no installed-binary IR trace"}


def summarize_pilot() -> dict:
    relative = "results/phase9_novelty/paired_sift_dev_q0000_n1000000_k10_with_stable.json"
    pilot = read_json(relative)
    methods = {}
    for name, samples in pilot["samples"].items():
        times = np.asarray([s["elapsed_ms"] for s in samples], dtype=np.float64)
        methods[name] = {"observations": len(samples), "median_ms": float(np.median(times)),
                         "descriptive_p95_ms": float(np.percentile(times, 95)),
                         "all_correct": all(s["correct"] for s in samples)}
    stable = methods["stable_i32_npu"]["median_ms"]
    return {"raw_file": relative, "raw_sha256": sha256(ROOT / relative),
            "n": pilot["n"], "k": pilot["k"], "width": pilot["width"],
            "input_sha256": pilot["input_sha256"], "methods": methods,
            "direct_npu_over_stable_i32_ratio": methods["direct_npu"]["median_ms"] / stable,
            "split_key_over_stable_i32_ratio": methods["split_key_npu"]["median_ms"] / stable,
            "stable_i32_over_fp16_static_ratio": stable / methods["fp16_static_npu"]["median_ms"],
            "independent_sessions": 1,
            "confidence_interval": None, "p_value": None,
            "statistical_status": "exploratory; within-process repetitions are not independent sessions",
            "placement": pilot["execution_devices"]}


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/final_research/20261028/audit.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    manifests = [audit_manifest("results/phase9/manifest.json"),
                 audit_manifest("results/phase9_novelty/manifest.json")]
    encoding = exhaustive_encoding()
    conversions = conversion_predictions()
    pilot = summarize_pilot()
    stress = read_json("results/phase9_novelty/stable_hier_stress_100.json")
    all_checks = (all(not m["raw_hash_failures"] and m["source_revision_available"] for m in manifests)
                  and encoding["all_63488_sorted_positions_match"]
                  and encoding["signed_zeros_share_code"]
                  and conversions["int16_observed_values_match_fp16_roundtrip"]
                  and all(all(case.values()) for case in conversions["int64_cases"].values())
                  and all(m["all_correct"] for m in pilot["methods"].values()))
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "final audit of existing evidence; no new hardware experiment",
              "python": sys.version, "numpy": np.__version__,
              "audit_source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "audit_script_sha256": sha256(Path(__file__).resolve()),
              "manifests": manifests, "encoding_order_check": encoding,
              "compiler_conversion_consistency": conversions,
              "existing_paired_pilot": pilot,
              "existing_stress": {"cases": stress["cases"],
                                  "failure_count": stress["failure_count"],
                                  "independent_sessions": False},
              "all_audit_checks_passed": bool(all_checks),
              "audit_does_not_establish": ["scientific novelty", "universal hardware exactness",
                                            "held-out performance", "statistical significance"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "raw_files_checked": sum(
        m["raw_files_checked"] for m in manifests),
        "all_audit_checks_passed": bool(all_checks),
        "finite_bit_patterns": encoding["finite_bit_patterns"]}, indent=2))
    return 0 if all_checks else 2


if __name__ == "__main__":
    raise SystemExit(main())
