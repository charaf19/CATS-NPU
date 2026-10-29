"""Recompute the bounded Phase 12 findings from saved raw outputs, without inference."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_final_research import audit_manifest
from scripts.probe_topk_numerical_path import check_output


def main():
    p = ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/phase12/fast_track"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "purpose": "recomputed saved evidence; no new hardware execution",
               "historical_integrity": [audit_manifest("results/phase9/manifest.json"),
                                        audit_manifest("results/phase9_novelty/manifest.json")],
               "adjacent_pairs": {}, "repair_correctness": {}, "performance": {},
               "known_harness_errors": [
                   {"session": "isolation_session01", "issue": "mixed Node/Output hierarchy constructor",
                    "disposition": "harness error; fixed and superseded by isolation_session02"},
                   {"sessions": ["sift_session02", "scifact_session01", "fiqa_session01"],
                    "method": "native_npu", "issue": "missing MethodRunner.prepare()",
                    "disposition": "harness error; exclude this method's timings, use final sessions"}]}
    for session in ("adjacent_default_session01", "adjacent_precision_session01"):
        report = json.loads((args.root / session / "results.json").read_text())
        summary["adjacent_pairs"][session] = {}
        for case in report["cases"]:
            path = args.root / session / case["name"]
            data = np.load(path / "input.npy", allow_pickle=False)
            with np.load(path / "NPU_0.npz", allow_pickle=False) as raw:
                checks = check_output(data, raw["output0"], raw["output1"], 1, True)
            computed = {"rows": len(checks),
                        "failed_rows": sum(not c["passed"] for c in checks),
                        "membership_failures": sum(not c["membership"] for c in checks),
                        "value_failures": sum(not c["value_match"] for c in checks)}
            saved = next(r for r in case["runs"] if r["backend"] == "NPU")["check_summary"]
            assert all(saved[k] == v for k, v in computed.items())
            summary["adjacent_pairs"][session][case["name"]] = computed
    for session in ("repair_session01", "repair_session02", "composition_session01", "composition_session02"):
        r = json.loads((args.root / session / "results.json").read_text())
        rows = []
        for case in r["cases"]:
            a = np.load(args.root / session / case["name"] / "input.npy", allow_pickle=False)
            expected = sorted(range(len(a)), key=lambda i: (-float(a[i]), i))[:case["k"]]
            raw_path = args.root / session / case["name"] / "fp32_island_raw.npz"
            with np.load(raw_path, allow_pickle=False) as raw:
                checks = check_output(a.reshape(1, -1), raw["output0"], raw["output1"],
                                      case["k"], True)
                assert raw["output1"].reshape(-1).tolist() == expected
                assert checks[0]["passed"]
            method = case["methods"]["fp32_island"]
            assert method["all_correct"]
            rows.append({"case": case["name"], "n": case["n"], "k": case["k"],
                         "saved_raw_preflight_verified": True,
                         "timed_repetitions_passed": sum(s["correct"] for s in method["samples"])})
        summary["repair_correctness"][session] = rows
    for session in ("sift_session01", "sift_session02", "sift_session03",
                    "scifact_session02", "fiqa_session02"):
        r = json.loads((args.root / session / "results.json").read_text())
        case = r["cases"][0]
        methods = {}
        for method, values in case["methods"].items():
            if method == "native_npu" and session == "sift_session02":
                continue
            assert values["all_correct"], (session, method)
            times = np.asarray([s["elapsed_ms"] for s in values["samples"]])
            methods[method] = {"median_ms": float(np.median(times)),
                               "descriptive_p95_ms": float(np.percentile(times, 95)),
                               "median_sequential_queries_per_s": float(1000 / np.median(times)),
                               "timed_repetitions": len(times)}
        summary["performance"][session] = {
            "input_path": case["input_path"], "n": case["n"], "k": case["k"],
            "methods": methods,
            "i32_over_fp32_island_ratio": methods["stable_i32"]["median_ms"] /
            methods["fp32_island"]["median_ms"]}
    # Bind every saved result file, including initial harness failures.
    summary["result_sha256"] = {
        p.relative_to(args.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.root.glob("*/results.json"))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"historical_raw_files_verified": sum(
        x["raw_files_checked"] for x in summary["historical_integrity"]),
        "historical_hash_failures": sum(len(x["raw_hash_failures"]) for x in summary["historical_integrity"]),
        "repair_cases_verified": sum(len(x) for x in summary["repair_correctness"].values()),
        "ratios": {s: r["i32_over_fp32_island_ratio"] for s, r in summary["performance"].items()}}, indent=2))


if __name__ == "__main__":
    main()
