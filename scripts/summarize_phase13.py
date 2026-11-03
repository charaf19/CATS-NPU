"""Post-freeze descriptive checks; no new tests or changes to primary inference."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.phase13_common import BASE, ROOT, reference, sha, write_json
from scripts.phase13_design import frozen
from scripts.probe_topk_numerical_path import check_output


def summarize(out):
    protocol, lock = frozen()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((BASE / "inputs/manifest.json").read_text())
    scores = {}
    score_stats = []
    for item in manifest["scores"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
        a = np.load(ROOT / item["path"], allow_pickle=False)
        scores[item["dataset"], item["query_id"]] = a
        subnormal = (a != 0) & (np.abs(a) < np.finfo(np.float16).tiny)
        score_stats.append({"dataset": item["dataset"], "query_id": item["query_id"], "n": len(a),
                            "subnormal_scores": int(subnormal.sum()),
                            "zero_scores": int((a == 0).sum()),
                            "subnormal_top100": int(subnormal[reference(a, 100)].sum())})
    cases = {c["id"]: c for c in protocol["performance_cases"]}
    expected = {key: reference(scores[c["dataset"], c["query_id"]][:c["n"]], c["k"]).tolist()
                for key, c in cases.items()}
    counts = Counter()
    components = defaultdict(list)
    for session in range(1, protocol["performance_sessions"] + 1):
        raw = json.loads((BASE / f"performance/session{session:02d}/results.json").read_text())
        assert raw["status"] == "complete" and raw["condition_valid_start"] and raw["condition_valid_end"]
        assert raw["freeze"] == lock
        assert len(raw["observations"]) == 48 * 6 * 6 and len(raw["warmups"]) == 48 * 6 * 2
        positions = defaultdict(list)
        for kind in ("observations", "warmups"):
            for obs in raw[kind]:
                counts[kind] += 1
                if obs["accepted"]:
                    assert (obs["indices"] == expected[obs["case_id"]]) == obs["correct"]
                counts[kind + "_rejections"] += not obs["accepted"]
                counts[kind + "_accepted_errors"] += obs["accepted"] and not obs["correct"]
                if kind != "observations":
                    continue
                positions[obs["case_id"], obs["method"]].append(obs["position"])
                case = cases[obs["case_id"]]
                if case["role"] not in ("primary", "dataset"):
                    continue
                for name in ("elapsed_ns", "input_preparation_ns", "infer_ns", "output_and_certificate_ns",
                             "certificate_ns", "cpu_tie_repairs", "device_submissions",
                             "backend_execute_sync_ns", "backend_validation_ns"):
                    if name in obs:
                        components[case["dataset"], obs["method"], name].append(obs[name])
        assert all(sorted(v) == list(range(6)) for v in positions.values())
        for group in raw["groups"]:
            assert not group["construction_errors"]
            for name in ("fp32", "int32", "fp16_static"):
                assert group["placement"][name] == ["NPU"]
            assert all(v == ["NPU"] for v in
                       group["placement"]["native_npu"]["reported_execution_devices"].values())
    component_rows = [{"dataset": ds, "method": method, "field": name, "observations": len(v),
                       "median": float(np.median(v)), "sum": int(sum(v)),
                       "unit": "nanoseconds" if name.endswith("_ns") else "count",
                       "interpretation": "pooled descriptive; components are not independent inference costs"}
                      for (ds, method, name), v in sorted(components.items())]
    write_json(out / "performance_integrity_and_components.json",
               {"counts": counts, "balanced_positions_verified": True, "placement_verified": True,
                "fresh_score_domain": score_stats, "components": component_rows})

    categories = defaultdict(Counter)
    outputs = defaultdict(set)
    failures = defaultdict(set)
    for session in range(1, protocol["numerical_sessions"] + 1):
        raw = json.loads((BASE / f"numerical/session{session:02d}/results.json").read_text())
        assert raw["freeze"] == lock
        for case in raw["cases"]:
            for name, entry in case["configurations"].items():
                assert entry["execution_devices"] == ["NPU"]
                for obs in entry["runs"]:
                    if obs["backend"] != "NPU":
                        continue
                    c = obs["checks"]
                    category = next((label for field, label in (
                        ("valid_ids", "invalid_or_duplicate_ids"), ("membership", "wrong_membership"),
                        ("value_match", "wrong_values_only_after_valid_membership"),
                        ("source_value_sorted", "wrong_source_order_only"),
                        ("stable_ids_match", "stable_sequence_only")) if not c[field]), "pass")
                    categories[name, case["family"]][category] += 1
                    if not c["passed"]:
                        failures[name, session].add(case["id"])
                    with np.load(ROOT / obs["raw"], allow_pickle=False) as a:
                        digest = hashlib.sha256(a["values"].tobytes() + a["ids"].tobytes()).hexdigest()
                    outputs[name, case["id"]].add(digest)
    write_json(out / "numerical_disjoint_categories.json", {
        "note": "Post-freeze descriptive partition by first failed check; categories are not independent defects",
        "rows": [{"configuration": cfg, "family": family, **c} for (cfg, family), c in sorted(categories.items())],
        "distinct_raw_outputs_per_case": [{"configuration": cfg, "case_id": case,
                                           "distinct_outputs": len(v)} for (cfg, case), v in sorted(outputs.items())],
        "failed_case_ids": [{"configuration": cfg, "session": session, "cases": sorted(v)}
                             for (cfg, session), v in sorted(failures.items())]})

    adjacent = []
    for path in sorted((BASE / "numerical").glob("adjacent_*/results.json")):
        raw = json.loads(path.read_text())
        for case in raw["cases"]:
            folder = path.parent / case["name"]
            data = np.load(folder / "input.npy", allow_pickle=False)
            assert case["execution_devices"] == ["NPU"]
            for obs in case["runs"]:
                with np.load(folder / f"{obs['backend']}_{obs['repetition']}.npz", allow_pickle=False) as a:
                    checks = check_output(data, a["output0"], a["output1"], case["k"], case["stable"])
                count = {"rows": len(checks), "failed_rows": sum(not c["passed"] for c in checks),
                         "membership_failures": sum(not c["membership"] for c in checks),
                         "value_failures": sum(not c["value_match"] for c in checks)}
                assert all(obs["check_summary"][k] == v for k, v in count.items())
                adjacent.append({"source": str(path.relative_to(ROOT)), "case": case["name"],
                                 "backend": obs["backend"], **count})
    write_json(out / "adjacent_replication.json", adjacent)
    write_json(out / "table6_scope_and_limitations.json", {
        "performance_sensitivity": "../analysis/table6_sensitivity.json",
        "numerical_failures": "numerical_disjoint_categories.json",
        "rejections_and_correctness": "performance_integrity_and_components.json",
        "scope": {"devices": 1, "npu_architecture": "3720", "driver": "1004512", "compiler": "458778",
                  "unique_fresh_queries": 24, "queries_per_dataset": 8,
                  "performance_processes": 12, "numerical_processes": 3,
                  "temporal_scope": "sequential sessions on one date; no cross-day or cross-machine replication",
                  "numerical_scope": "fixed adversarial panel plus bounded adjacent-pair batch; no population failure CI",
                  "exactness": "over supplied finite FP16 scores; checked wrappers are exact-or-reject",
                  "unsupported": ["NaN or infinite user scores", "uncompiled shapes or invalid integer dimensions"],
                  "not_established": ["universal device correctness", "unique silicon/compiler cause",
                                      "new algorithm or precision technique", "concurrent throughput"]}})
    print(json.dumps({"performance": counts, "numerical_case_configurations": len(outputs),
                      "adjacent_outputs_rechecked": len(adjacent)}))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.out)
