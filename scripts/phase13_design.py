"""Build the prospective design from exposure metadata and development pilots."""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.stats import t

from scripts.phase13_common import BASE, ROOT, sha, write_json


METHODS = ["fp32", "int32", "fp16_static", "native_npu", "cpp", "cuda"]


def numerical_cases():
    cases = []
    for width in (128, 256, 512):
        for k in (1, 10, 100):
            for family in ("positive_subnormal", "negative_subnormal", "normal_neighbors",
                           "mixed_ties_extremes", "signed_zero", "cross_tile_boundary"):
                n = (width + 1 if family in ("mixed_ties_extremes", "signed_zero")
                     else 2 * width - 1 if family == "negative_subnormal" else 2 * width + 1)
                cases.append({"id": f"{family}_W{width}_K{k}", "family": family,
                              "n": n, "k": k, "w": width, "seed": 130000 + len(cases)})
    return cases


def score_cases(selection):
    cases = []
    sizes = {"sift": 1000000, "scifact": 5183, "fiqa": 57638}
    for dataset, selection_info in selection.items():
        for i, qid in enumerate(selection_info["selected_ids"]):
            configs = [(sizes[dataset], 10, 256, "primary" if dataset == "sift" else "dataset")]
            if i < 2:
                configs.extend([(sizes[dataset], 1, 128, "sensitivity"),
                                (sizes[dataset], 100, 512, "sensitivity")])
                if dataset == "sift":
                    configs.extend([(16384, 10, 256, "sensitivity"),
                                    (131072, 10, 256, "sensitivity"),
                                    (1000000, 10, 128, "sensitivity"),
                                    (1000000, 10, 512, "sensitivity"),
                                    (1000000, 1, 256, "sensitivity"),
                                    (1000000, 100, 256, "sensitivity")])
            for n, k, w, role in configs:
                cases.append({"id": f"{dataset}_q{qid}_N{n}_K{k}_W{w}", "dataset": dataset,
                              "query_id": qid, "n": n, "k": k, "w": w, "role": role})
    return cases


def main():
    out = BASE / "protocol.json"
    if out.exists() and (BASE / "freeze.json").exists():
        raise FileExistsError(out)
    selection = json.loads((BASE / "query_selection.json").read_text())
    current_logs = []
    for path in sorted((BASE / "pilot").glob("session*/results.json")):
        r = json.loads(path.read_text())
        med = {m: np.median([x["elapsed_ns"] for x in r["observations"] if x["method"] == m])
               for m in ("fp32", "int32")}
        assert all(x["correct"] for x in r["observations"])
        current_logs.append(float(np.log(med["int32"] / med["fp32"])))
    old_sd = json.loads((BASE / "audit/audit.json").read_text())["phase12_log_ratio_sample_sd"]
    sigma = max(0.07, 2 * old_sd, 2 * float(np.std(current_logs, ddof=1)))
    paths = [p for p in (ROOT / "scripts").glob("phase13_*.py")]
    paths += [ROOT / p for p in ("src/cats_npu/precision_topk.py", "src/cats_npu/static_graph.py",
                                "src/cats_npu/order_encoding.py", "src/cats_npu/reference.py",
                                "src/cats_npu/algorithms.py", "src/cats_npu/backends.py",
                                "src/cats_npu/native.py",
                                "scripts/evaluate_precision_repair.py", "scripts/probe_stable_integer_hierarchy.py",
                                "scripts/probe_topk_numerical_path.py")]
    paths += list((ROOT / "src/cats_npu").glob("*cpp*.py"))
    paths += list((ROOT / "src/cats_npu").glob("*torch*.py"))
    paths += list((ROOT / "native").glob("*.cpp"))
    paths += list((ROOT / "native").glob("CMakeLists.txt"))
    d = {"version": 1, "client_date": "2026-11-03", "methods": METHODS,
         "performance_sessions": 12, "numerical_sessions": 3, "warmups": 2,
         "timed_repetitions": 6, "numerical_repetitions": 2, "session_idle_seconds": 10,
         "query_selection": selection, "performance_cases": score_cases(selection),
         "numerical_cases": numerical_cases(), "bootstrap_resamples": 50000,
         "crossed_bootstrap_resamples": 20000, "analysis_seed": 131313,
         "primary": {"dataset": "sift", "n": 1000000, "k": 10, "w": 256,
                     "numerator": "int32", "denominator": "fp32", "alpha": 0.05,
                     "practical_ratio": 1.10,
                     "estimand": "exp(mean session mean query paired log median latency ratio))",
                     "test": "two-sided exact session sign-flip versus log ratio zero",
                     "interval": "95% percentile session-cluster bootstrap",
                     "benefit_rule": "p<0.05 and session-bootstrap lower bound>1.10, with no accepted errors or rejected primary checked executions"},
         "secondary_holm_family": [
             {"dataset": ds, "numerator": method, "denominator": "fp32"}
             for ds in ("sift", "scifact", "fiqa") for method in ("native_npu", "fp16_static", "cpp", "cuda")]
             + [{"dataset": ds, "numerator": "int32", "denominator": "fp32"}
                for ds in ("scifact", "fiqa")],
         "pilot_justification": {"current_log_ratios": current_logs,
                                 "current_log_sd": float(np.std(current_logs, ddof=1)),
                                 "historical_log_sd": old_sd, "conservative_planning_sd": sigma,
                                 "t_halfwidth_at_12": float(t.ppf(0.975, 11) * sigma / math.sqrt(12)),
                                 "target": "approximately 4.5% log-ratio halfwidth with planning SD 0.07; not guaranteed power for a narrow 1.10 practical margin"},
         "environment_requirement": json.loads((BASE / "audit/audit.json").read_text())["environment"],
         "model": "sentence-transformers/all-MiniLM-L6-v2",
         "model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
         "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}
    write_json(out, d)
    print({"workload_cases": len(d["performance_cases"]), "numerical_cases": len(d["numerical_cases"]),
           "planning_log_halfwidth": d["pilot_justification"]["t_halfwidth_at_12"]})


def frozen():
    lock = json.loads((BASE / "freeze.json").read_text())
    protocol = json.loads((BASE / "protocol.json").read_text())
    assert sha(BASE / "protocol.json") == lock["protocol_sha256"], "Protocol changed"
    for relative, digest in protocol["source_sha256"].items():
        assert sha(ROOT / relative) == digest, f"Frozen source changed: {relative}"
    return protocol, lock


if __name__ == "__main__":
    main()
