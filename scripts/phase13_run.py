"""Prospectively frozen Phase 13 physical sessions and development smoke."""
from __future__ import annotations

from argparse import ArgumentParser
import gc
import json
from pathlib import Path

import numpy as np

from scripts.phase13_common import BASE, ROOT, environment, measure, power, reference, runners, sha, write_json
from scripts.phase13_design import frozen


def condition_matches(env, planned):
    return (env["npu"] == planned["npu"] and env["packages"] == planned["packages"]
            and env["power"]["ac_connected"] and not env["power"]["battery_saver"]
            and env["power"]["power_plan"] == planned["power"]["power_plan"])


def performance(session):
    protocol, lock = frozen()
    out = BASE / "performance" / f"session{session:02d}"
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((BASE / "inputs/manifest.json").read_text())
    inputs = {(v["dataset"], v["query_id"]): v for v in manifest["scores"]}
    env = environment()
    record = {"session": session, "freeze": lock, "environment_start": env,
              "condition_valid_start": condition_matches(env, protocol["environment_requirement"]),
              "observations": [], "warmups": [], "groups": [], "status": "running"}
    if not record["condition_valid_start"]:
        record["status"] = "invalid_environment_before_measurement"
        write_json(out / "results.json", record)
        return 2
    rng = np.random.default_rng(1300000 + session)
    groups = {}
    for case in protocol["performance_cases"]:
        groups.setdefault((case["dataset"], case["n"], case["k"], case["w"]), []).append(case)
    keys = list(groups)
    rng.shuffle(keys)
    for dataset, n, k, width in keys:
        rr, placement, preparation_ms, errors = runners(n, k, width)
        group = {"dataset": dataset, "n": n, "k": k, "w": width,
                 "placement": placement, "preparation_ms": preparation_ms, "construction_errors": errors}
        record["groups"].append(group)
        group_cases = list(groups[(dataset, n, k, width)])
        rng.shuffle(group_cases)
        for case in group_cases:
            info = inputs[dataset, case["query_id"]]
            path = ROOT / info["path"]
            assert sha(path) == info["sha256"]
            a = np.ascontiguousarray(np.load(path, allow_pickle=False)[:n])
            expected = reference(a, k)
            base_order = list(protocol["methods"])
            rng.shuffle(base_order)
            for warmup in range(protocol["warmups"]):
                for method in base_order:
                    if method in rr:
                        obs = measure(rr[method], a, expected)
                        record["warmups"].append({"case_id": case["id"], "method": method,
                                                  "warmup": warmup, **obs})
            # Six cyclic rotations put every method in every position once.
            for block in range(protocol["timed_repetitions"]):
                offset = (block + session) % len(base_order)
                order = base_order[offset:] + base_order[:offset]
                if session % 2:
                    order = order[::-1]
                for position, method in enumerate(order):
                    if method not in rr:
                        obs = {"elapsed_ns": None, "accepted": False, "correct": False,
                               "error": "construction: " + errors[method]}
                    else:
                        obs = measure(rr[method], a, expected)
                    record["observations"].append({"case_id": case["id"], "method": method,
                                                   "block": block, "position": position,
                                                   "input_sha256": info["sha256"], **obs})
            write_json(out / "results.json", record)
        del rr
        gc.collect()
        print(json.dumps({"session": session, "group": [dataset, n, k, width],
                          "observations": len(record["observations"])}), flush=True)
    record["power_end"] = power()
    record["condition_valid_end"] = (record["power_end"]["ac_connected"]
                                        and not record["power_end"]["battery_saver"]
                                        and record["power_end"]["power_plan"] == env["power"]["power_plan"])
    record["status"] = "complete"
    write_json(out / "results.json", record)
    return 0


def numerical(session):
    import openvino as ov

    from cats_npu.order_encoding import score_codes
    from cats_npu.precision_topk import build_precision_graph
    from scripts.probe_stable_integer_hierarchy import graph as integer_graph
    from scripts.probe_topk_numerical_path import build_model, check_output

    protocol, lock = frozen()
    out = BASE / "numerical" / f"session{session:02d}"
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((BASE / "inputs/manifest.json").read_text())
    env = environment()
    result = {"session": session, "freeze": lock, "environment": env,
              "environment_valid": condition_matches(env, protocol["environment_requirement"]),
              "operator_contract": "stable=true, mode=max, sort=value; raw outputs, no repair",
              "cases": [], "status": "running"}
    assert result["environment_valid"]
    core, cache = ov.Core(), {}
    names = ["fp16", "promotion", "topk", "reshape", "combined", "int32"]
    for case in manifest["numerical"]:
        assert sha(ROOT / case["path"]) == case["sha256"]
        a = np.load(ROOT / case["path"], allow_pickle=False)
        n, k, width = case["n"], case["k"], case["w"]
        item = {**case, "configurations": {}}
        result["cases"].append(item)
        folder = out / case["id"]
        folder.mkdir()
        for name in names:
            entry = {"runs": [], "error": None}
            item["configurations"][name] = entry
            source = score_codes(a) if name == "int32" else a
            data = np.full(((n + width - 1) // width, width),
                           np.iinfo(np.int32).min if name == "int32" else -np.inf, source.dtype)
            data.ravel()[:n] = source
            signature = (name, n, k, width)
            try:
                if signature not in cache:
                    model = (integer_graph(n, k, width) if name == "int32" else
                             build_model(data.shape, k, np.float16, "hierarchy", True) if name == "fp16"
                             else build_precision_graph(n, k, width))
                    cfg = {"NPU_COMPILER_TYPE": "PLUGIN"}
                    layers = {"topk": "TopK", "reshape": "Reshape", "combined": "TopK,Reshape"}.get(name)
                    if layers:
                        cfg["NPU_COMPILATION_MODE_PARAMS"] = "compute-layers-with-higher-precision=" + layers
                    model_path = folder / f"{name}.xml"
                    ov.serialize(model, model_path)
                    npu = core.compile_model(model, "NPU", cfg)
                    cpu = core.compile_model(model, "CPU")
                    devices = npu.get_property("EXECUTION_DEVICES")
                    devices = [devices] if isinstance(devices, str) else list(devices)
                    if devices != ["NPU"]:
                        raise RuntimeError(f"Placement: {devices}")
                    cache[signature] = (npu, cpu, cfg, devices, str(model_path.relative_to(ROOT)))
                npu, cpu, cfg, devices, model_path = cache[signature]
                entry.update({"compile_config": cfg, "execution_devices": devices, "model_path": model_path})
                for backend, executable in (("CPU", cpu), ("NPU", npu)):
                    req = executable.create_infer_request()
                    for repeat in range(1 if backend == "CPU" else protocol["numerical_repetitions"]):
                        req.infer({0: data})
                        values = np.array(req.get_output_tensor(0).data, copy=True).reshape(1, k)
                        ids = np.array(req.get_output_tensor(1).data, copy=True).reshape(1, k)
                        checks = check_output(source.reshape(1, -1), values, ids, k, True)[0]
                        value_path = folder / f"{name}_{backend}_{repeat}.npz"
                        np.savez_compressed(value_path, values=values, ids=ids)
                        entry["runs"].append({"backend": backend, "repeat": repeat, "checks": checks,
                                              "raw": str(value_path.relative_to(ROOT))})
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "results.json", result)
    result["power_end"] = power()
    result["condition_valid_end"] = (result["power_end"]["ac_connected"]
                                       and not result["power_end"]["battery_saver"]
                                       and result["power_end"]["power_plan"] == env["power"]["power_plan"])
    result["status"] = "complete"
    write_json(out / "results.json", result)
    print(json.dumps({"numerical_session": session, "cases": len(result["cases"])}), flush=True)
    return 0


def smoke(out):
    out.mkdir(parents=True, exist_ok=False)
    # Never use a fresh query in harness development.
    a = np.load("results/phase5/sift1m/q0000_n1000000_neg_l2_f16.npy", allow_pickle=False)
    result = []
    from scripts.phase13_design import score_cases
    selection = json.loads((BASE / "query_selection.json").read_text())
    configs = sorted({(c["n"], c["k"], c["w"]) for c in score_cases(selection)})
    for n, k, w in configs:
        rr, placement, prep, errors = runners(n, k, w)
        expected = reference(a[:n], k)
        observed = {name: measure(r, a[:n], expected) for name, r in rr.items()}
        result.append({"n": n, "k": k, "w": w, "placement": placement,
                       "preparation_ms": prep, "errors": errors, "observed": observed})
        write_json(out / "results.json", result)
        print({"smoke": [n, k, w], "errors": errors,
               "correct": {m: r["correct"] for m, r in observed.items()}}, flush=True)
        del rr
        gc.collect()
    return 0 if all(not r["errors"] and all(x["correct"] for x in r["observed"].values()) for r in result) else 2


def main():
    p = ArgumentParser()
    p.add_argument("kind", choices=("smoke", "numerical", "performance"))
    p.add_argument("--session", type=int)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.kind == "smoke":
        return smoke(args.out)
    if args.session is None or args.session < 1:
        raise ValueError("Positive session number required")
    return numerical(args.session) if args.kind == "numerical" else performance(args.session)


if __name__ == "__main__":
    raise SystemExit(main())
