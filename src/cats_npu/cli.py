"""Windows-friendly command-line interface: python -m cats_npu <command>."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import numpy as np

from .algorithms import MethodRunner
from .backends import NPU_COMPILER_TYPES
from .reference import exact_topk
from .workloads import DISTRIBUTIONS, scores_for
from .probe import OPS, doctor, probe, save_json
from .benchmark import bench
from .statistics import read_observations, analyze, save_report
from .data_import import make_scores_from_embeddings


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _ints(value: str) -> list[int]:
    return [int(x) for x in _csv(value)]


def parser() -> ArgumentParser:
    p = ArgumentParser(prog="cats-npu", description="CATS-NPU exact Top-K research toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("doctor", help="Report hardware and software without running models")
    a.add_argument("--out", default="results/doctor.json")
    a = sub.add_parser("probe", help="Actually compile, infer and check device operators")
    a.add_argument("--device", default="NPU")
    a.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    a.add_argument("--ops", default=",".join(OPS))
    a.add_argument("--compiler-type", choices=NPU_COMPILER_TYPES, default="DEFAULT",
                   help="NPU compiler selection; DRIVER and PLUGIN can differ in support/correctness")
    a.add_argument("--out", default="results/probe_npu.json")
    a = sub.add_parser("smoke", help="Correctness smoke test for one method")
    a.add_argument("--method", default="numpy_adaptive")
    a.add_argument("--n", type=int, default=1024)
    a.add_argument("--k", type=int, default=10)
    a.add_argument("--tile", type=int, default=128)
    a.add_argument("--batch", type=int, default=8)
    a.add_argument("--initial-budget", type=int, default=1)
    a.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    a.add_argument("--compiler-type", choices=NPU_COMPILER_TYPES, default="DEFAULT")
    a.add_argument("--distribution", choices=DISTRIBUTIONS, default="tied")
    a.add_argument("--seed", type=int, default=42)
    a = sub.add_parser("generate", help="Export synthetic score vector for repeatable tests")
    a.add_argument("--n", type=int, default=16384)
    a.add_argument("--k", type=int, default=10)
    a.add_argument("--dtype", choices=("f16", "f32", "f64"), default="f16")
    a.add_argument("--distribution", choices=DISTRIBUTIONS, default="clustered")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--out", default="results/scores.npy")
    a = sub.add_parser("from-embeddings", help="Create local cosine/dot-product score vector from .npy embeddings")
    a.add_argument("--embeddings", required=True)
    a.add_argument("--queries", required=True)
    a.add_argument("--query-index", type=int, default=0)
    a.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    a.add_argument("--no-normalize", action="store_true")
    a.add_argument("--chunk", type=int, default=65536)
    a.add_argument("--out", default="results/real_scores.npy")
    a = sub.add_parser("bench", help="Run verified timed comparisons")
    a.add_argument("--methods", default="cpu_exact,numpy_hier,numpy_adaptive")
    a.add_argument("--ns", default="1024,16384")
    a.add_argument("--ks", default="1,10")
    a.add_argument("--distributions", default="normal,tied,clustered")
    a.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    a.add_argument("--tile", type=int, default=256)
    a.add_argument("--batch", type=int, default=8)
    a.add_argument("--initial-budget", type=int, default=1)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--warmup", type=int, default=3)
    a.add_argument("--repeats", type=int, default=20)
    a.add_argument("--blocks", type=int, default=2)
    a.add_argument("--session-id", default=None)
    a.add_argument("--out", default="results/bench")
    a.add_argument("--input", default=None, help="Optional .npy scores; N must equal --ns")
    a.add_argument("--compiler-type", choices=NPU_COMPILER_TYPES, default="DEFAULT")
    a.add_argument("--allow-skips", action="store_true", help="Return success despite explicit failures")
    a = sub.add_parser("analyze", help="Paired session-level statistics")
    a.add_argument("inputs", nargs="+", help="JSONL files or directories")
    a.add_argument("--baseline", default="cpu_exact")
    a.add_argument("--min-sessions", type=int, default=6)
    a.add_argument("--out", default="results/analysis.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.cmd == "doctor":
            result = doctor()
            save_json(result, args.out)
            print(json.dumps(result, indent=2))
        elif args.cmd == "probe":
            names = tuple(_csv(args.ops))
            if any(op not in OPS for op in names):
                raise ValueError(f"Unknown ops; allowed {OPS}")
            result = probe(args.device, args.dtype, names, args.compiler_type)
            save_json(result, args.out)
            print(json.dumps(result, indent=2))
            if not result["all_passed"]:
                print("NOTE: not every requested operator passed. Inspect the JSON for exact failures.")
                return 1
        elif args.cmd == "smoke":
            scores = scores_for(args.n, args.k, args.distribution, args.seed, args.dtype)
            ref = exact_topk(scores, args.k)
            runner = MethodRunner(args.method, args.n, args.k, args.tile,
                                  args.batch, args.dtype, args.initial_budget,
                                  compiler_type=args.compiler_type)
            runner.prepare()
            result = runner.run(scores)
            passed = bool(np.array_equal(result.indices, ref))
            print(json.dumps({"method": args.method, "correct": passed,
                              "expected": ref.tolist(), "actual": result.indices.tolist(),
                              "compile_ms_excluded": runner.compile_ms,
                              "device_info": runner.device_info, **result.metadata()}, indent=2))
            return 0 if passed else 1
        elif args.cmd == "generate":
            x = scores_for(args.n, args.k, args.distribution, args.seed, args.dtype)
            p = Path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, x, allow_pickle=False)
            print(f"Wrote {x.size} {x.dtype} scores to {p}")
        elif args.cmd == "from-embeddings":
            result = make_scores_from_embeddings(
                args.embeddings, args.queries, args.query_index,
                args.out, dtype=args.dtype, normalize=not args.no_normalize, chunk=args.chunk,
            )
            print(json.dumps(result, indent=2))
        elif args.cmd == "bench":
            result = bench(methods=_csv(args.methods), ns=_ints(args.ns), ks=_ints(args.ks),
                           distributions=_csv(args.distributions), dtype=args.dtype,
                           seed=args.seed, tile=args.tile, batch=args.batch,
                           initial_budget=args.initial_budget, warmup=args.warmup,
                           repeats=args.repeats, blocks=args.blocks, out=Path(args.out),
                           session_id=args.session_id, allow_skips=args.allow_skips,
                           input_path=args.input, compiler_type=args.compiler_type)
            print(json.dumps(result, indent=2))
        elif args.cmd == "analyze":
            report = analyze(read_observations([Path(x) for x in args.inputs]),
                             args.baseline, args.min_sessions)
            save_report(report, Path(args.out))
            print(f"Report written: {args.out}")
            for case, result in report["cases"].items():
                print(f"\n{case}")
                for method, item in result["methods"].items():
                    print(f"  {method:30s} median={item.get('median_ms','NA')} ms "
                          f"p95={item.get('p95_ms','NA')} ms invalid={item['incorrect_runs']}")
                for method, item in result["comparisons_vs_baseline"].items():
                    print(f"    vs {args.baseline}: {method} "
                          f"speedup={item['geometric_speedup']:.3f} "
                          f"sessions={item['paired_sessions']} "
                          f"status={item['significance_status']}")
        return 0
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
