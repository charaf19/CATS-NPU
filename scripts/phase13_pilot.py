"""Current-stack pilot using only the old Phase 12 development query."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from cats_npu.precision_topk import PrecisionNPUTopK
from scripts.evaluate_precision_repair import IntegerRunner
from scripts.phase13_common import environment, measure, reference, write_json


def main():
    p = ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    a = np.load("results/phase5/sift1m/q0000_n1000000_neg_l2_f16.npy", allow_pickle=False)
    expected = reference(a, 10)
    rr = {"fp32": PrecisionNPUTopK(len(a), 10), "int32": IntegerRunner(len(a), 10, 256)}
    result = {"status": "pilot, previously accessed development input only",
              "environment": environment(), "seed": args.seed, "observations": []}
    for r in rr.values():
        for _ in range(2):
            assert measure(r, a, expected)["correct"]
    rng = np.random.default_rng(args.seed)
    for block in range(6):
        for method in rng.permutation(list(rr)):
            result["observations"].append({"method": str(method), "block": block,
                                           **measure(rr[method], a, expected)})
    write_json(args.out / "results.json", result)
    medians = {m: float(np.median([x["elapsed_ns"] for x in result["observations"] if x["method"] == m]))
               for m in rr}
    print({"path": str(args.out), "int32_over_fp32": medians["int32"] / medians["fp32"]})


if __name__ == "__main__":
    main()
