"""Measure the historical global tie-repair helper against its fast replacement."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from cats_npu.reference import exact_topk, topk_with_ids


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(
        "results/phase5/sift1m/q0015_n1000000_neg_l2_f16.npy"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path(
        "results/phase8/tie_repair_reference_audit.json"))
    args = parser.parse_args()
    scores = np.load(args.input, allow_pickle=False)
    indices = np.arange(scores.size, dtype=np.int64)
    def old() -> np.ndarray:
        return topk_with_ids(scores, indices, args.k)

    def new() -> np.ndarray:
        return exact_topk(scores, args.k)
    if not np.array_equal(old(), new()):
        raise AssertionError("Reference methods disagree")
    samples = {"historical_helper": [], "specialized_reference": []}
    methods = {"historical_helper": old, "specialized_reference": new}
    for repetition in range(9):
        order = list(methods) if repetition % 2 == 0 else list(reversed(methods))
        for name in order:
            started = perf_counter_ns()
            methods[name]()
            samples[name].append((perf_counter_ns() - started) / 1e6)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "exploratory microbenchmark, same input and process",
              "input": str(args.input),
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "n": int(scores.size), "k": args.k,
              "old_semantics": "topk_with_ids(scores, arange(N), K); checks uniqueness of known-unique IDs",
              "new_semantics": "exact_topk(scores, K); implicit sequential IDs",
              "same_exact_output": True,
              "samples_ms": samples,
              "median_ms": {name: median(values) for name, values in samples.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["median_ms"], indent=2))


if __name__ == "__main__":
    main()
