"""Physical NPU correctness probes for the Phase 9 static graph."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from cats_npu.reference import exact_topk
from cats_npu.static_graph import StaticNPUTopK


def main() -> None:
    tiny = np.nextafter(np.float16(0), np.float16(1))
    subnormal = np.full(512, -1, dtype=np.float16)
    subnormal[:64] = (np.arange(64) - 32).astype(np.float16) * tiny
    one_ulp = np.full(512, -1, dtype=np.float16)
    one_ulp[:16] = np.float16(1) + np.arange(16).astype(np.float16) * np.spacing(np.float16(1))
    signed_zero = np.zeros(513, dtype=np.float16)
    signed_zero[::2] = np.float16(-0.0)
    tie = np.full(5183, -1, dtype=np.float16)
    tie[:30] = 1
    cases = {
        "positive_negative_subnormal": (subnormal, 4),
        "one_ulp_at_one": (one_ulp, 4),
        "all_equal": (np.ones(513, dtype=np.float16), 10),
        "signed_zero": (signed_zero, 10),
        "boundary_tie": (tie, 10),
        "irregular_single_padded_tile": (np.arange(257, dtype=np.float16), 10),
    }
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "exploratory physical NPU edge probes", "cases": {}}
    for name, (scores, k) in cases.items():
        item = {"n": len(scores), "k": k, "expected": exact_topk(scores, k).tolist()}
        record["cases"][name] = item
        try:
            runner = StaticNPUTopK(len(scores), k)
            item["execution_devices"] = runner.execution_devices
            got = runner.run(scores)
            item["indices"] = got.indices.tolist()
            item["correct"] = bool(np.array_equal(got.indices, item["expected"]))
            item["infer_ms"] = got.infer_ns / 1e6
        except Exception as exc:
            item["correct"] = False
            item["error"] = f"{type(exc).__name__}: {exc}"
    out = Path("results/phase9/static_graph_edge_cases.json")
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
