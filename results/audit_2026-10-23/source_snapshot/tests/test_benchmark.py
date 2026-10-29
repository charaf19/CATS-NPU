from pathlib import Path
import json
from cats_npu.benchmark import bench
from cats_npu.statistics import read_observations, analyze


def test_benchmark_pipeline(tmp_path: Path):
    summary = bench(methods=["cpu_exact", "numpy_hier", "numpy_adaptive"],
                    ns=[65], ks=[3], distributions=["tied"], dtype="f16",
                    seed=3, tile=16, batch=2, initial_budget=1,
                    warmup=1, repeats=3, blocks=2, out=tmp_path,
                    session_id="unit_test", allow_skips=False)
    assert summary["observations"] == 18
    assert summary["wrong_results"] == 0
    rows = read_observations([tmp_path])
    assert len(rows) == 18
    assert all(r["correct"] for r in rows)
    assert all("backend_ns" in r for r in rows)
    assert analyze(rows)["cases"]
