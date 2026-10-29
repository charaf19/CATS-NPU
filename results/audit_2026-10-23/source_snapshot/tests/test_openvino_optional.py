import importlib.util
import numpy as np
import pytest
from cats_npu.algorithms import MethodRunner
from cats_npu.probe import probe
from cats_npu.reference import exact_topk
from cats_npu.workloads import scores_for


@pytest.mark.skipif(importlib.util.find_spec("openvino") is None, reason="OpenVINO not installed")
def test_ov_cpu_native_and_bitonic():
    x = scores_for(96, 4, "normal", 471, "f16")
    for method in ("ov_cpu_native_hier", "ov_cpu_bitonic_hier"):
        runner = MethodRunner(method, 96, 4, tile=16, batch=2, dtype="f16")
        runner.prepare()
        np.testing.assert_array_equal(runner.run(x).indices, exact_topk(x, 4))


@pytest.mark.skipif(importlib.util.find_spec("openvino") is None, reason="OpenVINO not installed")
def test_ov_cpu_probe():
    result = probe(device="CPU", dtype="f16", operations=("TopK", "Bitonic"))
    assert result["all_passed"], result
