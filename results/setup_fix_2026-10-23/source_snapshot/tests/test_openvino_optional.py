import importlib.util
import numpy as np
import pytest
from cats_npu.algorithms import MethodRunner
from cats_npu.backends import OpenVINOTileBackend
from cats_npu.probe import probe
from cats_npu.reference import exact_topk, topk_with_ids
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


@pytest.mark.skipif(importlib.util.find_spec("openvino") is None, reason="OpenVINO not installed")
@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("width", [2, 16])
def test_bitonic_rows_stable_with_padding(dtype, width):
    """Check device graph outputs directly, without host tie repair or merging."""
    backend = OpenVINOTileBackend(width, 3, "CPU", dtype, kernel="bitonic")
    ks = {1, width // 2, width}
    backend.prepare(ks)
    matrix = np.zeros((3, width), dtype=backend.dtype)
    matrix[0] = (np.arange(width) % 3) - 1
    # Row 1 is all tied; row 2 contains valid values plus internal padding.
    matrix[2] = -np.inf
    matrix[2, :width // 2] = np.arange(width // 2)
    local_ids = np.arange(width)
    for k in sorted(ks):
        values, indices = backend.select_rows(matrix, k)
        for row, found_values, found_indices in zip(matrix, values, indices):
            expected = topk_with_ids(row, local_ids, k)
            np.testing.assert_array_equal(found_indices, expected)
            np.testing.assert_array_equal(found_values, row[expected])
