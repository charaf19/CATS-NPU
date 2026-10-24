import importlib.util
import os
import numpy as np
import pytest
from cats_npu.algorithms import MethodRunner
from cats_npu.backends import (
    BackendUnavailable,
    OpenVINOTileBackend,
    _npu_cpu_exact_validation_rows,
)
from cats_npu.probe import _execution_devices, probe
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


def test_execution_devices_normalizes_scalar_and_sequence_forms():
    assert _execution_devices("NPU") == ["NPU"]
    assert _execution_devices(["CPU"]) == ["CPU"]


def test_strict_npu_numeric_policy_marks_subnormal_rows_and_rejects_f32():
    rows = np.array([
        [0.0, 1.0],
        [np.nextafter(np.float16(0), np.float16(1)), 1.0],
    ], dtype=np.float16)
    np.testing.assert_array_equal(_npu_cpu_exact_validation_rows(rows), [False, True])
    with pytest.raises(BackendUnavailable, match="FP32"):
        _npu_cpu_exact_validation_rows(np.array([[0.0, 1.0]], dtype=np.float32))


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
        for row, found_values, found_indices in zip(matrix, values, indices, strict=True):
            expected = topk_with_ids(row, local_ids, k)
            np.testing.assert_array_equal(found_indices, expected)
            np.testing.assert_array_equal(found_values, row[expected])


@pytest.mark.skipif(importlib.util.find_spec("openvino") is None, reason="OpenVINO not installed")
def test_openvino_cpu_sequential_and_batched_aliases_are_exact():
    x = scores_for(65, 10, "normal", 42, "f16")
    sequential = MethodRunner("ov_cpu_native_sequential_hier", 65, 10,
                              tile=16, batch=8, dtype="f16")
    batched = MethodRunner("ov_cpu_native_batched_hier", 65, 10,
                           tile=16, batch=8, dtype="f16")
    sequential.prepare()
    batched.prepare()
    seq_result = sequential.run(x)
    batch_result = batched.run(x)
    expected = exact_topk(x, 10)
    np.testing.assert_array_equal(seq_result.indices, expected)
    np.testing.assert_array_equal(batch_result.indices, expected)
    assert seq_result.backend_calls == 5
    assert batch_result.backend_calls == 1


@pytest.mark.skipif(os.environ.get("CATS_NPU_RUN_NPU_TESTS") != "1",
                    reason="set CATS_NPU_RUN_NPU_TESTS=1 for physical NPU regressions")
def test_npu_native_subnormal_never_accepts_wrong_ranking():
    tiny = np.nextafter(np.float16(0), np.float16(1))
    row = ((np.arange(16) - 8) * tiny).astype(np.float16).reshape(1, 16)
    backend = OpenVINOTileBackend(16, 1, "NPU", "f16", kernel="native",
                                  compiler_type="PLUGIN")
    backend.prepare({4})
    try:
        values, indices = backend.select_rows(row, 4)
    except RuntimeError as error:
        assert "exact ordering" in str(error) or "scores disagree" in str(error)
        return
    expected = topk_with_ids(row[0], np.arange(16), 4)
    np.testing.assert_array_equal(indices[0], expected)
    np.testing.assert_array_equal(values[0], row[0, expected])


@pytest.mark.skipif(os.environ.get("CATS_NPU_RUN_NPU_TESTS") != "1",
                    reason="set CATS_NPU_RUN_NPU_TESTS=1 for physical NPU regressions")
@pytest.mark.xfail(
    strict=True,
    reason="NPU PLUGIN changes a selected FP16 subnormal; strict validation rejects it",
)
def test_npu_bitonic_preserves_fp16_subnormal_score_index_pairs():
    row = np.array([
        0.13037109375, -0.11285400390625, -0.0401611328125, -0.412353515625,
        -0.07147216796875, -0.07989501953125, 0.06964111328125, -0.1666259765625,
        0.1414794921875, 0.3037109375, 0.046630859375, 0.10552978515625,
        0.0159454345703125, 2.5331974029541016e-05, -0.2164306640625,
        0.094970703125,
    ], dtype=np.float16).reshape(1, 16)
    backend = OpenVINOTileBackend(16, 1, "NPU", "f16", kernel="bitonic",
                                  compiler_type="PLUGIN")
    backend.prepare({11})
    values, indices = backend.select_rows(row, 11)
    expected = topk_with_ids(row[0], np.arange(16), 11)
    np.testing.assert_array_equal(indices[0], expected)
    np.testing.assert_array_equal(values[0], row[0, expected])


@pytest.mark.skipif(os.environ.get("CATS_NPU_RUN_NPU_TESTS") != "1",
                    reason="set CATS_NPU_RUN_NPU_TESTS=1 for physical NPU regressions")
@pytest.mark.xfail(strict=True, reason="NPU DRIVER bitonic returns repeated incorrect indices")
def test_npu_driver_bitonic_returns_exact_indices():
    row = np.arange(16, dtype=np.float16).reshape(1, 16)
    backend = OpenVINOTileBackend(16, 1, "NPU", "f16", kernel="bitonic",
                                  compiler_type="DRIVER")
    backend.prepare({4})
    values, indices = backend.select_rows(row, 4)
    expected = np.array([15, 14, 13, 12], dtype=np.int64)
    np.testing.assert_array_equal(indices[0], expected)
    np.testing.assert_array_equal(values[0], row[0, expected])
