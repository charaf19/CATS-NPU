import itertools

import numpy as np
import pytest

from cats_npu.dyadic_guard import UnsupportedScale, normal_scale
from cats_npu.precision_topk import build_precision_graph
from cats_npu.precision_topk import PrecisionNPUTopK
from cats_npu.static_graph import StaticGraphMismatch
from scripts.probe_topk_numerical_path import check_output


def test_oracle_does_not_invent_stable_tie_contract():
    a = np.array([[1, 1, 0]], dtype=np.float16)
    assert check_output(a, [[1]], np.array([[1]]), 1, False)[0]["passed"]
    assert not check_output(a, [[1]], np.array([[1]]), 1, True)[0]["passed"]
    result = check_output(a, [[0]], np.array([[1]]), 1, False)[0]
    assert result["membership"] and not result["value_match"]
    assert not result["passed"]


def test_oracle_detects_numerical_membership_and_order_failures():
    tiny = np.float16(2**-24)
    a = np.array([[0, tiny, 2 * tiny]], dtype=np.float16)
    assert not check_output(a, [[0]], np.array([[0]]), 1, False)[0]["membership"]
    assert not check_output(a, [[tiny, 2 * tiny]], np.array([[1, 2]]), 2, False)[0]["passed"]
    assert not check_output(a, [[tiny, tiny]], np.array([[1, 1]]), 2, False)[0]["valid_ids"]


def test_scaling_exact_roundtrip_on_every_finite_scalar():
    words = np.arange(65536, dtype=np.uint16)
    finite = words.view(np.float16)
    finite = finite[np.isfinite(finite)]
    for x in finite:
        result, shift = normal_scale(np.array([x], dtype=np.float16))
        assert np.ldexp(float(result[0]), -shift) == float(x)
        assert result[0] == 0 or abs(float(result[0])) >= 2**-14


def test_scaling_rejects_unrepresentable_dynamic_range():
    with pytest.raises(UnsupportedScale):
        normal_scale(np.array([2**-24, 65504], dtype=np.float16))
    scaled, shift = normal_scale(np.array([-0.0, 0.0], dtype=np.float16))
    assert shift == 0
    np.testing.assert_array_equal(scaled.view(np.uint16), [0x8000, 0])


def test_precision_graph_handles_ties_subnormals_padding_and_extremes_on_cpu():
    ov = pytest.importorskip("openvino")
    tiny = np.float16(2**-24)
    for n, k, width in ((2, 1, 2), (5, 2, 4)):
        model = build_precision_graph(n, k, width)
        compiled = ov.Core().compile_model(model, "CPU")
        for raw in itertools.product((-tiny, 0, tiny), repeat=n):
            x = np.asarray(raw, dtype=np.float16)
            data = np.full(((n + width - 1) // width, width), -np.inf, dtype=np.float16)
            data.ravel()[:n] = x
            out = list(compiled([data]).values())
            checks = check_output(x.reshape(1, n), *out, k, True)
            assert checks[0]["passed"]
    for bad in ((0, 1, 4), (4, 5, 4), (True, 1, 4), (2**31 + 1, 1, 256)):
        with pytest.raises((TypeError, ValueError)):
            build_precision_graph(*bad)


def test_runner_rejects_wrong_membership_and_values_without_replacing_device_answer():
    from types import SimpleNamespace

    class Request:
        def __init__(self, values, ids):
            self.outputs = [np.array(values, dtype=np.float32), np.array(ids, dtype=np.int32)]

        def infer(self, inputs):
            assert list(inputs) == [0]

        def get_output_tensor(self, i):
            return SimpleNamespace(data=self.outputs[i])

    runner = PrecisionNPUTopK.__new__(PrecisionNPUTopK)
    runner.n, runner.k = 2, 1
    runner.data = np.empty((1, 2), np.float16)
    source = np.array([0, 2**-24], np.float16)
    for values, ids in (([0], [0]), ([0], [1])):
        runner.request = Request(values, ids)
        with pytest.raises(StaticGraphMismatch):
            runner.run(source)
    runner.request = Request([2**-24], [1])
    np.testing.assert_array_equal(runner.run(source).indices, [1])
