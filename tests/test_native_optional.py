import numpy as np
import pytest
from cats_npu.algorithms import MethodRunner
from cats_npu.backends import BackendUnavailable
from cats_npu.reference import exact_topk
from cats_npu.workloads import scores_for


def test_cpp_reference_if_built():
    try:
        runner = MethodRunner("cpp_exact", 256, 10)
    except BackendUnavailable:
        pytest.skip("Optional compiled native C++ library unavailable")
    for dtype in ("f16", "f32"):
        for dist in ("normal", "tied", "near_ties", "adversarial"):
            x = scores_for(256, 10, dist, 99, dtype)
            np.testing.assert_array_equal(runner.run(x).indices, exact_topk(x, 10))
    tied = np.ones(257, dtype=np.float32)
    tied[128:] = 2.0
    tied_runner = MethodRunner("cpp_exact", tied.size, 32)
    np.testing.assert_array_equal(tied_runner.run(tied).indices, exact_topk(tied, 32))
