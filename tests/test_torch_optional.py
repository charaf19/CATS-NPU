import importlib.util

import numpy as np
import pytest

from cats_npu.algorithms import MethodRunner
from cats_npu.reference import exact_topk
from cats_npu.workloads import scores_for


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
@pytest.mark.parametrize("distribution", ["normal", "tied", "clustered", "adversarial"])
def test_torch_cpu_stable_matches_deterministic_reference(distribution):
    scores = scores_for(513, 10, distribution, 42, "f16")
    runner = MethodRunner("torch_cpu_stable", scores.size, 10)
    result = runner.run(scores)
    np.testing.assert_array_equal(result.indices, exact_topk(scores, 10))
    assert result.certificate in (
        "torch_topk_host_canonical_order",
        "torch_boundary_tie_cpu_repair",
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_torch_stable_all_equal_uses_explicit_cpu_tie_repair():
    scores = np.ones(257, dtype=np.float16)
    runner = MethodRunner("torch_cpu_stable", scores.size, 10)
    result = runner.run(scores)
    np.testing.assert_array_equal(result.indices, np.arange(10))
    assert result.cpu_tie_repairs == 1
    assert result.certificate == "torch_boundary_tie_cpu_repair"
