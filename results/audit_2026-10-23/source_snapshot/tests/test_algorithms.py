import numpy as np
import pytest
from cats_npu.algorithms import MethodRunner
from cats_npu.reference import exact_topk
from cats_npu.workloads import scores_for


@pytest.mark.parametrize("method", ["numpy_hier", "numpy_adaptive"])
@pytest.mark.parametrize("distribution", ["uniform", "normal", "clustered", "tied", "near_ties", "adversarial"])
@pytest.mark.parametrize("n,k,tile,batch", [(3, 1, 2, 2), (10, 4, 4, 2), (33, 10, 8, 4),
                                              (64, 1, 16, 2), (64, 64, 8, 3),
                                              (100, 15, 32, 4), (512, 10, 64, 8)])
def test_all_workload_cases(method, distribution, n, k, tile, batch):
    scores = scores_for(n, k, distribution, seed=1205, dtype="f16")
    r = MethodRunner(method, n, k, tile, batch, dtype="f16")
    r.prepare()
    got = r.run(scores)
    np.testing.assert_array_equal(got.indices, exact_topk(scores, k))
    assert got.certificate in ("local_k_exact", "strict_omission_bound")
    assert got.backend_calls >= 1


@pytest.mark.parametrize("method", ["numpy_hier", "numpy_adaptive"])
def test_adversarial_many_equal_scores(method):
    x = np.ones(513, dtype=np.float16)
    for k in [1, 2, 4, 10, 100, 512]:
        runner = MethodRunner(method, x.size, k, tile=64, batch=4)
        runner.prepare()
        res = runner.run(x)
        np.testing.assert_array_equal(res.indices, np.arange(k))
        if k < 64:
            assert res.cpu_tie_repairs >= 1


def test_unequal_tile_length_and_boundary():
    x = np.array([0, 0, 0, 0, 2, 2, 2, 2, 0, 1, 1], dtype=np.float32)
    for k in range(1, len(x) + 1):
        for method in ("numpy_hier", "numpy_adaptive"):
            runner = MethodRunner(method, x.size, k, tile=4, batch=2, dtype="f32")
            runner.prepare()
            np.testing.assert_array_equal(runner.run(x).indices, exact_topk(x, k))


def test_no_silent_precision_cast():
    x = np.array([0.1, 0.9, 0.3], dtype=np.float32)
    runner = MethodRunner("numpy_adaptive", 3, 2, tile=2, batch=1, dtype="f16")
    runner.prepare()
    with pytest.raises(ValueError, match="dtype"):
        runner.run(x)


def test_repeatability():
    x = scores_for(1024, 10, "tied", 42)
    runner = MethodRunner("numpy_adaptive", x.size, 10, tile=128, batch=8)
    runner.prepare()
    a = runner.run(x)
    b = runner.run(x)
    np.testing.assert_array_equal(a.indices, b.indices)


def test_early_certificate_occurs_on_easy_input():
    x = np.full(1024, -0.5, dtype=np.float16)
    x[::128] = np.arange(8, dtype=np.float16) / 8
    runner = MethodRunner("numpy_adaptive", x.size, 1, tile=128, batch=4)
    runner.prepare()
    res = runner.run(x)
    assert res.indices.tolist() == [896]
