import itertools
import numpy as np
import pytest

from cats_npu.reference import exact_topk, topk_with_ids, validate_scores


def brute_topk(x, k):
    return np.array(sorted(range(len(x)), key=lambda i: (-float(x[i]), i))[:k])


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_reference_brute_exhaustive_small(dtype):
    for n in range(1, 7):
        for tpl in itertools.product([-1, 0, 1], repeat=n):
            a = np.array(tpl, dtype=dtype)
            for k in range(1, n+1):
                np.testing.assert_array_equal(exact_topk(a, k), brute_topk(a, k))


def test_reference_custom_global_ids():
    scores = np.array([0.4, 0.4, 0.9, 0.4], dtype=np.float32)
    ids = np.array([90, 2, 4, 3])
    found = topk_with_ids(scores, ids, 3)
    np.testing.assert_array_equal(ids[found], [4, 2, 3])


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("k", [1, 10, 100, 4096])
def test_reference_large_boundary_ties_match_full_sort(dtype, k):
    rng = np.random.default_rng(19)
    # Many ties, including at the selection boundary, expose unstable TopK.
    scores = rng.integers(-5, 6, size=4096).astype(dtype)
    np.testing.assert_array_equal(exact_topk(scores, k), brute_topk(scores, k))


@pytest.mark.parametrize("bad", [np.array([0., np.nan]), np.array([np.inf, 1.]),
                                      np.array([]), np.array([[1., 2.]])])
def test_reject_bad_scores(bad):
    with pytest.raises(ValueError):
        validate_scores(bad, 1)


def test_invalid_k():
    with pytest.raises(ValueError):
        exact_topk(np.array([1, 2], dtype=np.float32), 3)
