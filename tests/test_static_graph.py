import itertools

import numpy as np
import pytest

from cats_npu.reference import exact_topk
from cats_npu.static_graph import StaticGraphMismatch, certify_rank


def test_rank_certificate_exhaustive_ties():
    for n in range(2, 7):
        for raw in itertools.product([-1, 0, 1], repeat=n):
            scores = np.asarray(raw, dtype=np.float16)
            for k in range(1, n + 1):
                correct = exact_topk(scores, k)
                np.testing.assert_array_equal(
                    certify_rank(scores, correct, scores[correct], k), correct)
                if k < n:
                    incorrect = correct.copy()
                    incorrect[-1] = next(i for i in range(n) if i not in correct)
                    if set(incorrect) != set(correct):
                        with pytest.raises(StaticGraphMismatch):
                            certify_rank(scores, incorrect, scores[incorrect], k)


def test_rank_certificate_catches_subnormal_misrank_and_invalid_output():
    tiny = np.nextafter(np.float16(0), np.float16(1))
    scores = np.array([tiny, 0, tiny * 2, -tiny], dtype=np.float16)
    np.testing.assert_array_equal(certify_rank(scores, [2, 0], scores[[2, 0]], 2), [2, 0])
    with pytest.raises(StaticGraphMismatch):
        certify_rank(scores, [2, 1], scores[[2, 1]], 2)
    with pytest.raises(StaticGraphMismatch):
        certify_rank(scores, [2, 2], scores[[2, 2]], 2)
    with pytest.raises(StaticGraphMismatch):
        certify_rank(scores, [2, 0], [float(scores[2]), 0], 2)


def test_rank_certificate_accepts_exactly_the_correct_subsets_on_bounded_domain():
    tiny = np.nextafter(np.float16(0), np.float16(1))
    alphabet = [-tiny, np.float16(-0.0), np.float16(0.0), tiny]
    for n in range(1, 5):
        for raw in itertools.product(alphabet, repeat=n):
            scores = np.asarray(raw, dtype=np.float16)
            for k in range(1, n + 1):
                expected = np.asarray(sorted(range(n), key=lambda i: (-float(scores[i]), i))[:k])
                for subset in itertools.combinations(range(n), k):
                    candidate = np.asarray(subset[::-1], dtype=np.int64)
                    correct_membership = set(candidate) == set(expected)
                    if correct_membership:
                        np.testing.assert_array_equal(
                            certify_rank(scores, candidate, scores[candidate], k), expected)
                    else:
                        with pytest.raises(StaticGraphMismatch):
                            certify_rank(scores, candidate, scores[candidate], k)
