import numpy as np
import pytest

from cats_npu.reference import exact_topk
from scripts.probe_order_encoded_npu import encode


def test_fp16_encoding_respects_every_representable_finite_score():
    bits = np.arange(65536, dtype=np.uint16)
    scores = bits.view(np.float16)
    scores = scores[np.isfinite(scores)]
    keys, info = encode(scores)
    expected = np.lexsort((np.arange(len(scores)), -scores.astype(np.float64)))
    np.testing.assert_array_equal(np.argsort(-keys.astype(np.int64), kind="stable"), expected)
    assert info["ordered_fp16_span"] > 0


def test_subnormals_zeros_and_boundary_ties_have_unique_stable_keys():
    tiny = np.nextafter(np.float16(0), np.float16(1))
    scores = np.array([tiny, 0, -tiny, -0.0, tiny, tiny * 2], dtype=np.float16)
    keys, _ = encode(scores)
    assert np.unique(keys).size == len(scores)
    np.testing.assert_array_equal(np.argsort(-keys.astype(np.int64)), exact_topk(scores, len(scores)))


def test_span_precondition_fails_closed_at_one_million():
    scores = np.zeros(1_000_000, dtype=np.float16)
    scores[0] = -np.finfo(np.float16).max
    scores[1] = np.finfo(np.float16).max
    with pytest.raises(ValueError, match="exceeds exact int32 key domain"):
        encode(scores)
