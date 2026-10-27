import numpy as np
import pytest

from cats_npu.reference import exact_topk
from cats_npu.order_encoding import score_codes
from scripts.probe_split_key_npu import graph


def test_score_code_preserves_all_finite_fp16_values():
    bits = np.arange(65536, dtype=np.uint16)
    scores = bits.view(np.float16)
    scores = scores[np.isfinite(scores)]
    codes = score_codes(scores)
    order = np.lexsort((np.arange(len(scores)), -codes.astype(np.int64)))
    np.testing.assert_array_equal(order[:100], exact_topk(scores, 100))
    assert int(codes.min()) > 0  # zero is reserved for padding


@pytest.mark.parametrize("k", [1, 10, 100])
def test_split_key_graph_cpu_finite_edge_distribution(k):
    ov = pytest.importorskip("openvino")
    rng = np.random.default_rng(31)
    scores = rng.choice(np.array([-65504, -1, -0.0, 0.0, 1, 65504], dtype=np.float16),
                        size=1025)
    scores[:4] = [np.nextafter(np.float16(0), np.float16(1)),
                  -np.nextafter(np.float16(0), np.float16(1)), 65504, 65504]
    width = 256
    data = np.zeros(((len(scores) + width - 1) // width) * width, dtype=np.int32)
    data[:len(scores)] = score_codes(scores)
    compiled = ov.Core().compile_model(graph(len(scores), k, width), "CPU")
    values, ids = list(compiled([data.reshape(-1, width)]).values())
    got = np.asarray(ids).reshape(-1).astype(np.int64)
    np.testing.assert_array_equal(np.asarray(values).reshape(-1), score_codes(scores)[got])
    got = got[np.lexsort((got, -scores[got].astype(np.float64)))]
    np.testing.assert_array_equal(got, exact_topk(scores, k))


def test_composite_keys_fit_exact_float32_integer_range():
    # Full finite-FP16 score codes are 1024..64511 after zero normalization.
    assert 64511 * (256 + 1) + 255 <= 2**24
    assert 3 * 5_592_404 + 2 <= 2**24
