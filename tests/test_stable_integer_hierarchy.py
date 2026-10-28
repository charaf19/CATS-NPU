import numpy as np
import pytest

from cats_npu.order_encoding import score_codes
from cats_npu.reference import exact_topk
from scripts.probe_stable_integer_hierarchy import graph


@pytest.mark.parametrize("k", [1, 4, 10, 100])
def test_stable_integer_graph_cpu_handles_subnormals_and_ties(k):
    ov = pytest.importorskip("openvino")
    tiny = np.nextafter(np.float16(0), np.float16(1))
    values = np.array([-tiny, -0.0, 0, tiny, 2 * tiny, 1, 1], dtype=np.float16)
    scores = np.resize(values, 513)
    width = 256
    data = np.zeros(((len(scores) + width - 1) // width) * width, dtype=np.int32)
    data[:len(scores)] = score_codes(scores)
    compiled = ov.Core().compile_model(graph(len(scores), k, width), "CPU")
    output_values, output_ids = list(compiled([data.reshape(-1, width)]).values())
    ids = np.asarray(output_ids).reshape(-1).astype(np.int64)
    np.testing.assert_array_equal(np.asarray(output_values).reshape(-1), score_codes(scores)[ids])
    ids = ids[np.lexsort((ids, -scores[ids].astype(np.float64)))]
    np.testing.assert_array_equal(ids, exact_topk(scores, k))


@pytest.mark.parametrize("n,k,width", [(256, 10, 256), (513, 0, 256),
                                      (513, 257, 256), (513, 10, 0),
                                      (2**31 - 1, 10, 256)])
def test_stable_integer_graph_rejects_unsupported_sizes_before_allocation(n, k, width):
    with pytest.raises(ValueError):
        graph(n, k, width)


def test_stable_integer_graph_rejects_boolean_shape():
    with pytest.raises(TypeError):
        graph(True, 1, 1)
