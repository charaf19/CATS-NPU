"""Exploratory one-submission, static-shape NPU Top-K.

The graph does local and final selection. A host rank certificate checks the
result and fails closed; it never repairs a device result with CPU selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
import math

import numpy as np

from .reference import validate_scores


class StaticGraphMismatch(RuntimeError):
    """The physical device result failed the exact stable rank certificate."""


def certify_rank(scores: np.ndarray, ids: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Return sorted IDs iff these K device candidates are the exact stable Top-K.

    A candidate's score is checked against the original input. Counting all
    pairs better than the selected Kth pair proves the complete set is exact.
    """
    a = validate_scores(scores, k)
    chosen = np.asarray(ids, dtype=np.int64).reshape(-1)
    returned = np.asarray(values).reshape(-1)
    if chosen.size != k or returned.size != k:
        raise StaticGraphMismatch("Wrong number of NPU outputs")
    if np.any(chosen < 0) or np.any(chosen >= a.size) or np.unique(chosen).size != k:
        raise StaticGraphMismatch("NPU returned invalid or duplicate IDs")
    if not np.array_equal(a[chosen], returned):
        raise StaticGraphMismatch("NPU returned scores inconsistent with host input")
    chosen = chosen[np.lexsort((chosen, -a[chosen].astype(np.float64)))]
    boundary_id = int(chosen[-1])
    threshold = a[boundary_id]
    better = int(np.count_nonzero(a > threshold))
    better += int(np.count_nonzero(a[:boundary_id] == threshold))
    if better != k - 1:
        raise StaticGraphMismatch(f"Boundary rank {better + 1} differs from K={k}")
    return chosen


def build_graph(n: int, k: int, width: int):
    import openvino as ov
    import openvino.opset13 as ops

    tiles = math.ceil(n / width)
    if n <= width or not 1 <= k <= width or tiles * k < k:
        raise ValueError("Require N > width and 1 <= K <= width")
    inp = ops.parameter([tiles, width], dtype=np.float16, name="scores")
    local = ops.topk(inp, k, axis=1, mode="max", sort="value", index_element_type="i32")
    bases = ops.constant((np.arange(tiles, dtype=np.int32) * width).reshape(tiles, 1))
    global_ids = ops.add(local.output(1), bases)
    shape = ops.constant(np.array([1, tiles * k], dtype=np.int64))
    candidates = ops.reshape(local.output(0), shape, False)
    candidate_ids = ops.reshape(global_ids, shape, False)
    final = ops.topk(candidates, k, axis=1, mode="max", sort="value", index_element_type="i32")
    selected_ids = ops.gather(candidate_ids, final.output(1),
                              ops.constant(np.array(1, dtype=np.int32)))
    return ov.Model([final.output(0), selected_ids.output(0)], [inp],
                    f"static_twolevel_N{n}_K{k}_W{width}")


@dataclass
class StaticRun:
    indices: np.ndarray
    input_preparation_ns: int
    infer_ns: int
    output_copy_ns: int
    certificate_ns: int
    staged_input_bytes: int
    retrieved_output_bytes: int


class StaticNPUTopK:
    def __init__(self, n: int, k: int, width: int = 256):
        import openvino as ov

        self.n, self.k, self.width = n, k, width
        self.tiles = math.ceil(n / width)
        self.core = ov.Core()
        start = perf_counter_ns()
        self.compiled = self.core.compile_model(build_graph(n, k, width), "NPU",
                                                {"NPU_COMPILER_TYPE": "PLUGIN"})
        self.compile_ms = (perf_counter_ns() - start) / 1e6
        devices = self.compiled.get_property("EXECUTION_DEVICES")
        self.execution_devices = [devices] if isinstance(devices, str) else list(devices)
        if self.execution_devices != ["NPU"]:
            raise RuntimeError(f"Strict NPU execution not established: {self.execution_devices}")
        self.request = self.compiled.create_infer_request()

    def run(self, scores: np.ndarray) -> StaticRun:
        a = validate_scores(scores, self.k)
        if a.size != self.n or a.dtype != np.float16:
            raise ValueError("Static graph requires its compiled N and finite FP16 scores")
        start = perf_counter_ns()
        data = np.full((self.tiles * self.width,), -np.inf, dtype=np.float16)
        data[:self.n] = a
        data = data.reshape(self.tiles, self.width)
        prepared = perf_counter_ns()
        self.request.infer({0: data})
        inferred = perf_counter_ns()
        values = np.array(self.request.get_output_tensor(0).data, copy=True).reshape(-1)
        ids = np.array(self.request.get_output_tensor(1).data, copy=True).reshape(-1)
        copied = perf_counter_ns()
        result = certify_rank(a, ids, values, self.k)
        certified = perf_counter_ns()
        return StaticRun(result, prepared - start, inferred - prepared,
                         copied - inferred, certified - copied, data.nbytes,
                         values.nbytes + ids.nbytes)
