"""Checked FP32 hierarchy using the existing NPU precision-preservation option.

This is a conventional precision-control baseline, not a novel compilation
algorithm. Mathematical correctness assumes exact stable TopK and routing.
The full host certificate makes successful returns exact-or-reject instead.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter_ns

import numpy as np

from .reference import validate_scores
from .static_graph import StaticGraphMismatch, certify_rank


def build_precision_graph(n: int, k: int, width: int = 256):
    import openvino as ov
    import openvino.opset13 as ops

    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           for v in (n, k, width)):
        raise TypeError("N, K and width must be integers")
    if not 1 <= k <= min(n, width):
        raise ValueError("Require 1 <= K <= min(N, width)")
    tiles = math.ceil(n / width)
    actual_width = min(n, width)
    if tiles * actual_width > np.iinfo(np.int32).max:
        raise ValueError("Padded index domain exceeds signed INT32")
    inp = ops.parameter([tiles, actual_width], dtype=np.float16, name="scores")
    promoted = ops.convert(inp, "f32")
    local = ops.topk(promoted, k, 1, "max", "value", index_element_type="i32", stable=True)
    if tiles == 1:
        return ov.Model([local.output(0), local.output(1)], [inp], "precision_direct")
    bases = ops.constant((np.arange(tiles, dtype=np.int32) * width).reshape(tiles, 1))
    ids = ops.add(local.output(1), bases)
    dims = ops.constant(np.array([1, tiles * k], dtype=np.int64))
    candidate_values = ops.reshape(local.output(0), dims, False)
    candidate_ids = ops.reshape(ids, dims, False)
    final = ops.topk(candidate_values, k, 1, "max", "value", index_element_type="i32", stable=True)
    selected_ids = ops.gather(candidate_ids, final.output(1),
                              ops.constant(np.array(1, dtype=np.int32)))
    return ov.Model([final.output(0), selected_ids.output(0)], [inp], "precision_hierarchy")


@dataclass
class PrecisionRun:
    indices: np.ndarray
    values: np.ndarray
    input_preparation_ns: int
    infer_ns: int
    output_and_certificate_ns: int
    end_to_end_ns: int


class PrecisionNPUTopK:
    """Finite FP16 host input to checked host indices/FP32 values, strict NPU.

    No CPU selection, retries, repairs or automatic algorithm fallback occurs.
    The caller must keep the source immutable for the duration of run().
    """

    def __init__(self, n: int, k: int, width: int = 256,
                 precision_layers: str = "TopK,Reshape"):
        import openvino as ov

        self.model = build_precision_graph(n, k, width)
        self.n, self.k, self.width = n, k, min(n, width)
        self.tiles = math.ceil(n / width)
        self.config = {"NPU_COMPILER_TYPE": "PLUGIN"}
        if precision_layers:
            self.config["NPU_COMPILATION_MODE_PARAMS"] = (
                "compute-layers-with-higher-precision=" + precision_layers)
        start = perf_counter_ns()
        self.compiled = ov.Core().compile_model(self.model, "NPU", self.config)
        self.compile_ms = (perf_counter_ns() - start) / 1e6
        devices = self.compiled.get_property("EXECUTION_DEVICES")
        self.execution_devices = [devices] if isinstance(devices, str) else list(devices)
        if self.execution_devices != ["NPU"]:
            raise RuntimeError(f"Strict NPU placement failed: {self.execution_devices}")
        self.request = self.compiled.create_infer_request()
        self.data = np.full((self.tiles, self.width), -np.inf, dtype=np.float16)

    def run(self, scores: np.ndarray) -> PrecisionRun:
        start = perf_counter_ns()
        a = validate_scores(scores, self.k)
        if a.dtype != np.float16 or len(a) != self.n:
            raise ValueError("Require the compiled N and finite FP16 scores")
        self.data.ravel()[:self.n] = a
        prepared = perf_counter_ns()
        self.request.infer({0: self.data})
        inferred = perf_counter_ns()
        values = np.array(self.request.get_output_tensor(0).data, copy=True).reshape(-1)
        ids = np.array(self.request.get_output_tensor(1).data, copy=True).reshape(-1)
        certified = certify_rank(a, ids, values, self.k)
        if not np.array_equal(ids, certified):
            raise StaticGraphMismatch("Device output violates the required stable order")
        end = perf_counter_ns()
        return PrecisionRun(ids, values, prepared - start, inferred - prepared,
                            end - inferred, end - start)
