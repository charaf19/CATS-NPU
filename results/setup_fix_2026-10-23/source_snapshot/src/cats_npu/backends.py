"""Select top-ranked elements per fixed-shape tile batch.

An OpenVINO NPU selector is explicitly compiled on 'NPU' (never AUTO/HETERO).
The *adaptive algorithm* still uses the host for routing, bounds and merging;
it is a HYBRID implementation, not an end-to-end NPU-native operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .reference import topk_with_ids


class BackendUnavailable(RuntimeError):
    pass


class TileBackend:
    name: str
    width: int
    batch: int
    dtype: np.dtype
    calls: int

    def prepare(self, ks: set[int]) -> None:
        raise NotImplementedError

    def select_rows(self, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Values and local column indices, each shaped [batch,k]."""
        raise NotImplementedError

    @property
    def device_info(self) -> dict[str, Any]:
        return {}


class NumpyTileBackend(TileBackend):
    def __init__(self, width: int, batch: int, dtype: str = "f16"):
        self.width = width
        self.batch = batch
        self.dtype = _np_dtype(dtype)
        self.name = "numpy_tile_cpu"
        self.calls = 0

    def prepare(self, ks: set[int]) -> None:
        if any(k < 1 or k > self.width for k in ks):
            raise ValueError("Requested K exceeds tile width")

    def select_rows(self, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        _validate_matrix(matrix, self.batch, self.width, k)
        values, indices = [], []
        for row in matrix:
            ids = topk_with_ids(row, np.arange(self.width), k)
            indices.append(ids)
            values.append(row[ids])
        self.calls += 1
        return np.asarray(values), np.asarray(indices, dtype=np.int64)


def _np_dtype(dtype: str) -> np.dtype:
    try:
        return {"f16": np.dtype("float16"), "f32": np.dtype("float32")}[dtype]
    except KeyError as e:
        raise ValueError("OpenVINO/NumPy tile backends accept f16 or f32 only") from e


def _validate_matrix(matrix: np.ndarray, batch: int, width: int, k: int) -> None:
    if matrix.shape != (batch, width):
        raise ValueError(f"Expected matrix {(batch, width)}, got {matrix.shape}")
    if not 1 <= k <= width:
        raise ValueError(f"K must be <= {width}")


def _ov_import():
    try:
        import openvino as ov  # type: ignore
        import openvino.opset13 as ops  # type: ignore
        return ov, ops
    except ImportError as e:
        raise BackendUnavailable(
            'OpenVINO is not installed. Run: python -m pip install -e ".[ov]"'
        ) from e


def _bitonic_model(ov: Any, ops: Any, batch: int, width: int, k: int, dtype: str) -> Any:
    """Static compare-exchange network, experimental fallback for native TopK.

    Complexity O(width * log(width)^2); width MUST be a power of 2, <=256.
    Padded -inf is allowed; ties compare smaller original local index first.
    This graph may fail NPU compilation if Select/Gather is unsupported.
    """
    if width < 2 or width & (width - 1) or width > 256:
        raise BackendUnavailable("Bitonic fallback requires power-of-two width 2..256")
    inp = ops.parameter([batch, width], dtype=_np_dtype(dtype), name="scores")
    values = inp
    idx = ops.broadcast(ops.constant(np.arange(width, dtype=np.int32).reshape(1, width)),
                        ops.constant(np.array([batch, width], dtype=np.int64)))
    # score-desc, index-asc comparator; sorting network is static (no dynamic axes).
    size = 2
    while size <= width:
        stride = size // 2
        while stride >= 1:
            # On Windows, XOR can yield np.intc rather than the np.int32
            # dtype identity accepted by OpenVINO, despite both printing int32.
            perm = (np.arange(width, dtype=np.int32) ^ stride).astype(np.int32)
            partner_v = ops.gather(values, ops.constant(perm), ops.constant(1, dtype=np.int32))
            partner_i = ops.gather(idx, ops.constant(perm), ops.constant(1, dtype=np.int32))
            better = ops.logical_or(
                ops.greater(values, partner_v),
                ops.logical_and(ops.equal(values, partner_v), ops.less(idx, partner_i)),
            )
            # Bitonic merge: pairs within a sub-sequence choose opposite orientations.
            i = np.arange(width)
            choose_better = ((i & size) == 0) == ((i & stride) == 0)
            mask = ops.constant(choose_better.reshape(1, width).astype(bool))
            choose_self = ops.select(mask, better, ops.logical_not(better))
            next_v = ops.select(choose_self, values, partner_v)
            next_i = ops.select(choose_self, idx, partner_i)
            values, idx = next_v, next_i
            stride //= 2
        size *= 2
    # Fixed output dimension, gather first K from sorted rows.
    cut = ops.constant(np.arange(k, dtype=np.int32))
    out_v = ops.gather(values, cut, ops.constant(1, dtype=np.int32))
    out_i = ops.gather(idx, cut, ops.constant(1, dtype=np.int32))
    return ov.Model([out_v, out_i], [inp], f"bitonic_B{batch}_W{width}_K{k}")


def _native_model(ov: Any, ops: Any, batch: int, width: int, k: int, dtype: str) -> Any:
    inp = ops.parameter([batch, width], dtype=_np_dtype(dtype), name="scores")
    top = ops.topk(inp, k, axis=1, mode="max", sort="value", index_element_type="i32")
    return ov.Model([top.output(0), top.output(1)], [inp], f"topk_B{batch}_W{width}_K{k}")


@dataclass
class ModelRuntime:
    model: Any
    request: Any
    reported_execution_devices: list[str]


class OpenVINOTileBackend(TileBackend):
    """One strict OpenVINO device backend with precompiled static configurations."""
    def __init__(self, width: int, batch: int, device: str, dtype: str = "f16",
                 kernel: str = "native", core: Any = None):
        ov, ops = _ov_import()
        self.ov, self.ops = ov, ops
        self.core = core if core is not None else ov.Core()
        self.width, self.batch = width, batch
        self.device = device.upper()
        if self.device.startswith("AUTO") or self.device.startswith("HETERO"):
            raise BackendUnavailable("AUTO/HETERO explicitly prohibited for execution proof")
        self.dtype = _np_dtype(dtype)
        self.dtype_tag = dtype
        if kernel not in ("native", "bitonic"):
            raise ValueError("kernel must be native or bitonic")
        self.kernel = kernel
        self.name = f"ov_{self.device.lower()}_{kernel}"
        self.calls = 0
        self._compiled: dict[int, ModelRuntime] = {}
        devices = list(self.core.available_devices)
        if self.device not in [d.upper() for d in devices]:
            raise BackendUnavailable(f"Device {self.device} unavailable. OpenVINO devices: {devices}")

    def prepare(self, ks: set[int]) -> None:
        for k in sorted(ks):
            if k in self._compiled:
                continue
            if not 1 <= k <= self.width:
                raise ValueError(f"K {k} out of bounds")
            try:
                if self.kernel == "native":
                    model = _native_model(self.ov, self.ops, self.batch, self.width, k, self.dtype_tag)
                else:
                    model = _bitonic_model(self.ov, self.ops, self.batch, self.width, k, self.dtype_tag)
                compiled = self.core.compile_model(model, self.device)
                request = compiled.create_infer_request()
                try:
                    devs = list(compiled.get_property("EXECUTION_DEVICES"))
                except Exception:
                    devs = []  # Unknown, not evidence of CPU/NPU placement.
                self._compiled[k] = ModelRuntime(compiled, request, devs)
            except BackendUnavailable:
                raise
            except Exception as e:
                raise BackendUnavailable(
                    f"OpenVINO compile failed on {self.device} "
                    f"(kernel={self.kernel}, B={self.batch}, W={self.width}, K={k}, "
                    f"dtype={self.dtype_tag}): {type(e).__name__}: {e}"
                ) from e

    def select_rows(self, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        _validate_matrix(matrix, self.batch, self.width, k)
        if k not in self._compiled:
            raise RuntimeError(f"K={k} not prepared; precompile outside timing")
        rt = self._compiled[k]
        x = np.ascontiguousarray(matrix, dtype=self.dtype)
        rt.request.infer({0: x})
        values = np.array(rt.request.get_output_tensor(0).data, copy=True)
        indices = np.array(rt.request.get_output_tensor(1).data, copy=True, dtype=np.int64)
        if values.shape != (self.batch, k) or indices.shape != (self.batch, k):
            raise RuntimeError("Unexpected OpenVINO output shape")
        if np.any(indices < 0) or np.any(indices >= self.width):
            raise RuntimeError("Invalid device index output")
        # Device output must reflect host input to within representable precision.
        # No implicit correction or CPU fallbacks for incorrect device results.
        expected = np.take_along_axis(x, indices, axis=1)
        if not np.array_equal(expected, values, equal_nan=False):
            raise RuntimeError("Device Top-K scores disagree with values at its returned indices")
        self.calls += 1
        return values, indices

    @property
    def device_info(self) -> dict[str, Any]:
        return {
            "requested": self.device,
            "kernel": self.kernel,
            "reported_execution_devices": {str(k): v.reported_execution_devices
                                           for k, v in self._compiled.items()},
            "prepared_k": sorted(self._compiled),
        }


def make_backend(method: str, width: int, batch: int, dtype: str) -> TileBackend:
    if method == "numpy":
        return NumpyTileBackend(width, batch, dtype)
    if not method.startswith("ov_"):
        raise ValueError(f"Unknown tile backend method: {method}")
    _, device, kernel = method.split("_", maxsplit=2)
    return OpenVINOTileBackend(width, batch, device, dtype, kernel)
