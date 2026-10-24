"""Select top-ranked elements per fixed-shape tile batch.

An OpenVINO NPU selector is explicitly compiled on 'NPU' (never AUTO/HETERO).
The *adaptive algorithm* still uses the host for routing, bounds and merging;
it is a HYBRID implementation, not an end-to-end NPU-native operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from time import perf_counter_ns
import numpy as np

from .reference import topk_with_ids


NPU_COMPILER_TYPES = ("DEFAULT", "PREFER_PLUGIN", "PLUGIN", "DRIVER")


class BackendUnavailable(RuntimeError):
    pass


class TileBackend:
    name: str
    width: int
    batch: int
    dtype: np.dtype
    calls: int
    last_profile: dict[str, int]

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
        self.last_profile = {}

    def prepare(self, ks: set[int]) -> None:
        if any(k < 1 or k > self.width for k in ks):
            raise ValueError("Requested K exceeds tile width")

    def select_rows(self, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        _validate_matrix(matrix, self.batch, self.width, k)
        t0 = perf_counter_ns()
        values, indices = [], []
        for row in matrix:
            ids = topk_with_ids(row, np.arange(self.width), k)
            indices.append(ids)
            values.append(row[ids])
        self.calls += 1
        self.last_profile = {
            "input_cast_ns": 0,
            "execute_sync_ns": perf_counter_ns() - t0,
            "output_copy_ns": 0,
            "validation_ns": 0,
            "staged_input_bytes": 0,
            "retrieved_output_bytes": 0,
        }
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


def _npu_cpu_exact_validation_rows(matrix: np.ndarray) -> np.ndarray:
    """Return rows that need measured CPU verification for exact NPU use."""
    if matrix.dtype == np.dtype("float32"):
        raise BackendUnavailable(
            "Strict exact NPU selection rejects FP32: reproduced one-ULP rankings "
            "are incorrect on this device/compiler"
        )
    if matrix.dtype == np.dtype("float16"):
        magnitude = np.abs(matrix)
        subnormal = (magnitude > 0) & (magnitude < np.finfo(np.float16).tiny)
        return np.any(subnormal, axis=1)
    return np.zeros(matrix.shape[0], dtype=bool)


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
            mask_i32 = ops.constant(choose_better.reshape(1, width).astype(np.int32))
            # The NPU driver rejects i8/bool as Select data and as Equal data.
            # Convert the dynamic comparator to a supported integer type, then
            # use Equal only to create the final boolean Select condition.
            better_i32 = ops.convert(better, "i32")
            choose_self = ops.equal(mask_i32, better_i32)
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
    reported_compiler_type: str | None


class OpenVINOTileBackend(TileBackend):
    """One strict OpenVINO device backend with precompiled static configurations."""
    def __init__(self, width: int, batch: int, device: str, dtype: str = "f16",
                 kernel: str = "native", core: Any = None,
                 compiler_type: str = "DEFAULT"):
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
        self.compiler_type = compiler_type.upper()
        if self.compiler_type not in NPU_COMPILER_TYPES:
            raise ValueError(f"compiler_type must be one of {NPU_COMPILER_TYPES}")
        if self.device != "NPU" and self.compiler_type != "DEFAULT":
            raise ValueError("NPU compiler type can only be selected for device NPU")
        self.compile_config = (
            {} if self.compiler_type == "DEFAULT"
            else {"NPU_COMPILER_TYPE": self.compiler_type}
        )
        self.name = f"ov_{self.device.lower()}_{kernel}"
        self.calls = 0
        self.last_profile = {}
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
                compiled = self.core.compile_model(model, self.device, self.compile_config)
                request = compiled.create_infer_request()
                try:
                    raw_devs = compiled.get_property("EXECUTION_DEVICES")
                    devs = [raw_devs] if isinstance(raw_devs, str) else list(raw_devs)
                except Exception:
                    devs = []  # Unknown, not evidence of CPU/NPU placement.
                compiler_used = None
                if self.device == "NPU":
                    try:
                        compiler_used = str(compiled.get_property("NPU_COMPILER_TYPE"))
                    except Exception:
                        pass
                self._compiled[k] = ModelRuntime(compiled, request, devs, compiler_used)
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
        cpu_exact_rows = (
            _npu_cpu_exact_validation_rows(matrix)
            if self.device == "NPU"
            else np.zeros(self.batch, dtype=bool)
        )
        if k not in self._compiled:
            raise RuntimeError(f"K={k} not prepared; precompile outside timing")
        rt = self._compiled[k]
        t0 = perf_counter_ns()
        x = np.ascontiguousarray(matrix, dtype=self.dtype)
        input_cast_ns = perf_counter_ns() - t0
        t0 = perf_counter_ns()
        rt.request.infer({0: x})
        execute_sync_ns = perf_counter_ns() - t0
        t0 = perf_counter_ns()
        values = np.array(rt.request.get_output_tensor(0).data, copy=True)
        raw_indices = np.array(rt.request.get_output_tensor(1).data, copy=True)
        indices = raw_indices.astype(np.int64, copy=False)
        output_copy_ns = perf_counter_ns() - t0
        t0 = perf_counter_ns()
        if values.shape != (self.batch, k) or indices.shape != (self.batch, k):
            raise RuntimeError("Unexpected OpenVINO output shape")
        if np.any(indices < 0) or np.any(indices >= self.width):
            raise RuntimeError("Invalid device index output")
        # Device output must reflect host input to within representable precision.
        # No implicit correction or CPU fallbacks for incorrect device results.
        expected = np.take_along_axis(x, indices, axis=1)
        if not np.array_equal(expected, values, equal_nan=False):
            self.last_profile = {
                "input_cast_ns": input_cast_ns,
                "execute_sync_ns": execute_sync_ns,
                "output_copy_ns": output_copy_ns,
                "validation_ns": perf_counter_ns() - t0,
                "staged_input_bytes": int(x.nbytes),
                "retrieved_output_bytes": int(values.nbytes + raw_indices.nbytes),
            }
            raise RuntimeError("Device Top-K scores disagree with values at its returned indices")
        # The measured NPU stack can mis-rank FP16 subnormals while returning
        # values that still match its chosen indices. Verify only affected rows
        # against the exact host ordering. This is validation, not correction;
        # a disagreement rejects the device result.
        exact_row_count = int(np.count_nonzero(cpu_exact_rows))
        local_ids = np.arange(self.width, dtype=np.int64)
        for row_no in np.flatnonzero(cpu_exact_rows):
            exact_indices = topk_with_ids(x[row_no], local_ids, k)
            if not np.array_equal(indices[row_no], exact_indices):
                self.last_profile = {
                    "input_cast_ns": input_cast_ns,
                    "execute_sync_ns": execute_sync_ns,
                    "output_copy_ns": output_copy_ns,
                    "validation_ns": perf_counter_ns() - t0,
                    "cpu_exact_validation_rows": exact_row_count,
                    "staged_input_bytes": int(x.nbytes),
                    "retrieved_output_bytes": int(values.nbytes + raw_indices.nbytes),
                }
                raise RuntimeError(
                    "Device Top-K indices disagree with exact ordering on an "
                    "FP16-subnormal row"
                )
        validation_ns = perf_counter_ns() - t0
        self.calls += 1
        self.last_profile = {
            "input_cast_ns": input_cast_ns,
            # Synchronous infer includes submission, device transfers,
            # execution, and synchronization; it is not a hardware-only timer.
            "execute_sync_ns": execute_sync_ns,
            "output_copy_ns": output_copy_ns,
            "validation_ns": validation_ns,
            "cpu_exact_validation_rows": exact_row_count,
            "staged_input_bytes": int(x.nbytes),
            "retrieved_output_bytes": int(values.nbytes + raw_indices.nbytes),
        }
        return values, indices

    @property
    def device_info(self) -> dict[str, Any]:
        return {
            "requested": self.device,
            "kernel": self.kernel,
            "compiler_type_requested": self.compiler_type,
            "reported_execution_devices": {str(k): v.reported_execution_devices
                                           for k, v in self._compiled.items()},
            "reported_compiler_type": {str(k): v.reported_compiler_type
                                       for k, v in self._compiled.items()},
            "prepared_k": sorted(self._compiled),
        }


def make_backend(method: str, width: int, batch: int, dtype: str,
                 compiler_type: str = "DEFAULT") -> TileBackend:
    if method == "numpy":
        return NumpyTileBackend(width, batch, dtype)
    if not method.startswith("ov_"):
        raise ValueError(f"Unknown tile backend method: {method}")
    _, device, kernel = method.split("_", maxsplit=2)
    return OpenVINOTileBackend(width, batch, device, dtype, kernel,
                               compiler_type=compiler_type)
