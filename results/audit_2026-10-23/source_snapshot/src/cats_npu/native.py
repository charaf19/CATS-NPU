"""Ctypes wrapper for optional C++ std::nth_element reference baseline."""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
import sys
import numpy as np
from .backends import BackendUnavailable


def _find_library() -> Path:
    override = os.environ.get("CATS_NPU_NATIVE_LIB")
    if override:
        path = Path(override)
        if not path.is_file():
            raise BackendUnavailable(f"CATS_NPU_NATIVE_LIB points to missing file: {path}")
        return path
    # Source checkout layout; wheel installations can set env var instead.
    root = Path(__file__).resolve().parents[2]
    patterns = (
        "native/build/Release/cats_topk.dll",
        "native/build/cats_topk.dll",
        "native/build/libcats_topk.so",
        "native/build/libcats_topk.dylib",
        "native/build/**/libcats_topk.so",
        "native/build/**/cats_topk.dll",
    )
    for pattern in patterns:
        found = sorted(root.glob(pattern))
        if found:
            return found[0]
    raise BackendUnavailable(
        "C++ DLL/SO not found. Build it: cmake -S native -B native/build; "
        "cmake --build native/build --config Release (requires C++ compiler + CMake)"
    )


class NativeCppTopK:
    def __init__(self):
        self.path = _find_library()
        self.lib = ctypes.CDLL(str(self.path))
        self.fn = self.lib.cats_topk_f32
        self.fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
                            ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]
        self.fn.restype = ctypes.c_int

    def select(self, scores: np.ndarray, k: int) -> np.ndarray:
        x = np.ascontiguousarray(scores, dtype=np.float32)
        out = np.empty(k, dtype=np.int64)
        code = self.fn(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       int(x.size), int(k),
                       out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        if code != 0:
            raise RuntimeError(f"native cats_topk_f32 returned error code {code}")
        return out
