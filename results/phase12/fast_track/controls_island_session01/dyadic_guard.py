"""Conventional guarded power-of-two scaling; not a new order encoding.

The transformation preserves finite FP16 ordering when all nonzero outputs
are normal and no output overflows. It does not prove an NPU primitive correct.
"""
from __future__ import annotations

import numpy as np


class UnsupportedScale(ValueError):
    """One finite FP16 scale cannot cover the input's magnitude range."""


def normal_scale(scores: np.ndarray) -> tuple[np.ndarray, int]:
    """Return exact FP16 scores * 2**shift, or explicitly reject the domain."""
    a = np.asarray(scores)
    if a.dtype != np.float16 or a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ValueError("Require a nonempty one-dimensional finite FP16 input")
    magnitude = np.abs(a.astype(np.float32))
    nonzero = magnitude[magnitude != 0]
    if not nonzero.size:
        return a.copy(), 0
    minimum = float(nonzero.min())
    maximum = float(nonzero.max())
    shift = 0
    while np.ldexp(minimum, shift) < np.finfo(np.float16).tiny:
        shift += 1
    if np.ldexp(maximum, shift) > np.finfo(np.float16).max:
        raise UnsupportedScale("No nonnegative dyadic scale makes all nonzeros normal and finite")
    result = np.ldexp(a.astype(np.float32), shift).astype(np.float16)
    restored = np.ldexp(result.astype(np.float32), -shift)
    if not np.array_equal(restored, a.astype(np.float32)):
        raise UnsupportedScale("Exact scaling round-trip failed")
    return result, shift
