"""Order-preserving integer codes for finite binary16 Top-K scores."""
from __future__ import annotations

import numpy as np

from .reference import validate_scores


def score_codes(scores: np.ndarray) -> np.ndarray:
    """Map finite FP16 scores to int32 codes with the same numeric order.

    Both signed zeros receive one code. Subnormals retain their distinct order.
    This is a representation transform only; it does not select winners.
    """
    a = np.asarray(scores)
    if a.dtype != np.float16:
        raise ValueError("Only finite FP16 input is supported")
    validate_scores(a, 1)
    bits = a.view(np.uint16).copy()
    bits[bits == 0x8000] = 0
    return np.where((bits & 0x8000) != 0, (~bits) & 0xffff,
                    bits ^ 0x8000).astype(np.int32)
