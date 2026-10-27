"""Deterministic CPU reference for exact Top-K on finite numeric input.

Ranking: score descending, then *global* zero-based index ascending.
The algorithm is linear-time selection on average, plus sorting of candidates;
no heuristic floating-point index perturbation is used.
"""
from __future__ import annotations

import numpy as np


def validate_scores(scores: np.ndarray, k: int) -> np.ndarray:
    a = np.asarray(scores)
    if a.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape={a.shape}")
    if a.size == 0:
        raise ValueError("Score vector cannot be empty")
    if a.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"Only float16/32/64 score vectors are accepted: {a.dtype}")
    if not np.isfinite(a).all():
        raise ValueError("Non-finite scores (NaN/+Inf/-Inf) are not supported")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("K must be an integer")
    if not 1 <= k <= a.size:
        raise ValueError(f"K must satisfy 1 <= K <= N={a.size}; got {k}")
    return a


def topk_with_ids(values: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    """Return positions within `values` of exact stable Top-K.

    `ids` are unique global indices (used to break score ties). `values` may
    contain -Inf for *internal* padding; callers should not expose -Inf inputs.
    """
    values = np.asarray(values)
    ids = np.asarray(ids, dtype=np.int64)
    if values.ndim != 1 or ids.ndim != 1 or values.size != ids.size:
        raise ValueError("Values and IDs must be same-length 1D arrays")
    if k < 1 or k > values.size:
        raise ValueError("K out of range")
    if np.isnan(values).any():
        raise ValueError("NaN is not accepted")
    if len(np.unique(ids)) != ids.size:
        raise ValueError("IDs must be unique")
    # An exact scalar cutoff; avoid FP32 perturbations or epsilon tie breaking.
    kth = np.partition(values, values.size - k)[values.size - k]
    higher = np.flatnonzero(values > kth)
    tied = np.flatnonzero(values == kth)
    needed = k - higher.size
    if not 0 <= needed <= tied.size:
        raise AssertionError("Invalid kth partition")
    ties_by_id = tied[np.argsort(ids[tied], kind="stable")[:needed]]
    chosen = np.concatenate([higher, ties_by_id])
    order = np.lexsort((ids[chosen], -values[chosen].astype(np.float64)))
    return chosen[order].astype(np.int64, copy=False)


def exact_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Deterministic reference returning sorted original indices."""
    a = validate_scores(scores, k)
    # IDs here are exactly 0..N-1, so the uniqueness check needed by
    # topk_with_ids would sort a million known-unique IDs during a large
    # boundary-tie repair. Select directly and retain the same stable order.
    kth = np.partition(a, a.size - k)[a.size - k]
    higher = np.flatnonzero(a > kth)
    needed = k - higher.size
    tied = np.flatnonzero(a == kth)[:needed]
    chosen = np.concatenate((higher, tied))
    order = np.lexsort((chosen, -a[chosen].astype(np.float64)))
    return chosen[order].astype(np.int64, copy=False)


def matches_reference(scores: np.ndarray, found: np.ndarray, k: int) -> bool:
    return np.array_equal(np.asarray(found, dtype=np.int64), exact_topk(scores, k))
