"""Reproducible score distributions for operator-level experiments."""
from __future__ import annotations

from pathlib import Path
import numpy as np

DISTRIBUTIONS = ("uniform", "normal", "clustered", "tied", "near_ties", "adversarial")


def scores_for(n: int, k: int, distribution: str, seed: int, dtype: str = "f16") -> np.ndarray:
    if n < 1 or not 1 <= k <= n:
        raise ValueError("Require N >= K >= 1")
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution {distribution}; choose from {DISTRIBUTIONS}")
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        a = rng.uniform(-1, 1, size=n)
    elif distribution == "normal":
        a = rng.normal(0, 0.3, size=n)
    elif distribution == "clustered":
        a = rng.normal(0, 0.1, size=n)
        block = min(n, max(2 * k, min(1024, n // 16)))
        a[:block] = rng.uniform(0.8, 1.0, size=block)
    elif distribution == "tied":
        a = np.round(rng.normal(0, 0.3, size=n), 1)
    elif distribution == "near_ties":
        a = rng.uniform(-1, 0.3, size=n)
        count = min(n, max(k + 10, 20))
        a[:count] = 0.8 + np.arange(count, dtype=np.float64) * 1e-5
    else:  # adversarial: most winners concentrated in an early tile
        a = rng.uniform(-1, 0, size=n)
        count = min(n, max(k + 20, 32))
        a[:count] = np.linspace(0.99, 0.5, count)
    kind = {"f16": np.float16, "f32": np.float32, "f64": np.float64}[dtype]
    return np.asarray(a, dtype=kind)


def load_input(path: str | Path, dtype: str) -> np.ndarray:
    p = Path(path)
    if p.suffix != ".npy":
        raise ValueError("Input must be a 1D .npy file containing finite similarity scores")
    a = np.load(p, allow_pickle=False)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("Loaded scores must be a finite 1D array")
    kind = {"f16": np.float16, "f32": np.float32, "f64": np.float64}[dtype]
    return np.asarray(a, dtype=kind)
