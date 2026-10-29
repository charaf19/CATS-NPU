"""Prepare real similarity-score test vectors from local .npy embeddings.

No network download, no text embedding model, no indexing. The purpose is to
create real score distributions to test the single Top-K operator.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


def make_scores_from_embeddings(embeddings: str | Path, queries: str | Path,
                                query_index: int, out: str | Path, dtype: str = "f16",
                                normalize: bool = True, chunk: int = 65536) -> dict:
    x = np.load(embeddings, mmap_mode="r", allow_pickle=False)
    q = np.load(queries, mmap_mode="r", allow_pickle=False)
    if x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[1]:
        raise ValueError("Embeddings must be [N,D], queries [Q,D] with matching D")
    if not 0 <= query_index < q.shape[0]:
        raise ValueError("query_index outside query matrix")
    if dtype not in ("f16", "f32"):
        raise ValueError("dtype must be f16 or f32")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    query = np.asarray(q[query_index], dtype=np.float32)
    if not np.isfinite(query).all():
        raise ValueError("Query contains non-finite values")
    if normalize:
        norm = np.linalg.norm(query)
        if norm <= 0:
            raise ValueError("Cannot normalize zero query vector")
        query = query / norm
    dst = np.empty(x.shape[0], dtype=np.float32)
    for first in range(0, x.shape[0], chunk):
        block = np.asarray(x[first:first+chunk], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"Non-finite embeddings detected near row {first}")
        if normalize:
            norms = np.linalg.norm(block, axis=1)
            if np.any(norms <= 0):
                raise ValueError(f"Zero embedding detected near row {first}")
            block = block / norms[:, None]
        dst[first:first+block.shape[0]] = block @ query
    if not np.isfinite(dst).all():
        raise RuntimeError("Produced non-finite score values")
    result = dst.astype(np.float16 if dtype == "f16" else np.float32)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, result, allow_pickle=False)
    manifest = {
        "source_embeddings": str(embeddings),
        "source_queries": str(queries),
        "query_index": query_index, "n": int(x.shape[0]), "d": int(x.shape[1]),
        "normalize": normalize, "dtype": dtype, "chunk": chunk,
        "score_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "note": "Cosine if normalized, otherwise inner-product similarities.",
    }
    p.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
