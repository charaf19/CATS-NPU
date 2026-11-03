"""Generate only prospectively selected fresh scores, after protocol commit."""
from __future__ import annotations

import json
import hashlib

import numpy as np

from scripts.phase13_common import BASE, ROOT, environment, sha, write_json
from scripts.phase13_design import frozen
from scripts.prepare_beir_scores import read_jsonl
from scripts.prepare_sift1m import fvecs, negative_l2


def numerical_array(case):
    n, k, width = case["n"], case["k"], case["w"]
    rng = np.random.default_rng(case["seed"])
    tiny = np.float16(2**-24)
    family = case["family"]
    if family in ("positive_subnormal", "negative_subnormal"):
        a = rng.integers(1, 1024, n).astype(np.float16) * tiny
        if family == "negative_subnormal":
            a *= -1
    elif family == "normal_neighbors":
        a = np.float16(2**-14) + np.arange(n).astype(np.float16) * tiny
        rng.shuffle(a)
    elif family == "signed_zero":
        a = np.zeros(n, np.float16)
        a[rng.integers(0, 2, n).astype(bool)] = -0.0
    elif family == "mixed_ties_extremes":
        alphabet = np.array([-65504, -1, -tiny, -0.0, 0.0, tiny, 1, 65504], np.float16)
        a = np.resize(alphabet, n)
        rng.shuffle(a)
    else:
        a = np.full(n, -1, np.float16)
        a[:k] = np.arange(1, k + 1).astype(np.float16) * tiny
        a[width:width + k] = np.arange(2 * k, k, -1).astype(np.float16) * tiny
        # A third short tile exercises padding without changing the winners.
        a[-1] = -tiny
    return np.ascontiguousarray(a)


def main():
    protocol, lock = frozen()
    out = BASE / "inputs"
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"freeze": lock, "environment": environment(), "scores": [], "numerical": [],
                "generation_in_topk_timing": False, "encoder_or_metric_changed": False}
    base = fvecs(ROOT / "data/extracted/sift1m/sift_base.fvecs")
    queries = fvecs(ROOT / "data/extracted/sift1m/sift_query.fvecs")
    for qid in protocol["query_selection"]["sift"]["selected_ids"]:
        a = negative_l2(base, np.asarray(queries[int(qid)]), 65536)
        a *= 1.0 / (128.0 * 255.0 * 255.0)
        a = a.astype(np.float16)
        p = out / f"sift_q{qid}.npy"
        np.save(p, a, allow_pickle=False)
        metadata["scores"].append({"dataset": "sift", "query_id": qid, "n": len(a),
                                   "path": p.relative_to(ROOT).as_posix(), "sha256": sha(p),
                                   "score_bytes_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                                   "metric": "(-squared_L2)/(128*255^2), then FP16"})
    from sentence_transformers import SentenceTransformer
    import torch

    model = SentenceTransformer(protocol["model"], revision=protocol["model_revision"],
                                device="cuda", local_files_only=True)
    cache = ROOT / "data/extracted/phase13_embeddings"
    cache.mkdir(exist_ok=False)
    metadata["embedding_parameters"] = {"model": protocol["model"], "revision": protocol["model_revision"],
                                        "batch_size": 256, "normalize_embeddings": True,
                                        "device": "cuda", "stored_embedding_dtype": "float32",
                                        "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    for dataset in ("scifact", "fiqa"):
        source = ROOT / "data/extracted/beir" / dataset / dataset
        corpus_path, query_path = source / "corpus.jsonl", source / "queries.jsonl"
        corpus = read_jsonl(corpus_path)
        query_map = {str(r["_id"]): r["text"] for r in read_jsonl(query_path)}
        text = [((r.get("title") or "") + " " + (r.get("text") or "")).strip() for r in corpus]
        ids = protocol["query_selection"][dataset]["selected_ids"]
        print(f"Encoding {dataset} original corpus ({len(text)} documents), fixed original model", flush=True)
        documents = model.encode(text, batch_size=256, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        qe = model.encode([query_map[q] for q in ids], batch_size=256, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        doc_path, qe_path = cache / f"{dataset}_documents.npy", out / f"{dataset}_query_embeddings.npy"
        np.save(doc_path, documents, allow_pickle=False)
        np.save(qe_path, qe, allow_pickle=False)
        metadata[f"{dataset}_provenance"] = {"corpus_jsonl_sha256": sha(corpus_path),
                                              "queries_jsonl_sha256": sha(query_path),
                                              "document_embedding_path": str(doc_path.relative_to(ROOT)),
                                              "document_embeddings_sha256": sha(doc_path),
                                              "query_embeddings_sha256": sha(qe_path)}
        for i, qid in enumerate(ids):
            score32 = documents @ qe[i]
            a = score32.astype(np.float16)
            assert np.isfinite(a).all()
            p = out / f"{dataset}_q{qid}.npy"
            np.save(p, a, allow_pickle=False)
            metadata["scores"].append({"dataset": dataset, "query_id": qid, "n": len(a),
                                       "path": p.relative_to(ROOT).as_posix(), "sha256": sha(p),
                                       "score_bytes_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                                       "metric": "cosine of original-model normalized FP32 embeddings, then FP16"})
        write_json(out / "manifest.json", metadata)
    for case in protocol["numerical_cases"]:
        a = numerical_array(case)
        p = out / (case["id"] + ".npy")
        np.save(p, a, allow_pickle=False)
        metadata["numerical"].append({**case, "path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)})
    write_json(out / "manifest.json", metadata)
    print(json.dumps({"fresh_query_vectors": len(metadata["scores"]),
                      "numerical_vectors": len(metadata["numerical"])}))


if __name__ == "__main__":
    main()
