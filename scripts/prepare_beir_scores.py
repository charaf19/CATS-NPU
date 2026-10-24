"""Create real semantic-similarity score arrays from BEIR test queries."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.request import urlopen
import zipfile

import numpy as np

from cats_npu.reference import exact_topk


DATASETS = {
    "scifact": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "md5": "5f7d1de60b170fc8027bb7898e2efca1",
    },
    "fiqa": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        "md5": "17918ed23cd04fb15047f73e6c3bd9d9",
    },
}


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url) as source, partial.open("wb") as target:  # nosec: fixed HTTPS URLs
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    partial.replace(destination)


def extract_safe(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            output = (destination / member.filename).resolve()
            if root not in output.parents and output != root:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        bundle.extractall(destination)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part
                 for part in re.split(r"(\d+)", value))


def test_query_ids(root: Path) -> list[str]:
    qrels = next(root.rglob("qrels/test.tsv"))
    lines = qrels.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    query_column = header.index("query-id") if "query-id" in header else 0
    return sorted({line.split("\t")[query_column] for line in lines[1:] if line.strip()},
                  key=natural_key)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--datasets", default="scifact,fiqa")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/beir"))
    parser.add_argument("--extract", type=Path, default=Path("data/extracted/beir"))
    parser.add_argument("--out", type=Path, default=Path("results/phase5/beir"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--query-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    selected = [name.strip() for name in args.datasets.split(",")]
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown BEIR datasets: {sorted(unknown)}")

    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    import torch

    resolved_revision = model_info(args.model).sha
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model, revision=resolved_revision, device=device)
    manifests = []
    for dataset_name in selected:
        source = DATASETS[dataset_name]
        archive = args.raw / f"{dataset_name}.zip"
        download(source["url"], archive)
        actual_md5 = digest_file(archive, "md5")
        if actual_md5 != source["md5"]:
            raise ValueError(f"{dataset_name} archive MD5 {actual_md5} != {source['md5']}")
        extract_root = args.extract / dataset_name
        extract_safe(archive, extract_root)
        corpus_path = next(extract_root.rglob("corpus.jsonl"))
        query_path = next(extract_root.rglob("queries.jsonl"))
        corpus = read_jsonl(corpus_path)
        query_map = {str(row["_id"]): row["text"] for row in read_jsonl(query_path)}
        ids = test_query_ids(extract_root)[:args.query_count]
        if len(ids) != args.query_count or any(query_id not in query_map for query_id in ids):
            raise ValueError(f"Could not select {args.query_count} test queries for {dataset_name}")
        corpus_text = [((row.get("title") or "") + " " + (row.get("text") or "")).strip()
                       for row in corpus]
        print(f"{dataset_name}: encoding {len(corpus_text)} documents on {device}", flush=True)
        corpus_embeddings = model.encode(
            corpus_text, batch_size=args.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
        ).astype(np.float32, copy=False)
        query_embeddings = model.encode(
            [query_map[query_id] for query_id in ids], batch_size=args.batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32, copy=False)
        dataset_out = args.out / dataset_name
        dataset_out.mkdir(parents=True, exist_ok=True)
        split_ids = {
            "development": ids[:4],
            "evaluation": ids[4:8],
            "confirmatory_holdout": ids[8:],
        }
        split_for = {query_id: split for split, values in split_ids.items()
                     for query_id in values}
        items = []
        for position, query_id in enumerate(ids):
            score32 = corpus_embeddings @ query_embeddings[position]
            score16 = score32.astype(np.float16)
            top32 = exact_topk(score32, 10)
            top16 = exact_topk(score16, 10)
            overlap = len(set(top32.tolist()) & set(top16.tolist())) / 10
            eleventh = np.partition(score16, -11)[-11]
            gap = float(score16[top16[-1]]) - float(eleventh)
            safe_query_id = re.sub(r"[^A-Za-z0-9_.-]", "_", query_id)
            output = dataset_out / f"q{position:04d}_{safe_query_id}_n{len(corpus)}_cosine_f16.npy"
            np.save(output, score16, allow_pickle=False)
            items.append({
                "dataset": f"BEIR/{dataset_name}", "query_position": position,
                "query_id": query_id, "split": split_for[query_id], "n": len(corpus),
                "model": args.model, "model_revision": resolved_revision,
                "metric": "cosine similarity of L2-normalized embeddings",
                "stored_dtype": "float16", "path": str(output),
                "score_sha256": digest_file(output),
                "fp16_fp32_top10_overlap": overlap,
                "top10_boundary_gap_f16": gap,
            })
        dataset_manifest = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": f"BEIR/{dataset_name}", "source_url": source["url"],
            "beir_source": "https://github.com/beir-cellar/beir/wiki/Datasets-available",
            "archive": str(archive), "archive_sha256": digest_file(archive),
            "archive_md5_expected_and_verified": source["md5"],
            "license_status": (
                "BEIR redistributes a prepared copy and explicitly requires users to verify "
                "the original dataset license and citation. The archive is not vendored."
            ),
            "model": args.model, "model_revision": resolved_revision,
            "sentence_transformers": sentence_transformers.__version__, "device": device,
            "embedding_time_in_topk_benchmark": False,
            "splits_fixed_before_embedding_inspection": split_ids,
            "corpus_size": len(corpus), "embedding_dimension": corpus_embeddings.shape[1],
            "items": items,
        }
        manifest_path = dataset_out / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(dataset_manifest, indent=2) + "\n")
        manifests.append(str(manifest_path))
        print(f"{dataset_name}: wrote {len(items)} score arrays", flush=True)
    print(json.dumps({"model_revision": resolved_revision, "manifests": manifests}, indent=2))


if __name__ == "__main__":
    main()
