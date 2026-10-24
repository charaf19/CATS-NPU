"""Acquire SIFT1M and generate real, query-specific Top-K score arrays.

Embedding/distance generation is deliberately outside benchmark timing. The
saved arrays contain negative squared L2 distances so larger scores rank first.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen

import numpy as np

from cats_npu.reference import exact_topk


DEFAULT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
REQUIRED = ("sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs")


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url) as source, partial.open("wb") as target:  # nosec: user-visible URL
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    partial.replace(destination)


def extract_required(archive: Path, destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        by_name = {Path(member.name).name: member for member in bundle.getmembers()
                   if member.isfile()}
        missing = [name for name in REQUIRED if name not in by_name]
        if missing:
            raise ValueError(f"Archive lacks required SIFT1M files: {missing}")
        for name in REQUIRED:
            output = destination / name
            if not output.is_file() or output.stat().st_size != by_name[name].size:
                source = bundle.extractfile(by_name[name])
                if source is None:
                    raise ValueError(f"Cannot read {name} from archive")
                with source, output.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            found[name] = output
    return found


def fvecs(path: Path) -> np.ndarray:
    raw = np.memmap(path, dtype="<i4", mode="r")
    if raw.size < 2:
        raise ValueError(f"Empty fvecs file: {path}")
    dimension = int(raw[0])
    if dimension <= 0 or raw.size % (dimension + 1):
        raise ValueError(f"Invalid fvecs layout: {path}")
    rows = raw.reshape(-1, dimension + 1)
    if not np.all(rows[:, 0] == dimension):
        raise ValueError(f"Inconsistent fvecs dimensions: {path}")
    return rows[:, 1:].view("<f4")


def ivecs(path: Path) -> np.ndarray:
    raw = np.memmap(path, dtype="<i4", mode="r")
    dimension = int(raw[0])
    if dimension <= 0 or raw.size % (dimension + 1):
        raise ValueError(f"Invalid ivecs layout: {path}")
    rows = raw.reshape(-1, dimension + 1)
    if not np.all(rows[:, 0] == dimension):
        raise ValueError(f"Inconsistent ivecs dimensions: {path}")
    return rows[:, 1:]


def negative_l2(base: np.ndarray, query: np.ndarray, chunk_rows: int) -> np.ndarray:
    scores = np.empty(base.shape[0], dtype=np.float32)
    for start in range(0, base.shape[0], chunk_rows):
        stop = min(start + chunk_rows, base.shape[0])
        # Keep the temporary bounded; the fvecs payload is a strided mmap view.
        block = np.asarray(base[start:stop], dtype=np.float32)
        delta = block - query
        scores[start:stop] = -np.einsum("ij,ij->i", delta, delta, optimize=True)
    return scores


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/sift.tar.gz"))
    parser.add_argument("--extract", type=Path, default=Path("data/extracted/sift1m"))
    parser.add_argument("--out", type=Path, default=Path("results/phase5/sift1m"))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--query-count", type=int, default=16)
    parser.add_argument("--sizes", default="1024,16384,131072,1000000")
    parser.add_argument("--chunk-rows", type=int, default=65536)
    args = parser.parse_args()
    sizes = sorted({int(value) for value in args.sizes.split(",")})
    if args.download:
        download(args.url, args.archive)
    if not args.archive.is_file():
        raise FileNotFoundError(f"Missing {args.archive}; pass --download")
    files = extract_required(args.archive, args.extract)
    base = fvecs(files["sift_base.fvecs"])
    queries = fvecs(files["sift_query.fvecs"])
    groundtruth = ivecs(files["sift_groundtruth.ivecs"])
    if base.shape != (1_000_000, 128) or queries.shape[1] != 128:
        raise ValueError(f"Unexpected SIFT1M shapes: base={base.shape}, queries={queries.shape}")
    if any(size < 1 or size > base.shape[0] for size in sizes):
        raise ValueError(f"Sizes must be within 1..{base.shape[0]}")
    if not 1 <= args.query_count <= min(queries.shape[0], groundtruth.shape[0]):
        raise ValueError("Invalid query count")
    args.out.mkdir(parents=True, exist_ok=True)
    splits = {
        "development": list(range(0, min(4, args.query_count))),
        "evaluation": list(range(4, min(8, args.query_count))),
        "confirmatory_holdout": list(range(8, args.query_count)),
    }
    split_for = {query_id: split for split, ids in splits.items() for query_id in ids}
    items = []
    # SIFT descriptors use 128 dimensions with byte-like values. Raw squared
    # distances can exceed finite FP16. One positive global scale preserves
    # FP32 ordering while mapping the theoretical range into [-1, 0].
    score_scale = 1.0 / (128.0 * 255.0 * 255.0)
    for query_id in range(args.query_count):
        score32 = negative_l2(base, np.asarray(queries[query_id]), args.chunk_rows)
        score32 *= score_scale
        expected100 = exact_topk(score32, min(100, score32.size))
        supplied100 = np.asarray(groundtruth[query_id, :expected100.size], dtype=np.int64)
        fp32_overlap = len(set(expected100.tolist()) & set(supplied100.tolist())) / expected100.size
        score16 = score32.astype(np.float16)
        fp16_top100 = exact_topk(score16, min(100, score16.size))
        fp16_overlap = len(set(fp16_top100.tolist()) & set(supplied100.tolist())) / fp16_top100.size
        for size in sizes:
            output = args.out / f"q{query_id:04d}_n{size}_neg_l2_f16.npy"
            np.save(output, score16[:size], allow_pickle=False)
            ids10 = exact_topk(score16[:size], min(10, size))
            gap = (float(score16[ids10[-1]]) -
                   float(np.partition(score16[:size], -(len(ids10) + 1))[-(len(ids10) + 1)])) \
                if size > len(ids10) else float("inf")
            items.append({
                "dataset": "ANN_SIFT1M",
                "query_id": query_id,
                "split": split_for[query_id],
                "n": size,
                "dimension": 128,
                "metric": "scaled negative squared Euclidean distance",
                "score_scale": score_scale,
                "stored_dtype": "float16",
                "path": str(output),
                "score_sha256": sha256_file(output),
                "top10_boundary_gap_f16": gap,
            })
        print(f"query {query_id + 1}/{args.query_count}: scores generated", flush=True)
        for item in items[-len(sizes):]:
            item["full_fp32_groundtruth_overlap_at_100"] = fp32_overlap
            item["full_fp16_groundtruth_overlap_at_100"] = fp16_overlap
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": args.url,
        "source_page": "http://corpus-texmex.irisa.fr/",
        "secondary_acquisition_reference": "https://github.com/facebookresearch/faiss/blob/main/benchs/README.md",
        "archive": str(args.archive),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": sha256_file(args.archive),
        "license_status": (
            "The source page supplies the research benchmark but no archive license was "
            "identified by this script. Users must verify reuse/citation terms."
        ),
        "distance_generation_in_benchmark_timing": False,
        "score_transform": "(-squared_L2) / (128 * 255^2); positive scaling preserves FP32 order",
        "ordering": "larger scaled negative squared L2 first; equal scores by smaller base index",
        "splits_fixed_before_score inspection": splits,
        "items": items,
    }
    (args.out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"queries": args.query_count, "arrays": len(items),
                      "manifest": str(args.out / "dataset_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
