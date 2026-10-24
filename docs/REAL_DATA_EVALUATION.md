# Real-data evaluation

Date: 2026-10-24  
Status: descriptive evaluation complete; confirmatory query splits preserved

## Datasets and provenance

### SIFT1M

The preparation script downloads `sift.tar.gz` from the original [TEXMEX/IRISA corpus](ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz), also referenced by the [Faiss benchmark instructions](https://github.com/facebookresearch/faiss/blob/main/benchs/README.md). The archive is 168,280,445 bytes with SHA-256 `92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`.

SIFT1M contains 1,000,000 base vectors and 10,000 queries of dimension 128. For each selected query, the script computes negative squared Euclidean distance outside the timed Top-K benchmark. Raw distances can exceed finite FP16, so every score receives the same positive scale `1 / (128 * 255^2)`. The scale preserves FP32 ordering and maps the theoretical descriptor range to `[-1, 0]` before FP16 conversion.

For all 16 selected queries, computed FP32 Top-100 had 100% set agreement with the supplied ground truth. FP16 Top-100 agreement ranged from 99% to 100%, median 100%. FP16 Top-10 boundary gaps ranged from zero to `1.75e-4`, showing that conversion sometimes creates boundary ties.

### BEIR SciFact and FiQA

The scripts download the official prepared archives listed by the [BEIR project](https://github.com/beir-cellar/beir/wiki/Datasets-available). Expected BEIR MD5 values are checked before extraction.

| Dataset | Documents | Archive MD5 | Archive SHA-256 |
|---|---:|---|---|
| SciFact | 5,183 | `5f7d1de60b170fc8027bb7898e2efca1` | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` |
| FiQA | 57,638 | `17918ed23cd04fb15047f73e6c3bd9d9` | `32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02` |

Documents and test queries are encoded with `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, using L2-normalized 384-dimensional embeddings. Saved scores are FP16 cosine similarities. Embedding time is outside Top-K timing.

Across 16 selected queries per dataset, FP16/FP32 Top-10 set overlap was 90%–100% for SciFact and 100% for FiQA. Precision conversion is therefore evaluated separately from exact selection: every benchmark method must exactly match the deterministic reference over the stored FP16 array, even when that array ranks one item differently from FP32.

BEIR states that its prepared copies do not grant or verify original dataset licenses. The repository records source and hashes but does not vendor the archives; users must verify original licenses and citation requirements. The SIFT source likewise did not expose a clear archive license in the acquisition path, so its raw archive is also excluded from version control.

## Predeclared splits

For each dataset, selected query identifiers were sorted before score inspection:

- first four: development;
- next four: descriptive evaluation used here;
- final eight: untouched confirmatory holdout.

SIFT1M evaluation uses query indices 4–7 at prefixes 1,024, 16,384, 131,072, and 1,000,000. SciFact and FiQA use four evaluation queries each over their complete corpora. Score arrays, hashes, query identifiers, transforms, model revision, and split assignments are in the dataset manifests under `results/phase5/`.

## Methods and timing

Every evaluation uses T256/B32, query batch one, `K` 4 and 10, two warmup calls, two randomized blocks, and five timed repetitions per block. Methods are optimized C++, NumPy exact, stable CUDA, native direct NPU TopK, fixed hierarchical NPU, geometric certified NPU, and submission-aware certified NPU. All methods consume identical stored FP16 scores and return host indices with score-descending/index-ascending order.

Compilation, distance generation, and embedding inference are excluded. Host-to-host elapsed time includes transfers, synchronization, host merging, certification, and any measured tie repair.

## SIFT1M results

The SIFT evaluation plus direct-NPU control contains 2,240 observations across 32 query/size/K cases, with zero skips and zero wrong results. The following values are medians across the four evaluation-query medians.

| N | K | C++ | CUDA | NPU direct | NPU fixed | NPU geometric | NPU submission-aware |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 4 | 0.050 ms | 0.220 ms | 0.952 ms | 1.219 ms | 2.310 ms | 2.293 ms |
| 1,024 | 10 | 0.036 ms | 0.204 ms | 0.960 ms | 1.268 ms | 4.416 ms | 2.325 ms |
| 16,384 | 4 | 0.179 ms | 0.247 ms | 2.716 ms | 2.846 ms | 3.554 ms | 3.833 ms |
| 16,384 | 10 | 0.189 ms | 0.257 ms | 2.727 ms | 2.864 ms | 3.731 ms | 3.616 ms |
| 131,072 | 4 | 1.572 ms | 0.603 ms | 18.009 ms | 17.178 ms | 15.881 ms | 15.417 ms |
| 131,072 | 10 | 1.634 ms | 0.602 ms | 17.996 ms | 19.662 ms | 14.933 ms | 14.498 ms |
| 1,000,000 | 4 | 15.179 ms | 2.208 ms | 133.595 ms | 110.267 ms | 98.595 ms | 96.927 ms |
| 1,000,000 | 10 | 16.175 ms | 4.006 ms | 133.920 ms | 119.223 ms | 97.928 ms | 101.187 ms |

Direct NPU is best among NPU methods at 1K and 16K. Certified candidate reduction overtakes direct and fixed NPU at 128K and 1M. At 1M/K=10, geometric is 1.37x faster than direct NPU and 1.22x faster than fixed NPU. CUDA remains 24.4x faster than geometric; C++ remains 6.1x faster.

Submission-aware versus geometric has minimum/median/maximum speedup 0.768x/1.014x/2.002x across 32 cases and wins 18. It halves submissions at 1K/K=10, but at 128K and 1M the active set exceeds one tile batch, the jump condition does not trigger, and it behaves much like geometric refinement.

## BEIR results

The SciFact/FiQA evaluation contains 1,120 observations across 16 query/dataset/K cases, with zero skips and zero wrong results.

| Dataset | K | C++ | CUDA | NPU direct | NPU fixed | NPU geometric | NPU submission-aware |
|---|---:|---:|---:|---:|---:|---:|---:|
| SciFact, N=5,183 | 4 | 0.092 ms | 0.247 ms | 1.813 ms | 1.484 ms | 1.401 ms | 1.408 ms |
| SciFact, N=5,183 | 10 | 0.081 ms | 0.203 ms | 1.921 ms | 1.463 ms | 2.421 ms | 2.665 ms |
| FiQA, N=57,638 | 4 | 0.860 ms | 0.702 ms | 16.733 ms | 10.997 ms | 10.668 ms | 9.503 ms |
| FiQA, N=57,638 | 10 | 0.675 ms | 0.526 ms | 14.294 ms | 10.562 ms | 8.495 ms | 9.819 ms |

Geometric is faster than fixed in 10 of 16 BEIR cases and has a median 1.052x speedup. Geometric is a median 1.452x faster than direct NPU. Submission-aware beats geometric in only 5 of 16 cases and has median speedup 0.961x, so it does not generalize as a default improvement on these semantic workloads.

## Interpretation and gate

The real data supports an NPU-internal crossover: native direct TopK is preferable for small score arrays; certified candidate reduction can reduce latency for larger arrays. It also confirms that score distribution changes which adaptive policy wins. No NPU method is competitive with the strongest CPU/GPU control in these experiments.

The Phase 5 gate is **passed for real-score feasibility, reproducibility, exactness, and distribution diversity**. It remains **descriptive** because queries were run in one experimental launch and no independent-session inference was attempted. The eight confirmatory queries per dataset are preserved for Phase 7.

## Reproduction

```powershell
.\scripts\17_prepare_sift1m.ps1 -Download
.\scripts\18_benchmark_sift1m.ps1
.\scripts\20_benchmark_direct_npu.ps1
.\scripts\22_prepare_beir.ps1
.\scripts\23_benchmark_beir.ps1
```

Pass the timestamped result directories to scripts 19, 21, and 24 for analysis. Generated score arrays, raw observations, tables, and SVG figures stay under `results/`.
