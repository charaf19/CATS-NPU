# Phase 13 reproducibility

This artifact characterizes one physical Intel AI Boost 3720 system. It reuses existing selection methods; it introduces no selection algorithm. The authoritative conclusions are in [the readiness report](PHASE13_FINAL_PUBLICATION_READINESS.md). No manuscript or paper figures are part of Phase 13.

## Source and protocol

Historical starting revision: `342b3726f531143ab5aac1666f2ec79a9baea839`. Prospective freeze commit: **`d97544ac4514bafbb81b0cf38044982f43fb1379`**, on `codex/phase12-boundary-counterexamples`. The protocol, query IDs, numerical generators, measurement harness and primary analysis were committed before fresh scores existed. `results/phase13/freeze.json` records that boundary. Every preparation/session/analysis invocation checks the protocol and frozen source digests. Post-freeze scripts `replay_phase13.py`, `summarize_phase13.py` and `check_phase13_artifacts.py` provide output routing, descriptive checks and artifact sealing; they do not change the frozen methods or hypothesis.

Use a byte-preserving checkout with `core.autocrlf=false` when obtaining the artifact (for example `git -c core.autocrlf=false clone <artifact-repository>`). All frozen source hashes and historical Phase 12 bindings were checked against Git blob bytes. Automatic CRLF rewriting can cause an integrity rejection even when Python behavior is unchanged; preserve the recorded bytes rather than editing the manifests to accept a changed checkout.

The environment is Windows 11 build 26300, Python 3.11.0, OpenVINO 2026.4.1, NumPy 2.4.6, PyTorch 2.11.0+cu128, SciPy 1.17.1 and sentence-transformers 5.1.2. CPU: Intel Core Ultra 7 255H; NPU architecture 3720, driver 1004512, compiler 458778; CUDA: NVIDIA RTX 5060 Laptop GPU. All session environments and power-plan checks are retained. AC connection and a fixed Balanced power plan are operating-condition controls, not energy metrics.

For a fresh checkout, install the matching NPU/NVIDIA drivers and Python 3.11, create `.venv`, then install `results/phase13/provenance/requirements_phase13.txt` and install the local project with `python -m pip install -e . --no-deps`. The requirements retain observed package pins and the CUDA wheel index; the original `pip_freeze.txt` is preserved separately, including its historical editable-project revision. Package availability and OS/driver installation are external prerequisites, not provided virtual-machine images.

The native baseline loads `native/build/cats_topk.dll`. On a fresh Windows x64 checkout, restore the byte-identical saved library before artifact verification:

```powershell
New-Item -ItemType Directory -Path native/build -Force | Out-Null
Copy-Item -LiteralPath results/phase13/provenance/cats_topk.dll -Destination native/build/cats_topk.dll
```

Its digest is checked against `native_library.json`. Alternatively build the frozen native source with the existing `scripts/08_build_cpp.ps1`; a different compiler can produce a different binary digest, which must be reported as a new replication environment rather than silently substituted into the sealed record.

## Artifact layout

All paths below are relative to the repository root.

| Path | Role |
|---|---|
| `results/phase13/audit/` | Historical hash audit and query exposure inventory |
| `results/phase13/pilot/`, `harness_smoke*` | Old-query development checks, excluded from confirmation |
| `results/phase13/protocol.json`, `query_selection.json`, `freeze.json` | Immutable prospective design and source binding |
| `results/phase13/inputs/` | Fresh FP16 scores, numerical arrays, query embeddings and provenance |
| `results/phase13/numerical/session01..03/` | CPU/NPU raw NPZ outputs, graph XML/BIN, placement and errors |
| `results/phase13/numerical/adjacent_*` | Independent bounded adjacent-pair replications |
| `results/phase13/performance/session01..12/` | Raw timed/warm-up IDs, latency, components and environment |
| `results/phase13/logs/`, `execution_log.json` | Exact commands, exit status and session logs |
| `results/phase13/analysis/` | Frozen analysis and six machine-readable JSON tables |
| `results/phase13/supplemental/` | Explicitly descriptive post-freeze raw-output and component checks |
| `results/phase13/provenance/` | Native library snapshot and its SHA256 |
| `results/phase13/artifact_manifest.json` | SHA256 binding of the final package |

The saved score arrays are sufficient to reproduce selection timings; source archives and the encoder are needed only to regenerate similarity scores. Timing excludes retrieval scoring. Scores represent the declared rounded FP16 inputs, not unrounded FP32 retrieval rankings.

## Inspect and recompute without running hardware

Run in PowerShell from the repository root, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_phase13_artifacts
.\.venv\Scripts\python.exe -m scripts.phase13_analyze --out results/phase13_analysis_recomputed
.\.venv\Scripts\python.exe -m scripts.summarize_phase13 --out results/phase13_supplement_recomputed
.\.venv\Scripts\python.exe -m pytest -q tests/test_phase13.py tests/test_precision_topk.py tests/test_static_graph.py tests/test_order_encoded_npu.py tests/test_stable_integer_hierarchy.py
```

Output directories must be new. The analysis loads original raw IDs/values, regenerates references, checks correctness flags and produces the same six tables. Fixed bootstrap seeds make the statistical outputs reproducible on the recorded software stack. Compare generated JSON contents with `results/phase13/analysis/`; directory names do not enter the statistical estimands. The checker verifies Phase 12 preservation, protocol/source digests, the native binary and every sealed Phase 13 file.

## Repeat physical sessions without overwriting confirmation

Connect AC power, use the recorded power plan and software/driver versions, close competing workloads, and ensure the native C++ library and CUDA baseline are available. Then:

```powershell
.\.venv\Scripts\python.exe -m scripts.replay_phase13 --out results/phase13_replay_01 --kind all --analyze
```

This starts three numerical and twelve performance processes sequentially, with the frozen ten-second idle before each performance process. It uses the original retained scores and fixed query IDs. It routes outputs to a new directory; existing directories are refused. New runs are replications on exposed inputs, **not new untouched confirmation**. A failed process is retained and stops the convenience launcher; do not replace its attempt or splice successful sessions into the original study.

The bounded adjacent test can be repeated separately, with new output paths:

```powershell
.\.venv\Scripts\python.exe -m scripts.probe_topk_numerical_path --suite adjacent --repeats 1 --out results/phase13_adjacent_replay_default
.\.venv\Scripts\python.exe -m scripts.probe_topk_numerical_path --suite adjacent --repeats 1 --higher-precision TopK --out results/phase13_adjacent_replay_protected
```

Do not launch overlapping NPU/performance jobs. Native NPU graphs use strict `NPU` compilation; the harness checks reported placement. CPU certification remains host work and its cost stays inside timing. No incorrect NPU output is replaced with a CPU answer in the checked FP32/INT32 methods.

## Source-data and score provenance

Original source archive SHA256 values:

| Dataset | SHA256 |
|---|---|
| SIFT1M | `92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a` |
| BEIR SciFact | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` |
| BEIR FiQA | `32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02` |

SIFT uses original fvecs and `(-squared_L2)/(128*255^2)`, then FP16. BEIR preserves corpus order and title-plus-text construction, using normalized FP32 embeddings from `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, batch 256 on CUDA, then FP32 dot products on the host and FP16 rounding. The input manifest retains source JSONL and embedding hashes, model settings, TF32 configuration and every score-file digest. Document embedding caches remain under ignored `data/extracted/phase13_embeddings/`; their hashes are retained, while original source archives remain under ignored `data/raw/`.

The original fresh-input command was `python -m scripts.phase13_prepare`, executed only after the freeze. It intentionally refuses to overwrite existing inputs. Regeneration requires an isolated copy with the recorded archives/model cache and a **new** input destination; never remove or replace the locked input directory. Small floating-point generation differences across libraries/devices can change FP16 boundary ties, so the retained score arrays and their hashes define this selection study's exact inputs.

## Interpretation limits

The numerical panel is a fixed adversarial design, with exact panel counts and no population error-rate confidence interval. The adjacent batch exhausts adjacent numerical values only in its declared `[63487,2]`, K=1 graph; it does not exhaust arbitrary tensors or compiled graphs. Twelve process launches are separate runtime sessions on one continuously operated laptop on one date. They do not prove independence of thermal state, OS interference or hardware noise. Session-cluster, crossed query/session and consecutive-pair sensitivity analyses expose some of that uncertainty; they cannot establish cross-machine or cross-day validity. Only eight newly selected queries per dataset support this experiment's query scope.

Public artifact verification establishes integrity relative to the recorded hashes and Git history; it is not a trusted external timestamp or a proof of opaque hardware behavior. Another NPU generation or compiler version is useful future external validation, not a claimed completed experiment.
