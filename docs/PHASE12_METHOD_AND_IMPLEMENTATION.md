# Phase 12 implementation and reproduction

**Artifact:** a conventional checked FP32 baseline and focused diagnostics. No novel generator, automated precision compiler, general reducer or modified compiler binary was implemented. [The final decision](PHASE12_FINAL_RESEARCH_DECISION.md) is authoritative.

| Source | Role |
|---|---|
| `src/cats_npu/precision_topk.py` | FP16 host payload, graph promotion, stable hierarchy, existing `TopK,Reshape` precision preservation, strict NPU and full rejection checks. |
| `src/cats_npu/dyadic_guard.py` | Conventional scaling control with explicit magnitude-domain rejection. |
| `scripts/probe_topk_numerical_path.py` | Direct/local/hierarchy, transport, comparison, promotion, scaling and adjacent-pair probes. Saves inputs, graphs, raw outputs, configuration and discovery. |
| `scripts/evaluate_precision_repair.py` | Interleaved checked baselines/precision ablations; adversarial cases; optional existing native/C++/CUDA controls. |
| `scripts/summarize_precision_repair.py` | Rechecks saved adjacent and repaired preflight outputs, descriptive timings and historical hashes; no inference. |
| `tests/test_precision_topk.py` | Oracle false-positive controls, every finite scalar's scaling check, vector rejection, CPU graph checks, and injected erroneous-request rejection without CPU replacement. |

The injected outputs are unit controls, not hardware defects. The prototype uses one request and a reusable input buffer and is not thread-safe. Input must remain immutable during `run`. Wrong runtime shape/dtype/nonfinite input rejects; compile/property failures propagate. Successful returned FP32 scores equal the supplied FP16 numeric scores. Signed-zero bit identity is not required by that numeric-value contract. No default CLI or previous baseline was changed.

**Reproduction from the repository root.** Every output must be new; scripts refuse overwriting.

```powershell
.\.venv\Scripts\python.exe -m scripts.probe_topk_numerical_path --suite pairs --out results/phase12/reproduced/pairs
.\.venv\Scripts\python.exe -m scripts.probe_topk_numerical_path --suite controls --higher-precision TopK,Reshape --out results/phase12/reproduced/controls
.\.venv\Scripts\python.exe -m scripts.probe_topk_numerical_path --suite adjacent --higher-precision TopK --repeats 1 --out results/phase12/reproduced/adjacent
.\.venv\Scripts\python.exe -m scripts.evaluate_precision_repair --case subnormal_across_tiles --out results/phase12/reproduced/composition --repeats 3
.\.venv\Scripts\python.exe -m scripts.evaluate_precision_repair --out results/phase12/reproduced/correctness --repeats 3
.\.venv\Scripts\python.exe -m scripts.evaluate_precision_repair --input results/phase5/sift1m/q0000_n1000000_neg_l2_f16.npy --out results/phase12/reproduced/sift --controls --repeats 5
.\.venv\Scripts\python.exe -m scripts.summarize_precision_repair --out results/phase12/reproduced_analysis.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_precision_topk.py tests/test_stable_integer_hierarchy.py tests/test_order_encoded_npu.py tests/test_static_graph.py
.\.venv\Scripts\python.exe -m ruff check scripts/evaluate_precision_repair.py scripts/probe_topk_numerical_path.py scripts/summarize_precision_repair.py src/cats_npu/precision_topk.py src/cats_npu/dyadic_guard.py tests/test_precision_topk.py
```

The summarizer defaults to the archived fast-track root; `--root` accepts another directory with the same session names/layout. New runs need not reproduce exact latency.

```python
import numpy as np
from cats_npu.precision_topk import PrecisionNPUTopK

scores = np.array([0, 2**-24], dtype=np.float16)
runner = PrecisionNPUTopK(n=2, k=1, width=2)
answer = runner.run(scores)  # tested stack returns [1], or explicitly rejects
```

`results/phase12/fast_track/` preserves successful and failed runs. Source snapshots and hashes bind changed code; unchanged dependencies are recoverable from the recorded Git base. Runtime XML snapshots are diagnostic API output, not verified lowering traces. The manifest binds the final reports/source/evidence. Prior results and production baselines remain intact.
