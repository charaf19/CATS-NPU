# Novelty assessment

Date searched: 2026-10-24  
Status: critical development assessment, not a systematic-review claim

## Comparison boundary

The relevant question is narrower than whether Intel consumer NPU measurements are uncommon. The question is whether certified tile refinement, adaptive candidate allocation, or submission-aware allocation is a clearly differentiated algorithmic or systems contribution.

| Work | Mechanism and hardware | Guarantee and communication behavior | Relationship to CATS-NPU |
|---|---|---|---|
| [DeepSelect](https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.md) (2026) | CUDA/Ascend block scan, threshold filter, candidate buffer, periodic radix compaction; random block order supports an expected work bound | Exact selection; reads every input once; repeated local compactions update a monotone threshold | Establishes that threshold-guided exact candidate reduction, hardware-specific batching, and limiting expensive selection work are prior ideas. It is device-resident and structurally different from host-certified static NPU tile refinement. |
| [RTop-K](https://openreview.net/forum?id=PHg4rAXFVH) (ICLR 2025) | GPU row-wise binary-search-based selection, with an optional early-stop mode | Exact mode is reported separately from approximate early stopping; GPU synchronization/data movement is kernel-oriented | Shows prior exact iterative threshold/search refinement on GPUs. CATS-NPU's omitted-score certificate and host/device scheduling differ, but iterative exact refinement itself is not novel. |
| [RadiK](https://arxiv.org/abs/2501.14336) (ICS 2024) | Scalable GPU radix selection with distribution-aware/adaptive engineering | Exact radix Top-K; designed to avoid GPU scalability and adversarial-distribution problems | Establishes strong exact GPU baselines and prior distribution-aware adaptation. CATS-NPU does not compete with its low-level device-resident implementation in this repository. |
| [Faster Approx. Top-K](https://arxiv.org/abs/2506.04165) (2025) | Cloud TPU two-stage partition Top-1 followed by selection over the reduced set | Approximate; gains parallelism by relaxing exactness | Closely motivates static accelerator-friendly hierarchy but does not provide CATS-NPU's exact certificate. Exactness is the meaningful distinction. |
| [Ascend-RaBitQ](https://arxiv.org/abs/2605.16007) (2026) | Ascend NPU coarse ranking and AI-CPU/CPU fine ranking for billion-scale vector search, with pipelining and load balancing | End-to-end approximate retrieval system rather than isolated exact score-vector Top-K | Establishes prior heterogeneous NPU/CPU ranking and explicit latency hiding. CATS-NPU is narrower and exact over an already computed score vector. |
| [Efficient Top-K Query Processing on Massively Parallel Hardware](https://doi.org/10.1145/3183713.3183735) (SIGMOD 2018) | GPU Top-K algorithms and a cost model for selecting among them | Exact query processing; chooses an implementation using hardware/workload costs | Cost-aware hardware selection is established. CATS-NPU prices a refinement jump inside one certified method rather than selecting a whole algorithm. |
| [AMD Adaptive Top-K](https://rocm.blogs.amd.com/software-tools-optimization/adaptive-topk/README.html) (2026) | MI300X switches between register bitonic and radix methods using K, N, and an empirically refined crossover | Exact GPU implementations; runtime dispatch follows a hardware-specific cost rule | Strong evidence that hardware-aware adaptive Top-K policy selection is already practiced. CATS-NPU's within-query submission/candidate trade is narrower. |
| [Faiss GPU selection](https://arxiv.org/abs/1702.08734) (2017) | CUDA warp-level k-selection integrated with similarity search | Exact device selection for supported K, with device-resident data paths | Provides a mature optimized GPU reference and shows the importance of avoiding host transfers. |
| [Sample-Guided Exact Top-K](https://arxiv.org/abs/2609.08450) (2026 preprint) | Sample-derived boundary, complete-row certification, candidate formation, and exact refinement on GPUs | Exact; explicitly certifies a reduced candidate set before refinement | Particularly close in the use of certification plus refinement. Its fused GPU design and sample-derived boundary differ, but it weakens any broad novelty claim around certified adaptive exact Top-K. |

The search also identified [Split-bucket partition](https://doi.org/10.1007/s11227-024-06031-x), which partitions GPU tiles into buckets and fixes per-block output amounts to reduce global communication. This further establishes hierarchical local work plus bounded merge traffic as prior systems practice.

## What is established here

The project has produced evidence for these engineering findings on one Intel consumer NPU stack:

1. Static tile batching and larger tile width reduce high-level OpenVINO submissions dramatically.
2. Exact host certification can preserve deterministic output over strict NPU local operations.
3. A simple online rule can trade extra candidates for fewer synchronous submissions and improve hard distributions in a development pilot.
4. The same rule can slow easy cases, and fixed hierarchical selection often remains faster.
5. The optimized NPU path remains much slower than optimized C++ and stable CUDA controls.

These are reproducible systems observations. They are not yet evidence of a new general Top-K algorithm.

## Unresolved and negative findings

- Local Top-K union, threshold filtering, iterative exact refinement, adaptive candidate control, hardware cost models, and hybrid accelerator/CPU ranking all have substantial prior art.
- The current cost model is a one-step heuristic derived from measurements already accumulated in the same query. It has no new competitive-ratio or expected-work bound.
- The high-level OpenVINO execution interval cannot separate transfer, firmware scheduling, device execution, and host wait, limiting architectural attribution.
- Evaluation currently covers one consumer NPU and one independent Phase 4 session.
- No direct publication with the identical rule was located in this bounded search, but failure to locate one is not proof of novelty.

## Phase 4 decision

**Current answer: no clearly established algorithmic novelty beyond established certified/adaptive Top-K ideas.** The strongest possible contribution at this stage is a narrowly scoped systems study of exact Top-K on a static-shape consumer NPU, including a measured submission/candidate tradeoff and negative CPU/GPU comparison. That contribution still needs real-score evidence, scaling, independent sessions, and a clearer model of when submissions dominate.

A defensible novelty claim would require at least one of:

- a general allocation policy with a proved property beyond exactness and termination;
- a device-resident design that removes the host refinement boundary and demonstrates a new static-graph technique;
- a validated performance model that predicts the optimal candidate/submission trade across held-out workloads and hardware;
- broader evidence showing a repeatable regime that prior GPU/TPU/NPU approaches do not address.

Until then, the research gate is **partially supported for systems insight and failed for a demonstrated novel algorithm**.
