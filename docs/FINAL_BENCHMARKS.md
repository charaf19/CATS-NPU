# Final descriptive CPU/GPU/NPU comparison

Date: 2026-10-24  
Status: configurations frozen for confirmatory work; results are descriptive

## Fairness contract

All methods receive the same host-resident FP16 score array. Top-K is exact over those stored values with score descending and original index ascending. Compilation and warmup are excluded. Timed latency begins before method input staging and ends after host indices, including transfer, synchronization, canonical ordering, host merge, certification, and measured tie repair.

The controls are:

- C++ `nth_element` exact Top-K on CPU;
- CUDA `torch.topk` plus host canonical ordering and exact boundary-tie repair;
- native OpenVINO TopK compiled directly for NPU;
- fixed static-tile NPU Top-K with host exact merge;
- geometric candidate refinement with the strict omission certificate;
- submission-aware certified refinement.

The NPU tiled methods are hybrid CPU/NPU implementations. The test does not compare device-resident input because the public path starts from host score arrays for every method. Query batch is one. No energy or peak-memory result is available.

## Scaling outcome

The complete SIFT1M scaling table is in `docs/REAL_DATA_EVALUATION.md`, with raw data under `results/phase5/sift1m_evaluation` and `results/phase6/direct_npu_sift1m`.

At small N, optimized C++ is fastest and native direct NPU is the best NPU path. CUDA becomes fastest by N=131,072 and remains fastest at one million scores. No NPU-versus-C++ or NPU-versus-CUDA crossover was observed.

Within the NPU methods, there is a reproducible descriptive crossover:

- at N=1,024/K=10: direct 0.960 ms, fixed 1.268 ms, geometric 4.416 ms;
- at N=16,384/K=10: direct 2.727 ms, fixed 2.864 ms, geometric 3.731 ms;
- at N=131,072/K=10: direct 17.996 ms, fixed 19.662 ms, geometric 14.933 ms;
- at N=1,000,000/K=10: direct 133.920 ms, fixed 119.223 ms, geometric 97.928 ms.

The crossover also appears on FiQA at N=57,638/K=10: direct 14.294 ms, fixed 10.562 ms, and geometric 8.495 ms. It does not hold on every distribution or K; SciFact K=10 favors fixed NPU at 1.463 ms over direct 1.921 ms and geometric 2.421 ms.

Across 32 SIFT cases, median slowdown was:

- fixed NPU versus C++: 13.92x;
- submission-aware NPU versus C++: 13.69x;
- fixed NPU versus CUDA: 19.89x;
- submission-aware NPU versus CUDA: 20.35x;
- direct NPU versus C++: 14.14x;
- direct NPU versus CUDA: 16.87x.

Across 16 BEIR cases, fixed NPU had median slowdowns of 16.09x versus C++ and 9.58x versus CUDA. Submission-aware had median slowdowns of 16.35x and 13.26x respectively.

## What drives the result

For SIFT K=10, the synchronous OpenVINO interval accounts for these median fractions of end-to-end latency:

| N | Direct NPU | Fixed NPU | Geometric NPU | Submission-aware NPU |
|---:|---:|---:|---:|---:|
| 1,024 | 73.8% | 73.0% | 79.8% | 77.2% |
| 16,384 | 87.0% | 70.4% | 72.0% | 73.8% |
| 131,072 | 89.1% | 60.6% | 67.3% | 65.3% |
| 1,000,000 | 91.7% | 60.4% | 64.6% | 62.3% |

This interval is directly measured but combines input transfer, runtime/driver scheduling, NPU execution, and host wait. It cannot be labeled pure device execution. At large N, adaptive methods issue roughly the same number of static tile batches as fixed selection but retrieve and merge fewer candidates, explaining their advantage over fixed and direct NPU without implying an advantage over CUDA.

The submission-aware policy helps when all uncertain tiles fit in one batch and a jump removes future refinement submissions. It provides little leverage at large N because the active set spans many batches. Its median real-data advantage over geometric is 1.014x on SIFT and 0.961x on BEIR, so it is not the final default policy.

## Answers to the Phase 6 questions

1. **Small N:** optimized C++ is fastest; direct NPU is the best NPU method.
2. **Large N:** CUDA is fastest at 128K and 1M; C++ is second at 1M.
3. **Batching:** tile batching is necessary for the NPU result, but query batching was not implemented.
4. **Adaptive versus fixed:** geometric adaptation helps mainly at larger N and some real distributions; it hurts some small cases.
5. **Real distributions:** they change refinement behavior and make the submission-aware heuristic inconsistent.
6. **Performance crossover:** an NPU-internal direct-to-adaptive crossover is observed; no NPU-to-CPU/GPU crossover is observed.
7. **NPU latency fraction:** synchronous OpenVINO work is roughly 60%–92% of measured NPU latency in the representative SIFT K=10 cases.
8. **Competitive conditions:** none against optimized C++/CUDA under the tested host-input, query-batch-one contract. Certified adaptation is competitive only relative to other NPU mappings at larger N.

## Gate decision

Phase 6 is **complete for descriptive comparison** and **negative for hardware superiority**. The within-NPU crossover is a useful systems result, but it requires independent-session validation before a scientific claim. A second competitive low-level GPU library was not added; stable CUDA PyTorch is the only GPU control. Device-resident inputs and query batching remain unevaluated.
