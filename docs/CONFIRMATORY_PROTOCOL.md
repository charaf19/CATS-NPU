# Locked confirmatory protocol

Locked: 2026-10-24, before any access to SIFT1M query indices 8–15 in a performance run.

## Primary question

At SIFT1M `N=1,000,000`, `K=10`, does frozen geometric certified NPU selection reduce host-to-host exact Top-K latency relative to native direct OpenVINO NPU TopK on this system?

The primary estimand is the geometric mean of the paired speedup `direct latency / geometric latency`, first aggregated across all eight queries within each separately initiated session and then across sessions.

The directional research expectation is that geometric is faster, but the inferential test is a two-sided exact sign-flip test over the six session-level mean log speedups. Alpha is 0.05. A 95% session-cluster bootstrap interval will be reported. Statistical support requires `p < 0.05` and a confidence interval entirely above one. Practical support additionally requires aggregate speedup of at least 1.10x.

Only this direct-versus-geometric comparison is primary, so no multiple-comparison correction is applied to it. Fixed NPU, submission-aware NPU, stable CUDA, and optimized C++ are prespecified secondary descriptive comparisons and will not be promoted to primary after results are observed.

## Frozen implementation and data

- SIFT1M archive SHA-256: `92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`.
- Queries: indices 8 through 15, preassigned to `confirmatory_holdout` before their scores were inspected.
- Scores: scaled negative squared L2, converted once to FP16 outside benchmark timing.
- Tile width: 256.
- Tile batch: 32.
- Query batch: 1.
- Initial adaptive budget: 1.
- NPU compiler type: `PLUGIN`.
- Methods and order source: `configs/phase7_confirmatory.json`.
- Exact ranking: score descending, original index ascending.

No parameter may be retuned using these queries. Any implementation or protocol change invalidates the current confirmatory collection and requires a newly named protocol with new untouched data.

## Sessions and timing

Six sequentially initiated sessions are required. Each session runs in fresh benchmark subprocesses and covers every query. Method and query order rotate cyclically by session to reduce fixed-order bias. Each query/method receives two warmups and ten timed observations in two randomized blocks. Repeats and blocks within a process are not independent samples.

Each session records timestamps, current Git commit, Python/NumPy/OpenVINO versions, device metadata, input hashes, raw observations, and correctness. Sessions are executed one after another, so their start times are separated by the duration of a complete session. No claim of independence from slowly varying thermal or operating-system effects is made; session-level pairing and counterbalancing reduce but do not eliminate those threats.

Any skip, runtime failure, incorrect result, changed input digest, or device-placement failure invalidates the affected session. Invalid sessions remain recorded and are not silently replaced. The decision to add sessions must be based on invalidity or prespecified precision needs, not the direction of the result.

## Secondary reporting

Secondary comparisons report session-level geometric-mean speedup and bootstrap intervals without confirmatory significance language:

- geometric versus fixed NPU;
- submission-aware versus geometric NPU;
- geometric NPU versus stable CUDA;
- geometric NPU versus optimized C++.

Median, P95, submissions, and synchronous OpenVINO shares are reported descriptively. No energy, device-resident-input, or multi-query-batch claim is in scope.
