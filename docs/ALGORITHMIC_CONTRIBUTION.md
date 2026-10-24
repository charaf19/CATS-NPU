# Submission-aware certified refinement

Date: 2026-10-24  
Status: exact prototype with positive development evidence; novelty and confirmatory performance remain unresolved

## Problem and mechanism

The input is a finite score vector partitioned into static tiles. Ranking is lexicographic: higher score first, then smaller original index. Each tile operation returns its best `r` elements and, when the tile is incomplete, the next local score as an upper bound on every omitted score.

The geometric policy begins with a small candidate budget and doubles each active tile's budget after a failed certificate. This is exact but can spend several high-latency synchronous NPU submissions to avoid transferring a small number of candidates.

The new `costadaptive` policy uses the same candidates, bounds, and exact stopping certificate. It changes only the next budget. When all active tiles fit in one static tile batch, it computes:

- observed submission cost = cumulative synchronous backend time / submissions so far;
- observed extraction cost per retained candidate = candidate extraction time / retained candidates;
- estimated jump cost = extraction cost per candidate multiplied by the extra candidates needed to jump from the next geometric budget to `K`.

If estimated jump cost is no larger than one observed synchronous submission, every active tile jumps directly to `K`. The purpose is to trade additional candidates and host work for avoidance of a possible later device submission. The fields `adaptive_policy`, `cost_aware_jumps`, `cost_model_launch_ns`, and `cost_model_extra_candidate_ns` record the decision.

This is an online heuristic. It does not predict the next certificate outcome, and it does not guarantee lower latency.

## Exactness proof

Let tile `j` contain set `X_j`. At budget `r_j`, the backend returns the exact lexicographic local top `r_j` set `C_j`. If `r_j < min(K, |X_j|)`, let `u_j` be the score of the next local element; otherwise tile `j` is complete. Let `C` be the union of all `C_j`, and let `tau` be the score of the lexicographic K-th element of `C`.

**Local-K lemma.** The union of each tile's exact local top `min(K, |X_j|)` contains the global top K. If an element has local rank greater than K, at least K elements in its own tile rank before it, so it cannot have global rank at most K.

**Certificate.** Suppose `|C| >= K` and every incomplete tile satisfies `u_j < tau`. Every omitted element in tile `j` has score at most `u_j`, hence strictly less than `tau`. It cannot displace any selected element. Completed tiles are covered by the local-K lemma. Therefore the lexicographic top K of `C` equals the global exact top K.

The inequality is strict. If `u_j == tau`, an omitted element can affect index-based tie ordering, so the algorithm refines. A local completed boundary with equal scores is repaired by the measured exact CPU path before the candidate set is trusted.

The submission-aware jump only replaces a proposed larger budget with `K`. Exact local prefixes remain exact, upper bounds remain valid, and the unchanged certificate decides termination. Therefore it preserves the proof above.

## Termination

Every active budget increases by at least one. The geometric step uses `min(K, max(r+1, 2r))`; the optional jump sets it to K. Once every tile reaches `min(K, tile size)`, the local-K lemma terminates the algorithm. The submission-aware policy cannot add rounds relative to geometric refinement and, after a jump, the jumped active tiles are complete after their next update.

## FP16 scope

The proof is over the finite values represented by the input array and assumes the device returns the correct local ordering for those values. The strict NPU implementation rejects FP32 on this measured stack. FP16 rows containing nonzero subnormals receive exact CPU validation because a device disagreement was reproduced previously; disagreement raises an error rather than being silently repaired. These checks establish implementation conditions for applying the proof, not a guarantee about arbitrary NPU arithmetic.

## Complexity

For `T = ceil(N/W)` tiles of width `W`:

- fixed hierarchical candidate storage is at most `T*K`;
- merge work is implementation-dependent and operates on at most `T*K` candidates;
- geometric refinement has at most `ceil(log2 K)+1` candidate levels from budget one, with at most `ceil(A_q/B)` submissions in round `q` for `A_q` active tiles and tile batch `B`;
- submission-aware refinement has the same worst-case candidate bound and no more rounds than geometric refinement;
- after its trigger, all then-active tiles fit in one batch and require at most one further submission for that jump.

No new asymptotic bound is claimed. The contribution being tested is a reduction in expensive static-graph submissions under a measured cost condition.

## Development ablation

The Phase 4 pilot under `results/phase4/policy_pilot` used T256/B32 and crossed two `N`, two `K`, and four distributions. It compared optimized C++, NumPy exact, fixed hierarchical NPU, geometric adaptive NPU, and submission-aware NPU. The run produced 1,600 correct observations with no skips or wrong results.

Across 16 cases, submission-aware was faster than geometric in 13 and slower in three. Geometric/submission-aware speedup had minimum 0.928x, median 1.272x, and maximum 2.263x. The median cost-aware jump count was one.

The mechanism is visible in the hard cases:

- `N=4,096, K=10`, adversarial: submissions fell 5 to 2 and latency 4.876 ms to 2.284 ms;
- `N=16,384, K=10`, adversarial: submissions fell 6 to 3 and latency 6.937 ms to 3.439 ms;
- `N=16,384, K=10`, clustered: submissions fell 5 to 3 and latency 6.029 ms to 3.854 ms.

It can lose when geometric refinement would already finish in the same number of submissions. For `N=4,096, K=10`, normal, both used two submissions, but the jump retained 61 candidates rather than 21 and latency increased from 2.262 ms to 2.438 ms.

Fixed hierarchical remained faster than submission-aware in most small cases and C++ remained much faster in every reported case. This pilot contains one independent process/session, so it establishes a reproducible development effect rather than statistical support.

## Gate decision

The prototype demonstrates a distinct behavior relative to the repository's geometric policy: it explicitly prices extra candidates against observed synchronous submission cost and conditionally eliminates refinements while keeping the certificate unchanged. Current evidence does not establish that this narrow heuristic is a novel Top-K algorithm. The Phase 4 algorithmic-performance gate is **partially supported**; the novelty gate remains open and currently leans negative. A publishable claim would require a stronger formal or systems distinction plus held-out, independent-session evidence.

## Reproduction

```powershell
.\scripts\15_phase4_policy_pilot.ps1
.\scripts\16_analyze_phase4.ps1 -ResultsPath results/phase4/policy_pilot
```
