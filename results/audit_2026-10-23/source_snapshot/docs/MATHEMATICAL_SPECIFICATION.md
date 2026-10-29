# Mathematical specification (working version 0.1)

## Problem
For finite scores `s[0..N-1]` with original global indices, find the K
highest-ranked `(score, -index)` pairs, i.e. score descending and index ascending.
All input scores are compared **at the supplied dtype**. Floating-point `NaN`
and +/- Infinity are rejected. Internal padded tiles use -Infinity only as a
sentinel; those padding columns must never enter returned candidates.

## Local-K lemma
Let `T_j` partition the N elements. Let `L_j = TopK(T_j, min(K,|T_j|))`
under the same global deterministic ranking. Then

    TopK(S, K) = TopK(union_j L_j, K).

*Proof.* If an element is not among the top K of its own tile, there are at
least K elements in that tile with better rank, so the omitted element cannot
be among the global Top-K. The union retains every possible global winner.

## Certified partial-tile selection
For each tile, retain r_j local elements and obtain an upper bound `u_j` on
scores of *all* omitted elements. A correct local Top-(r_j+1) operation supplies
a valid bound at rank r_j+1. Let `C` be the union of current candidates and
let `tau` be the Kth largest score in C if C has at least K elements.

An incomplete tile is safe to ignore if `u_j < tau`. For a complete tile
(r_j >= min(K, |T_j|)), its exact stable local-K already suffices by the
local-K lemma. All other incomplete tiles must be expanded until they are
certified or complete. This is a **sufficient** certificate and may expand
more than theoretically necessary in tied workloads.

*Proof.* For an incomplete tile with strict `u_j < tau`, each omitted element
has strictly lower score than the current Kth element, regardless of index.
For completed tiles, an omitted element cannot displace all K locally better
ranked candidates. Therefore, no omitted element can enter the global Top-K.

## Ties and stable correctness
Native OpenVINO TopK can differ in equal-score tie handling. At the final
local-K boundary, if the Kth and (K+1)th local scores tie, the *prototype*
repairs that tile using the exact host CPU reference. This repair count is
explicitly recorded. Interim partially selected tiles do **not** require
stable selection if their omission bounds are strictly below global tau.

Such a repair is a **hybrid CPU/NPU step**, not an NPU-native kernel. A future
study could try a static network that implements stable `(score, index)`
comparisons entirely on device.

## Termination
The budget starts at `r` and increases to `min(K, max(r+1, 2*r))` for
uncertified incomplete tiles. The budget increases monotonically and is
bounded above by `min(K, tile length)`. Therefore each tile is refined
O(log K) times and the algorithm terminates no later than all tiles reaching
local-K completion (barring backend errors).

## Complexity caveat
This is a correctness argument, **not a new asymptotic improvement**.
Hierarchical selection and threshold certificates already have precedents.
The potentially new contribution must be in the static-shape NPU-aware
candidate allocation/scheduling and measured reductions in time/transfer.

## Important numerical caveat
The proof assumes every local selection is correct on the input values.
Hardware may implement floating-point internal precision differently.
Independent correctness tests compare the final hardware indices against an
exact reference computed on the same host-provided FP16/FP32 scores. Any
mismatch is an experimental failure, not an approved approximate result.
