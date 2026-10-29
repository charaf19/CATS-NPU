# Phase 12 numerical and checking contracts

**These are conventional supporting arguments, not claimed novel theorems.** The compiled executable is not formally verified.

For immutable finite FP16 `x[0..N-1]`, rank i before j when `x[i]>x[j]`, or their numerical values are equal and `i<j`. Both signed zeros are equal. NaNs/infinities are excluded from user input. `T_K(x)` is the first K IDs. The checked runner requires this output order and rejects an incorrectly ordered device sequence.

**Diagnostic oracle.** With stable=false, a correct K-set contains every source score strictly above the source Kth score tau, none below tau, and K unique in-range IDs. Boundary-equal choices are allowed. Value-sorted output must be nonincreasing on the source scores. Returned values must equal the source at those IDs. Stable=true adds the exact tie-priority sequence. Construction/compile/runtime errors are distinct from numerical mismatches. The executed focus is max/value sorting; no claim covers unexecuted min or sort/index/none settings.

The independent scalar reference uses Python values directly: FP16 is exactly represented by Python floats, and integer `.item()` retains arbitrary integer values. It does not cast arbitrary INT64 controls through FP64. Oracle tests accept legal unspecified ties and reject wrong membership, values, duplicated IDs and ordering.

**Promotion.** Binary16 has at most 11 significant bits versus binary32's 24; its smallest nonzero magnitude 2^-24 is normal in binary32. Every finite FP16 value therefore promotes exactly, preserving numerical order, equality and the zero class. This concerns supplied FP16 scores, not retrieval order before quantization.

**Hierarchy.** An item omitted by exact stable local K already has K better pairs in its tile, so cannot be a global winner. Every global winner is retained. Local stable value sorting orders equal candidates by local index; contiguous tile-major concatenation therefore orders equal candidates by global index. Exact stable final K plus correct ID gather returns `T_K(x)`. Internal -infinity padding follows finite values; a short tile retains all its valid items if it has fewer than K.

This graph argument assumes promotion, comparisons, routing, index arithmetic and output conversions obey their semantics. A precision option alone does not prove that premise. The two-tile experiment shows the danger of preserving selectors while leaving the intervening value path subject to lowering.

Require integer N,K,W, `1<=K<=min(N,W)`, and padded size <=2^31-1 for signed INT32 routing/dimensions. For N<=W, use one direct row of length N. Abstract bounds do not promise compilation or allocation for every valid shape.

**Exact-or-reject proof.** First require K valid unique IDs and source-consistent returned values. Order their source pairs; let j be the worst selected ID. Compute

    R = count(x > x[j]) + count(x[:j] == x[j]).

The selected set already contains K-1 pairs better than j. If the full input has exactly K-1 such pairs, the set exhausts them and equals `T_K(x)`; otherwise reject. The runner additionally compares raw device order to the certified order and rejects differences. Thus successful returns are exact under trusted host comparison/checker execution and immutable input, independently of NPU selection correctness. No total-acceptance or universal-device guarantee follows. The host checks/sorts only the proposed K candidates and counts source ranks; it does not select a replacement answer from N scores.

**Dyadic control.** Choose the smallest nonnegative s making every nonzero `abs(x)*2^s >= 2^-14`. Accept only if maximum scaled magnitude <=65504 and exact round trip holds. A positive exact scale preserves ordering/equality. `[2^-24,65504]` requires s>=10 for the first condition but disallows positive s for the second, so rejects. Every scalar individually passing a test does not imply every vector admits a common scale.

**Finite-test boundary.** The adjacent batch covers 63,487 pairs formed from 63,488 finite words, including duplicate numerical zero. Passing establishes those executed rows only. Reducing arbitrary opaque behavior to adjacency would require justified fixed-elementwise-map and exact-selector assumptions; the suite cannot establish its own premises. Abstract transformation correctness, observed device agreement, a source-consistent explanation, and unique hardware attribution remain separate evidence levels.
