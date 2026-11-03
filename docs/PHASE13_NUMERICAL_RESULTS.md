# Phase 13 numerical fidelity results

**The numerical and graph-composition findings replicate on the physical NPU.** Existing combined precision preservation and stable INT32 pass this bounded panel. Neither passing method has thereby been proved universally correct on the NPU. These are empirical findings about existing techniques.

## Design and evidence

The [frozen protocol](PHASE13_CONFIRMATORY_PROTOCOL.md), commit `d97544ac4514bafbb81b0cf38044982f43fb1379`, precedes new input construction. Three separately launched processes each evaluate 54 numerical inputs with six graph configurations, two NPU calls and one CPU graph control per configuration: **972 case/configuration/session rows, 1,944 physical NPU calls and 972 CPU calls**. These are six executions per unique case/configuration, not 1,944 independent hardware experiments. No construction or runtime errors occurred; all CPU graph controls passed.

Widths are 128, 256 and 512; K is 1, 10 or 100. Six families cover positive and negative subnormals, normal neighbors, mixed signs/ties/extreme finite values, signed zeros, and separated-tile boundary witnesses. Irregular lengths are W+1, 2W−1 or 2W+1, exercising short padded tiles and local/global selection. Arrays, graph XML/BIN, raw CPU/NPU NPZ outputs, configuration and placement are retained under `results/phase13/numerical/session01..03/`. Saved outputs are rechecked by the frozen analysis; the supplementary script partitions overlapping errors descriptively.

Every numerical graph requests `mode=max`, `sort=value`, `stable=true`. The reference uses independent scalar numerical ordering and ascending original index. Checks distinguish unique in-range IDs, score-correct membership, source-consistent returned values, source score ordering and the stable ID sequence. Numerical equality treats +0 and −0 alike; bitwise preservation of the zero sign is not required. INT32 checks returned values against the exact key representation and ranking against its known source-score ordering.

## Counts from the fixed panel

Machine-readable source: `results/phase13/analysis/table2_numerical_correctness.json`; finer rows: `numerical_case_metrics.json`. Each row below has **324 NPU calls** (54 cases × three processes × two repeats).

| Graph configuration | Membership failures | Value failures | Stable sequence failures | Complete contract failures |
|---|---:|---:|---:|---:|
| Default FP16 | 168 | 54 | 174 | 174 |
| FP32 promotion, no preservation | 168 | 54 | 174 | 174 |
| Promotion + TopK preservation | 144 | 174 | 144 | 174 |
| Promotion + Reshape preservation | 168 | 54 | 174 | 174 |
| Promotion + TopK,Reshape preservation | 0 | 0 | 0 | 0 |
| Stable INT32 | 0 | 0 | 0 | 0 |

The complete failure fraction is 174/324 = **53.70%** for each of the first four configurations, in this deliberately difficult panel. It is not a random-input, production-query or population failure-rate estimate; no binomial interval is assigned. Columns overlap and must not be added. The same 29 of 54 input cases fail the complete contract under each unsafe configuration, although the failure category changes with precision settings. Repeated executions replicate cases rather than discover independent bugs.

All configurations pass all nine normal-neighbor cases and all nine signed-zero cases. The three subnormal-containing families (positive, negative, cross-tile) fail all nine cases per family under the first four configurations. Two mixed-tie/extreme cases also fail. No invalid/duplicate ID occurs. With TopK-only preservation, selected source scores remain in descending order even when membership or returned values are wrong. Therefore checking sortedness alone would miss these errors.

For each of the 324 unique case/configuration pairs, all six saved NPU value/ID arrays are byte-identical across the three processes and two repetitions. The supplementary disjoint partition identifies 168 membership-first failures and six source-order-only failures for each default/promotion/Reshape-only setting. TopK-only instead has 144 membership-first failures and 30 value-only failures after valid membership. There is no isolated legal-unstable-tie issue hidden in these counts: this panel explicitly requests stable ordering.

All requests report execution devices exactly `["NPU"]`; CPU calls are explicit reference controls. Compilation or placement alone is not counted as correctness evidence. The raw graph panel performs no host repair.

## Composition ablation

Machine-readable source: `results/phase13/analysis/table3_composition_ablation.json`. The cross-tile construction puts the true winners in the second tile while retaining a third short tile, so a corrupted local-candidate value path can change global selection.

Across **all nine W/K combinations**, each with six NPU executions, TopK-only preservation fails **54/54 memberships and 54/54 value associations**. Reshape-only and default/promotion-only each fail **54/54 memberships** and **18/54 value associations**. Combined preservation and INT32 pass all 54 calls. The nine combined settings are W ∈ {128,256,512}, K ∈ {1,10,100}, with N=2W+1.

This extends the original Phase 12 W=256/K=4 two-tile witness to separately generated widths, K and padded shapes. It establishes a concrete precision-sensitive composition interaction on this stack: preserving the selector alone does not preserve the tested hierarchical graph's output semantics. `Reshape` is mathematically a data-routing operation; this result does **not** mean abstract reshape performs arithmetic. Compiler/runtime handling of the intervening value path can matter even when selectors are protected.

Changing the preservation configuration changes the observed output. The pinned public compiler source audited in Phase 12 is consistent with F32-to-F16 lowering and operation exemptions. Its exact match to the packaged binary and the executed pass sequence were not established. No intermediate instruction trace or unique silicon/compiler root cause is claimed. Existing runtime-model exports are not treated as executable precision traces.

## Bounded adjacent-pair replication

`results/phase13/numerical/adjacent_default/` and `adjacent_protected/` independently rerun the existing `[63487,2]`, K=1 batch over neighboring finite FP16 representations, including the numerical-zero equality. Each directory contains direct FP16 and promoted-input variants, explicit CPU controls and raw arrays.

Default direct and unprotected promoted NPU executions each reproduce **2,046 membership failures and 2,047 complete output failures among 63,487 rows**. Promotion plus TopK preservation passes the promoted direct selector. The FP16 direct-input variant is retained as a separate control; preserving TopK without promoting its FP16 input does not establish subnormal safety. All CPU variants pass. Full per-variant counts are in `results/phase13/supplemental/adjacent_replication.json`.

This is exact coverage of the executed adjacent rows in one batch graph, not exhaustive arbitrary FP16 tensors, graph shapes, K, operator implementations or devices. An adjacency-to-universal-order proof would require an established fixed scalar transformation and correct selector; the opaque executable does not supply those premises. The 2,046 rows cannot be interpreted as 2,046 independent defects.

## Mathematical scope and conclusions

The [inherited mathematical foundation](PHASE12_MATHEMATICAL_FOUNDATION.md) remains valid: finite FP16 promotes exactly to FP32; stable exact local/global Top-K preserves global score-descending/index-ascending rank; correct host rank/value/order certification supplies **exact-or-reject** safety for immutable finite inputs under trusted host execution. Signed zeros are equal, NaN and infinite user scores are unsupported, and internal −infinity padding is below finite data. Integer dimensions/routing must fit the implementation's signed INT32 bounds and device compilation/allocation limits.

That argument distinguishes abstract transformation correctness, operator-semantic assumptions, real executable behavior, and instance certification. It does not prove total NPU acceptance or universal accuracy. The checked methods can reject unsupported input/shape or a wrong device proposal; they never replace a rejected answer with CPU Top-K. The host verification is measured in the performance study.

**C1 is supported** as bounded systematic numerical characterization, including counterexamples and passing controls. **C2 is supported** for the declared graph family/settings, with nine composition witnesses. Conventional mixed precision, integer encoding and certificates remain prior techniques. Safe passing behavior on this panel supplies empirical support, while the host certificate supplies the conditional per-result safety argument.
