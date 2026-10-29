# Phase 12 originality assessment

**2026-10-29 — NO-GO for the implemented precision repair as a novel method.** The user's latest direction selected the fastest empirical route. A working conventional baseline was developed; it is not represented as a novel boundary-directed generator.

The candidate promotes FP16 input, retains the existing stable hierarchy, preserves FP32 for TopK and its intervening Reshape, and checks output using the existing host certificate. The operation names were chosen manually from targeted experiments. No automatic analysis, optimal placement, synthesis or executable verification was introduced.

| Capability | Closest primary precedent | Consequence |
|---|---|---|
| Exempt operation types from FP16 lowering | Intel NPU public option and conversion pass [1,2] | The exact compiler control used here already exists. |
| Promote and protect precision-sensitive calculations and outputs | TensorRT precision/output-type control and accuracy guidance [3,4] | Established numerical/compiler engineering. |
| Stable local/global selection | Preserved CATS-NPU/DeepSelect comparison and OpenVINO TopK [5] | A payload-type change does not invent selection. |
| Reject incorrect answers | Existing `certify_rank` and its prior certifying-algorithm audit | The successful-return guarantee is inherited. |
| Cut graphs and observe outputs | Intel accuracy-debug guide [6] | Diagnostic isolation is established. |
| Scale values to a safer magnitude range | Conventional magnitude adjustment [4] | Useful restricted control, without a new representation mechanism. |

The reduction is constructive: replace the manual INT32 graph's encoded parameter with exact FP16-to-FP32 promotion, keep the selection/routing structure, and use Intel's existing option. The host safety argument is unchanged. No new source numerical domain beyond finite FP16, reduced trust base or distinct analysis has been established. New experiments show a useful configuration interaction but do not change this mechanism equivalence.

**New factual scope versus new methodology.** The direct repair, TopK-only value loss, two-tile membership failure and adjacent-pair observations add evidence to the repository. They demonstrate why protecting only the selector can be insufficient. They do not establish that protection of complete precision paths is new. Absence of an identical Intel example in search is not proof of novelty. A future mechanism would need a distinct technical property before more benchmarks are justified.

**Boundary-generation context.** The preceding review did not establish a differentiated generator beyond scalar witnesses, fault propagation and conventional shrinking. NNSmith [7] generates valid graphs and uses differential testing. Propilot [8] structures properties, input rules and oracles. FPRev [9] derives special numerical probes for an explicit accumulation model. Mimicry [10] targets classifier decision boundaries, which is a different task. None is claimed to implement this Intel repair; adapting a ranking oracle alone still supplies no new method over those paradigms. No head-to-head testing-effectiveness campaign was launched after the distinction failed.

**Primary references.** Accessed 2026-10-29; focused comparison, not a systematic-review completeness claim. Source/blob provenance and inspected ranges are in `results/phase12/fast_track/prior_art_review.json`. Public source is not identified as the installed binary's exact revision.

1. Intel, [NPU public precision option](https://github.com/openvinotoolkit/npu_compiler/blob/c74b1265f28dc5a8f08dd50babd2a063cb59ac6e/src/vpux_compiler/include/vpux/compiler/core/public_options.hpp), lines 41–44, revision `c74b1265f28dc5a8f08dd50babd2a063cb59ac6e`.
2. Intel, [ConvertPrecisionToFP16](https://github.com/openvinotoolkit/npu_compiler/blob/c74b1265f28dc5a8f08dd50babd2a063cb59ac6e/src/vpux_compiler/src/dialect/IE/transforms/passes/convert_precision_to_fp16.cpp), normal F32/F64 conversion and explicit operation exemptions.
3. NVIDIA, [TensorRT 8.6.1 Developer Guide, section 6.7.2](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html), layer input/output precision controls.
4. NVIDIA, [Accuracy Considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html), mixed precision and magnitude adjustment; not an Intel correctness proof.
5. Intel, [OpenVINO TopK-11](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/sort/top-k-11.html), configured stability and ordering.
6. Intel, [Accuracy debugging](https://github.com/openvinotoolkit/npu_compiler/blob/c74b1265f28dc5a8f08dd50babd2a063cb59ac6e/guides/how_to_debug_accuracy.md). Its intermediate-buffer facility requires a developer build and was not executed here.
7. Jiawei Liu et al., *NNSmith: Generating Diverse and Valid Test Cases for Deep Learning Compilers*. ASPLOS 2023, vol. 2, pp. 530–543. DOI [10.1145/3575693.3575707](https://doi.org/10.1145/3575693.3575707); [primary paper](https://arxiv.org/html/2207.13066v2).
8. Yuxin Qiu et al., *Tensor Algebraic Property Skeletons: Amplifying Property-Based Testing for AI Compilers*. June 2026 preprint, [arXiv:2606.06747v2](https://arxiv.org/html/2606.06747v2). No external execution or peer-reviewed venue claimed.
9. Peichen Xie, Yanjie Gao, Yang Wang, Jilong Xue, *Revealing Floating-Point Accumulation Orders in Software/Hardware Implementations*. USENIX ATC 2025, pp. 1425–1440. [Primary publication](https://www.usenix.org/conference/atc25/presentation/xie).
10. Oliver Weißl et al., *Targeted Deep Learning System Boundary Testing*. ACM TOSEM, DOI [10.1145/3771557](https://doi.org/10.1145/3771557); [author description and artifact links](https://oliverweissl.github.io/project_showcase/mimicry/). Its image-boundary search is not equated with numerical TopK testing.
