# Engineering notes and limitations

## Research status
Version 0.1 contains a correct reference, hierarchical exact operator,
certified adaptive hybrid prototype, OpenVINO TopK and bitonic graph builders,
multiple benchmark methods, session-based stats, and an optional C++ baseline.
It has NOT yet been evaluated on an Intel NPU. No NPU speedup is asserted.

## NPU compilation support
OpenVINO operation definitions do not guarantee NPU plugin support. Always
probe compile + infer + correctness on the actual NPU/driver/version.
Intel NPU 3720 currently requires static model shapes. A growing cache of
compiled shape variants can incur substantial startup cost, tracked separately.

## Hybrid overhead
Even for `ov_npu_native_adaptive`, the candidate merge, bounds/decisions,
partial-tile management, host input staging, and tie repairs remain on CPU.
This is not a graph fused into one NPU-native kernel. `backend_ns` is the
measured wall time spent in selection backend calls; host routing is measured
only through `elapsed_ns - backend_ns`, which also includes wrappers.

## Pure bitonic fallback
A static sorting network is provided for width 2..256 power of two.
It may run on CPU but fail to compile on NPU; this is a testable limitation.
For production-grade performance, graph fusion and minimal compilation cache
would be further engineering contributions. The network does full tile sorting
and only returns K outputs, so it is potentially far more work than TopK.

## GPU semantics
`torch_cuda` includes host->device input copy and device->host index copy.
Native torch TopK may choose different indices among exact equal scores. These
are flagged as mismatches, not silently re-ranked on CPU. If the aim is GPU
kernel-only comparison, implement a separate device-resident input protocol.

## Statistical limitations
The simple sign-flip test operates on session-level log ratios, and relies
on exchangeability under the null. Thermal drift, platform load, memory
pressure, and driver changes are confounders; control and report them.
Per-session medians are used for paired tests rather than hundreds of
correlated within-session iterations.

## Mathematical novelty
The certificate is standard threshold reasoning and the local-K lemma is
standard hierarchical selection. They are included to establish a sound
baseline and experimental starting point. A fresh and demonstrably superior
NPU-specific allocation / scheduling algorithm is still a research task.

## Literature and implementation links
- OpenVINO NPU documentation: https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- OpenVINO Python TopK API: https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.opset15.topk.html
- DeepSelect: https://github.com/deepseek-ai/DeepSelect
- Classical threshold algorithms: https://arxiv.org/abs/cs/0204046
