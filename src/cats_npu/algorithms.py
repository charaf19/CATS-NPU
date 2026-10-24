"""Certified adaptive and exact hierarchical Top-K selection.

The selection certificate relies on a correct per-tile Top-r primitive, operating
on the same finite input scores. FP16 scores are compared against FP16 inputs;
FP32 reference agreement must be tested separately and may fail on hardware
with lower internal compute precision.

IMPORTANT: these algorithms merge and route on the HOST. Using an NPU backend
makes them HYBRID CPU/NPU algorithms, not fully NPU-resident graph programs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
import re
from time import perf_counter_ns
import numpy as np

from .backends import TileBackend, NumpyTileBackend, OpenVINOTileBackend, BackendUnavailable
from .reference import validate_scores, exact_topk, topk_with_ids


@dataclass
class RunResult:
    indices: np.ndarray
    certificate: str
    rounds: int
    backend_calls: int
    candidate_count: int
    cpu_tie_repairs: int
    backend_ns: int
    host_routing_and_merge: bool
    host_input_preparation_ns: int = 0
    backend_input_cast_ns: int = 0
    backend_execute_sync_ns: int = 0
    backend_output_copy_ns: int = 0
    backend_validation_ns: int = 0
    cpu_exact_validation_rows: int = 0
    candidate_extraction_ns: int = 0
    merge_ns: int = 0
    certification_ns: int = 0
    device_submissions: int = 0
    staged_input_bytes: int = 0
    retrieved_output_bytes: int = 0
    adaptive_policy: str = "fixed"
    cost_aware_jumps: int = 0
    cost_model_launch_ns: int = 0
    cost_model_extra_candidate_ns: int = 0

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("indices")
        return d


@dataclass
class TileState:
    start: int
    size: int
    budget: int = 0
    upper_bound: float = float("inf")
    ids: np.ndarray | None = None
    values: np.ndarray | None = None
    tie_repairs: int = 0


class TileSelector:
    def __init__(self, backend: TileBackend, k: int, initial_budget: int = 1):
        self.backend = backend
        self.k = int(k)
        self.initial_budget = int(initial_budget)
        if self.initial_budget < 1:
            raise ValueError("initial_budget must be positive")
        self.backend_ns = 0
        self.cpu_tie_repairs = 0
        self.profile: dict[str, int] = {}

    def _reset_profile(self) -> None:
        self.profile = {
            "host_input_preparation_ns": 0,
            "backend_input_cast_ns": 0,
            "backend_execute_sync_ns": 0,
            "backend_output_copy_ns": 0,
            "backend_validation_ns": 0,
            "cpu_exact_validation_rows": 0,
            "candidate_extraction_ns": 0,
            "merge_ns": 0,
            "certification_ns": 0,
            "staged_input_bytes": 0,
            "retrieved_output_bytes": 0,
        }

    def _update(self, states: list[TileState], active: list[int], target: dict[int, int],
                scores: np.ndarray) -> None:
        """Group tiles by static requested K and submit fixed-shape batches."""
        width = self.backend.width
        groups: dict[int, list[int]] = defaultdict(list)
        for i in active:
            # One extra output is required to bound omitted elements.
            requested = min(width, min(target[i], self.k) + 1)
            groups[requested].append(i)
        for requested, ids in groups.items():
            for start in range(0, len(ids), self.backend.batch):
                prep_start = perf_counter_ns()
                chosen = ids[start:start + self.backend.batch]
                x = np.full((self.backend.batch, width), -np.inf, dtype=self.backend.dtype)
                for row_no, tile_id in enumerate(chosen):
                    st = states[tile_id]
                    x[row_no, :st.size] = scores[st.start:st.start + st.size]
                self.profile["host_input_preparation_ns"] += perf_counter_ns() - prep_start
                t0 = perf_counter_ns()
                got_values, got_indices = self.backend.select_rows(x, requested)
                self.backend_ns += perf_counter_ns() - t0
                backend_profile = self.backend.last_profile
                for source, target_name in (
                    ("input_cast_ns", "backend_input_cast_ns"),
                    ("execute_sync_ns", "backend_execute_sync_ns"),
                    ("output_copy_ns", "backend_output_copy_ns"),
                    ("validation_ns", "backend_validation_ns"),
                    ("cpu_exact_validation_rows", "cpu_exact_validation_rows"),
                    ("staged_input_bytes", "staged_input_bytes"),
                    ("retrieved_output_bytes", "retrieved_output_bytes"),
                ):
                    self.profile[target_name] += int(backend_profile.get(source, 0))
                extraction_start = perf_counter_ns()
                # Rows in one static request normally share a candidate budget.
                # Process each distinct budget as an array to avoid one NumPy
                # sort/uniqueness allocation per tile. Short final tiles form at
                # most one additional budget group.
                sizes = np.asarray([states[tile_id].size for tile_id in chosen],
                                   dtype=np.int64)
                starts = np.asarray([states[tile_id].start for tile_id in chosen],
                                    dtype=np.int64)
                budgets = np.asarray([
                    min(target[tile_id], self.k, states[tile_id].size)
                    for tile_id in chosen
                ], dtype=np.int64)
                for budget in np.unique(budgets):
                    positions = np.flatnonzero(budgets == budget)
                    local = np.asarray(got_indices[positions, :budget], dtype=np.int64)
                    group_sizes = sizes[positions]
                    invalid = np.any((local < 0) | (local >= group_sizes[:, None]), axis=1)
                    duplicate = (
                        np.any(np.diff(np.sort(local, axis=1), axis=1) == 0, axis=1)
                        if budget > 1 else np.zeros(positions.size, dtype=bool)
                    )
                    if np.any(invalid | duplicate):
                        bad_position = positions[int(np.flatnonzero(invalid | duplicate)[0])]
                        raise RuntimeError(
                            f"Backend yielded invalid tile indices for tile {chosen[bad_position]}"
                        )
                    group_starts = starts[positions]
                    orig = np.asarray(scores[group_starts[:, None] + local])
                    upper = np.full(positions.size, -np.inf, dtype=np.float64)
                    unfinished = budget < group_sizes
                    if np.any(unfinished):
                        upper[unfinished] = got_values[positions[unfinished], budget]
                    completed = budget >= np.minimum(self.k, group_sizes)
                    repair = completed & unfinished & (np.min(orig, axis=1) == upper)
                    ordinary = np.flatnonzero(~repair)
                    if ordinary.size:
                        order = np.lexsort(
                            (local[ordinary], -orig[ordinary].astype(np.float64)), axis=1
                        )
                        sorted_local = np.take_along_axis(local[ordinary], order, axis=1)
                        sorted_values = np.take_along_axis(orig[ordinary], order, axis=1)
                        for group_row, local_row, value_row in zip(
                            ordinary, sorted_local, sorted_values, strict=True
                        ):
                            position = positions[group_row]
                            st = states[chosen[position]]
                            st.budget = int(budget)
                            st.upper_bound = float(upper[group_row])
                            st.ids = st.start + local_row
                            st.values = value_row
                    for group_row in np.flatnonzero(repair):
                        position = positions[group_row]
                        st = states[chosen[position]]
                        repaired = exact_topk(scores[st.start:st.start + st.size], int(budget))
                        st.budget = int(budget)
                        st.upper_bound = float(upper[group_row])
                        st.ids = st.start + repaired
                        st.values = scores[st.start + repaired]
                        st.tie_repairs += 1
                        self.cpu_tie_repairs += 1
                self.profile["candidate_extraction_ns"] += perf_counter_ns() - extraction_start

    @staticmethod
    def _merge(states: list[TileState], k: int) -> tuple[np.ndarray, float, int]:
        cand_ids = np.concatenate([st.ids for st in states if st.ids is not None])
        cand_vals = np.concatenate([st.values for st in states if st.values is not None])
        if cand_ids.size < k:
            return np.empty((0,), dtype=np.int64), float("-inf"), int(cand_ids.size)
        order = topk_with_ids(cand_vals, cand_ids, k)
        return cand_ids[order], float(cand_vals[order[-1]]), int(cand_ids.size)

    def select(self, scores: np.ndarray, adaptive: bool,
               policy: str = "geometric") -> RunResult:
        validate_scores(scores, self.k)
        if policy not in ("geometric", "submission_aware"):
            raise ValueError("adaptive policy must be geometric or submission_aware")
        if np.dtype(scores.dtype) != self.backend.dtype:
            raise ValueError(f"score dtype {scores.dtype} != tile backend dtype {self.backend.dtype}")
        width = self.backend.width
        states = [TileState(i, min(width, scores.size - i)) for i in range(0, scores.size, width)]
        previous = self.backend.calls
        self.backend_ns = 0
        self.cpu_tie_repairs = 0
        self._reset_profile()
        first_r = min(self.k, self.initial_budget) if adaptive else self.k
        active = list(range(len(states)))
        # Keep one shared static request size per round. Short final tiles are
        # padded, so capping the target by tile length only creates extra,
        # unprepared K shapes and prevents batching.
        target = {i: first_r for i in range(len(states))}
        rounds = 0
        cost_aware_jumps = 0
        cost_model_launch_ns = 0
        cost_model_extra_candidate_ns = 0
        while True:
            rounds += 1
            self._update(states, active, target, scores)
            merge_start = perf_counter_ns()
            merged, tau, candidate_count = self._merge(states, self.k)
            self.profile["merge_ns"] += perf_counter_ns() - merge_start
            certificate_start = perf_counter_ns()
            done = [st.budget >= min(self.k, st.size) for st in states]
            # Local-K lemma: union of exact local TopK sets contains global TopK.
            if all(done):
                cert = "local_k_exact"
                self.profile["certification_ns"] += perf_counter_ns() - certificate_start
                break
            # Certified early stop: incomplete tiles can have no omitted item
            # with score >= global Kth candidate. Strict inequality avoids
            # ambiguous index tie-breaking at threshold.
            if candidate_count >= self.k and all(
                completed or st.upper_bound < tau
                for st, completed in zip(states, done, strict=True)
            ):
                cert = "strict_omission_bound"
                self.profile["certification_ns"] += perf_counter_ns() - certificate_start
                break
            if not adaptive:
                raise AssertionError("Fixed local-K must terminate after first pass")
            active = [i for i, st in enumerate(states)
                      if not done[i] and (candidate_count < self.k or st.upper_bound >= tau)]
            if not active:
                raise RuntimeError("No eligible tile to refine; certificate was not satisfied")
            geometric_target = {
                i: min(self.k, max(states[i].budget + 1, states[i].budget * 2))
                for i in active
            }
            target = geometric_target
            if policy == "submission_aware" and len(active) <= self.backend.batch:
                submissions = max(self.backend.calls - previous, 1)
                retained = sum(st.budget for st in states if st.ids is not None)
                cost_model_launch_ns = int(
                    self.profile["backend_execute_sync_ns"] / submissions
                )
                candidate_ns = self.profile["candidate_extraction_ns"] / max(retained, 1)
                extra_candidates = sum(
                    self.k - geometric_target[i] for i in active
                )
                cost_model_extra_candidate_ns = int(candidate_ns * extra_candidates)
                # One full active set fits in one static submission. Spend the
                # additional output/merge work now when it costs no more than
                # one observed synchronous submission, eliminating any later
                # refinement submission if the next certificate still fails.
                if (extra_candidates > 0
                        and cost_model_extra_candidate_ns <= cost_model_launch_ns):
                    target = {i: self.k for i in active}
                    cost_aware_jumps += 1
            if rounds > 1 + (self.k.bit_length() + 1):
                raise RuntimeError("Adaptive selection did not terminate in expected rounds")
            self.profile["certification_ns"] += perf_counter_ns() - certificate_start
        if merged.size != self.k:
            raise AssertionError("Wrong number of selected indices")
        return RunResult(
            indices=merged, certificate=cert, rounds=rounds,
            backend_calls=self.backend.calls - previous,
            candidate_count=candidate_count, cpu_tie_repairs=self.cpu_tie_repairs,
            backend_ns=self.backend_ns, host_routing_and_merge=True,
            device_submissions=(self.backend.calls - previous
                                if isinstance(self.backend, OpenVINOTileBackend) else 0),
            adaptive_policy=(policy if adaptive else "fixed"),
            cost_aware_jumps=cost_aware_jumps,
            cost_model_launch_ns=cost_model_launch_ns,
            cost_model_extra_candidate_ns=cost_model_extra_candidate_ns,
            **self.profile,
        )


class MethodRunner:
    """Prepare/execute an end-to-end selection method with compilation outside timing."""
    def __init__(self, method: str, n: int, k: int, tile: int = 256, batch: int = 8,
                 dtype: str = "f16", initial_budget: int = 1,
                 compiler_type: str = "DEFAULT"):
        self.method, self.n, self.k = method, n, k
        self.tile, self.batch = tile, batch
        self.dtype, self.initial_budget = dtype, initial_budget
        self.backend: TileBackend | None = None
        self.selector: TileSelector | None = None
        self.torch = None
        self.torch_device = None
        self.torch_stable = False
        self.compile_ms = 0.0
        self.kind = ""
        self.execution_mode = "configured"
        if method == "cpu_exact":
            self.kind = "exact"
        elif method == "cpp_exact":
            from .native import NativeCppTopK
            self.kind = "cpp"
            self.cpp = NativeCppTopK()
        elif method in ("torch_cpu", "torch_cuda", "torch_cpu_stable", "torch_cuda_stable"):
            self.kind = "torch"
            try:
                import torch  # type: ignore
            except ImportError as e:
                raise BackendUnavailable("PyTorch not installed; torch baselines are optional") from e
            self.torch = torch
            self.torch_device = "cuda" if "cuda" in method else "cpu"
            self.torch_stable = method.endswith("_stable")
            if self.torch_device == "cuda" and not torch.cuda.is_available():
                raise BackendUnavailable("CUDA is not available to PyTorch")
        elif method in ("numpy_hier", "numpy_adaptive", "numpy_costadaptive"):
            self.kind = method.rsplit("_", 1)[-1]
            self.backend = NumpyTileBackend(tile, batch, dtype)
        else:
            m = re.fullmatch(
                r"ov_([a-zA-Z0-9.]+)_(native|bitonic)_"
                r"(?:(sequential|batched)_)?(direct|hier|adaptive|costadaptive)", method,
            )
            if not m:
                raise ValueError(f"Unknown method {method}")
            device, kernel, mode, self.kind = m.groups()
            self.execution_mode = mode or "configured"
            backend_batch = 1 if mode == "sequential" else batch
            if mode == "batched" and batch < 2:
                raise ValueError("batched OpenVINO methods require --batch >= 2")
            if self.kind == "direct":
                if kernel == "bitonic":
                    raise ValueError("bitonic_direct not supported; use bitonic_hier")
                self.backend = OpenVINOTileBackend(
                    n, 1, device, dtype, kernel, compiler_type=compiler_type
                )
            else:
                self.backend = OpenVINOTileBackend(
                    tile, backend_batch, device, dtype, kernel, compiler_type=compiler_type
                )
        if self.backend is not None:
            self.selector = TileSelector(self.backend, k, initial_budget)

    def prepare(self) -> None:
        if self.backend is None:
            return
        t0 = perf_counter_ns()
        if self.kind == "direct":
            ks = {min(self.n, self.k + 1)}
        elif self.kind == "hier":
            ks = {min(self.tile, self.k + 1)}
        else:
            ks = set()
            r = min(self.k, self.initial_budget)
            while True:
                ks.add(min(self.tile, r + 1))
                if r >= self.k:
                    break
                r = min(self.k, max(r + 1, 2 * r))
        self.backend.prepare(ks)
        self.compile_ms = (perf_counter_ns() - t0) / 1e6

    @property
    def device_info(self) -> dict:
        if self.kind == "cpp":
            return {"requested": "CPU", "library": str(self.cpp.path)}
        if self.kind == "torch":
            info = {
                "requested": self.torch_device,
                "torch_version": self.torch.__version__,
                "stable_tie_policy": self.torch_stable,
            }
            if self.torch_device == "cuda":
                device_index = self.torch.cuda.current_device()
                info.update({
                    "device_index": device_index,
                    "device_name": self.torch.cuda.get_device_name(device_index),
                    "compute_capability": list(
                        self.torch.cuda.get_device_capability(device_index)
                    ),
                })
            return info
        if self.backend is None:
            return {}
        return {
            **self.backend.device_info,
            "execution_mode": self.execution_mode,
            "tile_batch": self.backend.batch,
            "adaptive_policy": (
                "submission_aware" if self.kind == "costadaptive"
                else "geometric" if self.kind == "adaptive"
                else "fixed"
            ),
        }

    def run(self, scores: np.ndarray) -> RunResult:
        validate_scores(scores, self.k)
        if self.kind == "exact":
            return RunResult(exact_topk(scores, self.k), "numpy_exact", 1, 0, self.k, 0, 0, True)
        if self.kind == "cpp":
            return RunResult(self.cpp.select(scores, self.k), "cpp_nth_element_exact",
                             1, 1, self.k, 0, 0, True)
        if self.kind == "torch":
            torch = self.torch
            # Host-input, host-output semantics. CUDA synchronization is part of
            # run time because indices are copied back to CPU.
            dt = torch.float16 if scores.dtype == np.float16 else torch.float32
            execute_start = perf_counter_ns()
            x = torch.from_numpy(np.ascontiguousarray(scores)).to(self.torch_device, dtype=dt)
            ids = torch.topk(x, self.k, largest=True, sorted=True).indices.cpu().numpy()
            execute_ns = perf_counter_ns() - execute_start
            ids = ids.astype(np.int64)
            repairs = 0
            certificate = "torch_unstable_ties"
            repair_ns = 0
            if self.torch_stable:
                repair_start = perf_counter_ns()
                selected_values = scores[ids]
                order = np.lexsort((ids, -selected_values.astype(np.float64)))
                ids = ids[order]
                threshold = scores[ids[-1]]
                selected_at_threshold = int(np.count_nonzero(scores[ids] == threshold))
                all_at_threshold = int(np.count_nonzero(scores == threshold))
                if all_at_threshold > selected_at_threshold:
                    ids = exact_topk(scores, self.k)
                    repairs = 1
                    certificate = "torch_boundary_tie_cpu_repair"
                else:
                    certificate = "torch_topk_host_canonical_order"
                repair_ns = perf_counter_ns() - repair_start
            return RunResult(
                ids, certificate, 1, 1, self.k, repairs, execute_ns, True,
                backend_execute_sync_ns=execute_ns,
                candidate_extraction_ns=repair_ns,
                device_submissions=1 if self.torch_device == "cuda" else 0,
                staged_input_bytes=int(scores.nbytes) if self.torch_device == "cuda" else 0,
                retrieved_output_bytes=(self.k * np.dtype(np.int64).itemsize
                                        if self.torch_device == "cuda" else 0),
            )
        if self.selector is None:
            raise AssertionError("Runner not initialized")
        adaptive = self.kind in ("adaptive", "costadaptive")
        policy = "submission_aware" if self.kind == "costadaptive" else "geometric"
        return self.selector.select(scores, adaptive=adaptive, policy=policy)
