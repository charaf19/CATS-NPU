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

    def _update(self, states: list[TileState], active: list[int], target: dict[int, int],
                scores: np.ndarray) -> None:
        """Group tiles by static requested K and submit fixed-shape batches."""
        width = self.backend.width
        groups: dict[int, list[int]] = defaultdict(list)
        for i in active:
            b = min(target[i], self.k, states[i].size)
            # One extra output is required to bound omitted elements.
            requested = min(width, min(target[i], self.k) + 1)
            groups[requested].append(i)
        for requested, ids in groups.items():
            for start in range(0, len(ids), self.backend.batch):
                chosen = ids[start:start + self.backend.batch]
                x = np.full((self.backend.batch, width), -np.inf, dtype=self.backend.dtype)
                for row_no, tile_id in enumerate(chosen):
                    st = states[tile_id]
                    x[row_no, :st.size] = scores[st.start:st.start + st.size]
                t0 = perf_counter_ns()
                got_values, got_indices = self.backend.select_rows(x, requested)
                self.backend_ns += perf_counter_ns() - t0
                for row_no, tile_id in enumerate(chosen):
                    st = states[tile_id]
                    b = min(target[tile_id], self.k, st.size)
                    local = np.asarray(got_indices[row_no, :b], dtype=np.int64)
                    if (np.any(local >= st.size) or np.any(local < 0)
                            or len(np.unique(local)) != b):
                        raise RuntimeError(f"Backend yielded invalid tile indices for tile {tile_id}")
                    orig = np.asarray(scores[st.start + local])
                    u = (float(got_values[row_no, b]) if b < st.size else float("-inf"))
                    st.budget = b
                    st.upper_bound = u
                    # Native TopK may be unstable for tied scores at local boundary.
                    # Completed (r=K) tiles require a deterministic exact local TopK.
                    # Repair ONLY when the local boundary can omit an equal-scored item.
                    if b >= min(self.k, st.size) and b < st.size and np.min(orig) == u:
                        repaired = exact_topk(scores[st.start:st.start + st.size], b)
                        local = repaired
                        orig = scores[st.start + local]
                        st.tie_repairs += 1
                        self.cpu_tie_repairs += 1
                    # Canonicalize ordering within retained candidates; selection
                    # correctness is independently verified against reference.
                    sort_idx = np.lexsort((local, -orig.astype(np.float64)))
                    st.ids = st.start + local[sort_idx]
                    st.values = orig[sort_idx]

    @staticmethod
    def _merge(states: list[TileState], k: int) -> tuple[np.ndarray, float, int]:
        cand_ids = np.concatenate([st.ids for st in states if st.ids is not None])
        cand_vals = np.concatenate([st.values for st in states if st.values is not None])
        if cand_ids.size < k:
            return np.empty((0,), dtype=np.int64), float("-inf"), int(cand_ids.size)
        order = topk_with_ids(cand_vals, cand_ids, k)
        return cand_ids[order], float(cand_vals[order[-1]]), int(cand_ids.size)

    def select(self, scores: np.ndarray, adaptive: bool) -> RunResult:
        validate_scores(scores, self.k)
        if np.dtype(scores.dtype) != self.backend.dtype:
            raise ValueError(f"score dtype {scores.dtype} != tile backend dtype {self.backend.dtype}")
        width = self.backend.width
        states = [TileState(i, min(width, scores.size - i)) for i in range(0, scores.size, width)]
        previous = self.backend.calls
        self.backend_ns = 0
        self.cpu_tie_repairs = 0
        first_r = min(self.k, self.initial_budget) if adaptive else self.k
        active = list(range(len(states)))
        target = {i: min(first_r, st.size) for i, st in enumerate(states)}
        rounds = 0
        while True:
            rounds += 1
            self._update(states, active, target, scores)
            merged, tau, candidate_count = self._merge(states, self.k)
            done = [st.budget >= min(self.k, st.size) for st in states]
            # Local-K lemma: union of exact local TopK sets contains global TopK.
            if all(done):
                cert = "local_k_exact"
                break
            # Certified early stop: incomplete tiles can have no omitted item
            # with score >= global Kth candidate. Strict inequality avoids
            # ambiguous index tie-breaking at threshold.
            if candidate_count >= self.k and all(
                completed or st.upper_bound < tau
                for st, completed in zip(states, done)
            ):
                cert = "strict_omission_bound"
                break
            if not adaptive:
                raise AssertionError("Fixed local-K must terminate after first pass")
            active = [i for i, st in enumerate(states)
                      if not done[i] and (candidate_count < self.k or st.upper_bound >= tau)]
            if not active:
                raise RuntimeError("No eligible tile to refine; certificate was not satisfied")
            target = {i: min(self.k, max(states[i].budget + 1, states[i].budget * 2))
                      for i in active}
            if rounds > 1 + (self.k.bit_length() + 1):
                raise RuntimeError("Adaptive selection did not terminate in expected rounds")
        if merged.size != self.k:
            raise AssertionError("Wrong number of selected indices")
        return RunResult(
            indices=merged, certificate=cert, rounds=rounds,
            backend_calls=self.backend.calls - previous,
            candidate_count=candidate_count, cpu_tie_repairs=self.cpu_tie_repairs,
            backend_ns=self.backend_ns, host_routing_and_merge=True,
        )


class MethodRunner:
    """Prepare/execute an end-to-end selection method with compilation outside timing."""
    def __init__(self, method: str, n: int, k: int, tile: int = 256, batch: int = 8,
                 dtype: str = "f16", initial_budget: int = 1):
        self.method, self.n, self.k = method, n, k
        self.tile, self.batch = tile, batch
        self.dtype, self.initial_budget = dtype, initial_budget
        self.backend: TileBackend | None = None
        self.selector: TileSelector | None = None
        self.torch = None
        self.torch_device = None
        self.compile_ms = 0.0
        self.kind = ""
        if method == "cpu_exact":
            self.kind = "exact"
        elif method == "cpp_exact":
            from .native import NativeCppTopK
            self.kind = "cpp"
            self.cpp = NativeCppTopK()
        elif method in ("torch_cpu", "torch_cuda"):
            self.kind = "torch"
            try:
                import torch  # type: ignore
            except ImportError as e:
                raise BackendUnavailable("PyTorch not installed; torch baselines are optional") from e
            self.torch = torch
            self.torch_device = "cuda" if method == "torch_cuda" else "cpu"
            if self.torch_device == "cuda" and not torch.cuda.is_available():
                raise BackendUnavailable("CUDA is not available to PyTorch")
        elif method in ("numpy_hier", "numpy_adaptive"):
            self.kind = method.rsplit("_", 1)[-1]
            self.backend = NumpyTileBackend(tile, batch, dtype)
        else:
            m = re.fullmatch(r"ov_([a-zA-Z0-9.]+)_(native|bitonic)_(direct|hier|adaptive)", method)
            if not m:
                raise ValueError(f"Unknown method {method}")
            device, kernel, self.kind = m.groups()
            if self.kind == "direct":
                if kernel == "bitonic":
                    raise ValueError("bitonic_direct not supported; use bitonic_hier")
                self.backend = OpenVINOTileBackend(n, 1, device, dtype, kernel)
            else:
                self.backend = OpenVINOTileBackend(tile, batch, device, dtype, kernel)
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
        return self.backend.device_info if self.backend is not None else {}

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
            x = torch.from_numpy(np.ascontiguousarray(scores)).to(self.torch_device, dtype=dt)
            ids = torch.topk(x, self.k, largest=True, sorted=True).indices.cpu().numpy()
            return RunResult(ids.astype(np.int64), "torch_unstable_ties", 1, 1, self.k, 0, 0, True)
        if self.selector is None:
            raise AssertionError("Runner not initialized")
        return self.selector.select(scores, adaptive=(self.kind == "adaptive"))
