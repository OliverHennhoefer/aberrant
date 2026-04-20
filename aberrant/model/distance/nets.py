"""NETS detector for streaming distance-based anomaly detection."""

from __future__ import annotations

from collections import deque
from itertools import product
from typing import TypeAlias

import numpy as np

from aberrant.base.model import BaseModel

_Cell: TypeAlias = tuple[int, ...]
_Entry: TypeAlias = tuple[int, np.ndarray, _Cell, _Cell, int]
_MAX_NEIGHBOR_OFFSETS = 20_000


class NETS(BaseModel):
    """
    NETS-style streaming outlier detector.

    The detector keeps a bounded sliding window and quantizes samples into
    full-space and random-subspace cells. Scores are based on neighborhood
    density within ``radius``:
    ``score = 1 - min(neighbor_count / k, 1)``.

    NETS-style set-based processing is approximated via slide-wise net-effect
    bookkeeping and cell-level upper-bound pruning before exact refinement.

    Notes:
    - Scores are continuous and bounded in ``[0, 1]``.
    - State is bounded by ``window_size``.
    - Feature schema is fixed after the first ``learn_one`` call.
    - Distance metric is Euclidean.
    """

    def __init__(
        self,
        k: int = 50,
        radius: float = 1.5,
        window_size: int = 10_000,
        slide_size: int = 500,
        subspace_dim: int | None = None,
        time_key: str | None = None,
        warm_up_slides: int = 1,
        predict_threshold: float = 0.5,
        seed: int | None = None,
        eps: float = 1e-9,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_size <= k:
            raise ValueError("window_size must be greater than k")
        if slide_size <= 0:
            raise ValueError("slide_size must be positive")
        if subspace_dim is not None and subspace_dim <= 0:
            raise ValueError("subspace_dim must be positive or None")
        if time_key is not None and (not isinstance(time_key, str) or not time_key):
            raise ValueError("time_key must be a non-empty string or None")
        if warm_up_slides <= 0:
            raise ValueError("warm_up_slides must be positive")
        if not (0.0 <= predict_threshold <= 1.0):
            raise ValueError("predict_threshold must be in [0, 1]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.k = k
        self.radius = radius
        self.window_size = window_size
        self.slide_size = slide_size
        self.subspace_dim = subspace_dim
        self.time_key = time_key
        self.warm_up_slides = warm_up_slides
        self.predict_threshold = predict_threshold
        self.seed = seed
        self.eps = eps

        self._radius_sq = self.radius * self.radius

        self._reset_state()

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self.seed)

        self._feature_order: tuple[str, ...] | None = None
        self._subspace_indices: np.ndarray | None = None
        self._active_subspace_dim = 0

        self._window_entries: deque[_Entry] = deque()
        self._entries_by_id: dict[int, np.ndarray] = {}

        self._full_cell_members: dict[_Cell, set[int]] = {}
        self._sub_cell_members: dict[_Cell, set[int]] = {}
        self._full_cell_counts: dict[_Cell, int] = {}
        self._sub_cell_counts: dict[_Cell, int] = {}

        self._upper_bound_cache: dict[tuple[_Cell, _Cell], float] = {}
        self._neighbor_offsets_cache: dict[int, tuple[_Cell, ...]] = {}

        self._next_entry_id = 0
        self._samples_seen = 0
        self._arrival_index = 0
        self._max_learned_time = float("-inf")

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed via ``learn_one``."""
        return self._samples_seen

    def _coerce_time(self, value: float) -> float:
        if not isinstance(value, int | float | np.number):
            raise ValueError("Timestamp value must be numeric")
        as_float = float(value)
        if not np.isfinite(as_float):
            raise ValueError("Timestamp value must be finite")
        return as_float

    def _split_input(self, x: dict[str, float]) -> tuple[float, dict[str, float]]:
        if not x:
            raise ValueError("Input dictionary cannot be empty")

        for key, value in x.items():
            if not isinstance(key, str):
                raise ValueError("All feature keys must be strings")
            if not isinstance(value, int | float | np.number):
                raise ValueError(f"Feature '{key}' is not numeric")
            if not np.isfinite(float(value)):
                raise ValueError(f"Feature '{key}' must be finite")

        if self.time_key is None:
            current_time = float(self._arrival_index + 1)
            features = x
        else:
            if self.time_key not in x:
                raise ValueError(f"Missing time_key '{self.time_key}' in input sample")
            current_time = self._coerce_time(x[self.time_key])
            features = {
                name: value for name, value in x.items() if name != self.time_key
            }

        if not features:
            raise ValueError("Input must contain at least one non-time feature")

        if current_time < self._max_learned_time:
            raise ValueError(
                f"Non-monotonic timestamp: received {current_time}, "
                f"current {self._max_learned_time}"
            )
        return current_time, features

    def _set_or_validate_feature_order(
        self,
        features: dict[str, float],
        *,
        mutate_schema: bool,
    ) -> tuple[str, ...] | None:
        if self._feature_order is None:
            if not mutate_schema:
                return None
            inferred = tuple(sorted(features.keys()))
            self._feature_order = inferred
            return inferred

        received = set(features.keys())
        expected = set(self._feature_order)
        if received != expected:
            expected_keys = ", ".join(self._feature_order)
            received_keys = ", ".join(sorted(features.keys()))
            raise ValueError(
                "Inconsistent feature keys. "
                f"Expected [{expected_keys}], received [{received_keys}]."
            )
        return self._feature_order

    def _init_subspace_if_needed(
        self,
        n_features: int,
        *,
        mutate_schema: bool,
    ) -> np.ndarray | None:
        if self._subspace_indices is not None:
            return self._subspace_indices
        if not mutate_schema:
            return None

        if self.subspace_dim is None:
            selected_dim = max(1, int(np.ceil(n_features / 2.0)))
        else:
            selected_dim = self.subspace_dim

        if selected_dim > n_features:
            raise ValueError(
                f"subspace_dim ({selected_dim}) cannot exceed number of features "
                f"({n_features})"
            )

        indices: np.ndarray
        if selected_dim == n_features:
            indices = np.arange(n_features, dtype=np.intp)
        else:
            indices = np.sort(
                self._rng.choice(n_features, size=selected_dim, replace=False)
            ).astype(np.intp)

        self._subspace_indices = indices
        self._active_subspace_dim = selected_dim
        return indices

    def _vectorize(
        self,
        features: dict[str, float],
        *,
        mutate_schema: bool,
    ) -> np.ndarray | None:
        feature_order = self._set_or_validate_feature_order(
            features,
            mutate_schema=mutate_schema,
        )
        if feature_order is None:
            return None

        vector = np.fromiter(
            (float(features[name]) for name in feature_order),
            dtype=np.float64,
            count=len(feature_order),
        )

        if self._init_subspace_if_needed(
            n_features=vector.shape[0],
            mutate_schema=mutate_schema,
        ) is None:
            return None
        return vector

    def _prepare_input(
        self,
        x: dict[str, float],
        *,
        mutate_schema: bool,
    ) -> tuple[float, np.ndarray | None]:
        current_time, features = self._split_input(x)
        vector = self._vectorize(features, mutate_schema=mutate_schema)
        return current_time, vector

    def _full_cell_id(self, vector: np.ndarray) -> _Cell:
        return tuple(int(value) for value in np.floor(vector / self.radius))

    def _sub_cell_id(self, vector: np.ndarray) -> _Cell:
        if self._subspace_indices is None:
            raise RuntimeError("Subspace indices are not initialized")
        projected = vector[self._subspace_indices]
        return tuple(int(value) for value in np.floor(projected / self.radius))

    def _are_neighbor_cells(self, left: _Cell, right: _Cell) -> bool:
        for left_dim, right_dim in zip(left, right, strict=False):
            if abs(left_dim - right_dim) > 1:
                return False
        return True

    def _neighbor_cells(self, cell: _Cell, cell_counts: dict[_Cell, int]) -> list[_Cell]:
        n_active_cells = len(cell_counts)
        if n_active_cells == 0:
            return []

        n_dims = len(cell)
        neighbor_offsets = 3**n_dims
        if neighbor_offsets <= n_active_cells and neighbor_offsets <= _MAX_NEIGHBOR_OFFSETS:
            offsets = self._neighbor_offsets_for_dims(n_dims)
            return [
                candidate
                for candidate in (
                    tuple(
                        base_coord + offset_coord
                        for base_coord, offset_coord in zip(cell, offset, strict=True)
                    )
                    for offset in offsets
                )
                if candidate in cell_counts
            ]

        return [
            existing_cell
            for existing_cell in cell_counts
            if self._are_neighbor_cells(existing_cell, cell)
        ]

    def _neighbor_offsets_for_dims(self, n_dims: int) -> tuple[_Cell, ...]:
        cached = self._neighbor_offsets_cache.get(n_dims)
        if cached is not None:
            return cached

        offsets = tuple(
            tuple(int(delta) for delta in offset)
            for offset in product((-1, 0, 1), repeat=n_dims)
        )
        self._neighbor_offsets_cache[n_dims] = offsets
        return offsets

    def _add_entry(
        self,
        vector: np.ndarray,
        full_cell: _Cell,
        sub_cell: _Cell,
        slide_id: int,
    ) -> None:
        entry_id = self._next_entry_id
        self._next_entry_id += 1

        self._window_entries.append((entry_id, vector, full_cell, sub_cell, slide_id))
        self._entries_by_id[entry_id] = vector

        self._full_cell_members.setdefault(full_cell, set()).add(entry_id)
        self._sub_cell_members.setdefault(sub_cell, set()).add(entry_id)
        self._full_cell_counts[full_cell] = self._full_cell_counts.get(full_cell, 0) + 1
        self._sub_cell_counts[sub_cell] = self._sub_cell_counts.get(sub_cell, 0) + 1

    def _remove_oldest_entry(self) -> None:
        if not self._window_entries:
            return

        old_id, _vector, full_cell, sub_cell, _slide_id = self._window_entries.popleft()
        self._entries_by_id.pop(old_id, None)

        full_members = self._full_cell_members.get(full_cell)
        if full_members is not None:
            full_members.discard(old_id)
            if not full_members:
                self._full_cell_members.pop(full_cell, None)

        sub_members = self._sub_cell_members.get(sub_cell)
        if sub_members is not None:
            sub_members.discard(old_id)
            if not sub_members:
                self._sub_cell_members.pop(sub_cell, None)

        new_full_count = self._full_cell_counts.get(full_cell, 0) - 1
        if new_full_count > 0:
            self._full_cell_counts[full_cell] = new_full_count
        else:
            self._full_cell_counts.pop(full_cell, None)

        new_sub_count = self._sub_cell_counts.get(sub_cell, 0) - 1
        if new_sub_count > 0:
            self._sub_cell_counts[sub_cell] = new_sub_count
        else:
            self._sub_cell_counts.pop(sub_cell, None)

    def _is_warm(self) -> bool:
        warm_samples = self.warm_up_slides * self.slide_size
        return self._samples_seen >= warm_samples and len(self._window_entries) >= (
            self.k + 1
        )

    def _upper_neighbor_bound(self, full_cell: _Cell, sub_cell: _Cell) -> float:
        cache_key = (full_cell, sub_cell)
        cached = self._upper_bound_cache.get(cache_key)
        if cached is not None:
            return cached

        full_upper = float(
            sum(
                self._full_cell_counts[cell]
                for cell in self._neighbor_cells(full_cell, self._full_cell_counts)
            )
        )
        sub_upper = float(
            sum(
                self._sub_cell_counts[cell]
                for cell in self._neighbor_cells(sub_cell, self._sub_cell_counts)
            )
        )
        upper = min(full_upper, sub_upper)

        if full_cell in self._full_cell_counts or sub_cell in self._sub_cell_counts:
            self._upper_bound_cache[cache_key] = upper

        return upper

    def _count_same_cell_neighbors(self, vector: np.ndarray, full_cell: _Cell) -> float:
        member_ids = self._full_cell_members.get(full_cell)
        if not member_ids:
            return 0.0

        count = 0
        for entry_id in member_ids:
            point = self._entries_by_id[entry_id]
            diff = point - vector
            if float(np.dot(diff, diff)) <= (self._radius_sq + self.eps):
                count += 1
        return float(count)

    def _candidate_ids_for_cells(
        self,
        members: dict[_Cell, set[int]],
        neighbor_cells: list[_Cell],
    ) -> set[int]:
        candidate_ids: set[int] = set()
        for cell in neighbor_cells:
            candidate_ids.update(members.get(cell, set()))
        return candidate_ids

    def _count_exact_neighbors(
        self,
        vector: np.ndarray,
        full_cell: _Cell,
        sub_cell: _Cell,
    ) -> float:
        full_neighbor_cells = self._neighbor_cells(full_cell, self._full_cell_counts)
        sub_neighbor_cells = self._neighbor_cells(sub_cell, self._sub_cell_counts)

        full_candidates = self._candidate_ids_for_cells(
            self._full_cell_members,
            full_neighbor_cells,
        )
        if not full_candidates:
            return 0.0

        sub_candidates = self._candidate_ids_for_cells(
            self._sub_cell_members,
            sub_neighbor_cells,
        )
        candidate_ids = full_candidates & sub_candidates
        if not candidate_ids:
            return 0.0

        count = 0
        for entry_id in candidate_ids:
            point = self._entries_by_id[entry_id]
            diff = point - vector
            if float(np.dot(diff, diff)) <= (self._radius_sq + self.eps):
                count += 1
        return float(count)

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one sample."""
        current_time, vector = self._prepare_input(x, mutate_schema=True)
        if vector is None:
            return

        full_cell = self._full_cell_id(vector)
        sub_cell = self._sub_cell_id(vector)
        slide_id = self._samples_seen // self.slide_size
        self._add_entry(vector, full_cell, sub_cell, slide_id)

        if len(self._window_entries) > self.window_size:
            self._remove_oldest_entry()
        # Keep cache semantics simple: any learning update invalidates all
        # upper-bound entries, while repeated score-only queries can still hit.
        self._upper_bound_cache.clear()

        self._samples_seen += 1
        self._max_learned_time = current_time
        if self.time_key is None:
            self._arrival_index += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for one sample."""
        _current_time, vector = self._prepare_input(x, mutate_schema=False)
        if vector is None or not self._is_warm():
            return 0.0

        full_cell = self._full_cell_id(vector)
        sub_cell = self._sub_cell_id(vector)

        upper_bound = self._upper_neighbor_bound(full_cell, sub_cell)
        if upper_bound < float(self.k):
            return 1.0

        lower_bound = self._count_same_cell_neighbors(vector, full_cell)
        if lower_bound >= float(self.k):
            return 0.0

        neighbors = self._count_exact_neighbors(vector, full_cell, sub_cell)
        score = 1.0 - min(neighbors / float(self.k), 1.0)
        return float(np.clip(score, 0.0, 1.0))

    def predict_one(self, x: dict[str, float]) -> int:
        """Return binary anomaly prediction using ``predict_threshold``."""
        return int(self.score_one(x) >= self.predict_threshold)

    def __repr__(self) -> str:
        return (
            "NETS("
            f"k={self.k}, radius={self.radius}, window_size={self.window_size}, "
            f"slide_size={self.slide_size}, subspace_dim={self.subspace_dim}, "
            f"time_key={self.time_key!r}, warm_up_slides={self.warm_up_slides}, "
            f"predict_threshold={self.predict_threshold}, seed={self.seed}, "
            f"samples_seen={self._samples_seen}, "
            f"active_full_cells={len(self._full_cell_counts)}, "
            f"active_sub_cells={len(self._sub_cell_counts)})"
        )
