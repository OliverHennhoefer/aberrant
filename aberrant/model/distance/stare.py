"""STARE-style detector for streaming local outlier detection."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TypeAlias

import numpy as np

from aberrant.base.model import BaseModel

_Cell: TypeAlias = tuple[int, ...]
_Entry: TypeAlias = tuple[int, np.ndarray, _Cell]


class STARE(BaseModel):
    """
    STARE-style local outlier detector for streaming data.

    The detector keeps a bounded sliding window and quantizes points into
    radius-sized grid cells. Scores are based on the number of neighbors within
    ``radius`` in the current window:
    ``score = 1 - min(neighbor_count / k, 1)``.

    A lightweight stationary-region skipping approximation is implemented via
    per-cell cache invalidation at slide boundaries: cells whose occupancy
    changed less than ``skip_threshold`` keep cached neighbor estimates. Unlike
    the paper, this class uses radius-neighbor counts rather than kernel-density
    estimates and returns a per-query score rather than a top-n outlier set.

    Notes:
    - Scores are continuous and bounded in ``[0, 1]``.
    - State is bounded by ``window_size``.
    - Feature schema is fixed after the first ``learn_one`` call.

    References:
        Yoon, S., Lee, J.-G., & Lee, B. S. (2020). Ultrafast Local Outlier
        Detection from a Data Stream with Stationary Region Skipping.
        https://doi.org/10.1145/3394486.3403171
    """

    def __init__(
        self,
        k: int = 50,
        radius: float = 1.0,
        window_size: int = 2048,
        slide_size: int = 128,
        skip_threshold: float = 0.1,
        time_key: str | None = None,
        warm_up_slides: int = 1,
        predict_threshold: float = 0.5,
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
        if not (0.0 <= skip_threshold <= 1.0):
            raise ValueError("skip_threshold must be in [0, 1]")
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
        self.skip_threshold = skip_threshold
        self.time_key = time_key
        self.warm_up_slides = warm_up_slides
        self.predict_threshold = predict_threshold
        self.eps = eps

        self._radius_sq = self.radius * self.radius

        self._reset_state()

    def _reset_state(self) -> None:
        self._feature_order: tuple[str, ...] | None = None
        self._window_entries: deque[_Entry] = deque()

        self._entries_by_id: dict[int, np.ndarray] = {}
        self._entry_cell_by_id: dict[int, _Cell] = {}
        self._cell_members: dict[_Cell, set[int]] = {}
        self._cell_counts: dict[_Cell, int] = {}

        self._slide_add: defaultdict[_Cell, int] = defaultdict(int)
        self._slide_remove: defaultdict[_Cell, int] = defaultdict(int)
        self._prev_cell_counts: dict[_Cell, int] = {}
        self._dirty_cells: set[_Cell] = set()
        self._neighbor_cache: dict[_Cell, float] = {}

        self._next_entry_id: int = 0
        self._samples_seen: int = 0
        self._arrival_index: int = 0
        self._max_learned_time: float = float("-inf")
        self._current_slide_count: int = 0

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

        return np.fromiter(
            (float(features[name]) for name in feature_order),
            dtype=np.float64,
            count=len(feature_order),
        )

    def _prepare_input(
        self,
        x: dict[str, float],
        *,
        mutate_schema: bool,
    ) -> tuple[float, np.ndarray | None]:
        current_time, features = self._split_input(x)
        vector = self._vectorize(features, mutate_schema=mutate_schema)
        return current_time, vector

    def _cell_id(self, vector: np.ndarray) -> _Cell:
        return tuple(int(value) for value in np.floor(vector / self.radius))

    def _are_neighbor_cells(self, left: _Cell, right: _Cell) -> bool:
        for left_dim, right_dim in zip(left, right, strict=False):
            if abs(left_dim - right_dim) > 1:
                return False
        return True

    def _candidate_cells(self, cell: _Cell) -> list[_Cell]:
        candidates: list[_Cell] = []
        for existing_cell in self._cell_members:
            if self._are_neighbor_cells(existing_cell, cell):
                candidates.append(existing_cell)
        return candidates

    def _count_neighbors_within_radius(self, vector: np.ndarray, cell: _Cell) -> float:
        if not self._cell_members:
            return 0.0

        count = 0
        for candidate_cell in self._candidate_cells(cell):
            member_ids = self._cell_members.get(candidate_cell)
            if not member_ids:
                continue

            for entry_id in member_ids:
                point = self._entries_by_id[entry_id]
                diff = point - vector
                if float(np.dot(diff, diff)) <= (self._radius_sq + self.eps):
                    count += 1

        return float(count)

    def _add_entry(self, vector: np.ndarray, cell: _Cell) -> None:
        entry_id = self._next_entry_id
        self._next_entry_id += 1

        self._window_entries.append((entry_id, vector, cell))
        self._entries_by_id[entry_id] = vector
        self._entry_cell_by_id[entry_id] = cell
        self._cell_members.setdefault(cell, set()).add(entry_id)
        self._cell_counts[cell] = self._cell_counts.get(cell, 0) + 1

        self._slide_add[cell] += 1
        self._dirty_cells.add(cell)
        self._neighbor_cache.pop(cell, None)

    def _remove_oldest_entry(self) -> None:
        if not self._window_entries:
            return

        old_id, _old_vector, old_cell = self._window_entries.popleft()
        self._entries_by_id.pop(old_id, None)
        self._entry_cell_by_id.pop(old_id, None)

        members = self._cell_members.get(old_cell)
        if members is not None:
            members.discard(old_id)
            if not members:
                self._cell_members.pop(old_cell, None)

        new_count = self._cell_counts.get(old_cell, 0) - 1
        if new_count > 0:
            self._cell_counts[old_cell] = new_count
            self._dirty_cells.add(old_cell)
        else:
            self._cell_counts.pop(old_cell, None)
            # Cell is no longer in the active window; avoid stale dirty markers.
            self._dirty_cells.discard(old_cell)

        self._slide_remove[old_cell] += 1
        self._neighbor_cache.pop(old_cell, None)

    def _expand_with_neighbors(self, cells: set[_Cell]) -> set[_Cell]:
        if not cells:
            return set()

        expanded = set(cells)
        for existing in self._cell_counts:
            for touched in cells:
                if self._are_neighbor_cells(existing, touched):
                    expanded.add(existing)
                    break
        return expanded

    def _is_warm(self) -> bool:
        warm_samples = self.warm_up_slides * self.slide_size
        return self._samples_seen >= warm_samples and len(self._window_entries) >= (
            self.k + 1
        )

    def _on_slide_boundary(self) -> None:
        touched = set(self._slide_add) | set(self._slide_remove)
        expanded = self._expand_with_neighbors(touched)

        for cell in expanded:
            current = self._cell_counts.get(cell, 0)
            if current == 0:
                self._dirty_cells.discard(cell)
                self._neighbor_cache.pop(cell, None)
                continue

            prev = self._prev_cell_counts.get(cell, 0)
            ratio = abs(current - prev) / float(max(prev, 1))
            if ratio > self.skip_threshold:
                self._dirty_cells.add(cell)
                self._neighbor_cache.pop(cell, None)

        self._dirty_cells.intersection_update(self._cell_counts)
        self._prev_cell_counts = dict(self._cell_counts)
        self._slide_add.clear()
        self._slide_remove.clear()

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one sample."""
        current_time, vector = self._prepare_input(x, mutate_schema=True)
        if vector is None:
            return

        cell = self._cell_id(vector)
        self._add_entry(vector, cell)

        if len(self._window_entries) > self.window_size:
            self._remove_oldest_entry()

        self._samples_seen += 1
        self._current_slide_count += 1
        self._max_learned_time = current_time
        if self.time_key is None:
            self._arrival_index += 1

        if self._current_slide_count >= self.slide_size:
            self._on_slide_boundary()
            self._current_slide_count = 0

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for one sample."""
        _current_time, vector = self._prepare_input(x, mutate_schema=False)
        if vector is None or not self._is_warm():
            return 0.0

        cell = self._cell_id(vector)
        if (
            cell in self._cell_counts
            and cell in self._neighbor_cache
            and cell not in self._dirty_cells
        ):
            neighbors = self._neighbor_cache[cell]
        else:
            neighbors = self._count_neighbors_within_radius(vector, cell)
            if cell in self._cell_counts:
                self._neighbor_cache[cell] = neighbors
                self._dirty_cells.discard(cell)
            else:
                self._neighbor_cache.pop(cell, None)
                self._dirty_cells.discard(cell)

        score = 1.0 - min(neighbors / float(self.k), 1.0)
        return float(np.clip(score, 0.0, 1.0))

    def predict_one(self, x: dict[str, float]) -> int:
        """Return binary anomaly prediction using ``predict_threshold``."""
        return int(self.score_one(x) >= self.predict_threshold)

    def __repr__(self) -> str:
        return (
            "STARE("
            f"k={self.k}, radius={self.radius}, window_size={self.window_size}, "
            f"slide_size={self.slide_size}, skip_threshold={self.skip_threshold}, "
            f"time_key={self.time_key!r}, warm_up_slides={self.warm_up_slides}, "
            f"predict_threshold={self.predict_threshold}, eps={self.eps}, "
            f"samples_seen={self._samples_seen}, active_cells={len(self._cell_counts)})"
        )
