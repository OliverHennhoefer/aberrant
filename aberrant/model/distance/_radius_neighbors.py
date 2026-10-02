"""Shared bounded radius-neighbor state for the cell-based detectors."""

from __future__ import annotations

import math
from collections import deque
from itertools import product
from typing import TypeAlias

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import NumericEventBoundary

_Cell: TypeAlias = tuple[int, ...]
_Entry: TypeAlias = tuple[int, _Cell]
_MAX_NEIGHBOR_OFFSETS = 20_000


class _RadiusNeighborDetector(BaseModel):
    """Own the window, cell index, and exact scarcity score in one place."""

    def __init__(
        self,
        *,
        k: int,
        radius: float,
        window_size: int,
        slide_size: int,
        time_key: str | None,
        warm_up_slides: int,
        predict_threshold: float,
        eps: float,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_size <= k:
            raise ValueError("window_size must be greater than k")
        if slide_size <= 0:
            raise ValueError("slide_size must be positive")
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
        self.time_key = time_key
        self.warm_up_slides = warm_up_slides
        self.predict_threshold = predict_threshold
        self.eps = eps
        self._radius_ratio = float(radius).as_integer_ratio()
        self._distance_limit_sq = radius * radius + eps
        # Squared distances can round down, especially in the subnormal range.
        distance_bound = math.nextafter(self._distance_limit_sq, math.inf)
        accepted_radius = math.nextafter(math.sqrt(distance_bound), math.inf)
        cell_radius = math.nextafter(accepted_radius / radius, math.inf)
        # An overflowing squared-distance limit requires scanning every cell.
        # Include touching boundary cells even when float arithmetic rounds a
        # just-outside point onto the exact accepted distance.
        self._max_cell_delta = (
            max(2, math.floor(cell_radius) + 1) if math.isfinite(cell_radius) else None
        )
        self._reset_state()

    def _reset_state(self) -> None:
        self._boundary = NumericEventBoundary(time_key=self.time_key)
        self._window_entries: deque[_Entry] = deque()
        self._cell_members: dict[_Cell, dict[int, np.ndarray]] = {}
        self._neighbor_cache: dict[_Cell, tuple[_Cell, ...]] = {}
        self._neighbor_offsets_cache: dict[int, tuple[_Cell, ...]] = {}
        self._samples_seen = 0

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed via ``learn_one``."""
        return self._samples_seen

    def _cell_id(self, vector: np.ndarray) -> _Cell:
        """Take the mathematical cell floor without float quotient rounding."""
        radius_numerator, radius_denominator = self._radius_ratio
        coordinates = []
        for value in vector:
            numerator, denominator = float(value).as_integer_ratio()
            coordinates.append(
                (numerator * radius_denominator) // (denominator * radius_numerator)
            )
        return tuple(coordinates)

    def _are_neighbor_cells(self, left: _Cell, right: _Cell) -> bool:
        return self._max_cell_delta is None or all(
            abs(a - b) <= self._max_cell_delta for a, b in zip(left, right, strict=True)
        )

    def _candidate_cells(self, cell: _Cell) -> tuple[_Cell, ...]:
        cached = self._neighbor_cache.get(cell)
        if cached is not None:
            return cached

        n_dims = len(cell)
        span = self._max_cell_delta
        offset_limit = min(len(self._cell_members), _MAX_NEIGHBOR_OFFSETS)
        offset_count = offset_limit + 1
        if span is not None:
            offset_count = 1
            for _ in range(n_dims):
                offset_count *= 2 * span + 1
                if offset_count > offset_limit:
                    break
        if offset_count <= offset_limit:
            offsets = self._neighbor_offsets_cache.get(n_dims)
            if offsets is None:
                assert span is not None
                offsets = tuple(product(range(-span, span + 1), repeat=n_dims))
                self._neighbor_offsets_cache[n_dims] = offsets
            neighbors = []
            for offset in offsets:
                candidate = tuple(
                    base + delta for base, delta in zip(cell, offset, strict=True)
                )
                if candidate in self._cell_members:
                    neighbors.append(candidate)
            candidates = tuple(neighbors)
        else:
            candidates = tuple(
                existing
                for existing in self._cell_members
                if self._are_neighbor_cells(existing, cell)
            )

        # Score-only queries cannot accumulate state for arbitrary new cells.
        if cell in self._cell_members:
            self._neighbor_cache.clear()
            self._neighbor_cache[cell] = candidates
        return candidates

    def _validate_feature_count(self, n_features: int) -> None:
        """Allow public compatibility parameters to constrain the schema."""

    def learn_one(self, x: dict[str, float]) -> None:
        """Retain one validated observation and evict the oldest when full."""
        event = self._boundary.preview(x)
        vector = event.features.values
        self._validate_feature_count(len(vector))
        cell = self._cell_id(vector)
        if cell not in self._cell_members:
            self._neighbor_cache.clear()
        entry_id = self._samples_seen
        self._window_entries.append((entry_id, cell))
        self._cell_members.setdefault(cell, {})[entry_id] = vector

        if len(self._window_entries) > self.window_size:
            old_id, old_cell = self._window_entries.popleft()
            members = self._cell_members[old_cell]
            del members[old_id]
            if not members:
                del self._cell_members[old_cell]
                self._neighbor_cache.clear()
        self._samples_seen += 1
        self._boundary.commit(event)

    def score_one(self, x: dict[str, float]) -> float:
        """Count exact neighbors, stopping once the scarcity score is zero."""
        event = self._boundary.preview(x)
        if self._samples_seen < self.warm_up_slides * self.slide_size or (
            len(self._window_entries) <= self.k
        ):
            return 0.0
        vector = event.features.values
        neighbors = 0
        for cell in self._candidate_cells(self._cell_id(vector)):
            for point in self._cell_members[cell].values():
                diff = point - vector
                if float(np.dot(diff, diff)) <= self._distance_limit_sq:
                    neighbors += 1
                    if neighbors >= self.k:
                        return 0.0
        return 1.0 - neighbors / float(self.k)

    def predict_one(self, x: dict[str, float]) -> int:
        """Return binary anomaly prediction using ``predict_threshold``."""
        return int(self.score_one(x) >= self.predict_threshold)
