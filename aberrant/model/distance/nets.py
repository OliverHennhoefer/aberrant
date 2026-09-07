"""Bounded cell-neighborhood detector for streaming anomaly detection."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from itertools import product
from typing import TypeAlias

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import NumericEventBoundary

_Cell: TypeAlias = tuple[int, ...]
_Entry: TypeAlias = tuple[int, _Cell]
_MAX_NEIGHBOR_OFFSETS = 20_000


class CellNeighborhoodDetector(BaseModel):
    """
    Bounded cell-neighborhood streaming outlier detector.

    The detector keeps a bounded sliding window and quantizes samples into
    full-space cells. Scores are based on neighborhood
    density within ``radius``:
    ``score = 1 - min(neighbor_count / k, 1)``.

    NETS-inspired cell indexing limits exact distance checks to neighboring cells.
    This class returns a continuous score for one query point; it does not
    reproduce the paper's exact window-level inlier/outlier set algorithm.

    Notes:
    - Scores are continuous and bounded in ``[0, 1]``.
    - State is bounded by ``window_size``.
    - Feature schema is fixed after the first ``learn_one`` call.
    - Distance metric is Euclidean.

    Args:
        k: Neighbor count at which the scarcity score reaches zero.
        radius: Positive Euclidean neighborhood radius and grid-cell width.
        window_size: Maximum number of learned points retained. It must exceed
            ``k``.
        slide_size: Number of learned events per warm-up slide.
        subspace_dim: Compatibility parameter, validated against the feature count.
            The former subspace index was redundant and has been removed.
        time_key: Event-time field to exclude from the feature vector. ``None``
            uses one-based arrival order; explicit times must be finite and
            non-decreasing.
        warm_up_slides: Complete slides required before non-zero scoring.
        predict_threshold: Score boundary used by ``predict_one``.
        seed: Retained for constructor compatibility; scoring is deterministic.
        eps: Positive numerical floor used in bound calculations.

    References:
        Yoon, S., Lee, J.-G., & Lee, B. S. (2019). NETS: Extremely Fast
        Outlier Detection from a Data Stream via Set-Based Processing.
        https://doi.org/10.14778/3342263.3342269
        Original implementation: https://github.com/kaist-dmlab/NETS
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
        self._boundary = NumericEventBoundary(time_key=self.time_key)
        self._window_entries: deque[_Entry] = deque()
        self._full_cell_members: dict[_Cell, dict[int, np.ndarray]] = {}
        self._neighbor_offsets_cache: dict[int, tuple[_Cell, ...]] = {}
        self._samples_seen = 0

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed via ``learn_one``."""
        return self._samples_seen

    def _full_cell_id(self, vector: np.ndarray) -> _Cell:
        return tuple(int(value) for value in np.floor(vector / self.radius))

    def _are_neighbor_cells(self, left: _Cell, right: _Cell) -> bool:
        for left_dim, right_dim in zip(left, right, strict=False):
            if abs(left_dim - right_dim) > 1:
                return False
        return True

    def _neighbor_cells(
        self, cell: _Cell, cell_counts: Mapping[_Cell, object]
    ) -> list[_Cell]:
        n_active_cells = len(cell_counts)
        if n_active_cells == 0:
            return []

        n_dims = len(cell)
        neighbor_offsets = 3**n_dims
        if (
            neighbor_offsets <= n_active_cells
            and neighbor_offsets <= _MAX_NEIGHBOR_OFFSETS
        ):
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

    def _is_warm(self) -> bool:
        warm_samples = self.warm_up_slides * self.slide_size
        return self._samples_seen >= warm_samples and len(self._window_entries) > self.k

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the single bounded cell index with one sample."""
        event = self._boundary.preview(x)
        vector = event.features.values
        if self.subspace_dim is not None and self.subspace_dim > len(vector):
            raise ValueError(
                f"subspace_dim ({self.subspace_dim}) cannot exceed number of features "
                f"({len(vector)})"
            )
        cell = self._full_cell_id(vector)
        entry_id = self._samples_seen
        self._window_entries.append((entry_id, cell))
        self._full_cell_members.setdefault(cell, {})[entry_id] = vector
        if len(self._window_entries) > self.window_size:
            old_id, old_cell = self._window_entries.popleft()
            members = self._full_cell_members[old_cell]
            del members[old_id]
            if not members:
                del self._full_cell_members[old_cell]
        self._samples_seen += 1
        self._boundary.commit(event)

    def score_one(self, x: dict[str, float]) -> float:
        """Count exact neighbors, stopping once the score must be zero."""
        event = self._boundary.preview(x)
        if not self._boundary.schema.is_established or not self._is_warm():
            return 0.0
        vector = event.features.values
        cell = self._full_cell_id(vector)
        neighbors = 0
        for candidate in self._neighbor_cells(cell, self._full_cell_members):
            for point in self._full_cell_members[candidate].values():
                diff = point - vector
                if float(np.dot(diff, diff)) <= self._radius_sq + self.eps:
                    neighbors += 1
                    if neighbors >= self.k:
                        return 0.0
        return 1.0 - neighbors / float(self.k)

    def predict_one(self, x: dict[str, float]) -> int:
        """Return binary anomaly prediction using ``predict_threshold``."""
        return int(self.score_one(x) >= self.predict_threshold)

    def __repr__(self) -> str:
        return (
            "CellNeighborhoodDetector("
            f"k={self.k}, radius={self.radius}, window_size={self.window_size}, "
            f"slide_size={self.slide_size}, subspace_dim={self.subspace_dim}, "
            f"time_key={self.time_key!r}, warm_up_slides={self.warm_up_slides}, "
            f"predict_threshold={self.predict_threshold}, seed={self.seed}, "
            f"samples_seen={self._samples_seen}, "
            f"active_full_cells={len(self._full_cell_members)})"
        )
