"""Bounded cell-neighborhood detector for streaming anomaly detection."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from itertools import product

import numpy as np

from aberrant._utils.validation import NumericEventBoundary, coerce_integer_count
from aberrant.base.model import BaseModel
from aberrant.base.readiness import Readiness

type _Cell = tuple[int, ...]
type _Entry = tuple[int, _Cell]
_MAX_NEIGHBOR_OFFSETS = 20_000
# Added to the squared radius so rounding cannot drop a neighbor on the boundary.
_DISTANCE_TOLERANCE = 1e-9
# Absolute rounding allowance for squared distances in the subnormal range.
_CLOSE_ABSOLUTE = 1e-300


class _CellMembers:
    """The points of one cell, oldest first, stored once as rows of a buffer.

    Points arrive at the end and leave from the front, as the window evicts
    them, so ``rows[start:start + len(ids)]`` holds them without copying.
    Iterating yields their entry ids. The buffer doubles when it is full and
    halves its slack when it is under a quarter full, so it holds at most
    four times its points. Pickles keep only the live rows.
    """

    __slots__ = ("ids", "rows", "start")

    def __init__(self, n_features: int) -> None:
        self.ids: deque[int] = deque()
        self.rows = np.empty((1, n_features))
        self.start = 0

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.ids)

    def points(self) -> np.ndarray:
        """Return the live rows as a view, oldest first."""
        return self.rows[self.start : self.start + len(self.ids)]

    def append(self, entry_id: int, point: np.ndarray) -> None:
        end = self.start + len(self.ids)
        if end == len(self.rows):
            self._resize(2 * len(self.ids) + 1)
            end = len(self.ids)
        self.rows[end] = point
        self.ids.append(entry_id)

    def pop_oldest(self) -> int:
        entry_id = self.ids.popleft()
        self.start += 1
        if len(self.rows) > 2 and 4 * len(self.ids) < len(self.rows):
            self._resize(2 * len(self.ids) + 1)
        return entry_id

    def _resize(self, capacity: int) -> None:
        live = self.points()
        rows = np.empty((capacity, self.rows.shape[1]))
        rows[: len(live)] = live
        self.rows, self.start = rows, 0

    def __getstate__(self) -> tuple[list[int], np.ndarray]:
        return list(self.ids), self.points().copy()

    def __setstate__(self, state: tuple[list[int], np.ndarray]) -> None:
        ids, rows = state
        self.ids = deque(ids)
        self.rows = rows if len(rows) else np.empty((1, rows.shape[1]))
        self.start = 0


class CellNeighborhoodDetector(BaseModel):
    """
    Bounded cell-neighborhood streaming outlier detector.

    The detector keeps a bounded sliding window and quantizes samples into
    full-space cells. Scores are based on neighborhood
    density within ``radius``:
    ``score = 1 - min(neighbor_count / k, 1)``.

    The score follows the distance-based outlier rule that NETS accelerates: a
    point is an outlier when fewer than ``k`` window points lie within
    ``radius``. Radius-wide grid cells limit the exact distance checks to
    neighboring cells. The class returns that rule as a continuous score per
    query instead of the paper's window-level outlier set.

    Notes:
    - Scores are continuous and bounded in ``[0, 1]``.
    - State is bounded by ``window_size``.
    - Feature schema is fixed after the first ``learn_one`` call.
    - Distance metric is Euclidean.

    Args:
        k: Neighbor count at which the scarcity score reaches zero.
        radius: Finite positive Euclidean neighborhood radius and grid-cell
            width, in the units of the features. The default of 1.5 suits
            standardized features; on features scaled to ``[0, 1]`` it covers
            most of the space, so every point finds ``k`` neighbors and
            scores 0.
        window_size: Maximum number of learned points retained. It must exceed
            ``k``.
        slide_size: Number of learned events per warm-up slide.
        time_key: Event-time field to exclude from the feature vector. ``None``
            uses one-based arrival order; explicit times must be finite and
            non-decreasing.
        warm_up: Complete slides required before non-zero scoring.

    References:
        Yoon, S., Lee, J.-G., & Lee, B. S. (2019). NETS: Extremely Fast
        Outlier Detection from a Data Stream via Set-Based Processing.
        https://doi.org/10.14778/3342263.3342269
        Original implementation: https://github.com/kaist-dmlab/NETS
    """

    def __init__(
        self,
        *,
        k: int = 50,
        radius: float = 1.5,
        window_size: int = 10_000,
        slide_size: int = 500,
        time_key: str | None = None,
        warm_up: int = 1,
    ) -> None:
        k = coerce_integer_count(k, label="k", minimum=1)
        window_size = coerce_integer_count(window_size, label="window_size", minimum=1)
        slide_size = coerce_integer_count(slide_size, label="slide_size", minimum=1)
        warm_up = coerce_integer_count(warm_up, label="warm_up", minimum=1)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive")
        if window_size <= k:
            raise ValueError("window_size must be greater than k")

        self.k = k
        self.radius = radius
        self.window_size = window_size
        self.slide_size = slide_size
        self.time_key = time_key
        self.warm_up = warm_up
        self._radius_ratio = float(radius).as_integer_ratio()
        self._distance_limit_sq = radius * radius + _DISTANCE_TOLERANCE
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
        self._cell_members: dict[_Cell, _CellMembers] = {}
        self._neighbor_cache: dict[_Cell, tuple[_Cell, ...]] = {}
        self._neighbor_offsets_cache: dict[int, tuple[_Cell, ...]] = {}
        self._samples_seen = 0

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

    def _scan_neighbor_cells(self, cell: _Cell) -> tuple[_Cell, ...]:
        """Return the retained cells near ``cell``, in insertion order."""
        existing = list(self._cell_members)
        span = self._max_cell_delta
        if span is None:
            return tuple(existing)
        try:
            coordinates = np.array(existing, dtype=np.int64)
            center = np.array(cell, dtype=np.int64)
        except OverflowError:
            # Cells of extreme values exceed int64; compare them exactly.
            return tuple(
                other for other in existing if self._are_neighbor_cells(other, cell)
            )
        # Far from zero, the int64 difference could overflow, so those cells
        # are compared as Python integers instead.
        limit = 2**61
        if (np.abs(center) < limit).all() and (
            (coordinates > -limit) & (coordinates < limit)
        ).all():
            near = (np.abs(coordinates - center) <= span).all(axis=1)
            return tuple(existing[index] for index in np.flatnonzero(near).tolist())
        return tuple(
            other for other in existing if self._are_neighbor_cells(other, cell)
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
            candidates = self._scan_neighbor_cells(cell)

        # Score-only queries cannot accumulate state for arbitrary new cells.
        if cell in self._cell_members:
            self._neighbor_cache.clear()
            self._neighbor_cache[cell] = candidates
        return candidates

    @property
    def readiness(self) -> Readiness:
        """Ready after ``warm_up`` slides and more than ``k`` retained events."""
        required = max(self.warm_up * self.slide_size, self.k + 1)
        ready = self._samples_seen >= self.warm_up * self.slide_size and (
            len(self._window_entries) > self.k
        )
        return Readiness(ready, self._samples_seen, required)

    def learn_one(self, x: dict[str, float]) -> None:
        """Retain one validated observation and evict the oldest when full."""
        event = self._boundary.preview(x)
        vector = event.features.values
        cell = self._cell_id(vector)
        if cell not in self._cell_members:
            self._neighbor_cache.clear()
        entry_id = self._samples_seen
        self._window_entries.append((entry_id, cell))
        members = self._cell_members.get(cell)
        if members is None:
            members = self._cell_members[cell] = _CellMembers(vector.size)
        members.append(entry_id, vector)

        if len(self._window_entries) > self.window_size:
            # The window is first in, first out, so the oldest entry is also
            # the oldest member of its cell.
            _, old_cell = self._window_entries.popleft()
            old_members = self._cell_members[old_cell]
            old_members.pop_oldest()
            if not old_members:
                del self._cell_members[old_cell]
                self._neighbor_cache.clear()
        self._samples_seen += 1
        self._boundary.commit(event)

    def score_one(self, x: dict[str, float]) -> float:
        """Count exact neighbors, stopping once the scarcity score is zero."""
        event = self._boundary.preview(x)
        if not self.readiness.ready:
            return 0.0
        vector = event.features.values
        limit = self._distance_limit_sq
        # Summing the squares of a batch may round differently from
        # ``np.dot``, by far less than this relative margin. Distances that
        # close to the limit, or not finite, are summed again with ``np.dot``
        # so that every comparison matches the one point at a time.
        margin = 4.0 * (vector.size + 2) * np.finfo(np.float64).eps
        close = margin * abs(limit) + _CLOSE_ABSOLUTE
        neighbors = 0
        for cell in self._candidate_cells(self._cell_id(vector)):
            points = self._cell_members[cell].points()
            # A crowded cell is measured ``4 * k`` points at a time, so that
            # stopping once ``k`` neighbors are found skips most of it.
            size = 4 * self.k
            while len(points):
                diffs = (points if len(points) <= size else points[:size]) - vector
                points = points[size:]
                distances = np.einsum("ij,ij->i", diffs, diffs)
                inside = distances <= limit
                unsure = ~(np.abs(distances - limit) > close)
                if unsure.any():
                    for row in np.flatnonzero(unsure).tolist():
                        diff = diffs[row]
                        inside[row] = float(np.dot(diff, diff)) <= limit
                neighbors += int(np.count_nonzero(inside))
                if neighbors >= self.k:
                    return 0.0
        return 1.0 - neighbors / float(self.k)
