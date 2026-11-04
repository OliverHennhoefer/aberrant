"""Local Outlier Factor (LOF) anomaly detection model."""

import bisect
import math
from typing import Literal

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema


class LocalOutlierFactor(BaseModel):
    """
    Local Outlier Factor (LOF) for online anomaly detection.

    LOF identifies anomalies by comparing the local density of a point
    with the local densities of its neighbors. Points with significantly
    lower density than their neighbors are considered outliers.

    This implementation maintains a sliding window of observations and
    computes LOF scores on demand.

    Note:
        Learning keeps a sorted list of up to ``2 * k`` nearest other window
        points for every window point, so each k-distance is known without a
        search. Scoring measures the query against the window once and reads
        the neighborhoods of its neighbors from those lists. Both cost
        ``O(window_size * n_features)`` per event. A neighborhood that may
        reach past its list, or whose tied distances must keep the window's
        order, is measured against the whole window again, so heavily
        duplicated data scores slower.

    Args:
        k: Number of neighbors to use for density estimation. Default is 10.
        window_size: Maximum number of points to keep in the window.
            Default is 1000.
        distance: Distance metric to use. Either "euclidean" or "manhattan".
            Default is "euclidean".

    Examples:
        ```python
        from aberrant.model.distance import LocalOutlierFactor

        stream = [
            {"x": 0.0, "y": 0.0},
            {"x": 0.1, "y": 0.0},
            {"x": 0.0, "y": 0.1},
            {"x": 3.0, "y": 3.0},
        ]
        lof = LocalOutlierFactor(k=2, window_size=20)
        scores = []
        for point in stream:
            score = lof.score_one(point)
            scores.append(score)
            lof.learn_one(point)
        assert len(scores) == len(stream)
        ```

    References:
        Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000).
        LOF: identifying density-based local outliers. In Proceedings
        of the 2000 ACM SIGMOD International Conference on Management
        of Data (pp. 93-104).
        https://doi.org/10.1145/342009.335388

        Pokrajac, D., Lazarevic, A., & Latecki, L. J. (2007). Incremental
        local outlier detection for data streams. In 2007 IEEE Symposium
        on Computational Intelligence and Data Mining (pp. 504-515).
    """

    def __init__(
        self,
        k: int = 10,
        window_size: int = 1000,
        distance: Literal["euclidean", "manhattan"] = "euclidean",
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if k >= window_size:
            raise ValueError("k must be less than window_size")
        if distance not in ("euclidean", "manhattan"):
            raise ValueError("distance must be 'euclidean' or 'manhattan'")

        self.k = k
        self.window_size = window_size
        self.distance = distance

        self._reset_state()

    def _reset_state(self) -> None:
        """Initialize or reset internal state."""
        self._schema = FeatureSchema()
        self._n_points = 0
        # Slot of the oldest point, which the next point replaces once the
        # window is full. Slots are in arrival order from it, wrapping around.
        self._oldest = 0
        # Each list holds at most this many neighbors of its point.
        self._list_size = min(2 * self.k, self.window_size - 1)
        # Window points by slot, one row per feature so distances to all
        # points take a few vectorized operations. The arrays grow with the
        # window up to ``window_size`` slots.
        self._window = np.empty((0, 0))
        # Per slot, the distances to its nearest other window points in
        # ascending order and their slots. A list holds every other point
        # (``_complete``) or at least ``k`` of them, and no point left out is
        # nearer than a listed one, so its k-th distance is the k-distance.
        self._near: list[list[float]] = []
        self._near_slots: list[list[int]] = []
        # Per slot, the slots whose lists hold it.
        self._holders: list[set[int]] = []
        # Per slot, the k-distance, or ``inf`` with fewer than k other points.
        self._k_distances: list[float] = []
        self._complete = np.empty(0, dtype=bool)
        # Per slot, the last listed distance, which new points must undercut
        # to enter an incomplete list.
        self._last = np.empty(0)

    @property
    def n_points(self) -> int:
        """Number of points currently in the window."""
        return self._n_points

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Add a new point to the window.

        Args:
            x: Feature dictionary with string keys and float values.
        """
        prepared = self._schema.preview(x)
        if self._n_points == self.window_size:
            slot = self._oldest
            self._oldest = (slot + 1) % self.window_size
            shortened = self._forget(slot)
        else:
            slot = self._n_points
            self._grow(prepared.values.size)
            shortened = set()
        self._window[:, slot] = prepared.values
        # Huge coordinates overflow to infinite distances, as when scoring.
        with np.errstate(over="ignore"):
            self._insert(slot, shortened)
        self._schema.commit(prepared)

    def _grow(self, n_features: int) -> None:
        """Add a slot, doubling the window arrays when they are full."""
        capacity = self._window.shape[1]
        if self._n_points == capacity:
            size = min(max(2 * capacity, 16), self.window_size)
            window = np.empty((n_features, size))
            if capacity:
                window[:, :capacity] = self._window
            self._window = window
            self._complete = np.resize(self._complete, size)
            self._last = np.resize(self._last, size)
        self._n_points += 1
        self._near.append([])
        self._near_slots.append([])
        self._holders.append(set())
        self._k_distances.append(math.inf)

    def _forget(self, slot: int) -> set[int]:
        """Remove ``slot`` from the lists that hold it and return those lists."""
        holders = self._holders[slot]
        self._holders[slot] = set()
        for row in holders:
            slots = self._near_slots[row]
            index = slots.index(slot)
            del slots[index]
            del self._near[row][index]
            self._listed(row)
        return holders

    def _insert(self, slot: int, shortened: set[int]) -> None:
        """List the point in ``slot`` and offer it to the other lists."""
        n = self._n_points
        size = self._list_size
        distances = self._distances(self._window[:, slot])
        # An incomplete list only takes points nearer than its last entry.
        rows = np.flatnonzero(self._complete[:n] | (distances < self._last[:n]))
        holders = self._holders[slot]
        for row, distance in zip(rows.tolist(), distances[rows].tolist(), strict=True):
            if row == slot:
                continue
            near = self._near[row]
            slots = self._near_slots[row]
            # The new point goes before equal distances.
            index = bisect.bisect_left(near, distance)
            near.insert(index, distance)
            slots.insert(index, slot)
            holders.add(row)
            if len(near) > size:
                near.pop()
                self._holders[slots.pop()].discard(row)
                self._complete[row] = False
            self._listed(row)
        self._set_list(slot, distances)
        # Lists that lost points to the eviction are rebuilt once they run short.
        for row in shortened:
            if len(self._near[row]) < self.k and not self._complete[row]:
                self._set_list(row, self._distances(self._window[:, row]))

    def _set_list(self, slot: int, distances: np.ndarray) -> None:
        """Rebuild the list of ``slot`` from its distances to the window."""
        for other in self._near_slots[slot]:
            self._holders[other].discard(slot)
        size = self._list_size
        complete = distances.size - 1 <= size
        if complete:
            nearest = np.argsort(distances, kind="stable")
        else:
            # The point itself, at distance zero, is among the size + 1
            # nearest unless that many other points share its position.
            nearest = np.argpartition(distances, size)[: size + 1]
            nearest = nearest[np.argsort(distances[nearest], kind="stable")]
        nearest = nearest[nearest != slot][:size]
        slots = nearest.tolist()
        self._near[slot] = distances[nearest].tolist()
        self._near_slots[slot] = slots
        for other in slots:
            self._holders[other].add(slot)
        self._complete[slot] = complete
        self._listed(slot)

    def _listed(self, slot: int) -> None:
        """Refresh the k-distance and last distance of a changed list."""
        near = self._near[slot]
        self._k_distances[slot] = near[self.k - 1] if len(near) >= self.k else math.inf
        self._last[slot] = near[-1] if near else math.inf

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute the LOF score for a point.

        The LOF score indicates how anomalous a point is:
        - LOF ≈ 1: Point has similar density to its neighbors (normal)
        - LOF > 1: Point has lower density than neighbors (potential outlier)
        - LOF >> 1: Point is significantly less dense (strong outlier)

        Args:
            x: Feature dictionary with string keys and float values.

        Returns:
            LOF score. Higher values indicate more anomalous points.
        """
        prepared = self._schema.preview(x)

        # Need at least k+1 points to compute LOF
        if self._n_points <= self.k:
            return 0.0
        if not self._schema.is_established:
            return 0.0

        distances = self._distances(prepared.values)
        k_distance = np.partition(distances, self.k - 1)[self.k - 1]
        # Get the k-distance neighborhood, including all ties at k-distance.
        neighbors = self._neighborhood(distances, k_distance)

        # Compute local reachability density for query point
        lrd_query = self._lrd(distances[neighbors].tolist(), neighbors.tolist())
        if lrd_query <= 0:
            return 0.0

        # Compute LOF as average ratio of neighbor LRDs to query LRD
        lof_sum = 0.0
        for neighbor in neighbors.tolist():
            lrd_neighbor = self._neighbor_lrd(neighbor)
            if lrd_neighbor > 0:
                if math.isinf(lrd_neighbor) and math.isinf(lrd_query):
                    # Identical dense neighborhoods should behave like LOF ~= 1.
                    lof_sum += 1.0
                elif math.isinf(lrd_query):
                    lof_sum += 0.0
                else:
                    lof_sum += lrd_neighbor / lrd_query

        return lof_sum / len(neighbors)

    def _neighbor_lrd(self, slot: int) -> float:
        """Return the local reachability density of a window point."""
        near = self._near[slot]
        k_distance = self._k_distances[slot]
        size = bisect.bisect_right(near, k_distance)
        # A list that ends inside the neighborhood may miss points tied at the
        # k-distance.
        if size < len(near) or self._complete[slot]:
            k_distances = self._k_distances
            reach = [
                max(k_distances[other], distance)
                for distance, other in zip(
                    near[:size], self._near_slots[slot][:size], strict=True
                )
            ]
            # Tied distances are summed in the window's order unless their
            # reach distances agree, which makes the order irrelevant.
            if len(set(near[:size])) == size or not any(
                near[index] == near[index + 1] and reach[index] != reach[index + 1]
                for index in range(size - 1)
            ):
                return self._reachability(reach)
        distances = self._distances(self._window[:, slot])
        members = self._neighborhood(distances, k_distance, exclude=slot)
        return self._lrd(distances[members].tolist(), members.tolist())

    def _neighborhood(
        self, distances: np.ndarray, radius: float, exclude: int | None = None
    ) -> np.ndarray:
        """
        Return the slots within ``radius`` in the order a full sort gives them.

        Distinct distances have one sorted order. Ties fall back to sorting the
        distances in arrival order, as ``_get_k_neighbors`` does.
        """
        members = np.flatnonzero(distances <= radius)
        if exclude is not None:
            members = members[members != exclude]
        values = distances[members]
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        if not np.any(sorted_values[1:] == sorted_values[:-1]):
            return members[order]
        oldest = self._oldest
        n = distances.size
        arrival = np.roll(distances, -oldest)
        position = None if exclude is None else (exclude - oldest) % n
        ordered = np.asarray(self._get_k_neighbors(arrival, exclude_index=position))
        return (ordered + oldest) % n

    def _lrd(self, distances: list[float], slots: list[int]) -> float:
        """
        Compute local reachability density from neighbor distances.

        LRD(p) = 1 / (sum of reach_dist(p, neighbors) / k), where
        reach_dist(p, o) = max(k-distance(o), dist(p, o)).
        """
        k_distances = self._k_distances
        return self._reachability(
            [
                max(k_distances[slot], distance)
                for distance, slot in zip(distances, slots, strict=True)
            ]
        )

    @staticmethod
    def _reachability(reach_dists: list[float]) -> float:
        """Turn reachability distances, in neighbor order, into a density."""
        reach_dist_sum = 0.0
        for reach_dist in reach_dists:
            reach_dist_sum += reach_dist
        if reach_dist_sum <= 0:
            return float("inf")
        return len(reach_dists) / reach_dist_sum

    def _distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distances from ``point`` to every window point."""
        diff = self._window[:, : self._n_points] - point[:, np.newaxis]
        if self.distance == "euclidean":
            distances: np.ndarray = np.sqrt(_sum_rows(np.square(diff, out=diff)))
            return distances
        return _sum_rows(np.abs(diff, out=diff))

    def _get_k_neighbors(
        self, distances: np.ndarray, exclude_index: int | None = None
    ) -> list[int]:
        """Return all neighbors whose distance is at most the k-distance."""
        ordered = np.argsort(distances)
        if exclude_index is not None:
            ordered = ordered[ordered != exclude_index]
        if len(ordered) > self.k:
            k_distance = distances[ordered[self.k - 1]]
            ordered = ordered[distances[ordered] <= k_distance]
        neighbors: list[int] = ordered.tolist()
        return neighbors

    def __repr__(self) -> str:
        return (
            f"LocalOutlierFactor(k={self.k}, window_size={self.window_size}, "
            f"distance='{self.distance}', n_points={self.n_points})"
        )


def _sum_rows(values: np.ndarray) -> np.ndarray:
    """
    Sum the rows of ``values`` in the order ``np.sum`` adds a contiguous run.

    NumPy sums a run of fewer than eight floats in sequence, a run of at most
    128 with eight interleaved partial sums, and a longer run as two halves.
    Following that order with whole rows gives each column the same bits as
    ``np.sum`` over that column, in a few vectorized additions instead of one
    short reduction per column.
    """
    n_rows = values.shape[0]
    total: np.ndarray
    if n_rows < 8:
        total = values[0].copy()
        for row in values[1:]:
            total += row
        return total
    if n_rows <= 128:
        partial = values[:8].copy()
        stop = n_rows - n_rows % 8
        for start in range(8, stop, 8):
            partial += values[start : start + 8]
        total = (partial[0] + partial[1]) + (partial[2] + partial[3])
        total += (partial[4] + partial[5]) + (partial[6] + partial[7])
        for row in values[stop:]:
            total += row
        return total
    half = n_rows // 2
    half -= half % 8
    total = _sum_rows(values[:half]) + _sum_rows(values[half:])
    return total
