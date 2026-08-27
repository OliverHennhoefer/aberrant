"""Local Outlier Factor (LOF) anomaly detection model."""

from collections import deque
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
        Time Complexity: ``score_one`` has O(k × n) complexity where n is the
        window size, as it computes distances for each of k neighbors. For
        high-throughput scenarios with large windows, consider using smaller
        k values or reducing window_size. River's ILOF uses incremental
        updates for better amortized performance.

    Args:
        k: Number of neighbors to use for density estimation. Default is 10.
        window_size: Maximum number of points to keep in the window.
            Default is 1000.
        distance: Distance metric to use. Either "euclidean" or "manhattan".
            Default is "euclidean".

    Example:
        >>> lof = LocalOutlierFactor(k=5, window_size=500)
        >>> for point in data_stream:
        ...     score = lof.score_one(point)
        ...     lof.learn_one(point)
        ...     if score > 1.5:  # LOF > 1 indicates outlier
        ...         print("Anomaly detected!")

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
        self._points: deque[np.ndarray] = deque(maxlen=self.window_size)

    @property
    def n_points(self) -> int:
        """Number of points currently in the window."""
        return len(self._points)

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Add a new point to the window.

        Args:
            x: Feature dictionary with string keys and float values.
        """
        prepared = self._schema.preview(x)
        self._points.append(prepared.values.copy())
        self._schema.commit(prepared)

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
        if len(self._points) <= self.k:
            return 0.0
        if not self._schema.is_established:
            return 0.0

        query = prepared.values

        # Compute distances to all points in window
        distances = self._compute_distances(query)

        # Get the k-distance neighborhood, including all ties at k-distance.
        k_neighbors = self._get_k_neighbors(distances)

        # Compute local reachability density for query point
        lrd_query = self._compute_lrd(query, k_neighbors, distances)

        if lrd_query <= 0:
            return 0.0

        # Compute LOF as average ratio of neighbor LRDs to query LRD
        lof_sum = 0.0
        for neighbor_idx in k_neighbors:
            neighbor_point = self._points[neighbor_idx]
            neighbor_distances = self._compute_distances(neighbor_point)
            neighbor_k_neighbors = self._get_k_neighbors(
                neighbor_distances, exclude_index=neighbor_idx
            )
            lrd_neighbor = self._compute_lrd(
                neighbor_point, neighbor_k_neighbors, neighbor_distances
            )
            if lrd_neighbor > 0:
                if np.isinf(lrd_neighbor) and np.isinf(lrd_query):
                    # Identical dense neighborhoods should behave like LOF ~= 1.
                    lof_sum += 1.0
                elif np.isinf(lrd_query):
                    lof_sum += 0.0
                else:
                    lof_sum += lrd_neighbor / lrd_query

        lof = lof_sum / len(k_neighbors)
        return lof

    def _compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distances from point to all points in window."""
        if len(self._points) == 0:
            return np.array([])

        points_array = np.array(self._points)

        if self.distance == "euclidean":
            diff = points_array - point
            distances = np.sqrt(np.sum(diff**2, axis=1))
        else:  # manhattan
            distances = np.sum(np.abs(points_array - point), axis=1)

        return np.asarray(distances, dtype=np.float64)

    def _get_k_neighbors(
        self, distances: np.ndarray, exclude_index: int | None = None
    ) -> list[int]:
        """Return all neighbors whose distance is at most the k-distance."""
        ordered = [
            int(idx) for idx in np.argsort(distances) if int(idx) != exclude_index
        ]
        if len(ordered) <= self.k:
            return ordered

        k_distance = distances[ordered[self.k - 1]]
        return [idx for idx in ordered if distances[idx] <= k_distance]

    def _compute_k_distance(self, idx: int) -> float:
        """Compute k-distance for point at index idx."""
        point = self._points[idx]
        distances = self._compute_distances(point)
        k_neighbors = self._get_k_neighbors(distances, exclude_index=idx)
        if len(k_neighbors) < self.k:
            return float("inf")
        return float(distances[k_neighbors[-1]])

    def _compute_reach_dist(
        self, point: np.ndarray, neighbor_idx: int, dist: float
    ) -> float:
        """
        Compute reachability distance from point to neighbor.

        reach_dist(p, o) = max(k-distance(o), dist(p, o))
        """
        k_dist_neighbor = self._compute_k_distance(neighbor_idx)
        return max(k_dist_neighbor, dist)

    def _compute_lrd(
        self,
        point: np.ndarray,
        k_neighbors: list[int],
        distances: np.ndarray,
    ) -> float:
        """
        Compute local reachability density for a point.

        LRD(p) = 1 / (sum of reach_dist(p, neighbors) / k)
        """
        if len(k_neighbors) == 0:
            return 0.0

        reach_dist_sum = 0.0
        for neighbor_idx in k_neighbors:
            dist = distances[neighbor_idx]
            reach_dist = self._compute_reach_dist(point, neighbor_idx, dist)
            reach_dist_sum += reach_dist

        if reach_dist_sum <= 0:
            return float("inf")

        return len(k_neighbors) / reach_dist_sum

    def __repr__(self) -> str:
        return (
            f"LocalOutlierFactor(k={self.k}, window_size={self.window_size}, "
            f"distance='{self.distance}', n_points={self.n_points})"
        )
