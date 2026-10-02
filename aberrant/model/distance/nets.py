"""Bounded cell-neighborhood detector for streaming anomaly detection."""

from __future__ import annotations

from aberrant.model.distance._radius_neighbors import _RadiusNeighborDetector


class CellNeighborhoodDetector(_RadiusNeighborDetector):
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
        radius: Finite positive Euclidean neighborhood radius and grid-cell width.
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
        eps: Positive tolerance added to squared-distance comparisons.

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
        if subspace_dim is not None and subspace_dim <= 0:
            raise ValueError("subspace_dim must be positive or None")
        self.subspace_dim = subspace_dim
        self.seed = seed
        super().__init__(
            k=k,
            radius=radius,
            window_size=window_size,
            slide_size=slide_size,
            time_key=time_key,
            warm_up_slides=warm_up_slides,
            predict_threshold=predict_threshold,
            eps=eps,
        )

    def _validate_feature_count(self, n_features: int) -> None:
        if self.subspace_dim is not None and self.subspace_dim > n_features:
            raise ValueError(
                f"subspace_dim ({self.subspace_dim}) cannot exceed number of features "
                f"({n_features})"
            )

    def __repr__(self) -> str:
        return (
            "CellNeighborhoodDetector("
            f"k={self.k}, radius={self.radius}, window_size={self.window_size}, "
            f"slide_size={self.slide_size}, subspace_dim={self.subspace_dim}, "
            f"time_key={self.time_key!r}, warm_up_slides={self.warm_up_slides}, "
            f"predict_threshold={self.predict_threshold}, seed={self.seed}, "
            f"samples_seen={self._samples_seen}, "
            f"active_full_cells={len(self._cell_members)})"
        )
