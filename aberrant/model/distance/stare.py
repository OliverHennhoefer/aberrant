"""Stationary-region neighbor detector for streaming local outlier detection."""

from __future__ import annotations

from aberrant.model.distance._radius_neighbors import _RadiusNeighborDetector


class StationaryRegionNeighborDetector(_RadiusNeighborDetector):
    """
    Stationary-region neighbor detector for streaming data.

    The detector keeps a bounded sliding window and quantizes points into
    radius-sized grid cells. Scores are based on the number of neighbors within
    ``radius`` in the current window:
    ``score = 1 - min(neighbor_count / k, 1)``.

    Candidate cell neighborhoods are reused while cell topology stays unchanged.
    Distances are always evaluated for the current query and live points. Unlike
    the paper, this class uses radius-neighbor counts rather than kernel-density
    estimates and returns a per-query score rather than a top-n outlier set.

    Notes:
    - Scores are continuous and bounded in ``[0, 1]``.
    - State is bounded by ``window_size``.
    - Feature schema is fixed after the first ``learn_one`` call.

    Args:
        k: Neighbor count at which the scarcity score reaches zero.
        radius: Finite positive Euclidean neighborhood radius and grid-cell width.
        window_size: Maximum number of learned points retained. It must exceed
            ``k``.
        slide_size: Number of learned events per warm-up slide.
        skip_threshold: Compatibility parameter in ``[0, 1]``. Approximate cached
            counts have been replaced by exact query-specific counts.
        time_key: Event-time field to exclude from the feature vector. ``None``
            uses one-based arrival order; explicit times must be finite and
            non-decreasing.
        warm_up_slides: Complete slides required before non-zero scoring.
        predict_threshold: Score boundary used by ``predict_one``.
        eps: Positive tolerance added to squared-distance comparisons.

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
        if not (0.0 <= skip_threshold <= 1.0):
            raise ValueError("skip_threshold must be in [0, 1]")
        self.skip_threshold = skip_threshold
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

    def __repr__(self) -> str:
        return (
            "StationaryRegionNeighborDetector("
            f"k={self.k}, radius={self.radius}, window_size={self.window_size}, "
            f"slide_size={self.slide_size}, skip_threshold={self.skip_threshold}, "
            f"time_key={self.time_key!r}, warm_up_slides={self.warm_up_slides}, "
            f"predict_threshold={self.predict_threshold}, eps={self.eps}, "
            f"samples_seen={self._samples_seen}, active_cells={len(self._cell_members)})"
        )
