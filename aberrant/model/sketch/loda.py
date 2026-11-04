"""Bounded streaming LODA adaptation for anomaly detection."""

from __future__ import annotations

from typing import cast

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import NumericEventBoundary, coerce_integer_count

# Bound on the comparisons of warm-up rows with interior bin edges held at
# once, so fitting the histograms needs about 1 MiB of temporaries.
_WARM_UP_COMPARISONS = 1 << 20


class StreamingLODA(BaseModel):
    """
    Bounded streaming adaptation of LODA.

    LODA projects each sample to multiple random one-dimensional views and
    maintains per-view streaming histograms. The anomaly score is the mean
    negative log-density across projections. It uses a fixed number of
    projections and fixed warm-up bin edges, so it is a bounded streaming
    adaptation rather than an exact reproduction of the paper's model-selection
    procedure.

    Notes:
    - Scores are continuous and non-negative.
    - Scores are ``0.0`` until warm-up is complete.
    - Memory is bounded by fixed-size projection and histogram state.
    - Feature schema is fixed after the first ``learn_one`` call.

    Args:
        n_projections: Number of random one-dimensional projections and
            histograms.
        n_bins: Fixed number of equal-width bins fitted per projection.
        sparsity: Fraction of input features active in each projection, in
            ``(0, 1]``. ``None`` uses ``min(1, 1 / sqrt(n_features))``.
        warm_up_samples: Number of projected learned samples used to fit the
            fixed histogram edges and initial counts.
        decay: Factor in ``(0, 1]`` applied to all histogram counts before each
            post-warm-up learned insertion.
        time_key: Event-time field to exclude from the feature vector. ``None``
            uses one-based arrival order; explicit times must be finite and
            non-decreasing. Time deltas do not alter ``decay``.
        pseudocount: Positive additive smoothing count for each bin.
        predict_threshold: Non-negative score boundary used by ``predict_one``.
        seed: Seed for model-local sparse projection generation.
        eps: Positive numerical floor for bin widths and log-density scoring.

    References:
        Pevny, T. (2016). Loda: Lightweight on-line detector of anomalies.
        Machine Learning, 102, 275-304.
        https://doi.org/10.1007/s10994-015-5521-0
    """

    def __init__(
        self,
        n_projections: int = 100,
        n_bins: int = 32,
        sparsity: float | None = None,
        warm_up_samples: int = 256,
        decay: float = 1.0,
        time_key: str | None = None,
        pseudocount: float = 1.0,
        predict_threshold: float = 0.5,
        seed: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        if n_projections <= 0:
            raise ValueError("n_projections must be positive")
        if n_bins <= 1:
            raise ValueError("n_bins must be greater than 1")
        if sparsity is not None and not (0.0 < sparsity <= 1.0):
            raise ValueError("sparsity must be in (0, 1] or None")
        warm_up_samples = coerce_integer_count(warm_up_samples, label="warm_up_samples")
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        if time_key is not None and (not isinstance(time_key, str) or not time_key):
            raise ValueError("time_key must be a non-empty string or None")
        if pseudocount <= 0.0:
            raise ValueError("pseudocount must be positive")
        if predict_threshold < 0.0:
            raise ValueError("predict_threshold must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.n_projections = n_projections
        self.n_bins = n_bins
        self.sparsity = sparsity
        self.warm_up_samples = warm_up_samples
        self.decay = decay
        self.time_key = time_key
        self.pseudocount = pseudocount
        self.predict_threshold = predict_threshold
        self.seed = seed
        self.eps = eps

        self._reset_state()

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self.seed)

        self._boundary = NumericEventBoundary(time_key=self.time_key)
        self._projection_matrix: np.ndarray | None = None
        self._bin_edges: np.ndarray | None = None
        self._bin_counts: np.ndarray | None = None
        self._bin_totals: np.ndarray | None = None
        # Fixed with the edges after warm-up: the interior edges transposed to
        # (n_bins - 1, n_projections), each bin's floored width, and the index
        # of each projection's last bin in the flattened (C-order) histograms.
        self._inner_edges: np.ndarray | None = None
        self._bin_widths: np.ndarray | None = None
        self._last_bins: np.ndarray = (
            np.arange(self.n_projections, dtype=np.intp) * self.n_bins + self.n_bins - 1
        )
        self._warmup_buffer: list[np.ndarray] = []
        self._ready = False

        self._samples_seen = 0

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed with ``learn_one``."""
        return self._samples_seen

    def _initialize_projection_matrix(self, n_features: int) -> None:
        if self._projection_matrix is not None:
            return

        if self.sparsity is None:
            density = min(1.0, 1.0 / float(np.sqrt(float(n_features))))
        else:
            density = self.sparsity
        active_features = max(1, int(round(density * n_features)))

        projection_matrix = np.zeros(
            (self.n_projections, n_features),
            dtype=np.float64,
        )
        scale = 1.0 / float(np.sqrt(float(active_features)))
        for projection in range(self.n_projections):
            indices = self._rng.choice(
                n_features,
                size=active_features,
                replace=False,
            )
            signs = self._rng.choice(
                np.array([-1.0, 1.0], dtype=np.float64),
                size=active_features,
                replace=True,
            )
            projection_matrix[projection, indices] = signs * scale
        self._projection_matrix = projection_matrix

    def _project(self, vector: np.ndarray) -> np.ndarray:
        if self._projection_matrix is None:
            raise RuntimeError("Projection matrix is not initialized")
        return cast(np.ndarray, self._projection_matrix @ vector)

    def _flat_bins(self, projected: np.ndarray, inner_edges: np.ndarray) -> np.ndarray:
        """Return each projection's bin as an index into the flattened histograms.

        Bin ``j`` covers ``[edges[j], edges[j + 1])`` and the outer bins also take
        values outside the fitted range. A value therefore lands in its
        projection's last bin, minus one bin for every interior edge above it.
        That is the bin ``np.searchsorted(edges, value, side="right") - 1`` finds
        after clipping, here for all projections at once. ``projected`` has
        shape ``(n_projections,)``, or ``(rows, 1, n_projections)`` for a batch.
        """
        above = np.add.reduce(projected < inner_edges, axis=-2)
        return cast(np.ndarray, self._last_bins - above)

    def _fit_histograms_from_buffer(self) -> None:
        if len(self._warmup_buffer) < self.warm_up_samples:
            return

        warmup = np.vstack(self._warmup_buffer)
        mins = np.min(warmup, axis=0)
        maxs = np.max(warmup, axis=0)

        lows = mins.copy()
        highs = maxs.copy()
        degenerate = np.abs(highs - lows) <= self.eps
        lows[degenerate] -= 0.5
        highs[degenerate] += 0.5

        edges = np.empty((self.n_projections, self.n_bins + 1), dtype=np.float64)
        for projection in range(self.n_projections):
            edges[projection] = np.linspace(
                float(lows[projection]),
                float(highs[projection]),
                num=self.n_bins + 1,
                dtype=np.float64,
            )
        inner_edges = np.ascontiguousarray(edges[:, 1:-1].T)

        counts = np.zeros(self.n_projections * self.n_bins, dtype=np.int64)
        rows = max(1, _WARM_UP_COMPARISONS // inner_edges.size)
        for start in range(0, len(warmup), rows):
            batch = warmup[start : start + rows, np.newaxis, :]
            bins = self._flat_bins(batch, inner_edges)
            counts += np.bincount(bins.ravel(), minlength=counts.size)

        self._bin_edges = edges
        self._inner_edges = inner_edges
        self._bin_widths = np.maximum(np.diff(edges, axis=1), self.eps)
        self._bin_counts = counts.astype(np.float64).reshape(
            self.n_projections, self.n_bins
        )
        self._bin_totals = np.full(
            self.n_projections, float(len(warmup)), dtype=np.float64
        )
        self._warmup_buffer.clear()
        self._ready = True

    def _update_histograms(self, projected: np.ndarray) -> None:
        if (
            self._inner_edges is None
            or self._bin_counts is None
            or self._bin_totals is None
        ):
            raise RuntimeError("Histogram state is not initialized")

        if self.decay < 1.0:
            self._bin_counts *= self.decay
            self._bin_totals *= self.decay

        bins = self._flat_bins(projected, self._inner_edges)
        # The counts are allocated C-contiguous and only updated in place, so
        # the flat reshape is a view. Each projection owns distinct bins.
        self._bin_counts.reshape(-1)[bins] += 1.0
        self._bin_totals += 1.0

    def _score_projected(self, projected: np.ndarray) -> float:
        if (
            self._inner_edges is None
            or self._bin_widths is None
            or self._bin_counts is None
            or self._bin_totals is None
        ):
            return 0.0

        bins = self._flat_bins(projected, self._inner_edges)
        counts = self._bin_counts.take(bins)
        widths = self._bin_widths.take(bins)

        mass = (counts + self.pseudocount) / (
            self._bin_totals + self.pseudocount * float(self.n_bins)
        )
        density = mass / widths
        # The mean of the negative log-densities; np.add.reduce is the sum that
        # np.mean takes, without its Python-level overhead.
        score = float(np.add.reduce(-np.log(density + self.eps))) / self.n_projections
        return float(max(score, 0.0))

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one sample."""
        event = self._boundary.preview(x)
        vector = event.features.values

        self._initialize_projection_matrix(n_features=vector.shape[0])
        projected = self._project(vector)

        if self._ready:
            self._update_histograms(projected)
        else:
            self._warmup_buffer.append(projected)
            if len(self._warmup_buffer) >= self.warm_up_samples:
                self._fit_histograms_from_buffer()

        self._samples_seen += 1
        self._boundary.commit(event)

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for one sample."""
        event = self._boundary.preview(x)
        if not self._boundary.schema.is_established or not self._ready:
            return 0.0

        projected = self._project(event.features.values)
        return self._score_projected(projected)

    def predict_one(self, x: dict[str, float]) -> int:
        """Return binary anomaly prediction using ``predict_threshold``."""
        return int(self.score_one(x) >= self.predict_threshold)

    def __repr__(self) -> str:
        return (
            "StreamingLODA("
            f"n_projections={self.n_projections}, n_bins={self.n_bins}, "
            f"sparsity={self.sparsity}, warm_up_samples={self.warm_up_samples}, "
            f"decay={self.decay}, time_key={self.time_key!r}, "
            f"pseudocount={self.pseudocount}, "
            f"predict_threshold={self.predict_threshold}, seed={self.seed}, "
            f"samples_seen={self._samples_seen}, ready={self._ready})"
        )
