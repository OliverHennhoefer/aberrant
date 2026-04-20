"""LODA sketch detector for streaming anomaly detection."""

from __future__ import annotations

from typing import cast

import numpy as np

from aberrant.base.model import BaseModel


class LODA(BaseModel):
    """
    Lightweight On-line Detector of Anomalies (LODA).

    LODA projects each sample to multiple random one-dimensional views and
    maintains per-view streaming histograms. The anomaly score is the mean
    negative log-density across projections.

    Notes:
    - Scores are continuous and non-negative.
    - Scores are ``0.0`` until warm-up is complete.
    - Memory is bounded by fixed-size projection and histogram state.
    - Feature schema is fixed after the first ``learn_one`` call.

    References:
        Pevny, T. (2016). Loda: Lightweight on-line detector of anomalies.
        Machine Learning, 102, 275-304.
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
        if warm_up_samples <= 0:
            raise ValueError("warm_up_samples must be positive")
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

        self._feature_order: tuple[str, ...] | None = None
        self._projection_matrix: np.ndarray | None = None
        self._bin_edges: np.ndarray | None = None
        self._bin_counts: np.ndarray | None = None
        self._bin_totals: np.ndarray | None = None
        self._warmup_buffer: list[np.ndarray] = []
        self._ready = False

        self._arrival_index = 0
        self._max_learned_time = float("-inf")
        self._samples_seen = 0

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed with ``learn_one``."""
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
            features = {key: value for key, value in x.items() if key != self.time_key}

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

    def _bin_indices_from_edges(
        self,
        projected: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        indices = np.empty(self.n_projections, dtype=np.intp)
        for projection in range(self.n_projections):
            edge_row = edges[projection]
            index = int(np.searchsorted(edge_row, projected[projection], side="right")) - 1
            indices[projection] = int(np.clip(index, 0, self.n_bins - 1))
        return indices

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

        counts = np.zeros((self.n_projections, self.n_bins), dtype=np.float64)
        totals = np.zeros(self.n_projections, dtype=np.float64)
        row_indices = np.arange(self.n_projections, dtype=np.intp)
        for projected in warmup:
            bins = self._bin_indices_from_edges(projected, edges)
            counts[row_indices, bins] += 1.0
            totals += 1.0

        self._bin_edges = edges
        self._bin_counts = counts
        self._bin_totals = totals
        self._warmup_buffer.clear()
        self._ready = True

    def _update_histograms(self, projected: np.ndarray) -> None:
        if self._bin_edges is None or self._bin_counts is None or self._bin_totals is None:
            raise RuntimeError("Histogram state is not initialized")

        if self.decay < 1.0:
            self._bin_counts *= self.decay
            self._bin_totals *= self.decay

        bins = self._bin_indices_from_edges(projected, self._bin_edges)
        row_indices = np.arange(self.n_projections, dtype=np.intp)
        self._bin_counts[row_indices, bins] += 1.0
        self._bin_totals += 1.0

    def _score_projected(self, projected: np.ndarray) -> float:
        if self._bin_edges is None or self._bin_counts is None or self._bin_totals is None:
            return 0.0

        bins = self._bin_indices_from_edges(projected, self._bin_edges)
        row_indices = np.arange(self.n_projections, dtype=np.intp)

        counts = self._bin_counts[row_indices, bins]
        totals = self._bin_totals
        left_edges = self._bin_edges[row_indices, bins]
        right_edges = self._bin_edges[row_indices, bins + 1]
        widths = np.maximum(right_edges - left_edges, self.eps)

        mass = (counts + self.pseudocount) / (
            totals + self.pseudocount * float(self.n_bins)
        )
        density = mass / widths
        score = float(np.mean(-np.log(density + self.eps)))
        return float(max(score, 0.0))

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one sample."""
        current_time, vector = self._prepare_input(x, mutate_schema=True)
        if vector is None:
            return

        self._initialize_projection_matrix(n_features=vector.shape[0])
        projected = self._project(vector)

        if self._ready:
            self._update_histograms(projected)
        else:
            self._warmup_buffer.append(projected)
            if len(self._warmup_buffer) >= self.warm_up_samples:
                self._fit_histograms_from_buffer()

        self._samples_seen += 1
        self._max_learned_time = current_time
        if self.time_key is None:
            self._arrival_index += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for one sample."""
        _current_time, vector = self._prepare_input(x, mutate_schema=False)
        if vector is None or not self._ready:
            return 0.0

        projected = self._project(vector)
        return self._score_projected(projected)

    def predict_one(self, x: dict[str, float]) -> int:
        """Return binary anomaly prediction using ``predict_threshold``."""
        return int(self.score_one(x) >= self.predict_threshold)

    def __repr__(self) -> str:
        return (
            "LODA("
            f"n_projections={self.n_projections}, n_bins={self.n_bins}, "
            f"sparsity={self.sparsity}, warm_up_samples={self.warm_up_samples}, "
            f"decay={self.decay}, time_key={self.time_key!r}, "
            f"pseudocount={self.pseudocount}, "
            f"predict_threshold={self.predict_threshold}, seed={self.seed}, "
            f"samples_seen={self._samples_seen}, ready={self._ready})"
        )
