"""SDOstream distance-based detector for streaming anomaly detection."""

from __future__ import annotations

from typing import Literal

import numpy as np

from aberrant._utils.validation import (
    NumericEventBoundary,
    coerce_finite_number,
    coerce_integer_count,
)
from aberrant.base.model import BaseModel
from aberrant.base.readiness import Readiness, ReadinessUnit


class SDOStream(BaseModel):
    """
    Streaming Density Observer detector.

    SDOStream maintains a fixed-size set of observers and an exponentially
    decayed activity score per observer. A sample is scored by the median
    distance to its nearest active observers, where active observers are
    selected via activity quantile filtering.

    Observers are sampled at a rate proportional to the time elapsed since the
    last sampled point, so ``time_scale`` is in the units of the event times. When the
    observer set is full, a sampled point replaces the observer with the lowest
    ``observations / (1 - exp(-age / time_scale))``: its decayed activity relative to the
    most it could have gathered since it was added, as in the authors'
    reference implementation.

    Notes:
    - Scores are continuous, non-negative distances.
    - State is bounded by ``max_observers`` and independent of stream length.
    - Feature schema is fixed after the first ``learn_one`` call.

    Args:
        max_observers: Maximum number of observers retained.
        time_scale: Positive fading time scale and observer-sampling time scale.
        inactive_quantile: Activity quantile below which observers are excluded, in
            ``[0, 1)``.
        n_neighbors: Number of nearest active observers whose distances enter
            the median score. It cannot exceed ``max_observers``.
        distance: Distance metric: ``"euclidean"``, ``"manhattan"``,
            ``"chebyshev"``, or ``"minkowski"``.
        minkowski_p: Positive order used only for Minkowski distance.
        time_key: Event-time field to exclude from the feature vector. ``None``
            uses one-based arrival order; explicit times must be finite and
            non-decreasing.
        warm_up: Observer count required before scoring. ``None``
            resolves to at least ``n_neighbors`` and at least two, capped at
            ``max_observers``.
        seed: Seed for the model-local observer-sampling generator.

    References:
        Hartl, A., Iglesias Vazquez, F., & Zseby, T. (2020). SDOstream:
        Low-Density Models for Streaming Outlier Detection.
        https://www.esann.org/sites/default/files/proceedings/2020/ES2020-143.pdf
    """

    def __init__(
        self,
        *,
        max_observers: int = 256,
        time_scale: float = 512.0,
        inactive_quantile: float = 0.3,
        n_neighbors: int = 6,
        distance: Literal["euclidean", "manhattan", "chebyshev", "minkowski"] = (
            "euclidean"
        ),
        minkowski_p: float = 2.0,
        time_key: str | None = None,
        warm_up: int | None = None,
        seed: int | None = None,
    ) -> None:
        max_observers = coerce_integer_count(
            max_observers, label="max_observers", minimum=1
        )
        n_neighbors = coerce_integer_count(n_neighbors, label="n_neighbors", minimum=1)
        if time_scale <= 0.0:
            raise ValueError("time_scale must be positive")
        if not (0.0 <= inactive_quantile < 1.0):
            raise ValueError("inactive_quantile must be in [0, 1)")
        if n_neighbors > max_observers:
            raise ValueError("n_neighbors must be less than or equal to max_observers")
        if distance not in ("euclidean", "manhattan", "chebyshev", "minkowski"):
            raise ValueError(
                "distance must be one of: euclidean, manhattan, chebyshev, minkowski"
            )
        if distance == "minkowski" and minkowski_p <= 0.0:
            raise ValueError("minkowski_p must be positive when distance='minkowski'")
        if time_key is not None and (not isinstance(time_key, str) or not time_key):
            raise ValueError("time_key must be a non-empty string or None")
        if warm_up is not None:
            if warm_up <= 0:
                raise ValueError("warm_up must be positive")
            if warm_up > max_observers:
                raise ValueError("warm_up must be less than or equal to max_observers")

        self.max_observers = max_observers
        self.time_scale = time_scale
        self.inactive_quantile = inactive_quantile
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.minkowski_p = minkowski_p
        self.time_key = time_key
        warm_up_default = (
            max(2, self.n_neighbors) if warm_up is None else max(2, warm_up)
        )
        self.warm_up = min(self.max_observers, warm_up_default)
        self.seed = seed

        self._fading = float(np.exp(-1.0 / self.time_scale))
        self._sampling_prefactor = (self.max_observers * self.max_observers) / (
            self.n_neighbors * self.time_scale
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self.seed)

        self._boundary = NumericEventBoundary(time_key=self.time_key)
        self._observers: np.ndarray | None = None
        self._observations: np.ndarray | None = None
        self._time_added: np.ndarray | None = None
        self._time_touched: np.ndarray | None = None

        self._n_observers = 0
        self._sample_index = 0
        self._last_added_index = -1
        self._last_added_time = 0.0

    @property
    def n_observers(self) -> int:
        """Number of observers currently maintained by the model."""
        return self._n_observers

    def _ensure_state_arrays(self, n_features: int) -> None:
        if self._observers is not None:
            return

        self._observers = np.zeros((self.max_observers, n_features), dtype=np.float64)
        self._observations = np.zeros(self.max_observers, dtype=np.float64)
        self._time_added = np.zeros(self.max_observers, dtype=np.float64)
        self._time_touched = np.zeros(self.max_observers, dtype=np.float64)

    def _decay_factor(self, delta: np.ndarray | float) -> np.ndarray | float:
        return np.power(self._fading, delta)

    def _decayed_observations(self, current_time: float) -> np.ndarray:
        if self._observations is None or self._time_touched is None:
            raise RuntimeError("Model state arrays are not initialized")
        if self._n_observers == 0:
            return np.zeros(0, dtype=np.float64)

        idx = slice(0, self._n_observers)
        deltas = current_time - self._time_touched[idx]
        return self._observations[idx] * self._decay_factor(deltas)

    def _pairwise_distances(
        self, vector: np.ndarray, observer_matrix: np.ndarray
    ) -> np.ndarray:
        if observer_matrix.size == 0:
            return np.zeros(0, dtype=np.float64)

        diff = observer_matrix - vector
        abs_diff = np.abs(diff)

        if self.distance == "euclidean":
            return np.asarray(np.sqrt(np.sum(diff * diff, axis=1)), dtype=np.float64)
        if self.distance == "manhattan":
            return np.asarray(np.sum(abs_diff, axis=1), dtype=np.float64)
        if self.distance == "chebyshev":
            return np.asarray(np.max(abs_diff, axis=1), dtype=np.float64)

        # Minkowski distance.
        return np.asarray(
            np.sum(abs_diff**self.minkowski_p, axis=1) ** (1.0 / self.minkowski_p),
            dtype=np.float64,
        )

    def _insert_observer(self, vector: np.ndarray, current_time: float) -> None:
        if (
            self._observers is None
            or self._observations is None
            or self._time_added is None
            or self._time_touched is None
        ):
            raise RuntimeError("Model state arrays are not initialized")

        idx = self._n_observers
        self._observers[idx] = vector
        self._observations[idx] = 1.0
        self._time_added[idx] = current_time
        self._time_touched[idx] = current_time
        self._n_observers += 1

    def _replace_observer(
        self, index: int, vector: np.ndarray, current_time: float
    ) -> None:
        if (
            self._observers is None
            or self._observations is None
            or self._time_added is None
            or self._time_touched is None
        ):
            raise RuntimeError("Model state arrays are not initialized")

        self._observers[index] = vector
        self._observations[index] = 1.0
        self._time_added[index] = current_time
        self._time_touched[index] = current_time

    def _nearest_indices(self, values: np.ndarray, count: int) -> np.ndarray:
        if values.size == 0:
            return np.zeros(0, dtype=np.intp)
        if count >= values.size:
            return np.arange(values.size, dtype=np.intp)

        return np.argpartition(values, count - 1)[:count]

    def _active_indices(self, current_time: float) -> np.ndarray:
        if self._n_observers == 0:
            return np.zeros(0, dtype=np.intp)

        decayed = self._decayed_observations(current_time)
        active_count = max(
            self.n_neighbors,
            int((1.0 - self.inactive_quantile) * self._n_observers),
        )
        active_count = min(active_count, self._n_observers)

        if active_count >= self._n_observers:
            return np.arange(self._n_observers, dtype=np.intp)

        threshold = self._n_observers - active_count
        return np.argpartition(decayed, threshold)[threshold:]

    def _choose_replacement_index(self, current_time: float) -> int:
        if (
            self._time_added is None
            or self._time_touched is None
            or self._observations is None
        ):
            raise RuntimeError("Model state arrays are not initialized")

        idx = slice(0, self._n_observers)
        decayed = self._observations[idx] * self._decay_factor(
            current_time - self._time_touched[idx]
        )
        # The activity an observer can have gathered since it was added
        # decays like its observations; one added at this instant is kept.
        reachable = 1.0 - self._decay_factor(current_time - self._time_added[idx])
        normalized = np.full(decayed.shape, np.inf)
        np.divide(decayed, reachable, out=normalized, where=reachable > 0.0)
        return int(np.argmin(normalized))

    def _advance(self) -> None:
        self._sample_index += 1

    @property
    def readiness(self) -> Readiness:
        """Ready once ``warm_up`` and ``n_neighbors`` observers exist."""
        return Readiness.after(
            self._n_observers,
            max(self.warm_up, self.n_neighbors),
            ReadinessUnit.OBSERVERS,
        )

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update model state with one sample.

        Args:
            x: Input feature dictionary.
        """
        event = self._boundary.preview(x)
        current_time = coerce_finite_number(
            event.timestamp.value, label="Timestamp value"
        )
        vector = event.features.values
        self._ensure_state_arrays(n_features=vector.shape[0])

        if self._n_observers == 0:
            self._insert_observer(vector, current_time)
            self._last_added_index = self._sample_index
            self._last_added_time = current_time
            self._advance()
            self._boundary.commit(event)
            return

        if (
            self._observers is None
            or self._observations is None
            or self._time_touched is None
        ):
            raise RuntimeError("Model state arrays are not initialized")

        current_index = self._sample_index
        all_observers = self._observers[: self._n_observers]
        distances = self._pairwise_distances(vector, all_observers)

        nn_count = min(self.n_neighbors, self._n_observers)
        nn_idx = self._nearest_indices(distances, nn_count)

        # Lazy decay update for touched observers only.
        for idx in nn_idx:
            delta = current_time - self._time_touched[idx]
            self._observations[idx] *= float(self._decay_factor(delta))
            self._observations[idx] += 1.0
            self._time_touched[idx] = current_time

        decayed = self._decayed_observations(current_time)
        obs_sum = float(np.sum(decayed))
        nn_obs_sum = float(np.sum(self._observations[nn_idx]))

        lhs = (
            float(self._rng.uniform())
            * obs_sum
            * (current_index - self._last_added_index)
        )
        rhs = (
            self._sampling_prefactor
            * nn_obs_sum
            * (current_time - self._last_added_time)
        )

        if lhs < rhs:
            if self._n_observers < self.max_observers:
                self._insert_observer(vector, current_time)
            else:
                replace_idx = self._choose_replacement_index(current_time)
                self._replace_observer(replace_idx, vector, current_time)

            self._last_added_index = current_index
            self._last_added_time = current_time

        self._advance()
        self._boundary.commit(event)

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score for one sample.

        Args:
            x: Input feature dictionary.

        Returns:
            Continuous non-negative anomaly score.
        """
        event = self._boundary.preview(x)
        current_time = coerce_finite_number(
            event.timestamp.value, label="Timestamp value"
        )
        vector = event.features.values

        if self._n_observers < self.warm_up:
            return 0.0
        if self._observers is None:
            return 0.0

        active_idx = self._active_indices(current_time)
        if active_idx.size < self.n_neighbors:
            return 0.0

        active_observers = self._observers[active_idx]
        distances = self._pairwise_distances(vector, active_observers)
        nn_dist_idx = self._nearest_indices(distances, self.n_neighbors)
        nearest = distances[nn_dist_idx]

        score = float(np.median(nearest))
        return float(max(score, 0.0))
