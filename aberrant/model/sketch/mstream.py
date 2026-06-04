"""MStream sketch-based detector for streaming anomaly detection."""

from __future__ import annotations

import math

import numpy as np

from aberrant.base.model import BaseModel


class MStream(BaseModel):
    """
    MStream detector for anomaly detection in multi-aspect streams.

    This implementation follows the authors' ``anom.cpp``, ``numerichash.cpp``,
    ``categhash.cpp``, and ``recordhash.cpp``:

    - every numeric and categorical attribute has its own current and
      cumulative count sketch,
    - the complete record is hashed into an additional random-hyperplane sketch,
    - current counts are decayed when a new timestamp arrives,
    - each candidate-inclusive count pair is converted with the published
      chi-square score and all contributions are combined with ``log1p``.

    Numeric attributes use the authors' ``log10(1 + x)`` transform and online
    min-max normalization. Names listed in ``categorical_features`` must contain
    integer-like values and use count-min hashing instead.

    ``score_one`` previews timestamp decay, online normalization, and the
    candidate insertion without mutating learned state. Calling
    ``score_one(x)`` followed by ``learn_one(x)`` therefore matches the
    authors' update-and-score order while preserving this library's
    score-before-learn convention.

    Notes:
    - Numeric values must be greater than ``-1`` for ``log10(1 + x)``.
    - Feature schema is fixed by the first sample, excluding ``time_key``.
    - Scores are continuous and non-negative.
    - State is bounded by the configured sketch dimensions.

    References:
        Bhatia, S., Jain, A., Li, P., Kumar, R., & Hooi, B. (2021). MStream:
        Fast Anomaly Detection in Multi-Aspect Streams.
        https://doi.org/10.1145/3442381.3450023
        Original implementation: https://github.com/Stream-AD/MStream
    """

    def __init__(
        self,
        rows: int = 2,
        buckets: int = 1024,
        alpha: float = 0.6,
        time_key: str | None = None,
        categorical_features: tuple[str, ...] = (),
        warm_up_buckets: int = 0,
        seed: int | None = None,
    ) -> None:
        if rows <= 0:
            raise ValueError("rows must be positive")
        if buckets <= 0:
            raise ValueError("buckets must be positive")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if time_key is not None and (not isinstance(time_key, str) or not time_key):
            raise ValueError("time_key must be a non-empty string or None")
        if any(not isinstance(name, str) or not name for name in categorical_features):
            raise ValueError("categorical_features must contain non-empty strings")
        if len(set(categorical_features)) != len(categorical_features):
            raise ValueError("categorical_features must not contain duplicates")
        if time_key is not None and time_key in categorical_features:
            raise ValueError("time_key cannot also be a categorical feature")
        if warm_up_buckets < 0:
            raise ValueError("warm_up_buckets must be non-negative")

        self.rows = rows
        self.buckets = buckets
        self.alpha = alpha
        self.time_key = time_key
        self.categorical_features = tuple(categorical_features)
        self.warm_up_buckets = warm_up_buckets
        self.seed = seed

        self._reset_state()

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._row_index = np.arange(self.rows, dtype=np.intp)

        self._feature_order: tuple[str, ...] | None = None
        self._numeric_indices: np.ndarray | None = None
        self._categorical_indices: np.ndarray | None = None

        self._numeric_min: np.ndarray | None = None
        self._numeric_max: np.ndarray | None = None
        self._numeric_current: np.ndarray | None = None
        self._numeric_total: np.ndarray | None = None

        self._categorical_hash_a: np.ndarray | None = None
        self._categorical_hash_b: np.ndarray | None = None
        self._categorical_current: np.ndarray | None = None
        self._categorical_total: np.ndarray | None = None

        self._record_numeric_planes: np.ndarray | None = None
        self._record_categorical_weights: np.ndarray | None = None
        self._record_current: np.ndarray | None = None
        self._record_total: np.ndarray | None = None

        self._current_bucket: int | None = None
        self._first_bucket: int | None = None
        self._arrival_index = 0
        self._samples_seen = 0

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of observed samples processed via learn_one."""
        return self._samples_seen

    @staticmethod
    def _coerce_bucket(value: float, key: str) -> int:
        if not isinstance(value, int | float | np.number):
            raise ValueError(f"Feature '{key}' must be numeric")
        as_float = float(value)
        if not np.isfinite(as_float):
            raise ValueError(f"Feature '{key}' must be finite")
        as_int = int(round(as_float))
        if not np.isclose(as_float, float(as_int), rtol=0.0, atol=1e-9):
            raise ValueError("Timestamp must be integer-like")
        return as_int

    def _split_input(self, x: dict[str, float]) -> tuple[int, dict[str, float]]:
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
            bucket = self._arrival_index + 1
            features = x
        else:
            if self.time_key not in x:
                raise ValueError(f"Missing time_key '{self.time_key}' in input sample")
            bucket = self._coerce_bucket(x[self.time_key], self.time_key)
            features = {key: value for key, value in x.items() if key != self.time_key}

        if not features:
            raise ValueError("Input must contain at least one non-time feature")
        if self._current_bucket is not None and bucket < self._current_bucket:
            raise ValueError(
                f"Non-monotonic timestamp: received {bucket}, current {self._current_bucket}"
            )
        return bucket, features

    def _initialize_state(self, features: dict[str, float]) -> None:
        categorical = set(self.categorical_features)
        missing_categorical = categorical.difference(features)
        if missing_categorical:
            names = ", ".join(sorted(missing_categorical))
            raise ValueError(f"Missing configured categorical features: [{names}]")

        self._feature_order = tuple(sorted(features))
        self._numeric_indices = np.asarray(
            [
                index
                for index, name in enumerate(self._feature_order)
                if name not in categorical
            ],
            dtype=np.intp,
        )
        self._categorical_indices = np.asarray(
            [
                index
                for index, name in enumerate(self._feature_order)
                if name in categorical
            ],
            dtype=np.intp,
        )
        n_numeric = len(self._numeric_indices)
        n_categorical = len(self._categorical_indices)

        self._numeric_min = np.full(n_numeric, np.inf, dtype=np.float64)
        self._numeric_max = np.full(n_numeric, -np.inf, dtype=np.float64)
        self._numeric_current = np.zeros((n_numeric, self.buckets), dtype=np.float64)
        self._numeric_total = np.zeros_like(self._numeric_current)

        high = max(self.buckets, 2)
        self._categorical_hash_a = self._rng.integers(
            1,
            high,
            size=(n_categorical, self.rows),
            dtype=np.int64,
        )
        self._categorical_hash_b = self._rng.integers(
            0,
            self.buckets,
            size=(n_categorical, self.rows),
            dtype=np.int64,
        )
        self._categorical_current = np.zeros(
            (n_categorical, self.rows, self.buckets), dtype=np.float64
        )
        self._categorical_total = np.zeros_like(self._categorical_current)

        n_record_bits = math.ceil(math.log2(self.buckets))
        self._record_numeric_planes = self._rng.normal(
            size=(self.rows, n_record_bits, n_numeric)
        )
        self._record_categorical_weights = self._rng.integers(
            0,
            self.buckets,
            size=(self.rows, n_categorical),
            dtype=np.int64,
        )
        self._record_current = np.zeros((self.rows, self.buckets), dtype=np.float64)
        self._record_total = np.zeros_like(self._record_current)

    def _set_or_validate_feature_order(self, features: dict[str, float]) -> None:
        if self._feature_order is None:
            self._initialize_state(features)
            return

        received = set(features)
        expected = set(self._feature_order)
        if received != expected:
            expected_keys = ", ".join(self._feature_order)
            received_keys = ", ".join(sorted(features))
            raise ValueError(
                "Inconsistent feature keys. "
                f"Expected [{expected_keys}], received [{received_keys}]."
            )

    def _vectorize(self, features: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        if (
            self._feature_order is None
            or self._numeric_indices is None
            or self._categorical_indices is None
        ):
            raise RuntimeError("Feature schema is not initialized")

        values = np.fromiter(
            (float(features[key]) for key in self._feature_order),
            dtype=np.float64,
            count=len(self._feature_order),
        )
        numeric = values[self._numeric_indices]
        if np.any(numeric <= -1.0):
            raise ValueError("Numeric MStream features must be greater than -1")

        categorical_values = values[self._categorical_indices]
        categorical = np.rint(categorical_values).astype(np.int64)
        if not np.allclose(categorical_values, categorical, rtol=0.0, atol=1e-9):
            raise ValueError("Configured categorical features must be integer-like")
        return numeric, categorical

    def _prepare_sample(
        self, x: dict[str, float]
    ) -> tuple[int, np.ndarray, np.ndarray]:
        bucket, features = self._split_input(x)
        self._set_or_validate_feature_order(features)
        numeric, categorical = self._vectorize(features)
        return bucket, numeric, categorical

    def _rollover_for_learning(self, bucket: int) -> None:
        if self._current_bucket is None:
            self._current_bucket = bucket
            self._first_bucket = bucket
            return
        if bucket == self._current_bucket:
            return

        for current in (
            self._numeric_current,
            self._categorical_current,
            self._record_current,
        ):
            if current is not None:
                current[...] *= self.alpha
        self._current_bucket = bucket

    def _time_index(self, bucket: int) -> int:
        if self._first_bucket is None:
            return 1
        return bucket - self._first_bucket + 1

    def _is_rollover(self, bucket: int) -> bool:
        return self._current_bucket is not None and bucket > self._current_bucket

    def _normalize_numeric(self, numeric: np.ndarray, *, update: bool) -> np.ndarray:
        if self._numeric_min is None or self._numeric_max is None:
            raise RuntimeError("Numeric state is not initialized")
        transformed = np.log10(1.0 + numeric)
        minimum = np.minimum(self._numeric_min, transformed)
        maximum = np.maximum(self._numeric_max, transformed)
        ranges = maximum - minimum
        normalized = np.divide(
            transformed - minimum,
            ranges,
            out=np.zeros_like(transformed),
            where=ranges > 0.0,
        )
        if update:
            self._numeric_min[:] = minimum
            self._numeric_max[:] = maximum
        return normalized

    def _numeric_bins(self, normalized: np.ndarray) -> np.ndarray:
        return np.floor(normalized * float(self.buckets - 1)).astype(np.intp)

    def _categorical_bins(self, categorical: np.ndarray) -> np.ndarray:
        if self._categorical_hash_a is None or self._categorical_hash_b is None:
            raise RuntimeError("Categorical hash state is not initialized")
        if categorical.size == 0:
            return np.empty((0, self.rows), dtype=np.intp)
        return np.asarray(
            (
                categorical[:, np.newaxis] * self._categorical_hash_a
                + self._categorical_hash_b
            )
            % self.buckets,
            dtype=np.intp,
        )

    def _record_bins(
        self, normalized: np.ndarray, categorical: np.ndarray
    ) -> np.ndarray:
        if (
            self._record_numeric_planes is None
            or self._record_categorical_weights is None
        ):
            raise RuntimeError("Record hash state is not initialized")

        if self._record_numeric_planes.shape[1] == 0:
            numeric_hash = np.zeros(self.rows, dtype=np.int64)
        else:
            signs = (
                np.einsum("rbn,n->rb", self._record_numeric_planes, normalized) >= 0.0
            )
            bit_weights = np.left_shift(
                np.int64(1),
                np.arange(signs.shape[1], dtype=np.int64),
            )
            numeric_hash = signs.astype(np.int64) @ bit_weights

        categorical_hash = (
            self._record_categorical_weights @ categorical
            if categorical.size
            else np.zeros(self.rows, dtype=np.int64)
        )
        return np.asarray(
            (numeric_hash + categorical_hash) % self.buckets, dtype=np.intp
        )

    @staticmethod
    def _counts_to_anomaly(total: float, current: float, time_index: int) -> float:
        """Convert count estimates with the formula from ``anom.cpp``."""
        current_mean = total / float(time_index)
        squared_error = max(0.0, current - current_mean) ** 2
        return squared_error / current_mean + squared_error / (
            current_mean * float(max(1, time_index - 1))
        )

    def _candidate_counts(
        self,
        current: np.ndarray,
        total: np.ndarray,
        indices: np.ndarray,
        *,
        rollover: bool,
    ) -> tuple[float, float]:
        current_values = current[self._row_index, indices]
        total_values = total[self._row_index, indices]
        decay = self.alpha if rollover else 1.0
        return (
            float(np.min(current_values)) * decay + 1.0,
            float(np.min(total_values)) + 1.0,
        )

    def _score_prepared(
        self,
        bucket: int,
        normalized: np.ndarray,
        categorical: np.ndarray,
    ) -> float:
        if (
            self._numeric_current is None
            or self._numeric_total is None
            or self._categorical_current is None
            or self._categorical_total is None
            or self._record_current is None
            or self._record_total is None
        ):
            raise RuntimeError("Sketch state is not initialized")

        rollover = self._is_rollover(bucket)
        time_index = self._time_index(bucket)
        total_score = 0.0

        numeric_bins = self._numeric_bins(normalized)
        numeric_decay = self.alpha if rollover else 1.0
        for feature_index, bin_index in enumerate(numeric_bins):
            current = (
                self._numeric_current[feature_index, bin_index] * numeric_decay + 1.0
            )
            total = self._numeric_total[feature_index, bin_index] + 1.0
            total_score += self._counts_to_anomaly(total, current, time_index)

        categorical_bins = self._categorical_bins(categorical)
        for feature_index, indices in enumerate(categorical_bins):
            current, total = self._candidate_counts(
                self._categorical_current[feature_index],
                self._categorical_total[feature_index],
                indices,
                rollover=rollover,
            )
            total_score += self._counts_to_anomaly(total, current, time_index)

        record_bins = self._record_bins(normalized, categorical)
        current, total = self._candidate_counts(
            self._record_current,
            self._record_total,
            record_bins,
            rollover=rollover,
        )
        total_score += self._counts_to_anomaly(total, current, time_index)
        return float(np.log1p(total_score))

    def learn_one(self, x: dict[str, float]) -> None:
        """Update model state with a single sample."""
        bucket, numeric, categorical = self._prepare_sample(x)
        self._rollover_for_learning(bucket)
        normalized = self._normalize_numeric(numeric, update=True)

        if (
            self._numeric_current is None
            or self._numeric_total is None
            or self._categorical_current is None
            or self._categorical_total is None
            or self._record_current is None
            or self._record_total is None
        ):
            raise RuntimeError("Sketch state is not initialized")

        for feature_index, bin_index in enumerate(self._numeric_bins(normalized)):
            self._numeric_current[feature_index, bin_index] += 1.0
            self._numeric_total[feature_index, bin_index] += 1.0

        for feature_index, indices in enumerate(self._categorical_bins(categorical)):
            self._categorical_current[feature_index, self._row_index, indices] += 1.0
            self._categorical_total[feature_index, self._row_index, indices] += 1.0

        record_bins = self._record_bins(normalized, categorical)
        self._record_current[self._row_index, record_bins] += 1.0
        self._record_total[self._row_index, record_bins] += 1.0

        self._samples_seen += 1
        if self.time_key is None:
            self._arrival_index += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute the candidate-inclusive anomaly score without mutating state."""
        bucket, numeric, categorical = self._prepare_sample(x)
        if self._time_index(bucket) - 1 < self.warm_up_buckets:
            return 0.0

        normalized = self._normalize_numeric(numeric, update=False)
        return self._score_prepared(bucket, normalized, categorical)

    def __repr__(self) -> str:
        return (
            f"MStream(rows={self.rows}, buckets={self.buckets}, alpha={self.alpha}, "
            f"time_key={self.time_key!r}, "
            f"categorical_features={self.categorical_features!r}, "
            f"warm_up_buckets={self.warm_up_buckets}, seed={self.seed}, "
            f"samples_seen={self._samples_seen})"
        )
