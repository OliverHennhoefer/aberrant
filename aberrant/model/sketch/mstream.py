"""MStream sketch-based detector for streaming anomaly detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import NumericEventBoundary, PreparedFeatures


@dataclass(slots=True)
class _CountState:
    """Current-bucket and cumulative count arrays."""

    current: np.ndarray
    total: np.ndarray


@dataclass(slots=True)
class _NumericState:
    """Initialized state for numeric attributes."""

    indices: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    counts: _CountState


@dataclass(slots=True)
class _CategoricalState:
    """Initialized state for categorical attributes."""

    indices: np.ndarray
    hash_a: np.ndarray
    hash_b: np.ndarray
    counts: _CountState


@dataclass(slots=True)
class _RecordState:
    """Initialized state for complete-record hashing."""

    numeric_planes: np.ndarray
    categorical_weights: np.ndarray
    counts: _CountState


@dataclass(slots=True)
class _MStreamState:
    """Complete learned state; every field is valid once constructed."""

    numeric: _NumericState
    categorical: _CategoricalState
    record: _RecordState
    current_bucket: int
    first_bucket: int


class MStream(BaseModel):
    """Bounded multi-aspect stream anomaly detector.

    Numeric attributes use ``log10(1 + x)`` and online min-max normalization.
    Names in ``categorical_features`` must contain integer-like values. Scoring
    previews rollover, normalization, and insertion without mutating state.

    References:
        Bhatia, S., Jain, A., Li, P., Kumar, R., & Hooi, B. (2021). MStream:
        Fast Anomaly Detection in Multi-Aspect Streams.
        https://doi.org/10.1145/3442381.3450023
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
        self._boundary = NumericEventBoundary(
            time_key=self.time_key,
            integer_time=True,
        )
        self._state: _MStreamState | None = None
        self._samples_seen = 0

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of observed samples processed via learn_one."""
        return self._samples_seen

    def _schema_indices(
        self,
        prepared: PreparedFeatures,
    ) -> tuple[np.ndarray, np.ndarray]:
        categorical_names = set(self.categorical_features)
        missing = categorical_names.difference(prepared.names)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Missing configured categorical features: [{names}]")

        numeric_indices = np.asarray(
            [
                index
                for index, name in enumerate(prepared.names)
                if name not in categorical_names
            ],
            dtype=np.intp,
        )
        categorical_indices = np.asarray(
            [
                index
                for index, name in enumerate(prepared.names)
                if name in categorical_names
            ],
            dtype=np.intp,
        )
        return numeric_indices, categorical_indices

    @staticmethod
    def _split_values(
        values: np.ndarray,
        numeric_indices: np.ndarray,
        categorical_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        numeric = values[numeric_indices]
        if np.any(numeric <= -1.0):
            raise ValueError("Numeric MStream features must be greater than -1")

        categorical_values = values[categorical_indices]
        categorical = np.rint(categorical_values).astype(np.int64)
        if not np.allclose(categorical_values, categorical, rtol=0.0, atol=1e-9):
            raise ValueError("Configured categorical features must be integer-like")
        return numeric, categorical

    def _create_state(
        self,
        *,
        numeric_indices: np.ndarray,
        categorical_indices: np.ndarray,
        bucket: int,
    ) -> _MStreamState:
        n_numeric = len(numeric_indices)
        n_categorical = len(categorical_indices)

        numeric_current = np.zeros(
            (n_numeric, self.buckets),
            dtype=np.float64,
        )
        numeric = _NumericState(
            indices=numeric_indices,
            minimum=np.full(n_numeric, np.inf, dtype=np.float64),
            maximum=np.full(n_numeric, -np.inf, dtype=np.float64),
            counts=_CountState(
                current=numeric_current,
                total=np.zeros_like(numeric_current),
            ),
        )

        high = max(self.buckets, 2)
        categorical_current = np.zeros(
            (n_categorical, self.rows, self.buckets),
            dtype=np.float64,
        )
        categorical = _CategoricalState(
            indices=categorical_indices,
            hash_a=self._rng.integers(
                1,
                high,
                size=(n_categorical, self.rows),
                dtype=np.int64,
            ),
            hash_b=self._rng.integers(
                0,
                self.buckets,
                size=(n_categorical, self.rows),
                dtype=np.int64,
            ),
            counts=_CountState(
                current=categorical_current,
                total=np.zeros_like(categorical_current),
            ),
        )

        n_record_bits = math.ceil(math.log2(self.buckets))
        record_current = np.zeros((self.rows, self.buckets), dtype=np.float64)
        record = _RecordState(
            numeric_planes=self._rng.normal(
                size=(self.rows, n_record_bits, n_numeric)
            ),
            categorical_weights=self._rng.integers(
                0,
                self.buckets,
                size=(self.rows, n_categorical),
                dtype=np.int64,
            ),
            counts=_CountState(
                current=record_current,
                total=np.zeros_like(record_current),
            ),
        )
        return _MStreamState(
            numeric=numeric,
            categorical=categorical,
            record=record,
            current_bucket=bucket,
            first_bucket=bucket,
        )

    def _rollover_for_learning(self, state: _MStreamState, bucket: int) -> None:
        if bucket == state.current_bucket:
            return
        state.numeric.counts.current *= self.alpha
        state.categorical.counts.current *= self.alpha
        state.record.counts.current *= self.alpha
        state.current_bucket = bucket

    @staticmethod
    def _time_index(state: _MStreamState, bucket: int) -> int:
        return bucket - state.first_bucket + 1

    @staticmethod
    def _is_rollover(state: _MStreamState, bucket: int) -> bool:
        return bucket > state.current_bucket

    @staticmethod
    def _normalize_numeric(
        state: _NumericState,
        numeric: np.ndarray,
        *,
        update: bool,
    ) -> np.ndarray:
        transformed = np.log10(1.0 + numeric)
        minimum = np.minimum(state.minimum, transformed)
        maximum = np.maximum(state.maximum, transformed)
        ranges = maximum - minimum
        normalized = np.divide(
            transformed - minimum,
            ranges,
            out=np.zeros_like(transformed),
            where=ranges > 0.0,
        )
        if update:
            state.minimum[:] = minimum
            state.maximum[:] = maximum
        return np.asarray(normalized, dtype=np.float64)

    def _numeric_bins(self, normalized: np.ndarray) -> np.ndarray:
        return np.floor(normalized * float(self.buckets - 1)).astype(np.intp)

    def _categorical_bins(
        self,
        state: _CategoricalState,
        categorical: np.ndarray,
    ) -> np.ndarray:
        if categorical.size == 0:
            return np.empty((0, self.rows), dtype=np.intp)
        return np.asarray(
            (
                categorical[:, np.newaxis] * state.hash_a
                + state.hash_b
            )
            % self.buckets,
            dtype=np.intp,
        )

    def _record_bins(
        self,
        state: _RecordState,
        normalized: np.ndarray,
        categorical: np.ndarray,
    ) -> np.ndarray:
        if state.numeric_planes.shape[1] == 0:
            numeric_hash = np.zeros(self.rows, dtype=np.int64)
        else:
            signs = np.einsum("rbn,n->rb", state.numeric_planes, normalized) >= 0.0
            bit_weights = np.left_shift(
                np.int64(1),
                np.arange(signs.shape[1], dtype=np.int64),
            )
            numeric_hash = signs.astype(np.int64) @ bit_weights

        categorical_hash = (
            state.categorical_weights @ categorical
            if categorical.size
            else np.zeros(self.rows, dtype=np.int64)
        )
        return np.asarray(
            (numeric_hash + categorical_hash) % self.buckets,
            dtype=np.intp,
        )

    @staticmethod
    def _counts_to_anomaly(total: float, current: float, time_index: int) -> float:
        current_mean = total / float(time_index)
        squared_error = max(0.0, current - current_mean) ** 2
        return squared_error / current_mean + squared_error / (
            current_mean * float(max(1, time_index - 1))
        )

    def _candidate_counts(
        self,
        counts: _CountState,
        indices: np.ndarray,
        *,
        rollover: bool,
    ) -> tuple[float, float]:
        current_values = counts.current[self._row_index, indices]
        total_values = counts.total[self._row_index, indices]
        decay = self.alpha if rollover else 1.0
        return (
            float(np.min(current_values)) * decay + 1.0,
            float(np.min(total_values)) + 1.0,
        )

    def _score_prepared(
        self,
        state: _MStreamState,
        bucket: int,
        normalized: np.ndarray,
        categorical: np.ndarray,
    ) -> float:
        rollover = self._is_rollover(state, bucket)
        time_index = self._time_index(state, bucket)
        total_score = 0.0

        numeric_bins = self._numeric_bins(normalized)
        numeric_decay = self.alpha if rollover else 1.0
        for feature_index, bin_index in enumerate(numeric_bins):
            current = (
                state.numeric.counts.current[feature_index, bin_index]
                * numeric_decay
                + 1.0
            )
            total = state.numeric.counts.total[feature_index, bin_index] + 1.0
            total_score += self._counts_to_anomaly(total, current, time_index)

        categorical_bins = self._categorical_bins(state.categorical, categorical)
        for feature_index, indices in enumerate(categorical_bins):
            counts = _CountState(
                current=state.categorical.counts.current[feature_index],
                total=state.categorical.counts.total[feature_index],
            )
            current, total = self._candidate_counts(
                counts,
                indices,
                rollover=rollover,
            )
            total_score += self._counts_to_anomaly(total, current, time_index)

        record_bins = self._record_bins(state.record, normalized, categorical)
        current, total = self._candidate_counts(
            state.record.counts,
            record_bins,
            rollover=rollover,
        )
        total_score += self._counts_to_anomaly(total, current, time_index)
        return float(np.log1p(total_score))

    def learn_one(self, x: dict[str, float]) -> None:
        """Update model state with a single sample."""
        event = self._boundary.preview(x)
        bucket = int(event.timestamp.value)

        state = self._state
        if state is None:
            numeric_indices, categorical_indices = self._schema_indices(event.features)
        else:
            numeric_indices = state.numeric.indices
            categorical_indices = state.categorical.indices
        numeric, categorical = self._split_values(
            event.features.values,
            numeric_indices,
            categorical_indices,
        )

        if state is None:
            state = self._create_state(
                numeric_indices=numeric_indices,
                categorical_indices=categorical_indices,
                bucket=bucket,
            )
        else:
            self._rollover_for_learning(state, bucket)

        normalized = self._normalize_numeric(state.numeric, numeric, update=True)
        for feature_index, bin_index in enumerate(self._numeric_bins(normalized)):
            state.numeric.counts.current[feature_index, bin_index] += 1.0
            state.numeric.counts.total[feature_index, bin_index] += 1.0

        categorical_bins = self._categorical_bins(state.categorical, categorical)
        for feature_index, indices in enumerate(categorical_bins):
            state.categorical.counts.current[
                feature_index, self._row_index, indices
            ] += 1.0
            state.categorical.counts.total[
                feature_index, self._row_index, indices
            ] += 1.0

        record_bins = self._record_bins(state.record, normalized, categorical)
        state.record.counts.current[self._row_index, record_bins] += 1.0
        state.record.counts.total[self._row_index, record_bins] += 1.0

        self._state = state
        self._samples_seen += 1
        self._boundary.commit(event)

    def score_one(self, x: dict[str, float]) -> float:
        """Compute the candidate-inclusive anomaly score without mutation."""
        event = self._boundary.preview(x)
        state = self._state
        if state is None:
            numeric_indices, categorical_indices = self._schema_indices(event.features)
            self._split_values(
                event.features.values,
                numeric_indices,
                categorical_indices,
            )
            return 0.0

        bucket = int(event.timestamp.value)
        numeric, categorical = self._split_values(
            event.features.values,
            state.numeric.indices,
            state.categorical.indices,
        )
        if self._time_index(state, bucket) - 1 < self.warm_up_buckets:
            return 0.0

        normalized = self._normalize_numeric(state.numeric, numeric, update=False)
        return self._score_prepared(state, bucket, normalized, categorical)

    def __repr__(self) -> str:
        return (
            f"MStream(rows={self.rows}, buckets={self.buckets}, alpha={self.alpha}, "
            f"time_key={self.time_key!r}, "
            f"categorical_features={self.categorical_features!r}, "
            f"warm_up_buckets={self.warm_up_buckets}, seed={self.seed}, "
            f"samples_seen={self._samples_seen})"
        )
