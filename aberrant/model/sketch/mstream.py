"""MStream sketch-based detector for streaming anomaly detection."""

from __future__ import annotations

import hashlib
import itertools

import numpy as np

from aberrant.base.model import BaseModel


class MStream(BaseModel):
    """
    MStream-style sketch detector for online anomaly detection.

    The model keeps two fixed-size count-min sketch tensors:
    - current bucket counts
    - exponentially decayed historical counts

    For each sample and each configured view, the anomaly contribution is:
    ``((c - h) ** 2) / (h + eps)`` where ``c`` and ``h`` are count-min
    estimates from current and historical sketches. The final score is the
    mean of all view contributions.

    ``c`` is evaluated as the candidate-inclusive current estimate
    (``current + 1``) to match score-before-learn streaming usage.

    This implementation hashes singleton and optional pairwise feature views.
    The original MStream separately hashes every feature and the entire record,
    so scores are not expected to match the authors' implementation.

    Notes:
    - Scores are continuous and non-negative.
    - Scores are ``0.0`` until warm-up is complete.
    - Feature schema is fixed after the first sample (excluding ``time_key``).

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
        interaction_order: int = 2,
        max_interactions: int | None = 64,
        warm_up_buckets: int = 1,
        eps: float = 1e-9,
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
        if interaction_order not in (1, 2):
            raise ValueError("interaction_order must be 1 or 2")
        if max_interactions is not None and max_interactions < 0:
            raise ValueError("max_interactions must be non-negative or None")
        if warm_up_buckets <= 0:
            raise ValueError("warm_up_buckets must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.rows = rows
        self.buckets = buckets
        self.alpha = alpha
        self.time_key = time_key
        self.interaction_order = interaction_order
        self.max_interactions = max_interactions
        self.warm_up_buckets = warm_up_buckets
        self.eps = eps
        self.seed = seed

        self._reset_state()

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._row_keys = self._make_row_keys()
        self._row_index = np.arange(self.rows, dtype=np.intp)

        self._feature_order: tuple[str, ...] | None = None
        self._views: tuple[tuple[int, ...], ...] | None = None
        self._view_label_bytes: tuple[bytes, ...] | None = None

        self._current_sketch: np.ndarray | None = None
        self._historical_sketch: np.ndarray | None = None

        self._current_bucket: int | None = None
        self._seen_buckets: int = 0
        self._ready: bool = False

        self._arrival_index: int = 0
        self._samples_seen: int = 0

    def _make_row_keys(self) -> tuple[bytes, ...]:
        salts = self._rng.integers(
            low=0,
            high=np.iinfo(np.uint64).max,
            size=self.rows,
            dtype=np.uint64,
        )
        return tuple(
            int(salt).to_bytes(16, byteorder="little", signed=False) for salt in salts
        )

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    def _coerce_bucket(self, value: float) -> int:
        if not isinstance(value, int | float | np.number):
            raise ValueError("Timestamp value must be numeric")

        as_float = float(value)
        if not np.isfinite(as_float):
            raise ValueError("Timestamp value must be finite")

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
            bucket = self._coerce_bucket(x[self.time_key])
            features = {key: value for key, value in x.items() if key != self.time_key}

        if not features:
            raise ValueError("Input must contain at least one non-time feature")
        return bucket, features

    def _initialize_views(self) -> None:
        if self._feature_order is None:
            raise RuntimeError("Feature order is not initialized")

        n_features = len(self._feature_order)
        views: list[tuple[int, ...]] = [(index,) for index in range(n_features)]

        if self.interaction_order == 2 and n_features > 1:
            pair_iter = itertools.combinations(range(n_features), 2)
            if self.max_interactions is None:
                views.extend(pair_iter)
            else:
                views.extend(itertools.islice(pair_iter, self.max_interactions))

        self._views = tuple(views)
        self._view_label_bytes = tuple(
            "|".join(self._feature_order[index] for index in view).encode("utf-8")
            for view in self._views
        )

        n_views = len(self._views)
        self._current_sketch = np.zeros(
            (n_views, self.rows, self.buckets), dtype=np.float64
        )
        self._historical_sketch = np.zeros_like(self._current_sketch)

    def _set_or_validate_feature_order(self, features: dict[str, float]) -> None:
        if self._feature_order is None:
            self._feature_order = tuple(sorted(features.keys()))
            self._initialize_views()
            return

        received = set(features.keys())
        expected = set(self._feature_order)
        if received != expected:
            expected_keys = ", ".join(self._feature_order)
            received_keys = ", ".join(sorted(features.keys()))
            raise ValueError(
                "Inconsistent feature keys. "
                f"Expected [{expected_keys}], received [{received_keys}]."
            )

    def _vectorize(self, features: dict[str, float]) -> np.ndarray:
        if self._feature_order is None:
            raise RuntimeError("Feature order is not initialized")
        return np.fromiter(
            (float(features[key]) for key in self._feature_order),
            dtype=np.float64,
            count=len(self._feature_order),
        )

    def _prepare_sample(self, x: dict[str, float]) -> tuple[int, np.ndarray]:
        bucket, features = self._split_input(x)
        self._set_or_validate_feature_order(features)
        return bucket, self._vectorize(features)

    def _rollover_if_needed(self, bucket: int) -> None:
        if self._current_sketch is None or self._historical_sketch is None:
            raise RuntimeError("Sketch tensors are not initialized")

        if self._current_bucket is None:
            self._current_bucket = bucket
            return

        if bucket < self._current_bucket:
            raise ValueError(
                f"Non-monotonic timestamp: received {bucket}, current {self._current_bucket}"
            )

        if bucket == self._current_bucket:
            return

        delta = bucket - self._current_bucket

        self._historical_sketch *= self.alpha
        self._historical_sketch += self._current_sketch
        self._current_sketch.fill(0.0)

        if delta > 1:
            self._historical_sketch *= self.alpha ** (delta - 1)

        self._current_bucket = bucket
        self._seen_buckets += delta
        self._ready = self._seen_buckets >= self.warm_up_buckets

    def _view_bucket_indices(self, x_vector: np.ndarray) -> np.ndarray:
        if self._views is None or self._view_label_bytes is None:
            raise RuntimeError("Views are not initialized")

        bucket_indices = np.zeros((len(self._views), self.rows), dtype=np.intp)

        for view_index, view in enumerate(self._views):
            payload = bytearray(self._view_label_bytes[view_index])
            for feature_index in view:
                payload.extend(np.float64(x_vector[feature_index]).tobytes())
            payload_bytes = bytes(payload)

            for row_index, row_key in enumerate(self._row_keys):
                digest = hashlib.blake2b(
                    payload_bytes,
                    digest_size=8,
                    key=row_key,
                ).digest()
                bucket_indices[view_index, row_index] = (
                    int.from_bytes(digest, byteorder="little", signed=False)
                    % self.buckets
                )

        return bucket_indices

    def _score_indices(self, bucket_indices: np.ndarray) -> float:
        if self._current_sketch is None or self._historical_sketch is None:
            raise RuntimeError("Sketch tensors are not initialized")

        n_views = bucket_indices.shape[0]
        if n_views == 0:
            return 0.0

        view_scores = np.empty(n_views, dtype=np.float64)

        for view_index in range(n_views):
            view_bins = bucket_indices[view_index]
            current_counts = self._current_sketch[
                view_index, self._row_index, view_bins
            ]
            historical_counts = self._historical_sketch[
                view_index, self._row_index, view_bins
            ]

            # score_one is typically called before learn_one, therefore use the
            # candidate-inclusive current estimate to avoid rank inversion.
            c = float(np.min(current_counts)) + 1.0
            h = float(np.min(historical_counts))
            view_scores[view_index] = ((c - h) ** 2) / (h + self.eps)

        return float(np.mean(view_scores))

    def learn_one(self, x: dict[str, float]) -> None:
        """Update model state with a single sample."""
        bucket, x_vector = self._prepare_sample(x)
        self._rollover_if_needed(bucket)

        if self._current_sketch is None:
            raise RuntimeError("Sketch tensors are not initialized")

        bucket_indices = self._view_bucket_indices(x_vector)
        for view_index in range(bucket_indices.shape[0]):
            self._current_sketch[
                view_index, self._row_index, bucket_indices[view_index]
            ] += 1.0

        self._samples_seen += 1
        if self.time_key is None:
            self._arrival_index += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for a single sample."""
        bucket, x_vector = self._prepare_sample(x)
        self._rollover_if_needed(bucket)

        if not self._ready:
            return 0.0

        bucket_indices = self._view_bucket_indices(x_vector)
        score = self._score_indices(bucket_indices)
        return float(max(score, 0.0))

    def __repr__(self) -> str:
        n_views = 0 if self._views is None else len(self._views)
        return (
            f"MStream(rows={self.rows}, buckets={self.buckets}, alpha={self.alpha}, "
            f"time_key={self.time_key!r}, interaction_order={self.interaction_order}, "
            f"max_interactions={self.max_interactions}, "
            f"warm_up_buckets={self.warm_up_buckets}, eps={self.eps}, seed={self.seed}, "
            f"n_views={n_views}, ready={self._ready}, seen_buckets={self._seen_buckets})"
        )
