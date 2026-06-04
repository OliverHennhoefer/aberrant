"""Bounded streaming RS-Hash adaptation for anomaly detection."""

from __future__ import annotations

import math

import numpy as np

from aberrant.base.model import BaseModel

_HASH_MODULUS = np.int64(2_147_483_647)  # Large Mersenne prime.


class StreamingRSHash(BaseModel):
    """
    Bounded-memory streaming adaptation of RS-Hash.

    RS-Hash keeps an ensemble of randomized feature subspaces and hashes each
    sample into fixed-size count tables with exponential fading. Low hashed
    occupancy indicates potentially anomalous behavior. Online z-score
    normalization and exponential fading are streaming adaptations in this
    implementation and do not reproduce the paper's exact window procedure.

    Notes:
    - Scores are continuous, non-negative, and bounded in ``[0, 1]``.
    - Memory usage is bounded by ``components_num * hash_num * bins``.
    - Feature schema is fixed after the first ``learn_one`` call.

    References:
        Sathe, S., & Aggarwal, C. C. (2016). Subspace Outlier Detection in
        Linear Time with Randomized Hashing. IEEE ICDM.
        https://www.charuaggarwal.net/linearout.pdf
    """

    def __init__(
        self,
        components_num: int = 24,
        hash_num: int = 4,
        bins: int = 256,
        subspace_size: int | None = None,
        bin_width: float = 1.0,
        decay: float = 0.01,
        warm_up_samples: int = 64,
        time_key: str | None = None,
        seed: int | None = None,
    ) -> None:
        if components_num <= 0:
            raise ValueError("components_num must be positive")
        if hash_num <= 0:
            raise ValueError("hash_num must be positive")
        if bins <= 0:
            raise ValueError("bins must be positive")
        if subspace_size is not None and subspace_size <= 0:
            raise ValueError("subspace_size must be positive or None")
        if bin_width <= 0.0:
            raise ValueError("bin_width must be positive")
        if decay < 0.0:
            raise ValueError("decay must be non-negative")
        if warm_up_samples <= 0:
            raise ValueError("warm_up_samples must be positive")
        if time_key is not None and (not isinstance(time_key, str) or not time_key):
            raise ValueError("time_key must be a non-empty string or None")

        self.components_num = components_num
        self.hash_num = hash_num
        self.bins = bins
        self.subspace_size = subspace_size
        self.bin_width = bin_width
        self.decay = decay
        self.warm_up_samples = warm_up_samples
        self.time_key = time_key
        self.seed = seed

        self._scale_renorm_threshold = 1e-12
        self._std_floor = 1e-6

        self._reset_state()

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self.seed)

        self._feature_order: tuple[str, ...] | None = None
        self._subspace_indices: np.ndarray | None = None
        self._subspace_shifts: np.ndarray | None = None
        self._hash_a: np.ndarray | None = None
        self._hash_b: np.ndarray | None = None
        self._counts: np.ndarray | None = None

        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._n_stats: int = 0

        self._samples_seen: int = 0
        self._arrival_index: int = 0
        self._scale: float = 1.0
        self._last_learned_time: float | None = None
        self._max_learned_time: float = float("-inf")

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed with learn_one."""
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
            features = {k: v for k, v in x.items() if k != self.time_key}

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

    def _initialize_hash_state(self, n_features: int) -> None:
        if self._counts is not None:
            return

        if self.subspace_size is None:
            resolved_subspace_size = max(1, int(math.ceil(math.sqrt(n_features))))
        else:
            resolved_subspace_size = self.subspace_size
        if resolved_subspace_size > n_features:
            raise ValueError(
                "subspace_size cannot exceed the number of input features "
                f"({n_features})"
            )

        subspace_indices = np.empty(
            (self.components_num, resolved_subspace_size),
            dtype=np.int32,
        )
        for index in range(self.components_num):
            chosen = self._rng.choice(
                n_features,
                size=resolved_subspace_size,
                replace=False,
            )
            subspace_indices[index] = chosen.astype(np.int32)

        self._subspace_indices = subspace_indices
        self._subspace_shifts = self._rng.uniform(
            low=0.0,
            high=self.bin_width,
            size=(self.components_num, resolved_subspace_size),
        )
        self._hash_a = self._rng.integers(
            low=1,
            high=_HASH_MODULUS,
            size=(self.components_num, self.hash_num, resolved_subspace_size),
            dtype=np.int64,
        )
        self._hash_b = self._rng.integers(
            low=0,
            high=_HASH_MODULUS,
            size=(self.components_num, self.hash_num),
            dtype=np.int64,
        )
        self._counts = np.zeros(
            (self.components_num, self.hash_num, self.bins),
            dtype=np.float64,
        )

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        if self._mean is None:
            return vector.copy()
        if self._n_stats < 2 or self._m2 is None:
            return vector - self._mean

        variance = self._m2 / float(self._n_stats - 1)
        std = np.sqrt(np.clip(variance, self._std_floor**2, None))
        return (vector - self._mean) / std

    def _update_running_stats(self, vector: np.ndarray) -> None:
        if self._mean is None or self._m2 is None:
            self._mean = vector.copy()
            self._m2 = np.zeros_like(vector)
            self._n_stats = 1
            return

        self._n_stats += 1
        delta = vector - self._mean
        self._mean += delta / float(self._n_stats)
        delta2 = vector - self._mean
        self._m2 += delta * delta2

    def _renormalize_counts(self) -> None:
        if self._counts is None:
            return
        if self._scale <= 0.0:
            raise RuntimeError("Internal scale must be positive")

        self._counts *= self._scale
        self._scale = 1.0

    def _apply_decay(self, current_time: float) -> None:
        if self._last_learned_time is None:
            self._last_learned_time = current_time
            return

        delta = current_time - self._last_learned_time
        if delta < 0.0:
            raise ValueError(
                f"Non-monotonic timestamp: received {current_time}, "
                f"current {self._last_learned_time}"
            )

        if delta > 0.0 and self.decay > 0.0:
            self._scale *= float(np.exp(-self.decay * delta))
            if self._scale < self._scale_renorm_threshold:
                self._renormalize_counts()

        self._last_learned_time = current_time

    def _query_scale(self, current_time: float) -> float:
        if self._last_learned_time is None:
            return self._scale

        delta = current_time - self._last_learned_time
        if delta < 0.0:
            raise ValueError(
                f"Non-monotonic timestamp: received {current_time}, "
                f"current {self._last_learned_time}"
            )
        if delta == 0.0 or self.decay == 0.0:
            return self._scale
        return self._scale * float(np.exp(-self.decay * delta))

    def _bucket_indices(self, normalized: np.ndarray) -> np.ndarray:
        if (
            self._subspace_indices is None
            or self._subspace_shifts is None
            or self._hash_a is None
            or self._hash_b is None
        ):
            raise RuntimeError("Hash state is not initialized")

        sub_values = normalized[self._subspace_indices]
        quantized = np.floor((sub_values + self._subspace_shifts) / self.bin_width)
        quantized_i64 = quantized.astype(np.int64)

        buckets = np.zeros((self.components_num, self.hash_num), dtype=np.intp)

        for component in range(self.components_num):
            q = quantized_i64[component]
            for hash_index in range(self.hash_num):
                acc = int(self._hash_b[component, hash_index])
                for dimension_index in range(q.shape[0]):
                    coefficient = int(
                        self._hash_a[component, hash_index, dimension_index]
                    )
                    q_value = int(q[dimension_index])
                    acc = (acc + coefficient * (q_value % int(_HASH_MODULUS))) % int(
                        _HASH_MODULUS
                    )
                buckets[component, hash_index] = acc % self.bins

        return buckets

    def _score_buckets(self, buckets: np.ndarray, query_scale: float) -> float:
        if self._counts is None:
            return 0.0

        normalizer = float(np.log1p(max(self._samples_seen, 1)))
        if normalizer <= 0.0:
            return 0.0

        component_scores = np.empty(self.components_num, dtype=np.float64)
        for component in range(self.components_num):
            occupancy_min = float("inf")
            for hash_index in range(self.hash_num):
                bucket = int(buckets[component, hash_index])
                occupancy = self._counts[component, hash_index, bucket] * query_scale
                occupancy_min = min(occupancy_min, float(occupancy))

            score_component = 1.0 - (np.log1p(max(occupancy_min, 0.0)) / normalizer)
            component_scores[component] = np.clip(score_component, 0.0, 1.0)

        return float(np.mean(component_scores))

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one sample."""
        current_time, vector = self._prepare_input(x, mutate_schema=True)
        if vector is None:
            return

        self._initialize_hash_state(n_features=vector.shape[0])
        self._apply_decay(current_time)

        normalized = self._normalize(vector)
        buckets = self._bucket_indices(normalized)

        if self._counts is None:
            raise RuntimeError("Count sketch is not initialized")
        if self._scale <= 0.0:
            raise RuntimeError("Internal scale must be positive")

        increment = 1.0 / self._scale
        for component in range(self.components_num):
            for hash_index in range(self.hash_num):
                bucket = int(buckets[component, hash_index])
                self._counts[component, hash_index, bucket] += increment

        self._update_running_stats(vector)

        self._samples_seen += 1
        self._max_learned_time = current_time
        if self.time_key is None:
            self._arrival_index += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for one sample."""
        current_time, vector = self._prepare_input(x, mutate_schema=False)
        if vector is None or self._counts is None:
            return 0.0
        if self._samples_seen < self.warm_up_samples:
            return 0.0

        normalized = self._normalize(vector)
        buckets = self._bucket_indices(normalized)
        query_scale = self._query_scale(current_time)
        return float(max(0.0, self._score_buckets(buckets, query_scale)))

    def __repr__(self) -> str:
        return (
            "StreamingRSHash("
            f"components_num={self.components_num}, hash_num={self.hash_num}, "
            f"bins={self.bins}, subspace_size={self.subspace_size}, "
            f"bin_width={self.bin_width}, decay={self.decay}, "
            f"warm_up_samples={self.warm_up_samples}, time_key={self.time_key!r}, "
            f"seed={self.seed}, samples_seen={self._samples_seen})"
        )


# Backward-compatible alias for the historical paper-derived public name.
RSHash = StreamingRSHash
