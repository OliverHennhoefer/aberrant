"""Bounded streaming RS-Hash adaptation for anomaly detection."""

from __future__ import annotations

import math

import numpy as np

from aberrant._utils.validation import NumericEventBoundary, coerce_integer_count
from aberrant.base.model import BaseModel

_HASH_MODULUS = np.int64(2_147_483_647)  # Large Mersenne prime.


def _time_difference_ratio(
    current: int | float, previous: int | float
) -> tuple[int, int]:
    """Subtract exact timestamp representations before any float conversion."""
    current_numerator, current_denominator = current.as_integer_ratio()
    previous_numerator, previous_denominator = previous.as_integer_ratio()
    numerator = (
        current_numerator * previous_denominator
        - previous_numerator * current_denominator
    )
    return numerator, current_denominator * previous_denominator


def _time_difference(current: int | float, previous: int | float) -> int | float:
    """Preserve mixed timestamp precision while retaining same-type arithmetic."""
    if isinstance(current, int) == isinstance(previous, int):
        return current - previous
    numerator, denominator = _time_difference_ratio(current, previous)
    return numerator if denominator == 1 else numerator / denominator


def _log_time_difference(current: int | float, previous: int | float) -> float:
    """Take the log of a positive elapsed time without float conversion."""
    numerator, denominator = _time_difference_ratio(current, previous)
    return math.log(numerator) - math.log(denominator)


class RSHash(BaseModel):
    """
    Bounded-memory streaming adaptation of RS-Hash.

    RS-Hash keeps an ensemble of randomized feature subspaces and hashes each
    sample into fixed-size count tables with exponential fading. Low hashed
    occupancy indicates potentially anomalous behavior. Online z-score
    normalization and exponential fading are streaming adaptations in this
    implementation and do not reproduce the paper's exact window procedure.

    Each subspace scores ``1 - log(1 + c) / log(1 + max(m, 1))``, where ``c`` is
    the faded count of the sample's least occupied bucket and ``m`` the faded
    number of learned samples, the most any bucket can hold. The score is the
    mean over subspaces.

    Notes:
    - Scores are continuous, non-negative, and bounded in ``[0, 1]``.
    - Memory usage is bounded by ``n_members * n_hashes * sketch_width``.
    - Feature schema is fixed after the first ``learn_one`` call.

    Args:
        n_members: Number of independently sampled feature subspaces.
        n_hashes: Number of independent hash tables per subspace.
        sketch_width: Number of counters in every hash table.
        subspace_size: Features sampled without replacement per component.
            ``None`` uses ``ceil(sqrt(n_features))``; an explicit value cannot
            exceed the established feature count.
        bin_width: Positive quantization width in online-standardized feature
            coordinates.
        decay: Non-negative exponential fading rate. Counts are scaled by
            ``exp(-decay * elapsed_time)``.
        warm_up: Number of learned samples required before scoring.
        time_key: Event-time field to exclude from the feature vector. ``None``
            uses one-based arrival order; explicit times must be finite and
            non-decreasing.
        seed: Seed for model-local subspace, shift, and hash generation.

    References:
        Sathe, S., & Aggarwal, C. C. (2016). Subspace Outlier Detection in
        Linear Time with Randomized Hashing. IEEE ICDM.
        https://www.charuaggarwal.net/linearout.pdf
    """

    def __init__(
        self,
        *,
        n_members: int = 24,
        n_hashes: int = 4,
        sketch_width: int = 256,
        subspace_size: int | None = None,
        bin_width: float = 1.0,
        decay: float = 0.01,
        warm_up: int = 64,
        time_key: str | None = None,
        seed: int | None = None,
    ) -> None:
        n_members = coerce_integer_count(n_members, label="n_members", minimum=1)
        n_hashes = coerce_integer_count(n_hashes, label="n_hashes", minimum=1)
        sketch_width = coerce_integer_count(
            sketch_width, label="sketch_width", minimum=1
        )
        warm_up = coerce_integer_count(warm_up, label="warm_up", minimum=1)
        if subspace_size is not None and subspace_size <= 0:
            raise ValueError("subspace_size must be positive or None")
        if bin_width <= 0.0:
            raise ValueError("bin_width must be positive")
        if decay < 0.0:
            raise ValueError("decay must be non-negative")
        if time_key is not None and (not isinstance(time_key, str) or not time_key):
            raise ValueError("time_key must be a non-empty string or None")

        self.n_members = n_members
        self.n_hashes = n_hashes
        self.sketch_width = sketch_width
        self.subspace_size = subspace_size
        self.bin_width = bin_width
        self.decay = decay
        self.warm_up = warm_up
        self.time_key = time_key
        self.seed = seed

        self._scale_renorm_threshold = 1e-12
        self._std_floor = 1e-6
        # With an array of buckets, these select one counter in every table.
        self._components = np.arange(n_members)[:, None]
        self._tables = np.arange(n_hashes)

        self._reset_state()

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self.seed)

        self._boundary = NumericEventBoundary(time_key=self.time_key)
        self._subspace_indices: np.ndarray | None = None
        self._subspace_shifts: np.ndarray | None = None
        self._hash_a: np.ndarray | None = None
        self._hash_b: np.ndarray | None = None
        self._counts: np.ndarray | None = None

        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._n_stats: int = 0

        self._samples_seen: int = 0
        # The faded number of learned samples, in the units of ``_counts``.
        self._mass: float = 0.0
        self._scale: float = 1.0
        # The value bytes and buckets of the last scored point. Only learning
        # changes the buckets of a point, so learning that point reuses them.
        self._scored: tuple[bytes, np.ndarray] | None = None

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        # The buckets of the last scored point are a cache, not model state.
        state["_scored"] = None
        return state

    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed with learn_one."""
        return self._samples_seen

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
            (self.n_members, resolved_subspace_size),
            dtype=np.int32,
        )
        for index in range(self.n_members):
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
            size=(self.n_members, resolved_subspace_size),
        )
        self._hash_a = self._rng.integers(
            low=1,
            high=_HASH_MODULUS,
            size=(self.n_members, self.n_hashes, resolved_subspace_size),
            dtype=np.int64,
        )
        self._hash_b = self._rng.integers(
            low=0,
            high=_HASH_MODULUS,
            size=(self.n_members, self.n_hashes),
            dtype=np.int64,
        )
        self._counts = np.zeros(
            (self.n_members, self.n_hashes, self.sketch_width),
            dtype=np.float64,
        )

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        if self._mean is None:
            return np.asarray(vector.copy(), dtype=np.float64)
        if self._n_stats < 2 or self._m2 is None:
            return np.asarray(vector - self._mean, dtype=np.float64)

        variance = self._m2 / float(self._n_stats - 1)
        std = np.sqrt(np.clip(variance, self._std_floor**2, None))
        return np.asarray((vector - self._mean) / std, dtype=np.float64)

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

    def _preview_scale(self, current_time: int | float) -> float:
        """Preview fading relative to the boundary's last committed timestamp."""
        if self._samples_seen == 0 or self.decay == 0.0:
            return self._scale
        previous_time = self._boundary.clock.max_time
        try:
            delta = _time_difference(current_time, previous_time)
        except OverflowError:
            log_delta = _log_time_difference(current_time, previous_time)
        else:
            if delta <= 745.0:
                return self._scale * float(np.exp(-self.decay * delta))
            log_delta = (
                _log_time_difference(current_time, previous_time)
                if isinstance(delta, float) and math.isinf(delta)
                else math.log(delta)
            )
        # Comparing logs handles huge elapsed times and tiny fading rates
        # without converting an unrepresentable difference or product first.
        log_elapsed_decay = log_delta + math.log(self.decay)
        if log_elapsed_decay > math.log(745.0):
            return 0.0
        return self._scale * math.exp(-math.exp(log_elapsed_decay))

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
        residues = np.remainder(quantized.astype(np.int64), _HASH_MODULUS)
        # Every coefficient and residue lies below the modulus 2**31 - 1, so
        # each product, and the sum of the reduced products, fits in int64.
        # Reducing each product first leaves (b + sum(a * q)) mod p unchanged.
        terms = np.remainder(self._hash_a * residues[:, None, :], _HASH_MODULUS)
        hashes = np.remainder(self._hash_b + terms.sum(axis=2), _HASH_MODULUS)
        buckets: np.ndarray = hashes % self.sketch_width
        return buckets

    def _score_buckets(self, buckets: np.ndarray, query_scale: float) -> float:
        if self._counts is None:
            return 0.0

        # No bucket can hold more than the faded mass of all learned samples,
        # so the score keeps its scale on an endless fading stream.
        mass = max(self._mass * query_scale, 1.0)
        normalizer = math.log(1.0 + mass)

        occupancy = self._counts[self._components, self._tables, buckets] * query_scale
        occupancy_min = np.maximum(occupancy.min(axis=1), 0.0)
        # A bucket holding the whole mass scores exactly 0. Its count and the
        # mass take the same float steps, but NumPy's log1p and math.log can
        # round one ulp apart, so the ratio of the two logs is not reliably 1.
        component_scores = np.where(
            occupancy_min >= mass,
            0.0,
            np.clip(1.0 - np.log1p(occupancy_min) / normalizer, 0.0, 1.0),
        )
        return float(np.mean(component_scores))

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one sample."""
        event = self._boundary.preview(x)
        current_time = event.timestamp.value
        vector = event.features.values
        scored, self._scored = self._scored, None

        self._initialize_hash_state(n_features=vector.shape[0])
        if scored is not None and scored[0] == vector.tobytes():
            buckets = scored[1]
        else:
            buckets = self._bucket_indices(self._normalize(vector))
        scale = self._preview_scale(current_time)

        if self._counts is None:
            raise RuntimeError("Count sketch is not initialized")

        # Materialize small scales before insertion. Underflow means complete
        # forgetting, so zero the old counts and start again at unit scale.
        if scale < self._scale_renorm_threshold:
            self._counts *= scale
            self._mass *= scale
            scale = 1.0
        self._scale = scale

        increment = 1.0 / self._scale
        self._counts[self._components, self._tables, buckets] += increment
        self._mass += increment

        self._update_running_stats(vector)

        self._samples_seen += 1
        self._boundary.commit(event)

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for one sample."""
        event = self._boundary.preview(x)
        if not self._boundary.schema.is_established or self._counts is None:
            return 0.0
        if self._samples_seen < self.warm_up:
            return 0.0

        current_time = event.timestamp.value
        vector = event.features.values
        buckets = self._bucket_indices(self._normalize(vector))
        self._scored = (vector.tobytes(), buckets)
        query_scale = self._preview_scale(current_time)
        return float(max(0.0, self._score_buckets(buckets, query_scale)))

    def __repr__(self) -> str:
        return (
            "RSHash("
            f"n_members={self.n_members}, n_hashes={self.n_hashes}, "
            f"sketch_width={self.sketch_width}, subspace_size={self.subspace_size}, "
            f"bin_width={self.bin_width}, decay={self.decay}, "
            f"warm_up={self.warm_up}, time_key={self.time_key!r}, "
            f"seed={self.seed}, samples_seen={self._samples_seen})"
        )
