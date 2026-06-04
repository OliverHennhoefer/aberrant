"""xStream anomaly detection for feature-evolving data streams."""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from collections.abc import Iterator

import numpy as np

from aberrant.base.model import BaseModel


class XStream(BaseModel):
    """
    xStream detector using StreamHash and half-space chains.

    xStream is a bounded-memory, streaming anomaly detector designed for
    evolving-feature data streams. It hashes sparse feature dictionaries into a
    fixed-dimensional projected space and maintains count-min sketches over
    recursively partitioned bins.

    The detector is sample-wise and stateful:
    - ``learn_one`` updates online sketches
    - ``score_one`` returns an anomaly score in ``[0, 1]`` (higher is more
      anomalous)

    Warm-up behavior:
    - Scores are ``0.0`` until the model is initialized from
      ``init_sample_size`` projected points and at least one reference window
      is available.

    Args:
        k: StreamHash projection dimensionality.
        n_chains: Number of half-space chains in the ensemble.
        depth: Number of levels per chain.
        cms_width: Width of each count-min sketch row.
        cms_num_hashes: Number of hash rows per count-min sketch.
        window_size: Number of samples per reference window.
        init_sample_size: Number of projected samples used to initialize range
            statistics and chain parameters.
        density: Fraction of non-zero StreamHash dimensions per feature.
        max_feature_cache_size: Maximum number of feature hash mappings to keep
            in memory. ``None`` disables eviction.
        seed: Random seed for reproducibility.

    References:
        Manzoor, E., Lamba, H., & Akoglu, L. (2018). xStream: Outlier
        Detection in Feature-Evolving Data Streams. KDD '18.
        https://doi.org/10.1145/3219819.3220107
        Original implementation: https://github.com/cmuxstream/cmuxstream-core
    """

    def __init__(
        self,
        k: int = 100,
        n_chains: int = 100,
        depth: int = 15,
        cms_width: int = 1024,
        cms_num_hashes: int = 4,
        window_size: int = 256,
        init_sample_size: int = 256,
        density: float = 1.0 / 3.0,
        max_feature_cache_size: int | None = 10_000,
        seed: int | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if n_chains <= 0:
            raise ValueError("n_chains must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if cms_width <= 0:
            raise ValueError("cms_width must be positive")
        if cms_num_hashes <= 0:
            raise ValueError("cms_num_hashes must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if init_sample_size <= 0:
            raise ValueError("init_sample_size must be positive")
        if not (0.0 < density <= 1.0):
            raise ValueError("density must be in (0, 1]")
        if max_feature_cache_size is not None and max_feature_cache_size <= 0:
            raise ValueError("max_feature_cache_size must be positive or None")

        self.k = k
        self.n_chains = n_chains
        self.depth = depth
        self.cms_width = cms_width
        self.cms_num_hashes = cms_num_hashes
        self.window_size = window_size
        self.init_sample_size = init_sample_size
        self.density = density
        self.max_feature_cache_size = max_feature_cache_size
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        """Initialize or reset internal state."""
        self._ready: bool = False
        self._reference_ready: bool = False
        self._samples_seen: int = 0
        self._samples_in_window: int = 0
        self._init_buffer: deque[np.ndarray] = deque(maxlen=self.init_sample_size)

        self._deltamax: np.ndarray | None = None
        self._chain_dims: np.ndarray | None = None
        self._shift: np.ndarray | None = None
        self._cms_current: np.ndarray | None = None
        self._cms_reference: np.ndarray | None = None

        self._hash_coeffs: np.ndarray | None = None
        self._hash_offsets: np.ndarray | None = None
        self._hash_coeffs_mod: np.ndarray | None = None
        self._hash_offsets_mod: np.ndarray | None = None
        self._scratch_z: np.ndarray | None = None
        self._scratch_bins: np.ndarray | None = None
        self._scratch_feature_visits: np.ndarray | None = None
        self._scratch_hash_dots: np.ndarray | None = None

        # Cache sparse StreamHash mappings for seen feature names:
        # feature -> (indices, signs)
        self._feature_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self.rng = np.random.default_rng(self.seed)
        self._reset_state()

    def _validate_input(self, x: dict[str, float]) -> None:
        """Validate one-sample input dictionary."""
        if not x:
            raise ValueError("Input dictionary cannot be empty")

        for key, value in x.items():
            if not isinstance(key, str):
                raise ValueError("All feature keys must be strings")
            if not isinstance(value, int | float | np.number):
                raise ValueError(f"Feature '{key}' is not numeric")
            if not np.isfinite(float(value)):
                raise ValueError(f"Feature '{key}' must be finite")

    def _feature_seed(self, feature: str) -> int:
        """Create deterministic seed for one feature name."""
        seed_prefix = "none" if self.seed is None else str(self.seed)
        payload = f"{seed_prefix}|{feature}".encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _feature_projection(self, feature: str) -> tuple[np.ndarray, np.ndarray]:
        """Get cached sparse projection indices and signs for one feature."""
        cached = self._feature_cache.get(feature)
        if cached is not None:
            self._feature_cache.move_to_end(feature)
            return cached

        nnz = max(1, int(round(self.k * self.density)))
        local_rng = np.random.default_rng(self._feature_seed(feature))
        indices = local_rng.choice(self.k, size=nnz, replace=False).astype(np.int32)
        signs = local_rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64), size=nnz, replace=True
        )

        mapping = (indices, signs)
        self._feature_cache[feature] = mapping
        if (
            self.max_feature_cache_size is not None
            and len(self._feature_cache) > self.max_feature_cache_size
        ):
            self._feature_cache.popitem(last=False)
        return mapping

    def _project_one(self, x: dict[str, float]) -> np.ndarray:
        """Project sparse dict sample into fixed-dimensional StreamHash vector."""
        y = np.zeros(self.k, dtype=np.float64)

        for feature, value in x.items():
            indices, signs = self._feature_projection(feature)
            y[indices] += float(value) * signs

        return y

    def _initialize_model(self) -> None:
        """Initialize chain/sketch state from buffered projected samples."""
        if len(self._init_buffer) < self.init_sample_size:
            return

        projected = np.vstack(self._init_buffer)

        deltamax = np.ptp(projected, axis=0) / 2.0
        deltamax[deltamax <= 0.0] = 1.0
        self._deltamax = deltamax

        self._chain_dims = self.rng.integers(
            0, self.k, size=(self.n_chains, self.depth), dtype=np.int32
        )
        self._shift = (
            self.rng.uniform(low=0.0, high=1.0, size=(self.n_chains, self.k))
            * self._deltamax
        )

        self._cms_current = np.zeros(
            (self.n_chains, self.depth, self.cms_num_hashes, self.cms_width),
            dtype=np.int32,
        )
        self._cms_reference = np.zeros_like(self._cms_current)

        self._hash_coeffs = self.rng.integers(
            1,
            np.iinfo(np.int32).max,
            size=(self.cms_num_hashes, self.k),
            dtype=np.int64,
        )
        self._hash_offsets = self.rng.integers(
            0,
            np.iinfo(np.int32).max,
            size=(self.n_chains, self.depth, self.cms_num_hashes),
            dtype=np.int64,
        )
        self._hash_coeffs_mod = self._hash_coeffs % self.cms_width
        self._hash_offsets_mod = self._hash_offsets % self.cms_width
        self._scratch_z = np.zeros(self.k, dtype=np.float64)
        # Use Python-int bins so large finite features cannot overflow int64.
        self._scratch_bins = np.zeros(self.k, dtype=object)
        self._scratch_feature_visits = np.zeros(self.k, dtype=np.int32)
        self._scratch_hash_dots = np.zeros(self.cms_num_hashes, dtype=np.int64)

        self._ready = True
        self._samples_in_window = 0

        for point in projected:
            self._learn_projected(point)

        self._init_buffer.clear()

    def _update_sketch(self, y: np.ndarray, sketch: np.ndarray) -> None:
        """Update one sketch tensor for a projected sample."""
        if (
            self._deltamax is None
            or self._chain_dims is None
            or self._shift is None
            or self._hash_coeffs_mod is None
            or self._hash_offsets_mod is None
            or self._scratch_z is None
            or self._scratch_bins is None
            or self._scratch_feature_visits is None
            or self._scratch_hash_dots is None
        ):
            raise RuntimeError("Model is not initialized")

        hash_index = np.arange(self.cms_num_hashes, dtype=np.intp)

        for chain in range(self.n_chains):
            for level, buckets in self._iter_chain_buckets(
                y,
                chain,
                self._scratch_z,
                self._scratch_bins,
                self._scratch_feature_visits,
                self._scratch_hash_dots,
            ):
                sketch[chain, level, hash_index, buckets] += 1

    def _iter_chain_buckets(
        self,
        y: np.ndarray,
        chain: int,
        z: np.ndarray,
        bins: np.ndarray,
        feature_visits: np.ndarray,
        hash_dots: np.ndarray,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """
        Yield traversal buckets for each level of one chain.

        This helper centralizes half-space chain traversal shared by sketch
        update and scoring paths.
        """
        if (
            self._deltamax is None
            or self._chain_dims is None
            or self._shift is None
            or self._hash_coeffs_mod is None
            or self._hash_offsets_mod is None
        ):
            raise RuntimeError("Model is not initialized")

        z.fill(0.0)
        bins.fill(0)
        feature_visits.fill(0)
        hash_dots.fill(0)
        hash_coeffs_mod = self._hash_coeffs_mod
        hash_offsets_mod = self._hash_offsets_mod[chain, :, :]

        for level in range(self.depth):
            feature = int(self._chain_dims[chain, level])
            feature_visits[feature] += 1

            if feature_visits[feature] == 1:
                z_new = (y[feature] + self._shift[chain, feature]) / self._deltamax[
                    feature
                ]
            else:
                z_new = (
                    2.0 * z[feature]
                    - self._shift[chain, feature] / self._deltamax[feature]
                )

            z[feature] = z_new
            bin_new = int(np.floor(z_new))

            if bin_new != bins[feature]:
                delta = bin_new - bins[feature]
                bins[feature] = bin_new
                delta_mod = int(delta % self.cms_width)
                if delta_mod:
                    hash_dots = (
                        hash_dots + delta_mod * hash_coeffs_mod[:, feature]
                    ) % self.cms_width

            buckets = (hash_dots + hash_offsets_mod[level, :]) % self.cms_width
            yield level, buckets.astype(np.intp)

    def _learn_projected(self, y: np.ndarray) -> None:
        """Learn one projected sample and maintain current/reference windows."""
        if self._cms_current is None:
            raise RuntimeError("Model is not initialized")

        self._update_sketch(y, self._cms_current)
        self._samples_seen += 1
        self._samples_in_window += 1

        if self._samples_in_window >= self.window_size:
            if self._cms_reference is None:
                raise RuntimeError("Reference sketch is not initialized")
            self._cms_reference[...] = self._cms_current
            self._cms_current.fill(0)
            self._samples_in_window = 0
            self._reference_ready = True

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update model state with one sample.

        Args:
            x: Feature dictionary with string keys and numeric values.
        """
        self._validate_input(x)
        y = self._project_one(x)

        if not self._ready:
            self._init_buffer.append(y)
            self._initialize_model()
            return

        self._learn_projected(y)

    def _score_projected(self, y: np.ndarray) -> float:
        """Compute anomaly score for one projected sample."""
        if (
            self._deltamax is None
            or self._chain_dims is None
            or self._shift is None
            or self._cms_reference is None
            or self._hash_coeffs_mod is None
            or self._hash_offsets_mod is None
            or self._scratch_z is None
            or self._scratch_bins is None
            or self._scratch_feature_visits is None
            or self._scratch_hash_dots is None
        ):
            return 0.0

        hash_index = np.arange(self.cms_num_hashes, dtype=np.intp)
        chain_scores = np.empty(self.n_chains, dtype=np.float64)

        for chain in range(self.n_chains):
            best_level_score = np.inf

            for level, buckets in self._iter_chain_buckets(
                y,
                chain,
                self._scratch_z,
                self._scratch_bins,
                self._scratch_feature_visits,
                self._scratch_hash_dots,
            ):
                counts = self._cms_reference[chain, level, hash_index, buckets]
                estimated_count = int(np.min(counts))

                level_score = np.log2(1.0 + estimated_count) + (level + 1.0)
                best_level_score = min(best_level_score, level_score)

            chain_scores[chain] = 2.0 ** (1.0 - best_level_score)

        score = float(np.mean(chain_scores))
        return float(np.clip(score, 0.0, 1.0))

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score for one sample.

        Returns:
            Score in ``[0, 1]`` where larger values indicate greater anomaly.
        """
        self._validate_input(x)

        if not self._ready or not self._reference_ready:
            return 0.0

        y = self._project_one(x)
        return self._score_projected(y)

    def __repr__(self) -> str:
        return (
            f"XStream(k={self.k}, n_chains={self.n_chains}, depth={self.depth}, "
            f"cms_width={self.cms_width}, cms_num_hashes={self.cms_num_hashes}, "
            f"window_size={self.window_size}, init_sample_size={self.init_sample_size}, "
            f"density={self.density}, max_feature_cache_size={self.max_feature_cache_size}, "
            f"feature_cache_size={len(self._feature_cache)}, ready={self._ready}, "
            f"reference_ready={self._reference_ready})"
        )
