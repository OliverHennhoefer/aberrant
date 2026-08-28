"""xStream anomaly detection for feature-evolving data streams."""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import coerce_finite_number


@dataclass(slots=True)
class _ScratchState:
    """Reusable traversal buffers."""

    z: np.ndarray
    bins: np.ndarray
    feature_visits: np.ndarray
    hash_dots: np.ndarray


@dataclass(slots=True)
class _XStreamState:
    """Complete initialized chain and sketch state."""

    deltamax: np.ndarray
    chain_dims: np.ndarray
    shift: np.ndarray
    cms_current: np.ndarray
    cms_reference: np.ndarray
    hash_coeffs_mod: np.ndarray
    hash_offsets_mod: np.ndarray
    scratch: _ScratchState
    samples_seen: int = 0
    samples_in_window: int = 0
    completed_windows: int = 0

    @property
    def reference_ready(self) -> bool:
        """Return whether a complete reference window exists."""
        return self.completed_windows > 0


class XStream(BaseModel):
    """StreamHash and half-space-chain detector for evolving feature streams.

    Scores remain zero until ``init_sample_size`` projected points initialize
    the chains and one complete reference window has been observed.

    Args:
        k: Dimension of the StreamHash feature projection.
        n_chains: Number of independently sampled half-space chains.
        depth: Number of levels and count sketches in each chain.
        cms_width: Number of counters in every count-min sketch row.
        cms_num_hashes: Number of independently hashed rows per count-min
            sketch.
        window_size: Number of learned projected points per current/reference
            window swap.
        init_sample_size: Number of projected points used to establish chain
            scales before window counting starts.
        density: Fraction of projected coordinates updated by each input
            feature's deterministic signed projection, in ``(0, 1]``.
        max_feature_cache_size: Maximum cached feature-name projections, with
            least-recently-used eviction. ``None`` disables this bound.
        seed: Seed for chain, shift, and sketch-hash generation. Feature-name
            projections are also derived deterministically from this seed.

    References:
        Manzoor, E., Lamba, H., & Akoglu, L. (2018). xStream: Outlier
        Detection in Feature-Evolving Data Streams. KDD '18.
        https://doi.org/10.1145/3219819.3220107
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

        self._reset_state()

    def _reset_state(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._state: _XStreamState | None = None
        self._init_buffer: deque[np.ndarray] = deque(maxlen=self.init_sample_size)
        self._feature_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @staticmethod
    def _validate_input(x: dict[str, float]) -> None:
        if not x:
            raise ValueError("Input dictionary cannot be empty")
        for key, value in x.items():
            if not isinstance(key, str):
                raise ValueError("All feature keys must be strings")
            coerce_finite_number(value, label=f"Feature '{key}'")

    def _feature_seed(self, feature: str) -> int:
        seed_prefix = "none" if self.seed is None else str(self.seed)
        payload = f"{seed_prefix}|{feature}".encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _feature_projection(self, feature: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._feature_cache.get(feature)
        if cached is not None:
            self._feature_cache.move_to_end(feature)
            return cached

        nnz = max(1, int(round(self.k * self.density)))
        local_rng = np.random.default_rng(self._feature_seed(feature))
        indices = local_rng.choice(self.k, size=nnz, replace=False).astype(np.int32)
        signs = local_rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=nnz,
            replace=True,
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
        y = np.zeros(self.k, dtype=np.float64)
        for feature, value in x.items():
            indices, signs = self._feature_projection(feature)
            y[indices] += float(value) * signs
        return y

    def _create_state(self) -> _XStreamState:
        projected = np.vstack(self._init_buffer)
        deltamax = np.ptp(projected, axis=0) / 2.0
        deltamax[deltamax <= 0.0] = 1.0

        chain_dims = self.rng.integers(
            0,
            self.k,
            size=(self.n_chains, self.depth),
            dtype=np.int32,
        )
        shift = (
            self.rng.uniform(low=0.0, high=1.0, size=(self.n_chains, self.k))
            * deltamax
        )
        cms_current = np.zeros(
            (self.n_chains, self.depth, self.cms_num_hashes, self.cms_width),
            dtype=np.int32,
        )
        hash_coeffs = self.rng.integers(
            1,
            np.iinfo(np.int32).max,
            size=(self.cms_num_hashes, self.k),
            dtype=np.int64,
        )
        hash_offsets = self.rng.integers(
            0,
            np.iinfo(np.int32).max,
            size=(self.n_chains, self.depth, self.cms_num_hashes),
            dtype=np.int64,
        )
        return _XStreamState(
            deltamax=deltamax,
            chain_dims=chain_dims,
            shift=shift,
            cms_current=cms_current,
            cms_reference=np.zeros_like(cms_current),
            hash_coeffs_mod=hash_coeffs % self.cms_width,
            hash_offsets_mod=hash_offsets % self.cms_width,
            scratch=_ScratchState(
                z=np.zeros(self.k, dtype=np.float64),
                # Python integers prevent overflow for very large finite features.
                bins=np.zeros(self.k, dtype=object),
                feature_visits=np.zeros(self.k, dtype=np.int32),
                hash_dots=np.zeros(self.cms_num_hashes, dtype=np.int64),
            ),
        )

    def _initialize_model(self) -> None:
        if len(self._init_buffer) < self.init_sample_size:
            return
        projected = tuple(self._init_buffer)
        state = self._create_state()
        for point in projected:
            self._learn_projected(state, point)
        self._state = state
        self._init_buffer.clear()

    def _iter_chain_buckets(
        self,
        state: _XStreamState,
        y: np.ndarray,
        chain: int,
    ) -> Iterator[tuple[int, np.ndarray]]:
        scratch = state.scratch
        scratch.z.fill(0.0)
        scratch.bins.fill(0)
        scratch.feature_visits.fill(0)
        scratch.hash_dots.fill(0)
        hash_offsets_mod = state.hash_offsets_mod[chain]

        for level in range(self.depth):
            feature = int(state.chain_dims[chain, level])
            scratch.feature_visits[feature] += 1
            if scratch.feature_visits[feature] == 1:
                z_new = (
                    y[feature] + state.shift[chain, feature]
                ) / state.deltamax[feature]
            else:
                z_new = (
                    2.0 * scratch.z[feature]
                    - state.shift[chain, feature] / state.deltamax[feature]
                )

            scratch.z[feature] = z_new
            bin_new = int(np.floor(z_new))
            if bin_new != scratch.bins[feature]:
                delta = bin_new - scratch.bins[feature]
                scratch.bins[feature] = bin_new
                delta_mod = int(delta % self.cms_width)
                if delta_mod:
                    scratch.hash_dots = (
                        scratch.hash_dots
                        + delta_mod * state.hash_coeffs_mod[:, feature]
                    ) % self.cms_width

            buckets = (
                scratch.hash_dots + hash_offsets_mod[level]
            ) % self.cms_width
            yield level, buckets.astype(np.intp)

    def _update_sketch(
        self,
        state: _XStreamState,
        y: np.ndarray,
        sketch: np.ndarray,
    ) -> None:
        hash_index = np.arange(self.cms_num_hashes, dtype=np.intp)
        for chain in range(self.n_chains):
            for level, buckets in self._iter_chain_buckets(state, y, chain):
                sketch[chain, level, hash_index, buckets] += 1

    def _learn_projected(self, state: _XStreamState, y: np.ndarray) -> None:
        self._update_sketch(state, y, state.cms_current)
        state.samples_seen += 1
        state.samples_in_window += 1
        if state.samples_in_window >= self.window_size:
            state.cms_reference[...] = state.cms_current
            state.cms_current.fill(0)
            state.samples_in_window = 0
            state.completed_windows += 1

    def learn_one(self, x: dict[str, float]) -> None:
        """Update model state with one sample."""
        self._validate_input(x)
        y = self._project_one(x)
        state = self._state
        if state is None:
            self._init_buffer.append(y)
            self._initialize_model()
            return
        self._learn_projected(state, y)

    def _score_projected(self, state: _XStreamState, y: np.ndarray) -> float:
        hash_index = np.arange(self.cms_num_hashes, dtype=np.intp)
        chain_scores = np.empty(self.n_chains, dtype=np.float64)
        for chain in range(self.n_chains):
            best_level_score = np.inf
            for level, buckets in self._iter_chain_buckets(state, y, chain):
                counts = state.cms_reference[chain, level, hash_index, buckets]
                estimated_count = int(np.min(counts))
                level_score = np.log2(1.0 + estimated_count) + (level + 1.0)
                best_level_score = min(best_level_score, level_score)
            chain_scores[chain] = 2.0 ** (1.0 - best_level_score)

        score = float(np.mean(chain_scores))
        return float(np.clip(score, 0.0, 1.0))

    def score_one(self, x: dict[str, float]) -> float:
        """Return an anomaly score in ``[0, 1]`` without learning."""
        self._validate_input(x)
        state = self._state
        if state is None or not state.reference_ready:
            return 0.0
        return self._score_projected(state, self._project_one(x))

    def __repr__(self) -> str:
        state = self._state
        return (
            f"XStream(k={self.k}, n_chains={self.n_chains}, depth={self.depth}, "
            f"cms_width={self.cms_width}, cms_num_hashes={self.cms_num_hashes}, "
            f"window_size={self.window_size}, init_sample_size={self.init_sample_size}, "
            f"density={self.density}, max_feature_cache_size={self.max_feature_cache_size}, "
            f"feature_cache_size={len(self._feature_cache)}, ready={state is not None}, "
            f"reference_ready={state is not None and state.reference_ready})"
        )
