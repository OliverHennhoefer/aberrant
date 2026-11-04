"""xStream anomaly detection for feature-evolving data streams."""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import coerce_finite_number, coerce_integer_count


@dataclass(slots=True)
class _ChainLevels:
    """Chain parameters per split, flattened as ``level * n_chains + chain``."""

    dims: np.ndarray
    shift: np.ndarray
    deltamax: np.ndarray
    # The shift in units of the dimension's scale, subtracted on a revisit.
    scaled_shift: np.ndarray
    # Splits that revisit a dimension and the chain's previous split on it,
    # one pair per visit count, so each round reads only finished splits.
    revisits: list[tuple[np.ndarray, np.ndarray]]
    # The previous split on the same dimension, or the number of splits for a
    # first visit.
    previous: np.ndarray
    # Hash coefficients of each split's dimension, ``(cms_num_hashes, splits)``.
    hash_coeffs: np.ndarray
    # Whether bin steps and their hash terms are reduced modulo the width
    # before summing, as int64 sums of unreduced terms could overflow.
    reduce_terms: bool
    # Hash offsets and the flat sketch index of each row's first counter,
    # shaped ``(cms_num_hashes, depth, n_chains)``.
    hash_offsets: np.ndarray
    row_starts: np.ndarray


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
    levels: _ChainLevels
    # The last projected point and its sketch cells, so that learning a point
    # right after scoring it descends the chains once.
    last_point: bytes = b""
    last_cells: np.ndarray | None = None
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
            scales. These points also populate the reference-window counters.
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
        window_size = coerce_integer_count(window_size, label="window_size")
        init_sample_size = coerce_integer_count(
            init_sample_size, label="init_sample_size"
        )
        if not (0.0 < density <= 1.0):
            raise ValueError("density must be in (0, 1]")
        if max_feature_cache_size is not None:
            max_feature_cache_size = coerce_integer_count(
                max_feature_cache_size, label="max_feature_cache_size"
            )

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
        projections = [self._feature_projection(feature) for feature in x]
        indices = np.concatenate([indices for indices, _ in projections])
        signs = np.concatenate([signs for _, signs in projections])
        values = np.array([float(value) for value in x.values()])
        # Every feature updates the same number of coordinates, and bincount
        # adds the weights in order, as adding each feature's terms in turn does.
        weights = np.repeat(values, len(projections[0][0])) * signs
        return np.bincount(indices, weights=weights, minlength=self.k)

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
            self.rng.uniform(low=0.0, high=1.0, size=(self.n_chains, self.k)) * deltamax
        )
        # A bucket can receive every observation in a window. Use the smallest
        # counters that hold a full window, since every learned point updates
        # counters spread across both sketches.
        for counter_dtype in (np.int8, np.int16, np.int32, np.int64):
            if self.window_size <= np.iinfo(counter_dtype).max:
                break
        cms_current = np.zeros(
            (self.n_chains, self.depth, self.cms_num_hashes, self.cms_width),
            dtype=counter_dtype,
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
        hash_coeffs_mod = hash_coeffs % self.cms_width
        hash_offsets_mod = hash_offsets % self.cms_width
        return _XStreamState(
            deltamax=deltamax,
            chain_dims=chain_dims,
            shift=shift,
            cms_current=cms_current,
            cms_reference=np.zeros_like(cms_current),
            hash_coeffs_mod=hash_coeffs_mod,
            hash_offsets_mod=hash_offsets_mod,
            levels=self._chain_levels(
                chain_dims, shift, deltamax, hash_coeffs_mod, hash_offsets_mod
            ),
        )

    def _chain_levels(
        self,
        chain_dims: np.ndarray,
        shift: np.ndarray,
        deltamax: np.ndarray,
        hash_coeffs_mod: np.ndarray,
        hash_offsets_mod: np.ndarray,
    ) -> _ChainLevels:
        n_chains, depth = chain_dims.shape
        n_splits = n_chains * depth
        previous = np.full(n_splits, n_splits, dtype=np.intp)
        revisits: list[tuple[list[int], list[int]]] = []
        for chain in range(n_chains):
            last_split: dict[int, int] = {}
            visits: dict[int, int] = {}
            for level, dim in enumerate(chain_dims[chain].tolist()):
                split = level * n_chains + chain
                count = visits.get(dim, 0)
                if count:
                    previous[split] = last_split[dim]
                    if len(revisits) < count:
                        revisits.append(([], []))
                    revisits[count - 1][0].append(split)
                    revisits[count - 1][1].append(last_split[dim])
                visits[dim] = count + 1
                last_split[dim] = split

        dims = chain_dims.T.ravel()
        level_shift = shift[np.tile(np.arange(n_chains), depth), dims]
        level_deltamax = deltamax[dims]
        width = self.cms_width
        # Unreduced terms lie within (-2 * width**2, 2 * width**2).
        reduce_terms = 2 * depth * width * width + width > np.iinfo(np.int64).max
        rows = np.arange(n_splits * self.cms_num_hashes, dtype=np.intp) * width
        return _ChainLevels(
            dims=dims,
            shift=level_shift,
            deltamax=level_deltamax,
            scaled_shift=level_shift / level_deltamax,
            revisits=[
                (np.array(splits, dtype=np.intp), np.array(earlier, dtype=np.intp))
                for splits, earlier in revisits
            ],
            previous=previous,
            hash_coeffs=hash_coeffs_mod[:, dims],
            reduce_terms=reduce_terms,
            hash_offsets=np.ascontiguousarray(hash_offsets_mod.transpose(2, 1, 0)),
            row_starts=np.ascontiguousarray(
                rows.reshape(n_chains, depth, self.cms_num_hashes).transpose(2, 1, 0)
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

    def _sketch_cells(self, state: _XStreamState, y: np.ndarray) -> np.ndarray:
        """Return the flat sketch index of every counter that ``y`` reaches.

        All chains descend together. The result is shaped
        ``(cms_num_hashes, depth, n_chains)``.
        """
        point = y.tobytes()
        if state.last_cells is not None and point == state.last_point:
            return state.last_cells
        levels = state.levels
        width = self.cms_width
        z = (y[levels.dims] + levels.shift) / levels.deltamax
        for splits, previous in levels.revisits:
            z[splits] = 2.0 * z[previous] - levels.scaled_shift[splits]
        bins = np.floor(z)
        finite = np.isfinite(bins)
        if not finite.all():
            # Raise what converting a chain's first such bin to int raises.
            by_chain = bins.reshape(self.depth, -1).T
            int(by_chain[~finite.reshape(self.depth, -1).T][0])
        # The hashes are linear modulo the width, so each bin only matters
        # modulo the width. ``fmod`` of an integral float is exact.
        bins_mod = np.zeros(bins.size + 1, dtype=np.int64)
        bins_mod[:-1] = np.fmod(bins, width)
        steps = bins_mod[:-1] - bins_mod[levels.previous]
        if levels.reduce_terms:
            steps %= width
            terms = steps * levels.hash_coeffs % width
        else:
            terms = steps * levels.hash_coeffs
        dots = np.cumsum(terms.reshape(levels.hash_offsets.shape), axis=1)
        cells: np.ndarray = levels.row_starts + (dots + levels.hash_offsets) % width
        state.last_point = point
        state.last_cells = cells
        return cells

    def _learn_projected(self, state: _XStreamState, y: np.ndarray) -> None:
        state.cms_current.reshape(-1)[self._sketch_cells(state, y)] += 1
        state.samples_seen += 1
        state.samples_in_window += 1
        if state.samples_in_window >= self.window_size:
            state.cms_reference, state.cms_current = (
                state.cms_current,
                state.cms_reference,
            )
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
        counts = state.cms_reference.reshape(-1)[self._sketch_cells(state, y)]
        level_scores = np.log2(1.0 + counts.min(axis=0)) + np.arange(
            1.0, self.depth + 1.0
        ).reshape(-1, 1)
        # Python's power, as in a per-chain loop: NumPy's vectorized power can
        # differ from it in the last bit.
        chain_scores = np.array(
            [2.0 ** (1.0 - best) for best in level_scores.min(axis=0).tolist()]
        )
        score = float(np.mean(chain_scores))
        return min(max(score, 0.0), 1.0)

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
