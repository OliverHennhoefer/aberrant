from __future__ import annotations

import collections
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from aberrant.base.similarity import BaseSimilaritySearchEngine
from aberrant.utils.validation import FeatureSchema, PreparedFeatures

if TYPE_CHECKING:
    import faiss as faiss_types

faiss: ModuleType | None
try:
    import faiss
except ModuleNotFoundError:  # pragma: no cover - exercised when optional extra missing
    faiss = None


class FaissSimilaritySearchEngine(BaseSimilaritySearchEngine):
    """Sliding-window exact Euclidean nearest-neighbor engine.

    Observations are stored in a bounded FIFO window and indexed with FAISS
    ``IndexFlatL2``. Although FAISS reports squared L2 distances, ``search``
    converts them to Euclidean distances and returns their arithmetic mean.
    Feature names are sorted when the first observation is appended and must
    then remain identical. The engine returns ``0.0`` until ``warm_up``
    observations have been retained.

    Args:
        window_size: Maximum number of data points to keep in the sliding window.
        warm_up: Minimum number of data points required before search can be performed.

    Note:
        Requires the ``faiss`` optional dependency group: install
        ``aberrant[faiss]``.
        The retained vector window is authoritative. The index is a derived
        cache: failed insertions discard it, and subsequent operations rebuild
        it from retained observations.
    """

    def __init__(self, window_size: int, warm_up: int) -> None:
        if faiss is None:
            raise RuntimeError(
                "faiss-cpu is required for FaissSimilaritySearchEngine. "
                'Install it via `pip install "aberrant[faiss]"`.'
            )

        self._check_params(window_size, warm_up)
        self.window_size = window_size
        self.warm_up: int = warm_up
        self._schema = FeatureSchema()
        self._window: collections.deque[np.ndarray] = collections.deque(
            maxlen=window_size
        )
        self.index: faiss_types.Index | None = None

    @property
    def keys(self) -> list[str] | None:
        """Return a snapshot of the established feature order."""
        names = self._schema.names
        return None if names is None else list(names)

    @property
    def window(self) -> collections.deque[dict[str, float]]:
        """Return a snapshot of retained observations, protecting indexed vectors."""
        names = self._schema.names or ()
        return collections.deque(
            (dict(zip(names, vector.tolist(), strict=True)) for vector in self._window),
            maxlen=self.window_size,
        )

    @staticmethod
    def _index_vector(prepared: PreparedFeatures) -> np.ndarray:
        """Convert a validated sample to the finite float32 representation FAISS uses."""
        with np.errstate(over="ignore"):
            vector = prepared.values.astype(np.float32)
        if not np.all(np.isfinite(vector)):
            raise ValueError("Feature values must fit the finite float32 range")
        return vector.reshape(1, -1)

    def append(self, x: dict[str, float]) -> None:
        """
        Add a data point to the search engine.

        Args:
            x: Dictionary representing a data point with feature names as keys.
        """
        prepared = self._schema.preview(x)
        vector = self._index_vector(prepared)
        if self.index is not None and len(self._window) < self.window_size:
            # FAISS may mutate before raising. Its cache is disposable; leave
            # the authoritative window and schema untouched on native failure.
            try:
                self.index.add(vector)
            except Exception:
                self.index = None
                raise
            self._schema.commit(prepared)
            self._window.append(prepared.values)
        else:
            # Stage eviction and reconstruction before publishing retained state.
            candidate_window = collections.deque(self._window, maxlen=self.window_size)
            candidate_window.append(prepared.values)
            candidate_index = self._build_index(candidate_window)
            self._schema.commit(prepared)
            self._window = candidate_window
            self.index = candidate_index

    @staticmethod
    def _build_index(window: collections.deque[np.ndarray]) -> faiss_types.Index:
        """Build a complete cache without publishing a partially populated index."""
        assert faiss is not None  # guaranteed by __init__ guard
        index: faiss_types.Index = faiss.IndexFlatL2(len(window[0]))
        index.add(np.asarray(window, dtype=np.float32))
        return index

    def search(self, item: dict[str, float], n_neighbors: int) -> float:
        """
        Search for the n nearest neighbors of a data point.

        Args:
            item: Dictionary representing the query data point.
            n_neighbors: Number of nearest neighbors to find.

        Returns:
            Mean Euclidean distance to the requested nearest neighbors, or
            ``0.0`` before warm-up completes.

        Raises:
            ValueError: If ``n_neighbors`` is not positive, exceeds the number
                of retained observations, or the feature schema differs.
        """
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")

        prepared = self._schema.preview(item)
        vector = self._index_vector(prepared)
        if len(self._window) < self.warm_up:
            return 0.0

        if n_neighbors > len(self._window):
            raise ValueError(
                f"n_neighbors ({n_neighbors}) cannot exceed window size ({len(self._window)})"
            )

        # Search for nearest neighbors
        if self.index is None:
            self.index = self._build_index(self._window)
        distances_sq, _ = self.index.search(vector, k=n_neighbors)
        # IndexFlatL2 returns squared L2 distances, while this public API promises
        # the mean distance.
        distances = np.sqrt(np.maximum(distances_sq[0], 0.0))
        return float(np.mean(distances))

    @staticmethod
    def _check_params(window_size: int, warm_up: int) -> None:
        """
        Validate constructor parameters.

        Args:
            window_size: Maximum window size.
            warm_up: Minimum data points for search.

        Raises:
            ValueError: If parameters are invalid.
        """
        if window_size <= 0:
            raise ValueError(f"window_size ({window_size}) must be positive")
        if warm_up <= 0:
            raise ValueError(f"warm_up ({warm_up}) must be positive")
        if window_size < warm_up:
            raise ValueError(
                f"window_size ({window_size}) must be >= warm_up ({warm_up})"
            )

    def __repr__(self) -> str:
        """Return a string representation of the FAISS engine."""
        return (
            f"FaissSimilaritySearchEngine(window_size={self.window_size}, "
            f"warm_up={self.warm_up}, current_size={len(self._window)})"
        )
