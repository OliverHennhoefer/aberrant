from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import numpy as np

from aberrant.base.similarity import BaseSimilaritySearchEngine

if TYPE_CHECKING:
    import faiss as faiss_types

try:
    import faiss
except ModuleNotFoundError:  # pragma: no cover - exercised when optional extra missing
    faiss = None


class FaissSimilaritySearchEngine(BaseSimilaritySearchEngine):
    """
    FAISS-based similarity search engine for efficient nearest neighbor search.

    Uses a sliding window of data points and incrementally updates the FAISS index
    for efficient similarity search operations.

    Args:
        window_size: Maximum number of data points to keep in the sliding window.
        warm_up: Minimum number of data points required before search can be performed.
    """

    def __init__(self, window_size: int, warm_up: int) -> None:
        if faiss is None:
            raise RuntimeError(
                "faiss-cpu is required for FaissSimilaritySearchEngine. "
                'Install it via `pip install "aberrant[faiss]"`.'
            )

        self.window: collections.deque[dict[str, float]] = collections.deque(
            maxlen=window_size
        )
        self.window_size = window_size

        self._check_params(window_size, warm_up)
        self.warm_up: int = warm_up

        self.index: faiss_types.Index | None = None
        self.keys: list[str] | None = None
        self._index_needs_rebuild = True

    def _dict_to_vector(self, x: dict[str, float]) -> np.ndarray:
        """Convert a feature dictionary to a row vector using the canonical key order."""
        if self.keys is None:
            raise ValueError("Feature keys are not initialized")
        return np.array([x[key] for key in self.keys], dtype=np.float32).reshape(1, -1)

    def append(self, x: dict[str, float]) -> None:
        """
        Add a data point to the search engine.

        Args:
            x: Dictionary representing a data point with feature names as keys.
        """
        # Initialize feature order on first sample.
        if self.keys is None:
            self.keys = sorted(x.keys())
            self._index_needs_rebuild = True
        else:
            current_keys = sorted(x.keys())
            if current_keys != self.keys:
                expected = ", ".join(self.keys)
                received = ", ".join(current_keys)
                raise ValueError(
                    "Inconsistent feature keys for similarity engine. "
                    f"Expected [{expected}], received [{received}]."
                )

        was_full = len(self.window) == self.window_size
        # Own the retained sample. Otherwise caller mutation can make the window
        # disagree with the vectors already stored in the FAISS index.
        stored = dict(x)
        self.window.append(stored)

        # Rebuild when index is uninitialized, key schema changed, or eviction occurred.
        if self._index_needs_rebuild or self.index is None or was_full:
            self._rebuild_index()
            self._index_needs_rebuild = False
        else:
            # For non-evicting appends, incrementally add one vector.
            self.index.add(self._dict_to_vector(stored))

    def search(self, item: dict[str, float], n_neighbors: int) -> float:
        """
        Search for the n nearest neighbors of a data point.

        Args:
            item: Dictionary representing the query data point.
            n_neighbors: Number of nearest neighbors to find.

        Returns:
            Mean distance to the k nearest neighbors, or 0.0 if not enough data.

        Raises:
            ValueError: If n_neighbors is not positive or exceeds available data points.
        """
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")

        if len(self.window) < self.warm_up:
            return 0.0

        if self.keys is None or self.index is None:
            return 0.0

        if n_neighbors > len(self.window):
            raise ValueError(
                f"n_neighbors ({n_neighbors}) cannot exceed window size ({len(self.window)})"
            )

        # Convert query point to vector
        query_keys = sorted(item.keys())
        if query_keys != self.keys:
            expected = ", ".join(self.keys)
            received = ", ".join(query_keys)
            raise ValueError(
                "Inconsistent feature keys for search. "
                f"Expected [{expected}], received [{received}]."
            )
        x = self._dict_to_vector(item)

        # Search for nearest neighbors
        distances_sq, _ = self.index.search(x, k=min(n_neighbors, self.index.ntotal))
        # IndexFlatL2 returns squared L2 distances, while this public API promises
        # the mean distance.
        distances = np.sqrt(np.maximum(distances_sq[0], 0.0))
        return float(np.mean(distances))

    def _rebuild_index(self) -> None:
        """
        Rebuild the FAISS index from current window data.

        This is called when new features are detected or when the index needs initialization.
        """
        if not self.window or self.keys is None:
            return

        # Convert all window data to matrix
        data_matrix = np.array(
            [[point.get(key, 0.0) for key in self.keys] for point in self.window],
            dtype=np.float32,
        )

        # Create new index and add all data
        assert faiss is not None  # guaranteed by __init__ guard
        self.index = faiss.IndexFlatL2(len(self.keys))
        if len(data_matrix) > 0:
            self.index.add(data_matrix)

    def _get_window_data(self) -> tuple[list[str], np.ndarray]:
        """
        Extract keys and data matrix from current window.

        Returns:
            Tuple of (sorted feature keys, data matrix).
        """
        keys = sorted({key for dict_ in self.window for key in dict_})
        return keys, np.array(
            [[dict_.get(key, 0.0) for key in keys] for dict_ in self.window],
            dtype=np.float32,
        )

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
            f"warm_up={self.warm_up}, current_size={len(self.window)})"
        )
