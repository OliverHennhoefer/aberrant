"""Nearest-neighbor distance over a sliding window of learned events."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Literal

import numpy as np

from aberrant._utils.validation import FeatureSchema, coerce_integer_count
from aberrant.base.model import BaseModel
from aberrant.base.readiness import Readiness
from aberrant.exceptions import MissingOptionalDependencyError, ValidationError


def _faiss() -> Any:
    try:
        return import_module("faiss")
    except ModuleNotFoundError as error:
        raise MissingOptionalDependencyError("KNN(backend='faiss')", "faiss") from error


class KNN(BaseModel):
    """Mean Euclidean distance to the ``k`` nearest recently learned events.

    ``learn_one`` appends the event to a window of the last ``window_size``
    learned events, and ``score_one`` returns the mean Euclidean distance from
    the candidate to its ``k`` nearest neighbors in that window. The score is
    ``0.0`` until the window holds ``k`` events. Feature names are sorted when
    the first event is learned and must stay the same.

    Args:
        k: Number of neighbors averaged for each score.
        window_size: Number of most recent learned events kept as reference.
            It must be at least ``k``.
        backend: ``"numpy"`` computes exact float64 distances to the whole
            window. ``"faiss"`` searches an exact float32 FAISS index, which
            is faster for wide or long windows and needs ``aberrant[faiss]``.

    References:
        Ramaswamy, S., Rastogi, R., & Shim, K. (2000). Efficient algorithms
        for mining outliers from large data sets. SIGMOD.
        https://doi.org/10.1145/335191.335437
    """

    _transient = ("_faiss_module", "_index")

    def __init__(
        self,
        *,
        k: int = 10,
        window_size: int = 1000,
        backend: Literal["numpy", "faiss"] = "numpy",
    ) -> None:
        """Initialize a nearest-neighbor scorer."""
        k = coerce_integer_count(k, label="k", minimum=1)
        window_size = coerce_integer_count(window_size, label="window_size", minimum=1)
        if window_size < k:
            raise ValueError("window_size must be at least k")
        if backend not in {"numpy", "faiss"}:
            raise ValueError("backend must be 'numpy' or 'faiss'")
        self._faiss_module = _faiss() if backend == "faiss" else None
        self.k = k
        self.window_size = window_size
        self.backend = backend
        self._schema = FeatureSchema()
        self._window: np.ndarray | None = None
        self._count = 0
        self._next = 0
        self._index: Any = None

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the window; the FAISS index is rebuilt from it."""
        super().__setstate__(state)
        if self.backend == "faiss":
            self._faiss_module = _faiss()

    @property
    def n_samples_seen(self) -> int:
        """Number of events in the reference window."""
        return self._count

    @property
    def readiness(self) -> Readiness:
        """Ready once the window holds ``k`` events."""
        return Readiness.after(self._count, self.k)

    def learn_one(self, x: dict[str, float]) -> None:
        """Append one event to the reference window, evicting the oldest."""
        prepared = self._schema.preview(x)
        vector = self._vector(prepared.values)
        self._schema.commit(prepared)
        if self._window is None:
            self._window = np.empty((self.window_size, vector.shape[0]), vector.dtype)
        self._window[self._next] = vector
        self._next = (self._next + 1) % self.window_size
        if self._count < self.window_size:
            if self._index is not None:
                self._index.add(vector.reshape(1, -1))
            self._count += 1
        else:
            self._index = None

    def score_one(self, x: dict[str, float]) -> float:
        """Return the mean distance to the ``k`` nearest window events."""
        prepared = self._schema.preview(x)
        vector = self._vector(prepared.values)
        if self._window is None or self._count < self.k:
            return 0.0
        if self.backend == "faiss":
            return self._faiss_score(vector)
        reference = self._window[: self._count]
        distances = np.sqrt(((reference - vector) ** 2).sum(axis=1))
        nearest = np.partition(distances, self.k - 1)[: self.k]
        return float(nearest.mean())

    def _vector(self, values: np.ndarray) -> np.ndarray:
        if self.backend == "numpy":
            return np.asarray(values, dtype=np.float64)
        with np.errstate(over="ignore"):
            vector = values.astype(np.float32)
        if not np.all(np.isfinite(vector)):
            raise ValidationError("Feature values must fit the finite float32 range")
        return vector

    def _faiss_score(self, vector: np.ndarray) -> float:
        if self._index is None:
            assert self._faiss_module is not None and self._window is not None
            index = self._faiss_module.IndexFlatL2(vector.shape[0])
            index.add(self._window[: self._count])
            self._index = index
        squared, _ = self._index.search(vector.reshape(1, -1), self.k)
        return float(np.sqrt(np.maximum(squared[0], 0.0)).mean())
