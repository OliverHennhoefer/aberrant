"""Public Online Isolation Forest model."""

from __future__ import annotations

import math
from collections import deque
from typing import Literal

import numpy as np

from aberrant._utils.validation import (
    FeatureSchema,
    PreparedFeatures,
    coerce_integer_count,
)
from aberrant.base.model import BaseModel
from aberrant.model.forest._online_tree import OnlineIsolationTree, _Bounds


class OnlineIsolationForest(BaseModel):
    """Incremental isolation forest with sliding-window unlearning.

    Scores are path-length isolation scores in ``[0, 1]``, and 0.0 until the
    forest has learned enough points to normalize path lengths. Learned points
    are retained until ``window_size`` is exceeded, then the oldest points are
    unlearned from the trees. The depth limit and the path-length
    normalization take logarithms to base ``2 * branching_factor``, as the
    authors' implementation does; the paper writes base 2.

    Args:
        n_trees: Number of independently seeded online isolation trees.
        max_leaf_samples: Base leaf population at which splitting becomes
            eligible.
        tree_type: ``"fixed"`` uses the base split threshold at every depth;
            ``"adaptive"`` multiplies it by ``2**depth``.
        subsample: Independent per-tree probability of selecting a point for a
            tree update, in ``(0, 1]``.
        window_size: Maximum number of learned points retained for sliding-window
            unlearning.
        branching_factor: Number of children created at each split. It must be
            greater than one.
        metric: Split geometry. Only ``"axisparallel"`` is implemented.
        seed: Root seed from which independent per-tree generators are spawned.

    References:
        Leveni, F., Weigert Cassales, G., Pfahringer, B., Bifet, A., &
        Boracchi, G. (2024). Online Isolation Forest.
        https://proceedings.mlr.press/v235/leveni24a.html
    """

    def __init__(
        self,
        *,
        n_trees: int = 100,
        max_leaf_samples: int = 32,
        tree_type: Literal["fixed", "adaptive"] = "adaptive",
        subsample: float = 1.0,
        window_size: int = 2048,
        branching_factor: int = 2,
        metric: Literal["axisparallel"] = "axisparallel",
        seed: int | None = None,
    ) -> None:
        """Initialize an Online Isolation Forest."""
        n_trees = coerce_integer_count(n_trees, label="n_trees", minimum=1)
        max_leaf_samples = coerce_integer_count(
            max_leaf_samples, label="max_leaf_samples", minimum=1
        )
        window_size = coerce_integer_count(window_size, label="window_size", minimum=1)
        branching_factor = coerce_integer_count(
            branching_factor, label="branching_factor", minimum=2
        )
        if tree_type not in {"fixed", "adaptive"}:
            raise ValueError("tree_type must be 'fixed' or 'adaptive'")
        if not (0.0 < subsample <= 1.0):
            raise ValueError("subsample must be in (0.0, 1.0]")
        if metric != "axisparallel":
            raise ValueError("metric must be 'axisparallel'")

        self.n_trees = n_trees
        self.max_leaf_samples = max_leaf_samples
        self.tree_type = tree_type
        self.subsample = subsample
        self.window_size = window_size
        self.branching_factor = branching_factor
        self.metric = metric
        self.seed = seed

        child_seeds = np.random.SeedSequence(seed).spawn(n_trees)
        self.trees = [
            OnlineIsolationTree(
                max_leaf_samples=max_leaf_samples,
                tree_type=tree_type,
                subsample=subsample,
                branching_factor=branching_factor,
                data_size=0,
                metric=metric,
                rng=np.random.default_rng(child_seed),
            )
            for child_seed in child_seeds
        ]
        self.data_window: deque[np.ndarray] = deque()
        self.data_size = 0
        self.normalization_factor = 0.0
        self._schema = FeatureSchema()
        self._n_features: int | None = None

    def _point(self, prepared: PreparedFeatures) -> tuple[np.ndarray, list[float]]:
        """Return validated features as a float32 row and as its values."""
        if self._n_features is not None and len(prepared.names) != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, received {len(prepared.names)}"
            )
        point = prepared.values.astype(np.float32)
        values: list[float] = point.tolist()
        # Finite float64 features can still overflow float32.
        if not all(map(math.isfinite, values)):
            raise ValueError("data must contain only finite values")
        return point, values

    def learn_one(self, x: dict[str, float]) -> None:
        """Learn one validated feature mapping."""
        prepared = self._schema.preview(x)
        point, values = self._point(prepared)
        if self._n_features is None:
            self._n_features = point.shape[0]

        # The trees do not read the normalization, so it is refreshed once at
        # the end.
        point_bounds = _Bounds.from_data(point.astype(np.float64)[np.newaxis])
        for tree in self.trees:
            tree.learn_point(point_bounds, values)
        self.data_size += 1
        self.data_window.append(point)
        if self.data_size > self.window_size:
            oldest_values = self.data_window.popleft().tolist()
            for tree in self.trees:
                tree.unlearn_point(oldest_values)
            self.data_size -= 1
        self._refresh_normalization()
        self._schema.commit(prepared)

    def score_one(self, x: dict[str, float]) -> float:
        """Score one validated feature mapping without mutating its schema."""
        prepared = self._schema.preview(x)
        _, values = self._point(prepared)
        if self.normalization_factor <= 0.0:
            return 0.0
        leaf_depths, leaf_sizes = zip(
            *(tree.leaf_path(values) for tree in self.trees), strict=True
        )
        path_lengths = OnlineIsolationTree.get_random_path_lengths(
            self.branching_factor,
            self.max_leaf_samples,
            np.array(leaf_sizes, dtype=np.float64),
        )
        depths = np.array(leaf_depths, dtype=np.float64) + path_lengths
        return float(2 ** (-depths.mean() / self.normalization_factor))

    def _refresh_normalization(self) -> None:
        self.normalization_factor = OnlineIsolationTree.get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            self.data_size * self.subsample,
        )
