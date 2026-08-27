"""Public Online Isolation Forest model."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.model.iforest._online_tree import OnlineIsolationTree
from aberrant.utils.validation import FeatureSchema, PreparedFeatures


class OnlineIsolationForest(BaseModel):
    """Incremental isolation forest with sliding-window unlearning.

    References:
        Leveni, F., Weigert Cassales, G., Pfahringer, B., Bifet, A., &
        Boracchi, G. (2024). Online Isolation Forest.
        https://proceedings.mlr.press/v235/leveni24a.html
    """

    def __init__(
        self,
        num_trees: int = 100,
        max_leaf_samples: int = 32,
        tree_type: str = "adaptive",
        subsample: float = 1.0,
        window_size: int = 2048,
        branching_factor: int = 2,
        metric: str = "axisparallel",
        n_jobs: int = 1,
        seed: int | None = None,
    ) -> None:
        """Initialize an Online Isolation Forest.

        ``n_jobs=1`` executes sequentially; ``n_jobs=-1`` uses all available
        logical CPUs.
        """
        if num_trees <= 0:
            raise ValueError("num_trees must be positive")
        if max_leaf_samples <= 0:
            raise ValueError("max_leaf_samples must be positive")
        if tree_type not in {"fixed", "adaptive"}:
            raise ValueError("tree_type must be 'fixed' or 'adaptive'")
        if not (0.0 < subsample <= 1.0):
            raise ValueError("subsample must be in (0.0, 1.0]")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if branching_factor <= 1:
            raise ValueError("branching_factor must be greater than 1")
        if metric != "axisparallel":
            raise ValueError("metric must be 'axisparallel'")
        if n_jobs == 0 or n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer")

        self.num_trees = num_trees
        self.max_leaf_samples = max_leaf_samples
        self.tree_type = tree_type
        self.subsample = subsample
        self.window_size = window_size
        self.branching_factor = branching_factor
        self.metric = metric
        self.n_jobs = n_jobs
        self._max_workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
        self.seed = seed

        child_seeds = np.random.SeedSequence(seed).spawn(num_trees)
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
        self.data_window: list[np.ndarray] = []
        self.data_size = 0
        self.normalization_factor = 0.0
        self._schema = FeatureSchema()
        self._n_features: int | None = None

    def _dict_to_array(self, prepared: PreparedFeatures) -> np.ndarray:
        if self._n_features is not None and len(prepared.names) != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, received {len(prepared.names)}"
            )
        return prepared.values.astype(np.float32).reshape(1, -1)

    def learn_one(self, x: dict[str, float]) -> None:
        """Learn one validated feature mapping."""
        prepared = self._schema.preview(x)
        self.learn_batch(self._dict_to_array(prepared))
        self._schema.commit(prepared)

    def score_one(self, x: dict[str, float]) -> float:
        """Score one validated feature mapping without mutating its schema."""
        prepared = self._schema.preview(x)
        scores = self.score_batch(self._dict_to_array(prepared))
        return float(scores[0])

    def _validate_batch(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            raise ValueError("data must be a 2D array of shape (n_samples, n_features)")
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("data must contain numeric values")
        if not np.all(np.isfinite(data)):
            raise ValueError("data must contain only finite values")
        if self._n_features is not None and data.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, received {data.shape[1]}"
            )

    def _run_tree_updates(self, method: str, data: np.ndarray) -> None:
        if method == "learn":
            functions = [tree.learn for tree in self.trees]
        elif method == "unlearn":
            functions = [tree.unlearn for tree in self.trees]
        else:
            raise ValueError(f"Unknown tree update method: {method}")

        if self._max_workers == 1:
            for function in functions:
                function(data)
            return
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            list(executor.map(lambda function: function(data), functions))

    def _refresh_normalization(self) -> None:
        self.normalization_factor = OnlineIsolationTree.get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            self.data_size * self.subsample,
        )

    def learn_batch(self, data: np.ndarray) -> None:
        """Learn a two-dimensional numeric batch."""
        self._validate_batch(data)
        if data.shape[0] == 0:
            return
        if self._n_features is None:
            self._n_features = data.shape[1]

        self.data_size += data.shape[0]
        self._refresh_normalization()
        self._run_tree_updates("learn", data)

        self.data_window.extend(data.copy())
        if self.data_size <= self.window_size:
            return

        old_data_count = self.data_size - self.window_size
        old_data = np.asarray(self.data_window[:old_data_count])
        self.data_window = self.data_window[old_data_count:]
        self.data_size -= old_data_count
        self._refresh_normalization()
        self._run_tree_updates("unlearn", old_data)

    def score_batch(self, data: np.ndarray) -> np.ndarray:
        """Score a two-dimensional numeric batch."""
        self._validate_batch(data)
        if data.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        functions = [tree.predict for tree in self.trees]
        if self._max_workers == 1:
            depths = np.asarray([function(data) for function in functions]).T
        else:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                depths = np.asarray(
                    list(executor.map(lambda function: function(data), functions))
                ).T

        mean_depths = depths.mean(axis=1)
        scores = 2 ** (
            -mean_depths / (self.normalization_factor + np.finfo(float).eps)
        )
        return np.asarray(scores, dtype=np.float64)
