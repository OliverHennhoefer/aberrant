"""Online ASD Isolation Forest for anomaly detection."""

import math
from collections import deque
from typing import Any

import numpy as np

from aberrant.base.model import BaseModel


class ASDIsolationForest(BaseModel):
    """
    Optimized Online ASD Isolation Forest for anomaly detection.

    This implementation uses NumPy for efficient computations and maintains
    a bounded queue of isolation trees built from disjoint stream chunks. It
    is a lightweight streaming adaptation, not an exact implementation of the
    published iForestASD sliding-window retraining algorithm.

    Args:
        n_estimators: Number of trees in the iforest.
        max_samples: Number of samples used to build each tree.
        seed: Random seed for reproducibility.

    References:
        Ding, Z., & Fei, M. (2013). An Anomaly Detection Approach Based on
        Isolation Forest Algorithm for Streaming Data using Sliding Window.
        https://www.sciencedirect.com/science/article/pii/S1474667016314999
        Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest.
        https://doi.org/10.1109/ICDM.2008.17
    """

    # Euler-Mascheroni constant for path length calculations
    EULER_MASCHERONI = 0.5772156649015329

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        # Validate parameters
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators}")
        if max_samples <= 1:
            raise ValueError(f"max_samples must be greater than 1, got {max_samples}")

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed

        # Initialize data structures
        self.feature_names: list[str] | None = None
        self.buffer: np.ndarray = np.empty((0, 0))
        self.buffer_count: int = 0
        self.trees: deque[dict[str, Any]] = deque()
        self.c_n: float = self._compute_c(max_samples)

        # Pre-allocate array for feature conversion optimization
        self._x_converted: np.ndarray = np.empty(0)

        # Initialize random number generator (unified approach)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _compute_c(n: int) -> float:
        """
        Compute the average path length adjustment term c(n).

        Args:
            n: Number of samples.

        Returns:
            Average path length adjustment value.
        """
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        harmonic = math.log(n - 1) + ASDIsolationForest.EULER_MASCHERONI
        return 2.0 * harmonic - 2.0 * (n - 1) / n

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a single data point.

        Args:
            x: Feature dictionary with string keys and float values.
        """
        if not x:
            raise ValueError("Input dictionary cannot be empty")

        # Initialize on first sample
        if self.feature_names is None:
            self.feature_names = sorted(x.keys())  # Ensure consistent ordering
            self.buffer = np.zeros((self.max_samples, len(self.feature_names)))
            self._x_converted = np.zeros(len(self.feature_names))
            self.buffer_count = 0

        # Convert dictionary to numpy array efficiently
        for i, feature in enumerate(self.feature_names):
            self._x_converted[i] = x.get(feature, 0.0)

        # Add current sample to buffer
        self.buffer[self.buffer_count] = self._x_converted
        self.buffer_count += 1

        # Check if buffer is now full and we should build a tree
        if self.buffer_count == self.max_samples:
            # Build new tree from current buffer
            new_tree = self._build_tree(self.buffer[: self.max_samples])
            self.trees.append(new_tree)
            if len(self.trees) > self.n_estimators:
                self.trees.popleft()

            # The completed chunk already contains the current sample.
            self.buffer_count = 0

    def _build_tree(self, data_arr: np.ndarray) -> dict[str, Any]:
        """
        Build an isolation tree from a NumPy array buffer.

        Args:
            data_arr: Data array to build tree from.

        Returns:
            Tree structure as nested dictionary.
        """
        n_samples = data_arr.shape[0]
        indices = np.arange(n_samples)
        max_height = math.ceil(math.log2(max(n_samples, 2)))  # Ensure valid log
        return self._build_tree_recursive(data_arr, indices, max_height)

    def _build_tree_recursive(
        self,
        data_arr: np.ndarray,
        indices: np.ndarray,
        max_height: int,
        current_height: int = 0,
    ) -> dict[str, Any]:
        """
        Recursively build isolation tree nodes with NumPy optimizations.

        Args:
            data_arr: Full dataset array.
            indices: Indices of samples for current node.
            max_height: Maximum tree height.
            current_height: Current depth in tree.

        Returns:
            Tree node as dictionary.
        """
        n = len(indices)
        if n <= 1 or current_height >= max_height:
            return {"size": n, "c": self._compute_c(n)}

        # Select only from features that can produce a non-empty split.
        node_values = data_arr[indices]
        variable_features = np.flatnonzero(np.ptp(node_values, axis=0) > 0)
        if len(variable_features) == 0:
            return {"size": n, "c": self._compute_c(n)}

        feature_idx = int(self.rng.choice(variable_features))
        feature_vals = data_arr[indices, feature_idx]
        min_val, max_val = np.min(feature_vals), np.max(feature_vals)

        # Generate split value using numpy's RNG
        split_val = self.rng.uniform(min_val, max_val)
        mask = feature_vals < split_val
        left_indices = indices[mask]
        right_indices = indices[~mask]

        return {
            "split_feature": self.feature_names[feature_idx],
            "split_val": split_val,
            "left": self._build_tree_recursive(
                data_arr, left_indices, max_height, current_height + 1
            ),
            "right": self._build_tree_recursive(
                data_arr, right_indices, max_height, current_height + 1
            ),
            "size": n,
        }

    def _compute_path_length(self, x: dict[str, float], tree: dict[str, Any]) -> float:
        """
        Compute path length for a sample through an isolation tree.

        Args:
            x: Sample dictionary.
            tree: Tree structure.

        Returns:
            Path length including adjustment term.
        """
        depth = 0
        current_node = tree

        while "split_feature" in current_node:
            feature_val = x.get(current_node["split_feature"], 0.0)
            if feature_val < current_node["split_val"]:
                current_node = current_node["left"]
            else:
                current_node = current_node["right"]
            depth += 1

        return depth + current_node["c"]

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score for a single data point.

        Args:
            x: Feature dictionary with string keys and float values.

        Returns:
            Anomaly score (higher values indicate more anomalous samples).
        """
        if not self.trees:
            return 0.0

        if not x:
            return 0.0

        total_path = sum(self._compute_path_length(x, tree) for tree in self.trees)
        avg_path = total_path / len(self.trees)
        return 2.0 ** (-avg_path / self.c_n)

    def __repr__(self) -> str:
        """Return a string representation of the ASD Isolation Forest."""
        return (
            f"ASDIsolationForest(n_estimators={self.n_estimators}, "
            f"max_samples={self.max_samples}, n_trees={len(self.trees)}, "
            f"seed={self.seed})"
        )
