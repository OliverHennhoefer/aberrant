"""Sliding-window iForestASD anomaly detector."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema


@dataclass(frozen=True, slots=True)
class _ASDLeaf:
    """Terminal isolation-tree node."""

    correction: float


@dataclass(frozen=True, slots=True)
class _ASDBranch:
    """Binary isolation-tree split."""

    feature: int
    threshold: float
    left: _ASDNode
    right: _ASDNode


_ASDNode: TypeAlias = _ASDLeaf | _ASDBranch


class ASDIsolationForest(BaseModel):
    """
    Isolation Forest for streaming data using a sliding reference window.

    iForestASD periodically discards the current batch Isolation Forest and
    trains a complete replacement forest on a recent reference window. This
    implementation keeps that window bounded, samples each isolation tree from
    it, and retrains after ``retrain_interval`` new observations.

    The original paper does not provide an author-maintained public repository.
    The window/retraining structure is cross-checked against the open-source
    PySAD ``IForestASD`` reference implementation, which implements Algorithm 2
    from the paper without its simulation-specific concept-drift step.

    Args:
        n_estimators: Number of isolation trees in each replacement forest.
        max_samples: Maximum reference-window samples used to build each tree.
        window_size: Number of recent samples retained for retraining. If
            ``None``, defaults to ``max_samples``.
        retrain_interval: New samples between forest replacements. If ``None``,
            defaults to ``window_size``.
        seed: Random seed for reproducibility.

    References:
        Ding, Z., & Fei, M. (2013). An Anomaly Detection Approach Based on
        Isolation Forest Algorithm for Streaming Data using Sliding Window.
        https://doi.org/10.3182/20130902-3-CN-3020.00044
        Reference implementation: https://github.com/selimfirat/pysad
        Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest.
        https://doi.org/10.1109/ICDM.2008.17
    """

    EULER_MASCHERONI = 0.5772156649015329

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        window_size: int | None = None,
        retrain_interval: int | None = None,
        seed: int | None = None,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators}")
        if max_samples <= 1:
            raise ValueError(f"max_samples must be greater than 1, got {max_samples}")

        resolved_window_size = max_samples if window_size is None else window_size
        if resolved_window_size <= 1:
            raise ValueError(
                f"window_size must be greater than 1, got {resolved_window_size}"
            )
        resolved_interval = (
            resolved_window_size if retrain_interval is None else retrain_interval
        )
        if resolved_interval <= 0:
            raise ValueError(
                f"retrain_interval must be positive, got {resolved_interval}"
            )

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.window_size = resolved_window_size
        self.retrain_interval = resolved_interval
        self.seed = seed

        self._schema = FeatureSchema()
        self.window: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.trees: list[_ASDNode] = []
        self.c_n = self._compute_c(min(self.max_samples, self.window_size))

        self._samples_since_retrain = 0
        self._fits_completed = 0
        self._last_fit_window: np.ndarray | None = None
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _compute_c(n: int) -> float:
        """Compute the average unsuccessful-search path length ``c(n)``."""
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        harmonic = math.log(n - 1) + ASDIsolationForest.EULER_MASCHERONI
        return 2.0 * harmonic - 2.0 * (n - 1) / n

    def learn_one(self, x: dict[str, float]) -> None:
        """Add one sample to the sliding window and retrain when due."""
        prepared = self._schema.preview(x)
        self.window.append(prepared.values.copy())
        self._samples_since_retrain += 1

        initial_fit_due = not self.trees and len(self.window) == self.window_size
        replacement_due = bool(self.trees) and (
            self._samples_since_retrain >= self.retrain_interval
        )
        if initial_fit_due or replacement_due:
            self._fit_forest()
        self._schema.commit(prepared)

    def _fit_forest(self) -> None:
        """Replace the complete forest using the current reference window."""
        data = np.asarray(self.window, dtype=np.float64)
        if len(data) <= 1:
            return

        sample_size = min(self.max_samples, len(data))
        trees: list[_ASDNode] = []
        for _ in range(self.n_estimators):
            indices = self.rng.choice(len(data), size=sample_size, replace=False)
            trees.append(self._build_tree(data[indices]))

        self.trees = trees
        self.c_n = self._compute_c(sample_size)
        self._samples_since_retrain = 0
        self._fits_completed += 1
        self._last_fit_window = data.copy()

    def _build_tree(self, data_arr: np.ndarray) -> _ASDNode:
        """Build one isolation tree from a sampled window."""
        indices = np.arange(data_arr.shape[0])
        max_height = math.ceil(math.log2(max(data_arr.shape[0], 2)))
        return self._build_tree_recursive(data_arr, indices, max_height)

    def _build_tree_recursive(
        self,
        data_arr: np.ndarray,
        indices: np.ndarray,
        max_height: int,
        current_height: int = 0,
    ) -> _ASDNode:
        """Recursively build an isolation tree."""
        n_samples = len(indices)
        if n_samples <= 1 or current_height >= max_height:
            return _ASDLeaf(correction=self._compute_c(n_samples))

        node_values = data_arr[indices]
        variable_features = np.flatnonzero(np.ptp(node_values, axis=0) > 0)
        if len(variable_features) == 0:
            return _ASDLeaf(correction=self._compute_c(n_samples))

        feature_index = int(self.rng.choice(variable_features))
        feature_values = data_arr[indices, feature_index]
        split_value = float(
            self.rng.uniform(np.min(feature_values), np.max(feature_values))
        )
        left_mask = feature_values < split_value
        return _ASDBranch(
            feature=feature_index,
            threshold=split_value,
            left=self._build_tree_recursive(
                data_arr,
                indices[left_mask],
                max_height,
                current_height + 1,
            ),
            right=self._build_tree_recursive(
                data_arr,
                indices[~left_mask],
                max_height,
                current_height + 1,
            ),
        )

    @staticmethod
    def _compute_path_length(values: np.ndarray, tree: _ASDNode) -> float:
        """Compute one sample's path length through an isolation tree."""
        depth = 0
        current_node = tree
        while isinstance(current_node, _ASDBranch):
            feature_value = values[current_node.feature]
            if feature_value < current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
            depth += 1
        return float(depth + current_node.correction)

    def score_one(self, x: dict[str, float]) -> float:
        """Compute the current replacement forest's anomaly score."""
        prepared = self._schema.preview(x)
        if not self.trees or self.c_n <= 0.0:
            return 0.0

        average_path = float(
            np.mean(
                [
                    self._compute_path_length(prepared.values, tree)
                    for tree in self.trees
                ]
            )
        )
        return float(2.0 ** (-average_path / self.c_n))

    def __repr__(self) -> str:
        """Return a string representation of iForestASD."""
        return (
            f"ASDIsolationForest(n_estimators={self.n_estimators}, "
            f"max_samples={self.max_samples}, window_size={self.window_size}, "
            f"retrain_interval={self.retrain_interval}, n_trees={len(self.trees)}, "
            f"fits_completed={self._fits_completed}, seed={self.seed})"
        )
