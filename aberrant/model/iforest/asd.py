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


@dataclass(frozen=True, slots=True)
class _FlatForest:
    """
    Every tree of a forest in flat arrays, to walk all trees at once.

    Node ``i`` splits on ``features[i]`` at ``thresholds[i]``. Its child for a
    value below the threshold is ``children[2 * i + 1]`` and otherwise
    ``children[2 * i]``. A leaf is its own child, so ``height`` steps take
    every walk to its leaf, where ``path_lengths`` holds the leaf's depth plus
    its correction.
    """

    roots: np.ndarray
    features: np.ndarray
    thresholds: np.ndarray
    children: np.ndarray
    path_lengths: np.ndarray
    height: int

    def path_lengths_of(self, values: np.ndarray) -> np.ndarray:
        """Return the path length of ``values`` through every tree."""
        nodes = self.roots
        for _ in range(self.height):
            below = values[self.features[nodes]] < self.thresholds[nodes]
            nodes = self.children[2 * nodes + below]
        path_lengths: np.ndarray = self.path_lengths[nodes]
        return path_lengths


class _FlatNodes:
    """Collect the nodes of growing trees in preorder for a ``_FlatForest``."""

    __slots__ = ("children", "features", "height", "path_lengths", "thresholds")

    def __init__(self) -> None:
        self.features: list[int] = []
        self.thresholds: list[float] = []
        self.children: list[int] = []
        self.path_lengths: list[float] = []
        self.height = 0

    def add_leaf(self, depth: int, correction: float) -> None:
        """Append a leaf at ``depth`` that points to itself."""
        index = len(self.features)
        self.features.append(0)
        self.thresholds.append(0.0)
        self.children += (index, index)
        self.path_lengths.append(float(depth + correction))
        self.height = max(self.height, depth)

    def add_branch(self, feature: int, threshold: float) -> int:
        """Append a branch whose children ``link`` sets later; return its index."""
        index = len(self.features)
        self.features.append(feature)
        self.thresholds.append(threshold)
        self.children += (index, index)
        self.path_lengths.append(0.0)
        return index

    def link(self, index: int, left: int, right: int) -> None:
        """Set the children of the branch at ``index``."""
        self.children[2 * index] = right
        self.children[2 * index + 1] = left

    def freeze(self, roots: list[int]) -> _FlatForest:
        """Return the collected trees, rooted at ``roots``, as arrays."""
        return _FlatForest(
            roots=np.array(roots, dtype=np.intp),
            features=np.array(self.features, dtype=np.intp),
            thresholds=np.array(self.thresholds, dtype=np.float64),
            children=np.array(self.children, dtype=np.intp),
            path_lengths=np.array(self.path_lengths, dtype=np.float64),
            height=self.height,
        )


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
        self._flat: _FlatForest | None = None
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
        nodes = _FlatNodes()
        roots: list[int] = []
        for _ in range(self.n_estimators):
            indices = self.rng.choice(len(data), size=sample_size, replace=False)
            roots.append(len(nodes.features))
            trees.append(self._build_tree(data[indices], nodes))

        self.trees = trees
        self._flat = nodes.freeze(roots)
        self.c_n = self._compute_c(sample_size)
        self._samples_since_retrain = 0
        self._fits_completed += 1
        self._last_fit_window = data.copy()

    def _build_tree(
        self, data_arr: np.ndarray, nodes: _FlatNodes | None = None
    ) -> _ASDNode:
        """Build one isolation tree from a sampled window, also into ``nodes``."""
        max_height = math.ceil(math.log2(max(data_arr.shape[0], 2)))
        # One row per feature, so that per-feature reductions run along rows.
        return self._build_tree_recursive(
            np.ascontiguousarray(data_arr.T),
            max_height,
            0,
            _FlatNodes() if nodes is None else nodes,
        )

    def _build_tree_recursive(
        self,
        columns: np.ndarray,
        max_height: int,
        current_height: int,
        nodes: _FlatNodes,
    ) -> _ASDNode:
        """Recursively build an isolation tree from a node's feature columns."""
        n_samples = columns.shape[1]
        if n_samples <= 1 or current_height >= max_height:
            return self._leaf(n_samples, current_height, nodes)

        # A feature varies exactly when its peak-to-peak range is positive.
        lows = np.minimum.reduce(columns, axis=1)
        highs = np.maximum.reduce(columns, axis=1)
        variable_features = (highs > lows).nonzero()[0]
        if len(variable_features) == 0:
            return self._leaf(n_samples, current_height, nodes)

        # Generator.choice draws one uniform index into the array.
        feature_index = int(
            variable_features[self.rng.integers(0, len(variable_features))]
        )
        split_value = float(self.rng.uniform(lows[feature_index], highs[feature_index]))
        left_mask = columns[feature_index] < split_value
        index = nodes.add_branch(feature_index, split_value)
        left = self._build_tree_recursive(
            columns.compress(left_mask, axis=1), max_height, current_height + 1, nodes
        )
        right_index = len(nodes.features)
        right = self._build_tree_recursive(
            columns.compress(~left_mask, axis=1), max_height, current_height + 1, nodes
        )
        nodes.link(index, left=index + 1, right=right_index)
        return _ASDBranch(
            feature=feature_index, threshold=split_value, left=left, right=right
        )

    def _leaf(self, n_samples: int, depth: int, nodes: _FlatNodes) -> _ASDLeaf:
        """Build a leaf for ``n_samples`` samples at ``depth``."""
        correction = self._compute_c(n_samples)
        nodes.add_leaf(depth, correction)
        return _ASDLeaf(correction=correction)

    def score_one(self, x: dict[str, float]) -> float:
        """Compute the current replacement forest's anomaly score."""
        prepared = self._schema.preview(x)
        if self._flat is None or self.c_n <= 0.0:
            return 0.0

        average_path = float(np.mean(self._flat.path_lengths_of(prepared.values)))
        return float(2.0 ** (-average_path / self.c_n))

    def __repr__(self) -> str:
        """Return a string representation of iForestASD."""
        return (
            f"ASDIsolationForest(n_estimators={self.n_estimators}, "
            f"max_samples={self.max_samples}, window_size={self.window_size}, "
            f"retrain_interval={self.retrain_interval}, n_trees={len(self.trees)}, "
            f"fits_completed={self._fits_completed}, seed={self.seed})"
        )
