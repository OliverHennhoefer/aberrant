"""STREamRHF tree-based streaming anomaly detector."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from aberrant.base.model import BaseModel

_OPEN_UNIT_LOW = float(np.nextafter(0.0, 1.0))


def _empty_moments(n_features: int) -> np.ndarray:
    """Create per-feature ``mean, M2, M3, M4, n`` moment state."""
    return np.zeros((n_features, 5), dtype=np.float64)


def _update_moments(moments: np.ndarray, point: np.ndarray) -> None:
    """Update moments with the online formulas used by the author code."""
    for feature, value in enumerate(point):
        mean, m2, m3, m4, n = moments[feature]
        previous_n = n
        n += 1.0
        delta = value - mean
        delta_n = delta / n
        delta_n_sq = delta_n * delta_n
        term = delta * delta_n * previous_n
        mean += delta_n
        m4 += (
            term * delta_n_sq * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n_sq * m2
            - 4.0 * delta_n * m3
        )
        m3 += term * delta_n * (n - 2.0) - 3.0 * delta_n * m2
        m2 += term
        moments[feature] = (mean, m2, m3, m4, n)


def _moments_for(points: list[np.ndarray], n_features: int) -> np.ndarray:
    moments = _empty_moments(n_features)
    for point in points:
        _update_moments(moments, point)
    return moments


def _kurtosis_weights(moments: np.ndarray) -> np.ndarray:
    """Return ``log(kurtosis + 1)`` weights from STREamRHF."""
    m2 = moments[:, 1]
    m4 = moments[:, 3]
    n = moments[:, 4]
    kurtosis = np.divide(
        n * m4,
        m2 * m2,
        out=np.zeros_like(m2),
        where=(m2 != 0.0) & (m4 != 0.0),
    )
    return np.log1p(np.maximum(kurtosis, 0.0))


@dataclass
class _RHFNode:
    """One STREamRHF node."""

    depth: int
    node_id: int
    moments: np.ndarray
    points: list[np.ndarray] | None = None
    split_feature: int | None = None
    split_value: float | None = None
    left: _RHFNode | None = None
    right: _RHFNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.split_feature is None

    @property
    def size(self) -> int:
        if self.is_leaf:
            return 0 if self.points is None else len(self.points)
        left_size = 0 if self.left is None else self.left.size
        right_size = 0 if self.right is None else self.right.size
        return left_size + right_size

    def collect_points(self) -> list[np.ndarray]:
        """Collect all points below this node."""
        if self.is_leaf:
            return [] if self.points is None else list(self.points)
        points: list[np.ndarray] = []
        if self.left is not None:
            points.extend(self.left.collect_points())
        if self.right is not None:
            points.extend(self.right.collect_points())
        return points


class _RandomHistogramTree:
    """Kurtosis-weighted random histogram tree."""

    def __init__(
        self,
        max_depth: int,
        n_features: int,
        seed_sequence: np.random.SeedSequence,
    ) -> None:
        self.max_depth = max_depth
        self.n_features = n_features
        self.seed_sequence = seed_sequence
        self._node_random: dict[int, tuple[float, float]] = {}
        self.root: _RHFNode | None = None

    def build(self, points: list[np.ndarray]) -> None:
        """Build a complete tree from one reference window."""
        self.root = self._build_node(points, depth=0, node_id=0)

    def _random_values(self, node_id: int) -> tuple[float, float]:
        """Return fixed feature/split quantiles for one visited node."""
        cached = self._node_random.get(node_id)
        if cached is None:
            words: list[int] = []
            remaining = node_id
            while remaining:
                words.append(remaining & 0xFFFFFFFF)
                remaining >>= 32
            if not words:
                words.append(0)
            node_seed = np.random.SeedSequence(
                entropy=self.seed_sequence.entropy,
                spawn_key=(*self.seed_sequence.spawn_key, len(words), *words),
                pool_size=self.seed_sequence.pool_size,
            )
            values = np.random.default_rng(node_seed).uniform(
                _OPEN_UNIT_LOW,
                1.0,
                size=2,
            )
            cached = (float(values[0]), float(values[1]))
            self._node_random[node_id] = cached
        return cached

    def _selected_feature(self, moments: np.ndarray, node_id: int) -> int | None:
        weights = _kurtosis_weights(moments)
        total = float(np.sum(weights))
        if total <= 0.0:
            return None
        feature_random, _ = self._random_values(node_id)
        target = feature_random * total
        feature = int(np.searchsorted(np.cumsum(weights), target, side="left"))
        return min(feature, self.n_features - 1)

    def _build_node(
        self,
        points: list[np.ndarray],
        *,
        depth: int,
        node_id: int,
    ) -> _RHFNode:
        moments = _moments_for(points, self.n_features)
        leaf = _RHFNode(
            depth=depth,
            node_id=node_id,
            moments=moments,
            points=list(points),
        )
        if len(points) <= 1 or depth >= self.max_depth:
            return leaf

        feature = self._selected_feature(moments, node_id)
        if feature is None:
            return leaf

        values = np.asarray([point[feature] for point in points], dtype=np.float64)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if minimum == maximum:
            return leaf
        _, split_random = self._random_values(node_id)
        split_value = minimum + split_random * (maximum - minimum)
        left_points = [point for point in points if point[feature] <= split_value]
        right_points = [point for point in points if point[feature] > split_value]
        if not left_points or not right_points:
            return leaf

        return _RHFNode(
            depth=depth,
            node_id=node_id,
            moments=moments,
            split_feature=feature,
            split_value=split_value,
            left=self._build_node(
                left_points,
                depth=depth + 1,
                node_id=2 * node_id + 1,
            ),
            right=self._build_node(
                right_points,
                depth=depth + 1,
                node_id=2 * node_id + 2,
            ),
        )

    def insert(self, point: np.ndarray) -> int:
        """Insert one point, rebuilding a subtree if its split feature changes."""
        if self.root is None:
            self.build([point])
        else:
            self.root = self._insert_node(self.root, point)
        return self.leaf_size(point)

    def _insert_node(self, node: _RHFNode, point: np.ndarray) -> _RHFNode:
        if node.is_leaf:
            points = node.collect_points()
            points.append(point)
            return self._build_node(points, depth=node.depth, node_id=node.node_id)

        updated_moments = node.moments.copy()
        _update_moments(updated_moments, point)
        selected_feature = self._selected_feature(updated_moments, node.node_id)
        if selected_feature != node.split_feature:
            points = node.collect_points()
            points.append(point)
            return self._build_node(points, depth=node.depth, node_id=node.node_id)

        node.moments = updated_moments
        if point[node.split_feature] <= node.split_value:  # type: ignore[index,operator]
            if node.left is None:
                raise RuntimeError("Split node has no left child")
            node.left = self._insert_node(node.left, point)
        else:
            if node.right is None:
                raise RuntimeError("Split node has no right child")
            node.right = self._insert_node(node.right, point)
        return node

    def leaf_size(self, point: np.ndarray) -> int:
        """Return the size of the leaf reached by point."""
        node = self.root
        if node is None:
            return 0
        while not node.is_leaf:
            if node.split_feature is None or node.split_value is None:
                raise RuntimeError("Invalid split node")
            if point[node.split_feature] <= node.split_value:
                if node.left is None:
                    raise RuntimeError("Split node has no left child")
                node = node.left
            else:
                if node.right is None:
                    raise RuntimeError("Split node has no right child")
                node = node.right
        return node.size


class StreamRandomHistogramForest(BaseModel):
    """
    STREamRHF tree-based unsupervised anomaly detector.

    The forest follows the authors' implementation structure:

    - split attributes are sampled proportionally to ``log(kurtosis + 1)``,
    - every node keeps fixed random quantiles for attribute and split selection,
    - an insertion rebuilds a subtree when its selected attribute changes,
    - each completed current window replaces the reference forest,
    - anomaly score is the sum of ``log(n / leaf_size)`` across trees after
      candidate insertion.

    ``score_one`` inserts into copies of the trees so it reproduces the
    candidate-inclusive score without mutating learned state.

    References:
        Nesic, S., et al. (2022). STREamRHF: Tree-Based Unsupervised Anomaly
        Detection for Data Streams.
        https://doi.org/10.1109/AICCSA56895.2022.10017876
        Original implementation: https://github.com/stefannesic/streamRHF
    """

    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 5,
        window_size: int = 256,
        seed: int | None = None,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if window_size <= 1:
            raise ValueError("window_size must be greater than 1")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.window_size = window_size
        self.seed = seed
        self.feature_names: list[str] | None = None

        self._tree_seed_sequences = np.random.SeedSequence(seed).spawn(n_estimators)
        self._initial_window: list[np.ndarray] = []
        self._current_window: list[np.ndarray] = []
        self._trees: list[_RandomHistogramTree] = []
        self._forest_size = 0

    def _set_or_validate_schema(
        self, x: dict[str, float], *, allow_initialize: bool
    ) -> None:
        if not x:
            raise ValueError("Input dictionary cannot be empty")
        for key, value in x.items():
            if not isinstance(key, str):
                raise ValueError("All feature keys must be strings")
            if not isinstance(value, int | float | np.number):
                raise ValueError(f"Feature '{key}' is not numeric")
            if not np.isfinite(float(value)):
                raise ValueError(f"Feature '{key}' must be finite")

        if self.feature_names is None:
            if allow_initialize:
                self.feature_names = sorted(x)
            return
        if set(x) != set(self.feature_names):
            expected = ", ".join(self.feature_names)
            received = ", ".join(sorted(x))
            raise ValueError(
                "Inconsistent feature keys. "
                f"Expected [{expected}], received [{received}]."
            )

    def _vectorize(self, x: dict[str, float]) -> np.ndarray:
        if self.feature_names is None:
            raise RuntimeError("Feature schema is not initialized")
        return np.fromiter(
            (float(x[feature]) for feature in self.feature_names),
            dtype=np.float64,
            count=len(self.feature_names),
        )

    def _build_forest(self, points: list[np.ndarray]) -> None:
        if self.feature_names is None:
            raise RuntimeError("Forest schema is not initialized")
        self._trees = []
        for tree_index in range(self.n_estimators):
            tree = _RandomHistogramTree(
                self.max_depth,
                len(self.feature_names),
                self._tree_seed_sequences[tree_index],
            )
            tree.build(points)
            self._trees.append(tree)
        self._forest_size = len(points)

    def learn_one(self, x: dict[str, float]) -> None:
        """Insert one sample and replace the forest at window boundaries."""
        self._set_or_validate_schema(x, allow_initialize=True)
        point = self._vectorize(x)

        if not self._trees:
            self._initial_window.append(point)
            if len(self._initial_window) == self.window_size:
                self._build_forest(self._initial_window)
                self._initial_window = []
            return

        for tree in self._trees:
            tree.insert(point)
        self._forest_size += 1
        self._current_window.append(point)
        if len(self._current_window) == self.window_size:
            self._build_forest(self._current_window)
            self._current_window = []

    def score_one(self, x: dict[str, float]) -> float:
        """Return the candidate-inclusive STREamRHF leaf-mass score."""
        self._set_or_validate_schema(x, allow_initialize=False)
        if not self._trees:
            return 0.0
        point = self._vectorize(x)
        candidate_size = self._forest_size + 1

        score = 0.0
        for learned_tree in self._trees:
            tree = copy.deepcopy(learned_tree)
            leaf_size = tree.insert(point)
            if leaf_size > 0:
                score += math.log(float(candidate_size) / float(leaf_size))
        return float(score)

    def __repr__(self) -> str:
        return (
            "StreamRandomHistogramForest("
            f"n_estimators={self.n_estimators}, max_depth={self.max_depth}, "
            f"window_size={self.window_size}, seed={self.seed}, "
            f"initialized={bool(self._trees)}, forest_size={self._forest_size})"
        )
