"""Mondrian-tree isolation forest for streaming anomaly detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema


def _average_path_length(n: int) -> float:
    """
    Expected path length of unsuccessful BST search for `n` samples.

    This is the standard isolation-forest normalization term c(n).
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    euler_mascheroni = 0.5772156649015329
    return 2.0 * (math.log(n - 1) + euler_mascheroni) - 2.0 * (n - 1) / n


@dataclass(slots=True)
class _MondrianBlock:
    """Bounds and population shared by every non-empty block."""

    min: np.ndarray
    max: np.ndarray
    count: int
    split_time: float


class MondrianLeaf(_MondrianBlock):
    """An occupied terminal block, with no split or child metadata."""

    __slots__ = ()

    @classmethod
    def from_point(cls, point: np.ndarray, split_time: float) -> MondrianLeaf:
        return cls(point.copy(), point.copy(), 1, split_time)

    def is_leaf(self) -> bool:
        return True

    def update_stats(self, point: np.ndarray) -> None:
        np.minimum(self.min, point, out=self.min)
        np.maximum(self.max, point, out=self.max)
        self.count += 1


@dataclass(slots=True)
class MondrianBranch(_MondrianBlock):
    """A split with exactly two occupied children."""

    split_feature: int
    split_threshold: float
    left_child: MondrianNode
    right_child: MondrianNode

    @classmethod
    def join(
        cls,
        left: MondrianNode,
        right: MondrianNode,
        split_feature: int,
        split_threshold: float,
        split_time: float,
    ) -> MondrianBranch:
        return cls(
            np.minimum(left.min, right.min),
            np.maximum(left.max, right.max),
            left.count + right.count,
            split_time,
            split_feature,
            split_threshold,
            left,
            right,
        )

    def is_leaf(self) -> bool:
        return False

    def recompute_from_children(self) -> None:
        np.minimum(self.left_child.min, self.right_child.min, out=self.min)
        np.maximum(self.left_child.max, self.right_child.max, out=self.max)
        self.count = self.left_child.count + self.right_child.count


MondrianNode = MondrianLeaf | MondrianBranch


class MondrianTree:
    """
    One online Mondrian tree.

    The tree follows the Mondrian extension logic using split-time sampling
    with exponential clocks over block extensions.
    """

    def __init__(
        self, selected_indices: np.ndarray, lambda_: float, rng: np.random.Generator
    ) -> None:
        self.selected_indices = selected_indices
        self.lambda_ = lambda_
        self.rng = rng
        self._projected_buffer = np.empty(len(selected_indices), dtype=np.float64)
        self.root: MondrianNode | None = None
        self.n_samples = 0

    def learn_one(self, x_projected: np.ndarray) -> None:
        """Insert one projected point with online Mondrian extension."""
        if self.root is None:
            self.root = MondrianLeaf.from_point(x_projected, self.lambda_)
        else:
            self.root = self._extend_block(
                self.root, x_projected, parent_split_time=0.0
            )
        self.n_samples += 1

    def learn_one_from_global(self, global_features: np.ndarray) -> None:
        """Project one global feature vector and update this tree."""
        np.take(global_features, self.selected_indices, out=self._projected_buffer)
        self.learn_one(self._projected_buffer)

    def _extend_block(
        self,
        node: MondrianNode,
        x_values: np.ndarray,
        parent_split_time: float,
    ) -> MondrianNode:
        """
        Extend one Mondrian block with a point.

        If a sampled split time occurs before the node's own split time, create
        a new parent above this node; otherwise recurse or absorb into leaf.
        """
        lower_extension = np.maximum(node.min - x_values, 0.0)
        upper_extension = np.maximum(x_values - node.max, 0.0)
        extension_weights = lower_extension + upper_extension
        extension_rate = float(np.sum(extension_weights))
        sampled_time = self._sample_exponential(extension_rate)

        if parent_split_time + sampled_time < node.split_time:
            split_feature = self._sample_split_feature(extension_weights)
            split_threshold = self._sample_split_threshold(
                x_values=x_values,
                node=node,
                split_feature=split_feature,
            )

            new_leaf = MondrianLeaf.from_point(x_values, self.lambda_)
            left, right = (
                (new_leaf, node)
                if x_values[split_feature] <= split_threshold
                else (node, new_leaf)
            )
            return MondrianBranch.join(
                left,
                right,
                split_feature,
                split_threshold,
                parent_split_time + sampled_time,
            )

        if isinstance(node, MondrianLeaf):
            node.update_stats(x_values)
            return node

        if x_values[node.split_feature] <= node.split_threshold:
            node.left_child = self._extend_block(
                node.left_child,
                x_values,
                parent_split_time=node.split_time,
            )
        else:
            node.right_child = self._extend_block(
                node.right_child,
                x_values,
                parent_split_time=node.split_time,
            )

        node.recompute_from_children()
        return node

    def _sample_exponential(self, rate: float) -> float:
        """Sample `Exp(rate)` and return `inf` when the rate is zero."""
        if rate <= 0.0:
            return math.inf
        return float(self.rng.exponential(scale=1.0 / rate))

    def _sample_split_feature(self, extension_weights: np.ndarray) -> int:
        """Sample split dimension proportional to extension magnitudes."""
        total = float(np.sum(extension_weights))
        if total <= 0.0:
            return 0
        offset = float(self.rng.random()) * total
        cumulative = 0.0
        for idx, weight in enumerate(extension_weights):
            cumulative += float(weight)
            if offset <= cumulative:
                return idx
        return len(extension_weights) - 1

    def _sample_split_threshold(
        self,
        x_values: np.ndarray,
        node: MondrianNode,
        split_feature: int,
    ) -> float:
        """Sample split threshold on the extension interval for one feature."""
        value = float(x_values[split_feature])
        lower = float(node.min[split_feature])
        upper = float(node.max[split_feature])

        if value > upper:
            return float(self.rng.uniform(upper, value))
        if value < lower:
            return float(self.rng.uniform(value, lower))
        return value

    def score_one(self, x_projected: np.ndarray) -> float:
        """Return path length plus leaf-size adjustment for one point."""
        if self.root is None:
            return 0.0

        path_length = 0
        current_node = self.root

        while isinstance(current_node, MondrianBranch):
            path_length += 1
            if x_projected[current_node.split_feature] <= current_node.split_threshold:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child

        return float(path_length + _average_path_length(current_node.count))

    def score_one_from_global(self, global_features: np.ndarray) -> float:
        """Project one global feature vector and score against this tree."""
        np.take(global_features, self.selected_indices, out=self._projected_buffer)
        return self.score_one(self._projected_buffer)


class MondrianIsolationForest(BaseModel):
    """
    Online isolation forest built from Mondrian trees.

    The tree update follows online Mondrian block-extension mechanics, while
    anomaly scoring uses Isolation Forest path-length normalization. The
    original Mondrian Forest is a supervised classification model and does not
    define this anomaly score, so this class is a custom hybrid.

    Args:
        n_estimators: Number of trees in the forest.
        subspace_size: Number of features sampled per tree.
        lambda_: Mondrian lifetime budget.
        seed: Random seed for reproducibility.

    References:
        Lakshminarayanan, B., Roy, D. M., & Teh, Y. W. (2014). Mondrian
        Forests: Efficient Online Random Forests.
        https://proceedings.neurips.cc/paper_files/paper/2014/hash/195f15384c2a79cedf293e4a847ce85c-Abstract.html
    """

    def __init__(
        self,
        n_estimators: int = 100,
        subspace_size: int = 256,
        lambda_: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if subspace_size <= 0:
            raise ValueError("subspace_size must be positive")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")

        self.n_estimators = n_estimators
        self.subspace_size = subspace_size
        self.lambda_ = lambda_
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.trees: list[MondrianTree] = []
        self.n_samples = 0
        self._schema = FeatureSchema()

    def learn_one(self, x: dict[str, float]) -> None:
        """Update all trees with one feature dictionary."""
        prepared = self._schema.preview(x)
        if not self._schema.is_established:
            self._initialize_features(len(prepared.names))

        for tree in self.trees:
            tree.learn_one_from_global(prepared.values)

        self.n_samples += 1
        self._schema.commit(prepared)

    def score_one(self, x: dict[str, float]) -> float:
        """Compute normalized anomaly score in [0, 1]."""
        prepared = self._schema.preview(x)
        if not self._schema.is_established or self.n_samples <= 1:
            return 0.0
        if not self.trees:
            return 0.0

        path_length_sum = 0.0
        for tree in self.trees:
            path_length_sum += tree.score_one_from_global(prepared.values)
        avg_path_length = path_length_sum / len(self.trees)
        c_factor = self._compute_c_factor()
        if c_factor <= 0.0:
            return 0.0

        score = 2.0 ** (-avg_path_length / c_factor)
        return float(np.clip(score, 0.0, 1.0))

    def _initialize_features(self, feature_count: int) -> None:
        """Create all trees for a validated feature width."""
        self.subspace_size = min(self.subspace_size, feature_count)
        self.trees = []

        max_seed = np.iinfo(np.int64).max
        for _ in range(self.n_estimators):
            selected_indices = np.asarray(
                self.rng.choice(
                    feature_count,
                    size=self.subspace_size,
                    replace=False,
                ),
                dtype=np.int64,
            )
            tree_seed = int(self.rng.integers(0, max_seed))
            tree_rng = np.random.default_rng(tree_seed)
            self.trees.append(MondrianTree(selected_indices, self.lambda_, tree_rng))

    def _compute_c_factor(self) -> float:
        """Compute forest-level isolation normalization term."""
        return _average_path_length(self.n_samples)

    def __repr__(self) -> str:
        """Return a string representation of the MondrianIsolationForest."""
        return (
            f"MondrianIsolationForest(n_estimators={self.n_estimators}, "
            f"subspace_size={self.subspace_size}, "
            f"lambda_={self.lambda_}, seed={self.seed})"
        )
