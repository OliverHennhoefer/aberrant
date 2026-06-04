"""Online Mondrian Forest for streaming anomaly detection."""

from __future__ import annotations

import math

import numpy as np

from aberrant.base.model import BaseModel


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


class MondrianNode:
    """
    Node in an online Mondrian tree.

    Nodes store an axis-aligned block (`min`, `max`), subtree sample count,
    split metadata for internal nodes, and split time `split_time`.
    """

    __slots__ = [
        "split_feature",
        "split_threshold",
        "split_time",
        "left_child",
        "right_child",
        "is_leaf_",
        "min",
        "max",
        "count",
    ]

    def __init__(self, split_time: float = math.inf) -> None:
        """Initialize a Mondrian node."""
        self.split_feature: int | None = None
        self.split_threshold: float | None = None
        self.split_time: float = split_time
        self.left_child: MondrianNode | None = None
        self.right_child: MondrianNode | None = None
        self.is_leaf_: bool = True
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None
        self.count: int = 0

    def is_leaf(self) -> bool:
        """Return whether this node is a leaf."""
        return self.is_leaf_

    def update_stats(self, x_values: np.ndarray) -> None:
        """Update this node's bounding block and sample count with one point."""
        if self.count == 0:
            self.min = x_values.copy()
            self.max = x_values.copy()
        else:
            np.minimum(self.min, x_values, out=self.min)
            np.maximum(self.max, x_values, out=self.max)
        self.count += 1

    def recompute_from_children(self) -> None:
        """Recompute block bounds and count from both children."""
        if self.left_child is None or self.right_child is None:
            raise RuntimeError("Internal node must have both children")
        if self.left_child.min is None or self.left_child.max is None:
            raise RuntimeError("Left child has incomplete bounds")
        if self.right_child.min is None or self.right_child.max is None:
            raise RuntimeError("Right child has incomplete bounds")

        if self.min is None:
            self.min = np.minimum(self.left_child.min, self.right_child.min)
        else:
            np.minimum(self.left_child.min, self.right_child.min, out=self.min)
        if self.max is None:
            self.max = np.maximum(self.left_child.max, self.right_child.max)
        else:
            np.maximum(self.left_child.max, self.right_child.max, out=self.max)
        self.count = self.left_child.count + self.right_child.count


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
        self.root = MondrianNode(split_time=lambda_)
        self.n_samples = 0

    def learn_one(self, x_projected: np.ndarray) -> None:
        """Insert one projected point with online Mondrian extension."""
        if self.root.count == 0:
            self.root.update_stats(x_projected)
            self.n_samples += 1
            return

        self.root = self._extend_block(self.root, x_projected, parent_split_time=0.0)
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
        if node.min is None or node.max is None:
            node.update_stats(x_values)
            return node

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

            new_leaf = MondrianNode(split_time=self.lambda_)
            new_leaf.update_stats(x_values)

            parent = MondrianNode(split_time=parent_split_time + sampled_time)
            parent.split_feature = split_feature
            parent.split_threshold = split_threshold
            parent.is_leaf_ = False

            if x_values[split_feature] <= split_threshold:
                parent.left_child = new_leaf
                parent.right_child = node
            else:
                parent.left_child = node
                parent.right_child = new_leaf

            parent.recompute_from_children()
            return parent

        if node.is_leaf():
            node.update_stats(x_values)
            return node

        if node.split_feature is None or node.split_threshold is None:
            raise RuntimeError("Internal node missing split metadata")

        if x_values[node.split_feature] <= node.split_threshold:
            if node.left_child is None:
                raise RuntimeError("Internal node missing left child")
            node.left_child = self._extend_block(
                node.left_child,
                x_values,
                parent_split_time=node.split_time,
            )
        else:
            if node.right_child is None:
                raise RuntimeError("Internal node missing right child")
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
        if node.min is None or node.max is None:
            return float(x_values[split_feature])

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
        if self.root.count == 0:
            return 0.0

        path_length = 0
        current_node = self.root

        while not current_node.is_leaf():
            path_length += 1
            if (
                current_node.split_feature is None
                or current_node.split_threshold is None
            ):
                raise RuntimeError("Internal node missing split metadata")
            if x_projected[current_node.split_feature] <= current_node.split_threshold:
                if current_node.left_child is None:
                    raise RuntimeError("Internal node missing left child")
                current_node = current_node.left_child
            else:
                if current_node.right_child is None:
                    raise RuntimeError("Internal node missing right child")
                current_node = current_node.right_child

        return float(path_length + _average_path_length(current_node.count))

    def score_one_from_global(self, global_features: np.ndarray) -> float:
        """Project one global feature vector and score against this tree."""
        np.take(global_features, self.selected_indices, out=self._projected_buffer)
        return self.score_one(self._projected_buffer)


class MondrianForest(BaseModel):
    """
    Online Mondrian Forest for anomaly detection.

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
        self._feature_order: list[str] | None = None
        self._feature_to_index: dict[str, int] | None = None
        self._feature_count: int = 0
        self._feature_vector: np.ndarray | None = None

    def learn_one(self, x: dict[str, float]) -> None:
        """Update all trees with one feature dictionary."""
        self._validate_input(x)
        if self._feature_order is None:
            self._initialize_features(x)

        if self._feature_order is None:
            raise RuntimeError("Feature initialization failed")

        global_features = self._dict_to_vector(x)

        for tree in self.trees:
            tree.learn_one_from_global(global_features)

        self.n_samples += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute normalized anomaly score in [0, 1]."""
        self._validate_input(x)
        if self._feature_order is None or self.n_samples <= 1:
            return 0.0
        if not self.trees:
            return 0.0

        global_features = self._dict_to_vector(x)

        path_length_sum = 0.0
        for tree in self.trees:
            path_length_sum += tree.score_one_from_global(global_features)
        avg_path_length = path_length_sum / len(self.trees)
        c_factor = self._compute_c_factor()
        if c_factor <= 0.0:
            return 0.0

        score = 2.0 ** (-avg_path_length / c_factor)
        return float(np.clip(score, 0.0, 1.0))

    def _initialize_features(self, x: dict[str, float]) -> None:
        """Initialize feature order and create all trees."""
        if not x:
            raise ValueError("Input dictionary cannot be empty")

        self._feature_order = sorted(x.keys())
        self._feature_count = len(self._feature_order)
        self.subspace_size = min(self.subspace_size, self._feature_count)
        self._feature_to_index = {
            feature: idx for idx, feature in enumerate(self._feature_order)
        }
        self._feature_vector = np.empty(self._feature_count, dtype=np.float64)
        self.trees = []

        max_seed = np.iinfo(np.int64).max
        for _ in range(self.n_estimators):
            selected_indices = np.asarray(
                self.rng.choice(
                    self._feature_count, size=self.subspace_size, replace=False
                ),
                dtype=np.int64,
            )
            tree_seed = int(self.rng.integers(0, max_seed))
            tree_rng = np.random.default_rng(tree_seed)
            self.trees.append(MondrianTree(selected_indices, self.lambda_, tree_rng))

    def _validate_input(self, x: dict[str, float]) -> None:
        """Validate one-sample dictionary input."""
        if not x:
            raise ValueError("Input dictionary cannot be empty")

        feature_to_index = self._feature_to_index
        if feature_to_index is not None and len(x) != self._feature_count:
            expected_keys = ", ".join(self._feature_order or [])
            received_keys = ", ".join(sorted(x.keys()))
            raise ValueError(
                "Inconsistent feature keys. "
                f"Expected [{expected_keys}], received [{received_keys}]."
            )

        has_bad_key = False
        for key, value in x.items():
            if not isinstance(key, str):
                raise ValueError("All feature keys must be strings")
            if not isinstance(value, int | float | np.number):
                raise ValueError(f"Feature '{key}' is not numeric")
            if not np.isfinite(float(value)):
                raise ValueError(f"Feature '{key}' must be finite")
            if feature_to_index is not None and key not in feature_to_index:
                has_bad_key = True

        if has_bad_key:
            expected_keys = ", ".join(self._feature_order)
            received_keys = ", ".join(sorted(x.keys()))
            raise ValueError(
                "Inconsistent feature keys. "
                f"Expected [{expected_keys}], received [{received_keys}]."
            )

    def _dict_to_vector(self, x: dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to stable-order vector without reallocating."""
        if self._feature_order is None or self._feature_vector is None:
            raise RuntimeError("Feature initialization failed")

        buffer = self._feature_vector
        for idx, feature in enumerate(self._feature_order):
            buffer[idx] = float(x[feature])
        return buffer

    def _compute_c_factor(self) -> float:
        """Compute forest-level isolation normalization term."""
        return _average_path_length(self.n_samples)

    def __repr__(self) -> str:
        """Return a string representation of the MondrianForest."""
        return (
            f"MondrianForest(n_estimators={self.n_estimators}, "
            f"subspace_size={self.subspace_size}, "
            f"lambda_={self.lambda_}, seed={self.seed})"
        )
