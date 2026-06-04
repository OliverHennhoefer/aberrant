from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy import argsort, empty, inf, log, ndarray, sort, split, vstack, zeros

from aberrant.base.model import BaseModel


class TrueOnlineINode:
    """
    Node in a True Online Isolation Tree.

    Represents either a leaf node (containing data points) or an internal node
    (containing split information for traversal). Maintains bounding box information
    for incremental learning and unlearning.
    """

    def __init__(
        self,
        data_size: int,
        children: ndarray | None,
        depth: int,
        node_index: int,
        min_values: ndarray | None = None,
        max_values: ndarray | None = None,
        projection_vector: ndarray | None = None,
        split_values: ndarray | None = None,
    ) -> None:
        """
        Initialize a TrueOnlineINode.

        Args:
            data_size: Number of data points in this node's subtree.
            children: Array of child nodes, None for leaf nodes.
            depth: Depth of this node in the tree.
            node_index: Unique index for this node.
            min_values: Minimum values for each feature in this node's bounding box.
            max_values: Maximum values for each feature in this node's bounding box.
            projection_vector: Axis-parallel projection vector for splitting.
            split_values: Array of split threshold values for multi-way splits.
        """
        self.data_size = data_size
        self.children = children
        self.depth = depth
        self.node_index = node_index
        self.min_values = min_values
        self.max_values = max_values
        self.projection_vector = projection_vector
        self.split_values = split_values


class TrueOnlineITree:
    """
    True Online Isolation Tree with incremental learning and unlearning.

    Implements the actual Online Isolation Forest algorithm with proper
    incremental updates, data removal, and axis-parallel splits.
    """

    @staticmethod
    def get_random_path_length(
        branching_factor: int, max_leaf_samples: int, num_samples: float
    ) -> float:
        """
        Compute the expected path length for n samples in a tree with given branching factor.

        This is the original formula from the Online Isolation Forest paper.

        Args:
            branching_factor: Number of children per internal node.
            max_leaf_samples: Maximum samples in leaf nodes.
            num_samples: Number of samples.

        Returns:
            Expected path length for num_samples.
        """
        if num_samples < max_leaf_samples:
            return 0
        else:
            return log(num_samples / max_leaf_samples) / log(2 * branching_factor)

    @staticmethod
    def get_multiplier(type: str, depth: int) -> int:
        """
        Compute the multiplier according to the type.

        Args:
            type: Type of multiplier ('fixed' or 'adaptive').
            depth: Current depth in the tree.

        Returns:
            Multiplier value.
        """
        if type == "fixed":
            return 1
        elif type == "adaptive":
            return 2**depth
        else:
            raise ValueError(f"Bad type {type}")

    def __init__(
        self,
        max_leaf_samples: int,
        type: str,
        subsample: float,
        branching_factor: int,
        data_size: int,
        metric: str = "axisparallel",
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Initialize a TrueOnlineITree.

        Args:
            max_leaf_samples: Maximum number of samples in a leaf node.
            type: Type of tree ('fixed' or 'adaptive').
            subsample: Subsampling rate for training.
            branching_factor: Number of children per internal node.
            data_size: Expected size of the data window.
            metric: Splitting metric ('axisparallel').
            rng: Instance-local random number generator.
        """
        self.max_leaf_samples = max_leaf_samples
        self.type = type
        self.subsample = subsample
        self.branching_factor = branching_factor
        self.data_size = data_size
        self.metric = metric
        self.rng = np.random.default_rng() if rng is None else rng
        self.depth_limit = self.get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            self.data_size * self.subsample,
        )
        self.root: TrueOnlineINode | None = None
        self.next_node_index = 0

    def learn(self, data: ndarray) -> "TrueOnlineITree":
        """
        Learn from new data using incremental updates.

        Args:
            data: Input data array of shape (n_samples, n_features).

        Returns:
            Self for method chaining.
        """
        # Subsample data to improve diversity among trees
        data = data[self.rng.random(data.shape[0]) < self.subsample]
        if data.shape[0] >= 1:
            # Update the counter of data seen so far
            self.data_size += data.shape[0]
            # Adjust depth limit according to data seen so far and branching factor
            self.depth_limit = self.get_random_path_length(
                self.branching_factor, self.max_leaf_samples, self.data_size
            )
            # Recursively update the tree
            if self.root is None:
                self.next_node_index, self.root = self.recursive_build(data)
            else:
                self.next_node_index, self.root = self.recursive_learn(
                    self.root, data, self.next_node_index
                )
        return self

    def unlearn(self, data: ndarray) -> "TrueOnlineITree":
        """
        Remove data influence from the tree.

        Args:
            data: Input data array of shape (n_samples, n_features).

        Returns:
            Self for method chaining.
        """
        # Subsample data to improve diversity among trees
        data = data[self.rng.random(data.shape[0]) < self.subsample]
        if data.shape[0] >= 1:
            # Update the counter of data seen so far
            self.data_size -= data.shape[0]
            # Adjust depth limit according to data seen so far and branching factor
            self.depth_limit = self.get_random_path_length(
                self.branching_factor, self.max_leaf_samples, self.data_size
            )
            # Recursively update the tree
            if self.root is not None:
                self.root = self.recursive_unlearn(self.root, data)
        return self

    def recursive_learn(
        self, node: TrueOnlineINode, data: ndarray, node_index: int
    ) -> tuple[int, TrueOnlineINode]:
        """
        Recursively learn from data by updating existing nodes.

        Args:
            node: Current node to update.
            data: Input data array for this subtree.
            node_index: Next available node index.

        Returns:
            Tuple of (next_node_index, updated_node).
        """
        # Update the number of data seen so far by the current node
        node.data_size += data.shape[0]

        # Update the vectors of minimum and maximum values seen so far by the current node
        if node.min_values is not None and node.max_values is not None:
            node.min_values = vstack([data, node.min_values]).min(axis=0)
            node.max_values = vstack([data, node.max_values]).max(axis=0)

        # If the current node is a leaf, try to split it
        if node.children is None:
            # If there are enough samples to be split according to the max leaf samples and
            # the depth limit has not been reached yet, split the node
            if (
                node.data_size
                >= self.max_leaf_samples * self.get_multiplier(self.type, node.depth)
                and node.depth < self.depth_limit
            ):
                # Sample data_size points uniformly at random within the bounding box defined by
                # the vectors of minimum and maximum values of data seen so far by the current node
                data_sampled = self.rng.uniform(
                    node.min_values,
                    node.max_values,
                    size=(node.data_size, data.shape[1]),
                )
                return self.recursive_build(
                    data_sampled, depth=node.depth, node_index=node_index
                )
            else:
                return node_index, node
        # If the current node is not a leaf, recursively update all its children
        else:
            # Partition data
            partition_indices = self.split_data(
                data, node.projection_vector, node.split_values
            )
            # Recursively update children
            for i, indices in enumerate(partition_indices):
                if len(indices) > 0:
                    node_index, node.children[i] = self.recursive_learn(
                        node.children[i], data[indices], node_index
                    )
            return node_index, node

    def recursive_unlearn(
        self, node: TrueOnlineINode, data: ndarray
    ) -> TrueOnlineINode:
        """
        Recursively remove data influence from nodes.

        Args:
            node: Current node to update.
            data: Input data array to remove influence from.

        Returns:
            Updated node.
        """
        # Update the number of data seen so far by the current node
        node.data_size -= data.shape[0]

        # If the current node is a leaf, return it
        if node.children is None:
            return node
        # If the current node is not a leaf, try to unsplit it
        # If there are not enough samples according to max leaf samples, unsplit the node
        elif node.data_size < self.max_leaf_samples * self.get_multiplier(
            self.type, node.depth
        ):
            return self.recursive_unbuild(node)
        # If there are enough samples according to max leaf samples, recursively update all its children
        else:
            # Partition data
            partition_indices = self.split_data(
                data, node.projection_vector, node.split_values
            )
            # Recursively update children
            for i, indices in enumerate(partition_indices):
                if len(indices) > 0:
                    node.children[i] = self.recursive_unlearn(
                        node.children[i], data[indices]
                    )

            # Update the vectors of minimum and maximum values seen so far by the current node
            if len(node.children) > 0:
                min_vals = [
                    child.min_values
                    for child in node.children
                    if child.min_values is not None
                ]
                max_vals = [
                    child.max_values
                    for child in node.children
                    if child.max_values is not None
                ]
                if min_vals and max_vals:
                    node.min_values = vstack(min_vals).min(axis=0)
                    node.max_values = vstack(max_vals).max(axis=0)
            return node

    def recursive_unbuild(self, node: TrueOnlineINode) -> TrueOnlineINode:
        """
        Recursively unbuild a node by converting it back to a leaf.

        Args:
            node: Node to unbuild.

        Returns:
            Unbuild node (converted to leaf).
        """
        # If the current node is a leaf, return it
        if node.children is None:
            return node
        # If the current node is not a leaf, unbuild it
        else:
            # Recursively unbuild children
            for i, child in enumerate(node.children):
                node.children[i] = self.recursive_unbuild(child)

            # Update the vectors of minimum and maximum values seen so far by the current node
            if len(node.children) > 0:
                min_vals = [
                    child.min_values
                    for child in node.children
                    if child.min_values is not None
                ]
                max_vals = [
                    child.max_values
                    for child in node.children
                    if child.max_values is not None
                ]
                if min_vals and max_vals:
                    node.min_values = vstack(min_vals).min(axis=0)
                    node.max_values = vstack(max_vals).max(axis=0)

            # Delete children nodes, projection vector and split values
            node.children = None
            node.projection_vector = None
            node.split_values = None
            return node

    def recursive_build(
        self, data: ndarray, depth: int = 0, node_index: int = 0
    ) -> tuple[int, TrueOnlineINode]:
        """
        Recursively build the isolation tree from data.

        Args:
            data: Input data array for this subtree.
            depth: Current depth in the tree.
            node_index: Current node index.

        Returns:
            Tuple of (next_node_index, node) for the created subtree.
        """
        # If there aren't enough samples to be split according to the max leaf samples or
        # the depth limit has been reached, build a leaf node
        if (
            data.shape[0]
            < self.max_leaf_samples * self.get_multiplier(self.type, depth)
            or depth >= self.depth_limit
        ):
            return node_index + 1, TrueOnlineINode(
                data_size=data.shape[0],
                children=None,
                depth=depth,
                node_index=node_index,
                min_values=data.min(axis=0, initial=inf),
                max_values=data.max(axis=0, initial=-inf),
                projection_vector=None,
                split_values=None,
            )
        else:
            # Sample projection vector (axis-parallel)
            if self.metric == "axisparallel":
                projection_vector = zeros(data.shape[1])
                projection_vector[self.rng.choice(projection_vector.shape[0])] = 1.0
            else:
                raise ValueError(f"Bad metric {self.metric}")

            # Project sampled data using projection vector
            projected_data = data @ projection_vector

            # Sample split values
            split_values = sort(
                self.rng.uniform(
                    min(projected_data),
                    max(projected_data),
                    size=self.branching_factor - 1,
                )
            )

            # Partition sampled data
            partition_indices = self.split_data(data, projection_vector, split_values)

            # Generate recursively children nodes
            children = empty(shape=(self.branching_factor,), dtype=object)
            for i, indices in enumerate(partition_indices):
                if len(indices) > 0:
                    node_index, children[i] = self.recursive_build(
                        data[indices], depth + 1, node_index
                    )
                else:
                    # Create empty leaf node if no data
                    children[i] = TrueOnlineINode(
                        data_size=0,
                        children=None,
                        depth=depth + 1,
                        node_index=node_index,
                        min_values=None,
                        max_values=None,
                        projection_vector=None,
                        split_values=None,
                    )
                    node_index += 1

            return node_index + 1, TrueOnlineINode(
                data_size=data.shape[0],
                children=children,
                depth=depth,
                node_index=node_index,
                min_values=data.min(axis=0),
                max_values=data.max(axis=0),
                projection_vector=projection_vector,
                split_values=split_values,
            )

    def predict(self, data: ndarray) -> ndarray:
        """
        Predict path lengths for multiple data points.

        Args:
            data: Input data array of shape (n_samples, n_features).

        Returns:
            Path lengths array of shape (n_samples,).
        """
        # Compute depth of each sample
        return self.recursive_depth_search(
            self.root, data, empty(shape=(data.shape[0],), dtype=float)
        )

    def recursive_depth_search(
        self, node: TrueOnlineINode, data: ndarray, depths: ndarray
    ) -> ndarray:
        """
        Recursively compute path lengths for data points.

        Args:
            node: Current node in traversal.
            data: Input data array.
            depths: Array to store computed depths.

        Returns:
            Array of path lengths.
        """
        # If the current node is a leaf, fill the depths vector with the current depth plus a normalization factor
        if node is None or node.children is None or data.shape[0] == 0:
            if node is not None:
                depths[:] = node.depth + self.get_random_path_length(
                    self.branching_factor, self.max_leaf_samples, node.data_size
                )
            else:
                depths[:] = 0
        else:
            # Partition data
            partition_indices = self.split_data(
                data, node.projection_vector, node.split_values
            )
            # Fill the vector of depths
            for i, indices in enumerate(partition_indices):
                if len(indices) > 0:
                    depths[indices] = self.recursive_depth_search(
                        node.children[i], data[indices], depths[indices]
                    )
        return depths

    def split_data(
        self, data: ndarray, projection_vector: ndarray, split_values: ndarray
    ) -> list[ndarray]:
        """
        Split data according to projection vector and split values.

        Args:
            data: Input data array.
            projection_vector: Projection vector for splitting.
            split_values: Array of split threshold values.

        Returns:
            List of index arrays for each partition.
        """
        # Project data using projection vector
        projected_data = data @ projection_vector
        # Sort projected data and keep sort indices
        sort_indices = argsort(projected_data)
        # Split data according to their membership
        partition = split(
            sort_indices, projected_data[sort_indices].searchsorted(split_values)
        )
        return partition


class OnlineIsolationForest(BaseModel):
    """
    True Online Isolation Forest with proper incremental learning and unlearning.

    Implements the actual Online Isolation Forest algorithm from the paper with
    sliding window management, incremental updates, and correct formulas.

    References:
        Leveni, F., Weigert Cassales, G., Pfahringer, B., Bifet, A., &
        Boracchi, G. (2024). Online Isolation Forest.
        https://proceedings.mlr.press/v235/leveni24a.html
        Original implementation:
        https://github.com/ineveLoppiliF/Online-Isolation-Forest
    """

    def __init__(
        self,
        num_trees: int = 100,
        max_leaf_samples: int = 32,
        type: str = "adaptive",
        subsample: float = 1.0,
        window_size: int = 2048,
        branching_factor: int = 2,
        metric: str = "axisparallel",
        n_jobs: int = 1,
        seed: int | None = None,
    ) -> None:
        """
        Initialize a TrueOnlineIForest.

        Args:
            num_trees: Number of isolation trees in the iforest.
            max_leaf_samples: Maximum samples in leaf nodes.
            type: Type of iforest implementation ('fixed' or 'adaptive').
            subsample: Subsampling rate for training.
            window_size: Size of the sliding window for data.
            branching_factor: Number of children per internal node.
            metric: Splitting metric ('axisparallel').
            n_jobs: Number of parallel jobs.
            seed: Random seed used to create independent per-tree generators.
        """
        # Parameter validation
        if num_trees <= 0:
            raise ValueError("num_trees must be positive")
        if max_leaf_samples <= 0:
            raise ValueError("max_leaf_samples must be positive")
        if type not in ["fixed", "adaptive"]:
            raise ValueError("type must be 'fixed' or 'adaptive'")
        if subsample <= 0.0 or subsample > 1.0:
            raise ValueError("subsample must be in (0.0, 1.0]")
        if branching_factor <= 1:
            raise ValueError("branching_factor must be greater than 1")

        self.num_trees = num_trees
        self.max_leaf_samples = max_leaf_samples
        self.type = type
        self.subsample = subsample
        self.window_size = window_size
        self.branching_factor = branching_factor
        self.metric = metric
        self.n_jobs = n_jobs if n_jobs != -1 else None
        self.seed = seed

        child_seeds = np.random.SeedSequence(seed).spawn(num_trees)
        self.trees: list[TrueOnlineITree] = [
            TrueOnlineITree(
                max_leaf_samples=max_leaf_samples,
                type=type,
                subsample=subsample,
                branching_factor=branching_factor,
                data_size=0,
                metric=metric,
                rng=np.random.default_rng(child_seed),
            )
            for child_seed in child_seeds
        ]

        self.data_window: list[ndarray] = []
        self.data_size: int = 0
        self.normalization_factor: float = 0.0
        self._feature_order: tuple[str, ...] | None = None
        self._n_features: int | None = None

    def _set_or_validate_feature_order(self, x: dict[str, float]) -> None:
        """Set feature order from first sample and validate future samples."""
        if self._feature_order is None:
            candidate_order = tuple(sorted(x.keys()))
            if (
                self._n_features is not None
                and len(candidate_order) != self._n_features
            ):
                raise ValueError(
                    f"Expected {self._n_features} features, received {len(candidate_order)}"
                )
            self._feature_order = candidate_order
            if self._n_features is None:
                self._n_features = len(self._feature_order)
            return

        received_keys = set(x.keys())
        expected_keys = set(self._feature_order)
        if received_keys != expected_keys:
            expected = ", ".join(self._feature_order)
            received = ", ".join(sorted(x.keys()))
            raise ValueError(
                "Inconsistent feature keys. "
                f"Expected [{expected}], received [{received}]."
            )

    def _dict_to_array(self, x: dict[str, float]) -> ndarray:
        """
        Convert a feature dictionary to a 2D float32 array with stable feature order.
        """
        self._set_or_validate_feature_order(x)
        if self._feature_order is None:
            return np.array([], dtype=np.float32).reshape(0, 0)

        row = np.fromiter(
            (x[feature] for feature in self._feature_order),
            dtype=np.float32,
            count=len(self._feature_order),
        )
        return row.reshape(1, -1)

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Learn from a single data point.

        Args:
            x: Input data point as a dictionary with feature names as keys
                and feature values as values.
        """
        if not x:
            return

        data_point = self._dict_to_array(x)
        self.learn_batch(data_point)

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score for a single data point.

        Args:
            x: Input data point as a dictionary with feature names as keys
                and feature values as values.

        Returns:
            Anomaly score. Higher values indicate greater anomaly likelihood.
        """
        if not x:
            return 0.0

        data_point = self._dict_to_array(x)
        scores = self.score_batch(data_point)
        return float(scores[0]) if len(scores) > 0 else 0.0

    def learn_batch(self, data: ndarray) -> None:
        """
        Learn from a batch of data points using incremental updates.

        Args:
            data: Input data array of shape (n_samples, n_features).
        """
        if data.size == 0:
            return
        if data.ndim != 2:
            raise ValueError("data must be a 2D array of shape (n_samples, n_features)")

        if self._n_features is None:
            self._n_features = data.shape[1]
        elif data.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, received {data.shape[1]}"
            )

        # Update the counter of data seen so far
        self.data_size += data.shape[0]

        # Compute the normalization factor
        self.normalization_factor = TrueOnlineITree.get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            self.data_size * self.subsample,
        )

        # Learn new data in all trees
        learn_funcs = [tree.learn for tree in self.trees]
        if self.n_jobs is None or self.n_jobs == 1:
            # Sequential execution
            for func in learn_funcs:
                func(data)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                list(executor.map(lambda f: f(data), learn_funcs))

        # If the window size is not None, add new data to the window and eventually remove old ones
        if self.window_size:
            # Update the window of data seen so far
            self.data_window.extend(data)

            # If the window size is smaller than the number of data seen so far, unlearn old data
            if self.data_size > self.window_size:
                # Extract old data and update the window of data seen so far
                old_data_count = self.data_size - self.window_size
                old_data = np.array(self.data_window[:old_data_count])
                self.data_window = self.data_window[old_data_count:]

                # Update the counter of data seen so far
                self.data_size -= old_data_count

                # Compute the normalization factor
                self.normalization_factor = TrueOnlineITree.get_random_path_length(
                    self.branching_factor,
                    self.max_leaf_samples,
                    self.data_size * self.subsample,
                )

                # Unlearn old data from all trees
                unlearn_funcs = [tree.unlearn for tree in self.trees]
                if self.n_jobs is None or self.n_jobs == 1:
                    # Sequential execution
                    for func in unlearn_funcs:
                        func(old_data)
                else:
                    # Parallel execution
                    with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                        list(executor.map(lambda f: f(old_data), unlearn_funcs))

    def score_batch(self, data: ndarray) -> ndarray:
        """
        Score a batch of data points.

        Args:
            data: Input data array of shape (n_samples, n_features).

        Returns:
            Anomaly scores array of shape (n_samples,).
        """
        if data.size == 0:
            return np.array([])
        if data.ndim != 2:
            raise ValueError("data must be a 2D array of shape (n_samples, n_features)")
        if self._n_features is not None and data.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, received {data.shape[1]}"
            )

        # Collect trees' predict functions
        predict_funcs = [tree.predict for tree in self.trees]

        # Compute the depths of all samples in each tree
        if self.n_jobs is None or self.n_jobs == 1:
            # Sequential execution
            depths = np.array([func(data) for func in predict_funcs]).T
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                depths = np.array(
                    list(executor.map(lambda f: f(data), predict_funcs))
                ).T

        # Compute the mean depth of each sample along all trees
        mean_depths = depths.mean(axis=1)

        # Compute normalized mean depths
        normalized_mean_depths = 2 ** (
            -mean_depths / (self.normalization_factor + np.finfo(float).eps)
        )

        return normalized_mean_depths
