"""Typed node and tree implementation for Online Isolation Forest."""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


@dataclass(slots=True)
class _Bounds:
    """Non-empty axis-aligned node bounds.

    ``box`` stacks the minimum over the negated maximum, shape
    ``(2, n_features)``, so the union of bounds is one elementwise minimum.
    Negation is exact, and the array is never modified in place, so bounds may
    share it.
    """

    box: np.ndarray

    @classmethod
    def from_data(cls, data: np.ndarray) -> _Bounds:
        return cls(np.vstack((data.min(axis=0), -data.max(axis=0))))

    @classmethod
    def union(cls, bounds: Sequence[_Bounds]) -> _Bounds:
        return cls(np.min([item.box for item in bounds], axis=0))

    @property
    def minimum(self) -> np.ndarray:
        minimum: np.ndarray = self.box[0]
        return minimum

    @property
    def maximum(self) -> np.ndarray:
        maximum: np.ndarray = -self.box[1]
        return maximum

    def include(self, other: _Bounds) -> None:
        self.box = np.minimum(self.box, other.box)


@dataclass(slots=True)
class _EmptyLeaf:
    """Leaf with no observations and therefore no bounding box."""

    depth: int
    node_index: int
    data_size: int = 0


@dataclass(slots=True)
class _OnlineLeaf:
    """Non-empty leaf node."""

    data_size: int
    depth: int
    node_index: int
    bounds: _Bounds


@dataclass(slots=True)
class _OnlineBranch:
    """Non-empty multiway branch node.

    Child ``i`` holds the values of ``feature`` in
    ``[split_values[i - 1], split_values[i])``, with open outer ends.

    A branch does not store its bounding box, which is only read when the
    branch collapses into a leaf. The box is derived from the subtree instead,
    see ``OnlineIsolationTree.node_bounds``. ``split_bounds`` holds the boxes of
    child leaves that were split since the branch was last unlearned, because
    a split replaces a leaf by a subtree built from points sampled inside it.
    """

    data_size: int
    depth: int
    node_index: int
    children: list[_OnlineNode]
    feature: int
    split_values: tuple[float, ...]
    split_bounds: _Bounds | None = None


_OnlineNode: TypeAlias = _EmptyLeaf | _OnlineLeaf | _OnlineBranch


class OnlineIsolationTree:
    """Incremental isolation tree with explicit leaf and branch states.

    ``learn``, ``unlearn`` and ``predict`` process batches. ``learn_point``,
    ``unlearn_point`` and ``leaf_path`` process one row with the same results
    as a one-row batch, without the batch partitioning.
    """

    @staticmethod
    def get_random_path_length(
        branching_factor: int,
        max_leaf_samples: int,
        num_samples: float,
    ) -> float:
        if num_samples < max_leaf_samples:
            return 0.0
        return float(
            np.log(num_samples / max_leaf_samples) / np.log(2 * branching_factor)
        )

    @staticmethod
    def get_random_path_lengths(
        branching_factor: int,
        max_leaf_samples: int,
        num_samples: np.ndarray,
    ) -> np.ndarray:
        """Vectorized ``get_random_path_length`` for non-negative populations."""
        # Populations below ``max_leaf_samples`` take log(1) = 0, as above.
        ratios = np.maximum(num_samples, max_leaf_samples) / max_leaf_samples
        lengths: np.ndarray = np.log(ratios) / np.log(2 * branching_factor)
        return lengths

    @staticmethod
    def get_multiplier(tree_type: str, depth: int) -> int:
        if tree_type == "fixed":
            return 1
        if tree_type == "adaptive":
            return int(2**depth)
        raise ValueError(f"Bad type {tree_type}")

    def __init__(
        self,
        max_leaf_samples: int,
        tree_type: str,
        subsample: float,
        branching_factor: int,
        data_size: int,
        metric: str = "axisparallel",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.max_leaf_samples = max_leaf_samples
        self.tree_type = tree_type
        self.subsample = subsample
        self.branching_factor = branching_factor
        self.data_size = data_size
        self.metric = metric
        self.rng = np.random.default_rng() if rng is None else rng
        # ``depth_limit`` is the random path length of this population: the
        # expected subsampled population at first, then ``data_size``. Updates
        # only record the population. The limit is computed when a leaf may
        # split and reused while the population is the same, as it is on every
        # learn of a full window that learns and unlearns one row per event.
        self._depth_limit_population: float = self.data_size * self.subsample
        self._depth_limit_memo: tuple[float, float] | None = None
        self.root: _OnlineNode | None = None
        self.next_node_index = 0
        self._sample_membership: deque[bool] = deque()

    @property
    def depth_limit(self) -> float:
        """Depth below which leaves may split, given the current population."""
        population = self._depth_limit_population
        memo = self._depth_limit_memo
        if memo is None or memo[0] != population:
            limit = self.get_random_path_length(
                self.branching_factor,
                self.max_leaf_samples,
                population,
            )
            memo = self._depth_limit_memo = (population, limit)
        return memo[1]

    def learn(self, data: np.ndarray) -> OnlineIsolationTree:
        membership = np.asarray(
            self.rng.random(data.shape[0]) < self.subsample, dtype=np.bool_
        )
        self._sample_membership.extend(bool(value) for value in membership)
        selected = data[membership]
        if selected.shape[0] == 0:
            return self

        self.data_size += selected.shape[0]
        self._depth_limit_population = self.data_size
        if self.root is None:
            self.next_node_index, self.root = self.recursive_build(selected)
        else:
            self.next_node_index, self.root = self.recursive_learn(
                self.root,
                selected,
                self.next_node_index,
            )
        return self

    def learn_point(self, point: _Bounds, values: list[float]) -> None:
        """Learn one row given as its own bounds and as a list of its values.

        Callers that update many trees build ``point`` once, preferably in
        float64 like the bounds of leaves split from sampled points, so that
        NumPy need not cast.
        """
        # ``random()`` draws the same number as ``random(1)`` in ``learn``.
        selected = self.rng.random() < self.subsample
        self._sample_membership.append(selected)
        if not selected:
            return

        self.data_size += 1
        self._depth_limit_population = self.data_size
        node = self.root
        if node is None:
            self.next_node_index, self.root = self.recursive_build(
                point.minimum[np.newaxis]
            )
            return

        parent: _OnlineBranch | None = None
        index = 0
        while isinstance(node, _OnlineBranch):
            node.data_size += 1
            parent = node
            index = bisect_right(node.split_values, values[node.feature])
            node = node.children[index]

        if isinstance(node, _EmptyLeaf):
            leaf = _OnlineLeaf(
                data_size=1,
                depth=node.depth,
                node_index=node.node_index,
                bounds=_Bounds(point.box),
            )
        else:
            leaf = node
            leaf.data_size += 1
            leaf.bounds.include(point)

        if self._is_full(leaf):
            self.next_node_index, replacement = self._split(
                leaf, parent, self.next_node_index
            )
            self._replace_child(parent, index, replacement)
        elif leaf is not node:
            self._replace_child(parent, index, leaf)

    def unlearn(self, data: np.ndarray) -> OnlineIsolationTree:
        """Forget a FIFO prefix using its original sampling decisions."""
        count = data.shape[0]
        if count > len(self._sample_membership):
            raise ValueError("Cannot unlearn more observations than were learned")
        membership = np.fromiter(
            (self._sample_membership.popleft() for _ in range(count)),
            dtype=np.bool_,
            count=count,
        )
        selected = data[membership]
        if selected.shape[0] == 0:
            return self

        self.data_size -= selected.shape[0]
        self._depth_limit_population = self.data_size
        if self.root is not None:
            self.root = self.recursive_unlearn(self.root, selected)
        return self

    def unlearn_point(self, values: list[float]) -> None:
        """Forget the oldest learned row, given as a list of its values."""
        if not self._sample_membership:
            raise ValueError("Cannot unlearn more observations than were learned")
        if not self._sample_membership.popleft():
            return

        self.data_size -= 1
        self._depth_limit_population = self.data_size
        node = self.root
        parent: _OnlineBranch | None = None
        index = 0
        while isinstance(node, _OnlineBranch):
            node.data_size -= 1
            if node.data_size < self._split_threshold(node.depth):
                self._replace_child(parent, index, self.recursive_unbuild(node))
                return
            # The subtree now bounds every point the branch still covers.
            node.split_bounds = None
            parent = node
            index = bisect_right(node.split_values, values[node.feature])
            node = node.children[index]

        if isinstance(node, _OnlineLeaf):
            node.data_size -= 1
            if node.data_size <= 0:
                empty = _EmptyLeaf(depth=node.depth, node_index=node.node_index)
                self._replace_child(parent, index, empty)

    def _replace_child(
        self,
        parent: _OnlineBranch | None,
        index: int,
        node: _OnlineNode,
    ) -> None:
        if parent is None:
            self.root = node
        else:
            parent.children[index] = node

    def _split_threshold(self, depth: int) -> int:
        """Return ``max_leaf_samples * get_multiplier(tree_type, depth)``."""
        # Inlined for the known types: unlearning checks every branch it visits.
        if self.tree_type == "adaptive":
            return self.max_leaf_samples << depth
        if self.tree_type == "fixed":
            return self.max_leaf_samples
        return self.max_leaf_samples * self.get_multiplier(self.tree_type, depth)

    def _is_full(self, leaf: _OnlineLeaf) -> bool:
        return (
            leaf.data_size >= self._split_threshold(leaf.depth)
            and leaf.depth < self.depth_limit
        )

    def _split(
        self,
        leaf: _OnlineLeaf,
        parent: _OnlineBranch | None,
        node_index: int,
    ) -> tuple[int, _OnlineNode]:
        """Rebuild a full leaf from points sampled uniformly inside its bounds."""
        sampled = self.rng.uniform(
            leaf.bounds.minimum,
            leaf.bounds.maximum,
            size=(leaf.data_size, leaf.bounds.minimum.shape[0]),
        )
        if parent is not None:
            # The parent still covers the leaf's points; its new subtree may not.
            parent.split_bounds = (
                leaf.bounds
                if parent.split_bounds is None
                else _Bounds.union([parent.split_bounds, leaf.bounds])
            )
        return self.recursive_build(sampled, depth=leaf.depth, node_index=node_index)

    def recursive_learn(
        self,
        node: _OnlineNode,
        data: np.ndarray,
        node_index: int,
        parent: _OnlineBranch | None = None,
    ) -> tuple[int, _OnlineNode]:
        if isinstance(node, _EmptyLeaf):
            node = _OnlineLeaf(
                data_size=data.shape[0],
                depth=node.depth,
                node_index=node.node_index,
                bounds=_Bounds.from_data(data),
            )
        else:
            node.data_size += data.shape[0]
            if isinstance(node, _OnlineLeaf):
                node.bounds.include(_Bounds.from_data(data))

        if isinstance(node, _OnlineLeaf):
            if self._is_full(node):
                return self._split(node, parent, node_index)
            return node_index, node

        partition_indices = self.split_data(
            data[:, node.feature],
            node.split_values,
        )
        for index, indices in enumerate(partition_indices):
            if len(indices) > 0:
                node_index, node.children[index] = self.recursive_learn(
                    node.children[index],
                    data[indices],
                    node_index,
                    node,
                )
        return node_index, node

    def recursive_unlearn(
        self,
        node: _OnlineNode,
        data: np.ndarray,
    ) -> _OnlineNode:
        if isinstance(node, _EmptyLeaf):
            return node

        node.data_size -= data.shape[0]
        if isinstance(node, _OnlineLeaf):
            if node.data_size <= 0:
                return _EmptyLeaf(depth=node.depth, node_index=node.node_index)
            return node

        if node.data_size < self._split_threshold(node.depth):
            return self.recursive_unbuild(node)

        partition_indices = self.split_data(
            data[:, node.feature],
            node.split_values,
        )
        for index, indices in enumerate(partition_indices):
            if len(indices) > 0:
                node.children[index] = self.recursive_unlearn(
                    node.children[index],
                    data[indices],
                )
        node.split_bounds = None
        return node

    @classmethod
    def node_bounds(cls, node: _OnlineLeaf | _OnlineBranch) -> _Bounds:
        """Return the bounding box of a non-empty node.

        A branch's box is the union of its non-empty children's boxes and its
        ``split_bounds``. Building, learning and unlearning keep this equal to
        a box kept up to date at every branch: building takes the box of the
        data, learning extends the boxes along the learned path, and unlearning
        sets it to the union of the children.
        """
        if isinstance(node, _OnlineLeaf):
            return node.bounds
        boxes = [
            cls.node_bounds(child)
            for child in node.children
            if not isinstance(child, _EmptyLeaf)
        ]
        if node.split_bounds is not None:
            boxes.append(node.split_bounds)
        return _Bounds.union(boxes)

    @classmethod
    def recursive_unbuild(cls, node: _OnlineBranch) -> _OnlineNode:
        if node.data_size <= 0:
            return _EmptyLeaf(depth=node.depth, node_index=node.node_index)
        return _OnlineLeaf(
            data_size=node.data_size,
            depth=node.depth,
            node_index=node.node_index,
            bounds=cls.node_bounds(node),
        )

    def recursive_build(
        self,
        data: np.ndarray,
        depth: int = 0,
        node_index: int = 0,
    ) -> tuple[int, _OnlineNode]:
        if data.shape[0] == 0:
            return node_index + 1, _EmptyLeaf(
                depth=depth,
                node_index=node_index,
            )
        if data.shape[0] < self._split_threshold(depth) or depth >= self.depth_limit:
            return node_index + 1, _OnlineLeaf(
                data_size=data.shape[0],
                depth=depth,
                node_index=node_index,
                bounds=_Bounds.from_data(data),
            )

        if self.metric != "axisparallel":
            raise ValueError(f"Bad metric {self.metric}")
        feature = int(self.rng.choice(data.shape[1]))
        column = data[:, feature]
        split_values = np.sort(
            self.rng.uniform(
                column.min(),
                column.max(),
                size=self.branching_factor - 1,
            )
        )
        partition_indices = self.split_data(column, split_values)

        children: list[_OnlineNode] = []
        for indices in partition_indices:
            if len(indices) > 0:
                node_index, child = self.recursive_build(
                    data[indices],
                    depth + 1,
                    node_index,
                )
            else:
                child = _EmptyLeaf(depth=depth + 1, node_index=node_index)
                node_index += 1
            children.append(child)

        return node_index + 1, _OnlineBranch(
            data_size=data.shape[0],
            depth=depth,
            node_index=node_index,
            children=children,
            feature=feature,
            split_values=tuple(split_values.tolist()),
        )

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.recursive_depth_search(
            self.root,
            data,
            np.empty(shape=(data.shape[0],), dtype=float),
        )

    def leaf_path(self, values: list[float]) -> tuple[int, int]:
        """Return the depth and population of the leaf that ``values`` reaches."""
        node = self.root
        if node is None:
            return 0, 0
        while isinstance(node, _OnlineBranch):
            node = node.children[bisect_right(node.split_values, values[node.feature])]
        return node.depth, node.data_size

    def recursive_depth_search(
        self,
        node: _OnlineNode | None,
        data: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        if node is None or not isinstance(node, _OnlineBranch) or data.shape[0] == 0:
            if node is None:
                depths[:] = 0
            else:
                depths[:] = node.depth + self.get_random_path_length(
                    self.branching_factor,
                    self.max_leaf_samples,
                    node.data_size,
                )
            return depths

        partition_indices = self.split_data(
            data[:, node.feature],
            node.split_values,
        )
        for index, indices in enumerate(partition_indices):
            if len(indices) > 0:
                depths[indices] = self.recursive_depth_search(
                    node.children[index],
                    data[indices],
                    depths[indices],
                )
        return depths

    @staticmethod
    def split_data(
        values: np.ndarray,
        split_values: Sequence[float] | np.ndarray,
    ) -> list[np.ndarray]:
        """Return the row indices of each child for one feature's values."""
        # Compare in float64 like the split values, whatever the data type.
        projected = np.asarray(values, dtype=np.float64)
        sort_indices = np.argsort(projected)
        return np.split(
            sort_indices,
            projected[sort_indices].searchsorted(split_values),
        )
