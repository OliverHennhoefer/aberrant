"""Typed node and tree implementation for Online Isolation Forest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


@dataclass(slots=True)
class _Bounds:
    """Non-empty axis-aligned node bounds."""

    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def from_data(cls, data: np.ndarray) -> _Bounds:
        return cls(minimum=data.min(axis=0), maximum=data.max(axis=0))

    def include(self, data: np.ndarray) -> None:
        self.minimum = np.minimum(self.minimum, data.min(axis=0))
        self.maximum = np.maximum(self.maximum, data.max(axis=0))


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
    """Non-empty multiway branch node."""

    data_size: int
    depth: int
    node_index: int
    bounds: _Bounds
    children: list[_OnlineNode]
    projection_vector: np.ndarray
    split_values: np.ndarray


_OnlineNode: TypeAlias = _EmptyLeaf | _OnlineLeaf | _OnlineBranch


class OnlineIsolationTree:
    """Incremental isolation tree with explicit leaf and branch states."""

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
        self.depth_limit = self.get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            self.data_size * self.subsample,
        )
        self.root: _OnlineNode | None = None
        self.next_node_index = 0

    def learn(self, data: np.ndarray) -> OnlineIsolationTree:
        selected = data[self.rng.random(data.shape[0]) < self.subsample]
        if selected.shape[0] == 0:
            return self

        self.data_size += selected.shape[0]
        self.depth_limit = self.get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            self.data_size,
        )
        if self.root is None:
            self.next_node_index, self.root = self.recursive_build(selected)
        else:
            self.next_node_index, self.root = self.recursive_learn(
                self.root,
                selected,
                self.next_node_index,
            )
        return self

    def unlearn(self, data: np.ndarray) -> OnlineIsolationTree:
        selected = data[self.rng.random(data.shape[0]) < self.subsample]
        if selected.shape[0] == 0:
            return self

        self.data_size -= selected.shape[0]
        self.depth_limit = self.get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            self.data_size,
        )
        if self.root is not None:
            self.root = self.recursive_unlearn(self.root, selected)
        return self

    def _split_threshold(self, depth: int) -> int:
        return self.max_leaf_samples * self.get_multiplier(self.tree_type, depth)

    def recursive_learn(
        self,
        node: _OnlineNode,
        data: np.ndarray,
        node_index: int,
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
            node.bounds.include(data)

        if isinstance(node, _OnlineLeaf):
            if node.data_size >= self._split_threshold(node.depth) and (
                node.depth < self.depth_limit
            ):
                sampled = self.rng.uniform(
                    node.bounds.minimum,
                    node.bounds.maximum,
                    size=(node.data_size, data.shape[1]),
                )
                return self.recursive_build(
                    sampled,
                    depth=node.depth,
                    node_index=node_index,
                )
            return node_index, node

        partition_indices = self.split_data(
            data,
            node.projection_vector,
            node.split_values,
        )
        for index, indices in enumerate(partition_indices):
            if len(indices) > 0:
                node_index, node.children[index] = self.recursive_learn(
                    node.children[index],
                    data[indices],
                    node_index,
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
            data,
            node.projection_vector,
            node.split_values,
        )
        for index, indices in enumerate(partition_indices):
            if len(indices) > 0:
                node.children[index] = self.recursive_unlearn(
                    node.children[index],
                    data[indices],
                )
        self._recompute_branch_bounds(node)
        return node

    @staticmethod
    def _recompute_branch_bounds(node: _OnlineBranch) -> None:
        bounded_children = [
            child
            for child in node.children
            if isinstance(child, _OnlineLeaf | _OnlineBranch)
        ]
        if not bounded_children:
            return
        node.bounds = _Bounds(
            minimum=np.vstack(
                [child.bounds.minimum for child in bounded_children]
            ).min(axis=0),
            maximum=np.vstack(
                [child.bounds.maximum for child in bounded_children]
            ).max(axis=0),
        )

    @staticmethod
    def recursive_unbuild(node: _OnlineBranch) -> _OnlineNode:
        if node.data_size <= 0:
            return _EmptyLeaf(depth=node.depth, node_index=node.node_index)
        return _OnlineLeaf(
            data_size=node.data_size,
            depth=node.depth,
            node_index=node.node_index,
            bounds=node.bounds,
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
        projection_vector = np.zeros(data.shape[1])
        projection_vector[self.rng.choice(projection_vector.shape[0])] = 1.0
        projected_data = data @ projection_vector
        split_values = np.sort(
            self.rng.uniform(
                min(projected_data),
                max(projected_data),
                size=self.branching_factor - 1,
            )
        )
        partition_indices = self.split_data(
            data,
            projection_vector,
            split_values,
        )

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
            bounds=_Bounds.from_data(data),
            children=children,
            projection_vector=projection_vector,
            split_values=split_values,
        )

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.recursive_depth_search(
            self.root,
            data,
            np.empty(shape=(data.shape[0],), dtype=float),
        )

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
            data,
            node.projection_vector,
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
        data: np.ndarray,
        projection_vector: np.ndarray,
        split_values: np.ndarray,
    ) -> list[np.ndarray]:
        projected_data = data @ projection_vector
        sort_indices = np.argsort(projected_data)
        return np.split(
            sort_indices,
            projected_data[sort_indices].searchsorted(split_values),
        )
