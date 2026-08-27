"""Random Cut Forest (RCF) for streaming anomaly detection."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema


@dataclass
class _RCFLeaf:
    """Leaf node for one or more identical points."""

    point: np.ndarray
    point_ids: list[int]
    parent: _RCFBranch | None = None
    size: int = 1

    @property
    def bbox_min(self) -> np.ndarray:
        return self.point

    @property
    def bbox_max(self) -> np.ndarray:
        return self.point


@dataclass
class _RCFBranch:
    """Branch node with random cut split."""

    left: _RCFLeaf | _RCFBranch
    right: _RCFLeaf | _RCFBranch
    cut_dim: int
    cut_value: float
    parent: _RCFBranch | None = None
    bbox_min: np.ndarray = field(init=False)
    bbox_max: np.ndarray = field(init=False)
    size: int = 0

    def recompute(self) -> None:
        """Recompute bounding box and subtree size."""
        self.size = self.left.size + self.right.size
        self.bbox_min = np.minimum(self.left.bbox_min, self.right.bbox_min)
        self.bbox_max = np.maximum(self.left.bbox_max, self.right.bbox_max)


class _RandomCutTree:
    """Internal Random Cut Tree implementation."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.root: _RCFLeaf | _RCFBranch | None = None
        self._id_to_leaf: dict[int, _RCFLeaf] = {}

    @property
    def size(self) -> int:
        """Return number of stored points (with duplicates)."""
        if self.root is None:
            return 0
        return self.root.size

    def insert(self, point_id: int, point: np.ndarray) -> None:
        """Insert one point ID into the tree."""
        if point_id in self._id_to_leaf:
            raise ValueError(f"Point ID {point_id} already exists in tree")

        new_leaf = _RCFLeaf(point=point.copy(), point_ids=[point_id], size=1)
        self.root = self._insert_node(self.root, new_leaf)
        if self.root is not None:
            self.root.parent = None

    def remove(self, point_id: int) -> None:
        """Remove one point ID from the tree."""
        leaf = self._id_to_leaf.pop(point_id, None)
        if leaf is None:
            return

        if len(leaf.point_ids) > 1:
            leaf.point_ids.remove(point_id)
            leaf.size -= 1
            self._recompute_upwards(leaf.parent)
            return

        parent = leaf.parent
        if parent is None:
            self.root = None
            return

        sibling = parent.right if parent.left is leaf else parent.left
        grandparent = parent.parent
        sibling.parent = grandparent

        if grandparent is None:
            self.root = sibling
            return

        if grandparent.left is parent:
            grandparent.left = sibling
        else:
            grandparent.right = sibling

        self._recompute_upwards(grandparent)

    def score_point(self, point: np.ndarray) -> float:
        """
        Preview the candidate's collusive displacement.

        Reference RRCF scoring inserts the candidate, computes its CoDisp, and
        then removes it. Restoring the generator state also makes this preview
        observationally non-mutating for future learned inserts.
        """
        if self.root is None:
            return 0.0

        temporary_id = -1
        if temporary_id in self._id_to_leaf:
            raise RuntimeError("Temporary score ID collides with a learned point ID")
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        self.insert(temporary_id, point)
        try:
            return self.codisp(temporary_id)
        finally:
            self.remove(temporary_id)
            self.rng.bit_generator.state = rng_state

    def codisp(self, point_id: int) -> float:
        """Return collusive displacement for a stored point ID."""
        leaf = self._id_to_leaf.get(point_id)
        if leaf is None:
            raise KeyError(f"Point ID {point_id} does not exist in tree")
        return self._codisp_from_leaf(leaf)

    def _insert_node(
        self,
        node: _RCFLeaf | _RCFBranch | None,
        new_leaf: _RCFLeaf,
    ) -> _RCFLeaf | _RCFBranch:
        """Recursively insert a leaf and return subtree root."""
        new_id = new_leaf.point_ids[0]

        if node is None:
            self._id_to_leaf[new_id] = new_leaf
            return new_leaf

        if isinstance(node, _RCFLeaf):
            if np.array_equal(node.point, new_leaf.point):
                node.point_ids.append(new_id)
                node.size += 1
                self._id_to_leaf[new_id] = node
                self._recompute_upwards(node.parent)
                return node

            branch = self._split_leaf(node, new_leaf)
            self._id_to_leaf[new_id] = new_leaf
            return branch

        expanded_min = np.minimum(node.bbox_min, new_leaf.point)
        expanded_max = np.maximum(node.bbox_max, new_leaf.point)
        sample = self._sample_cut(expanded_min, expanded_max)

        if sample is not None:
            cut_dim, cut_value = sample
            if (
                cut_value <= node.bbox_min[cut_dim]
                or cut_value >= node.bbox_max[cut_dim]
            ):
                if new_leaf.point[cut_dim] <= cut_value:
                    branch = _RCFBranch(
                        left=new_leaf,
                        right=node,
                        cut_dim=cut_dim,
                        cut_value=cut_value,
                    )
                else:
                    branch = _RCFBranch(
                        left=node,
                        right=new_leaf,
                        cut_dim=cut_dim,
                        cut_value=cut_value,
                    )
                branch.left.parent = branch
                branch.right.parent = branch
                branch.recompute()
                self._id_to_leaf[new_id] = new_leaf
                return branch

        if new_leaf.point[node.cut_dim] <= node.cut_value:
            node.left = self._insert_node(node.left, new_leaf)
            node.left.parent = node
        else:
            node.right = self._insert_node(node.right, new_leaf)
            node.right.parent = node

        node.recompute()
        return node

    def _split_leaf(
        self,
        existing_leaf: _RCFLeaf,
        new_leaf: _RCFLeaf,
    ) -> _RCFBranch:
        """Create a separating branch between two non-identical leaves."""
        diffs = np.abs(existing_leaf.point - new_leaf.point)
        total_diff = float(np.sum(diffs))

        if total_diff <= 0.0:
            cut_dim = 0
            cut_value = float(existing_leaf.point[0])
        else:
            probs = diffs / total_diff
            cut_dim = int(self.rng.choice(diffs.size, p=probs))
            low = float(min(existing_leaf.point[cut_dim], new_leaf.point[cut_dim]))
            high = float(max(existing_leaf.point[cut_dim], new_leaf.point[cut_dim]))
            cut_value = float(self.rng.uniform(low, high))

            left_existing = existing_leaf.point[cut_dim] <= cut_value
            left_new = new_leaf.point[cut_dim] <= cut_value
            if left_existing == left_new:
                cut_value = (low + high) / 2.0

        if new_leaf.point[cut_dim] <= cut_value:
            left, right = new_leaf, existing_leaf
        else:
            left, right = existing_leaf, new_leaf

        branch = _RCFBranch(
            left=left,
            right=right,
            cut_dim=cut_dim,
            cut_value=cut_value,
        )
        left.parent = branch
        right.parent = branch
        branch.recompute()
        return branch

    def _sample_cut(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
    ) -> tuple[int, float] | None:
        """Sample cut dimension/value proportional to bounding-box span."""
        spans = bbox_max - bbox_min
        total_span = float(np.sum(spans))
        if total_span <= 0.0:
            return None

        offset = float(self.rng.random()) * total_span
        cumulative = np.cumsum(spans)
        cut_dim = int(np.searchsorted(cumulative, offset, side="right"))
        if cut_dim >= spans.size:
            cut_dim = spans.size - 1

        previous = float(cumulative[cut_dim - 1]) if cut_dim > 0 else 0.0
        cut_value = float(bbox_min[cut_dim] + (offset - previous))
        return cut_dim, cut_value

    def _codisp_from_leaf(self, leaf: _RCFLeaf) -> float:
        """Compute the reference collusive displacement from leaf to root."""
        node: _RCFLeaf | _RCFBranch = leaf
        if node.parent is None:
            return 0.0

        max_ratio = 0.0
        while node.parent is not None:
            parent = node.parent
            sibling = parent.right if parent.left is node else parent.left
            if node.size > 0:
                ratio = float(sibling.size) / float(node.size)
                max_ratio = max(max_ratio, ratio)
            node = parent
        return max_ratio

    def _recompute_upwards(self, node: _RCFBranch | None) -> None:
        """Recompute stats up to root after local change."""
        while node is not None:
            node.recompute()
            node = node.parent


class RandomCutForest(BaseModel):
    """
    Random Cut Forest for online anomaly detection.

    This model keeps an ensemble of random cut trees over a bounded sample of
    recent shingled points. `learn_one` performs one-sample updates and
    forgetting, while `score_one` returns an anomaly score for one sample.

    Tree insertion/removal and raw anomaly scoring follow Random Cut Tree
    mechanics. ``score_one`` temporarily inserts each query, computes its
    collusive displacement (CoDisp), removes it, and restores RNG state. Optional
    exponential scaling is a monotonic presentation transform over raw CoDisp.

    Args:
        n_trees: Number of random cut trees.
        sample_size: Maximum number of stored shingled points.
        shingle_size: Number of consecutive points concatenated per tree insert.
        warmup_samples: Number of inserted shingles before non-zero scoring.
            If `None`, defaults to `sample_size`.
        normalize_score: If True, map raw CoDisp to [0, 1].
        score_scale: Scale for score normalization when enabled.
        seed: Random seed for reproducibility.

    References:
        Guha, S., Mishra, N., Roy, G., & Schrijvers, O. (2016). Robust Random
        Cut Forest Based Anomaly Detection on Streams.
        https://proceedings.mlr.press/v48/guha16.html
        Reference implementation: https://github.com/aws/random-cut-forest-by-aws
    """

    def __init__(
        self,
        n_trees: int = 40,
        sample_size: int = 256,
        shingle_size: int = 1,
        warmup_samples: int | None = None,
        normalize_score: bool = False,
        score_scale: float = 8.0,
        seed: int | None = None,
    ) -> None:
        if n_trees <= 0:
            raise ValueError("n_trees must be positive")
        if sample_size <= 1:
            raise ValueError("sample_size must be greater than 1")
        if shingle_size <= 0:
            raise ValueError("shingle_size must be positive")
        if warmup_samples is not None and warmup_samples <= 0:
            raise ValueError("warmup_samples must be positive or None")
        if score_scale <= 0.0:
            raise ValueError("score_scale must be positive")

        self.n_trees = n_trees
        self.sample_size = sample_size
        self.shingle_size = shingle_size
        self.warmup_samples = sample_size if warmup_samples is None else warmup_samples
        self.normalize_score = normalize_score
        self.score_scale = score_scale
        self.seed = seed

        self._reset_state()

    def _reset_state(self) -> None:
        """Initialize or clear learned state."""
        self._schema = FeatureSchema()
        self._history: deque[np.ndarray] = deque(maxlen=self.shingle_size)
        self._id_window: deque[int] = deque()
        self._next_point_id: int = 0
        self._inserted_shingles: int = 0
        self._ready: bool = False

        seed_sequence = np.random.SeedSequence(self.seed)
        child_sequences = seed_sequence.spawn(self.n_trees)
        self._trees = [
            _RandomCutTree(np.random.default_rng(seq)) for seq in child_sequences
        ]

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    def learn_one(self, x: dict[str, float]) -> None:
        """Update forest state with one sample."""
        prepared = self._schema.preview(x)
        vector = prepared.values.copy()
        self._history.append(vector)

        shingle = self._current_shingle()
        if shingle is None:
            self._schema.commit(prepared)
            return

        if len(self._id_window) >= self.sample_size:
            oldest = self._id_window.popleft()
            for tree in self._trees:
                tree.remove(oldest)

        point_id = self._next_point_id
        self._next_point_id += 1
        self._id_window.append(point_id)

        for tree in self._trees:
            tree.insert(point_id, shingle)

        self._inserted_shingles += 1
        self._ready = self._inserted_shingles >= self.warmup_samples
        self._schema.commit(prepared)

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for one sample."""
        prepared = self._schema.preview(x)
        if not self._ready or not self._id_window:
            return 0.0

        query = self._score_shingle(prepared.values)
        if query is None:
            return 0.0

        raw = float(np.mean([tree.score_point(query) for tree in self._trees]))
        if not self.normalize_score:
            return raw

        bounded = 1.0 - float(np.exp(-raw / self.score_scale))
        return float(np.clip(bounded, 0.0, 1.0))

    def _current_shingle(self) -> np.ndarray | None:
        """Return current learn shingle from history."""
        if len(self._history) < self.shingle_size:
            return None
        if self.shingle_size == 1:
            return np.asarray(self._history[-1].copy(), dtype=np.float64)
        return np.asarray(np.concatenate(list(self._history)), dtype=np.float64)

    def _score_shingle(self, vector: np.ndarray) -> np.ndarray | None:
        """Build score shingle without mutating state."""
        if self.shingle_size == 1:
            return vector

        required_history = self.shingle_size - 1
        if len(self._history) < required_history:
            return None

        history = list(self._history)[-required_history:]
        return np.asarray(np.concatenate([*history, vector]), dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"RandomCutForest(n_trees={self.n_trees}, sample_size={self.sample_size}, "
            f"shingle_size={self.shingle_size}, warmup_samples={self.warmup_samples}, "
            f"normalize_score={self.normalize_score}, score_scale={self.score_scale}, "
            f"ready={self._ready}, stored_points={len(self._id_window)})"
        )
