"""STREamRHF tree-based streaming anomaly detector."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema, coerce_integer_count

_OPEN_UNIT_LOW = float(np.nextafter(0.0, 1.0))


def _fold_moments(values: Iterable[float]) -> list[float]:
    """Return one feature's ``mean, M2, M3, M4, n`` with the author code's formulas.

    Plain floats round exactly like NumPy scalars, operation by operation, and
    are much faster to update one value at a time.
    """
    mean = m2 = m3 = m4 = n = 0.0
    for value in values:
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
    return [mean, m2, m3, m4, n]


def _moments_with(moments: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Return moments of shape ``(..., n_features, 5)`` updated with ``point``.

    The formulas of ``_fold_moments`` apply elementwise, so every entry
    rounds exactly as updating one value at a time does.
    """
    mean, m2, m3, m4, n = np.moveaxis(moments, -1, 0)
    previous_n = n
    n = n + 1.0
    delta = point - mean
    delta_n = delta / n
    delta_n_sq = delta_n * delta_n
    term = delta * delta_n * previous_n
    updated: np.ndarray = np.stack(
        (
            mean + delta_n,
            m2 + term,
            m3 + (term * delta_n * (n - 2.0) - 3.0 * delta_n * m2),
            m4
            + (
                term * delta_n_sq * (n * n - 3.0 * n + 3.0)
                + 6.0 * delta_n_sq * m2
                - 4.0 * delta_n * m3
            ),
            n,
        ),
        axis=-1,
    )
    return updated


def _kurtosis_weights(moments: np.ndarray) -> np.ndarray:
    """Return ``log(kurtosis + 1)`` weights from STREamRHF for each feature."""
    m2 = moments[..., 1]
    m4 = moments[..., 3]
    n = moments[..., 4]
    kurtosis = np.divide(
        n * m4,
        m2 * m2,
        out=np.zeros_like(m2),
        where=(m2 != 0.0) & (m4 != 0.0),
    )
    return np.asarray(np.log1p(np.maximum(kurtosis, 0.0)), dtype=np.float64)


def _selected_feature(weights: np.ndarray, total: float, feature_random: float) -> int:
    """Select the first feature whose cumulative weight reaches the target.

    The target is ``feature_random * total``, where ``total`` is the sum of
    the weights and is not zero.
    """
    target = feature_random * total
    feature = int(np.searchsorted(np.cumsum(weights), target, side="left"))
    return min(feature, len(weights) - 1)


def _selected_features(weights: np.ndarray, randoms: np.ndarray) -> np.ndarray:
    """Apply ``_selected_feature`` to each row of weights, or give ``-1``.

    Rows whose weights sum to zero select ``-1``. Weights are not negative,
    so counting the cumulative weights below the target finds what
    ``np.searchsorted`` finds; rows with infinite or NaN weights still use
    ``np.searchsorted`` for its ordering of NaN.
    """
    totals = weights.sum(axis=1)
    cumulative = np.cumsum(weights, axis=1)
    targets = randoms * totals
    counts = np.count_nonzero(cumulative < targets[:, np.newaxis], axis=1)
    for row in np.flatnonzero(~np.isfinite(totals)).tolist():
        counts[row] = np.searchsorted(cumulative[row], targets[row], side="left")
    features: np.ndarray = np.minimum(counts, weights.shape[1] - 1)
    features[totals <= 0.0] = -1
    return features


@dataclass(slots=True)
class _RHFLeaf:
    """A terminal bucket owns its observations.

    A leaf keeps no moments: inserting into it rebuilds it from its points.
    """

    depth: int
    node_id: int
    points: list[np.ndarray]

    @property
    def size(self) -> int:
        return len(self.points)

    def collect_points(self) -> list[np.ndarray]:
        return list(self.points)


@dataclass(slots=True)
class _RHFBranch:
    """An internal histogram split owns two complete subtrees."""

    depth: int
    node_id: int
    moments: np.ndarray
    split_feature: int
    split_value: float
    left: _RHFNode
    right: _RHFNode

    @property
    def size(self) -> int:
        return self.left.size + self.right.size

    def collect_points(self) -> list[np.ndarray]:
        return self.left.collect_points() + self.right.collect_points()

    def child(self, point: np.ndarray | list[float]) -> _RHFNode:
        if point[self.split_feature] <= self.split_value:
            return self.left
        return self.right


_RHFNode = _RHFLeaf | _RHFBranch


@dataclass(slots=True)
class _Insertion:
    """The changes that inserting one point makes to a tree.

    Branches on the point's path keep their split and take updated moments,
    and the node below the last of them is rebuilt with the point. ``draws``
    holds the node random values that the rebuild drew.
    """

    path: list[tuple[_RHFBranch, np.ndarray]]
    subtree: _RHFNode
    draws: dict[int, tuple[float, float]]


def _plan_insertions(
    trees: list[_RandomHistogramTree],
    point: np.ndarray,
    *,
    preview: bool,
) -> list[_Insertion]:
    """Plan inserting ``point`` into every tree without changing any of them.

    Each branch on the point's path takes the point's moments. The first
    branch whose selected attribute changes is rebuilt with the point, or else
    the leaf the point reaches is. The branches of all trees are updated and
    tested together. A preview collects new node random values apart from
    the tree's cache.
    """
    values = point.tolist()
    paths: list[list[_RHFBranch]] = []
    leaves: list[_RHFLeaf] = []
    for tree in trees:
        if tree.root is None:
            raise RuntimeError("Tree is not built")
        path = []
        node = tree.root
        while isinstance(node, _RHFBranch):
            path.append(node)
            node = node.child(values)
        paths.append(path)
        leaves.append(node)
    branches = [branch for path in paths for branch in path]
    moments = np.empty((0, 0, 5))
    features: list[int] = []
    if branches:
        moments = _moments_with(np.array([b.moments for b in branches]), point)
        # Existing branches drew their random values when they were built.
        randoms = np.array(
            [
                tree._node_random[branch.node_id][0]
                for tree, path in zip(trees, paths, strict=True)
                for branch in path
            ]
        )
        features = _selected_features(_kurtosis_weights(moments), randoms).tolist()

    insertions = []
    offset = 0
    for tree, path, leaf in zip(trees, paths, leaves, strict=True):
        kept = 0
        while kept < len(path) and features[offset + kept] == path[kept].split_feature:
            kept += 1
        rebuilt = path[kept] if kept < len(path) else leaf
        draws: dict[int, tuple[float, float]] = {} if preview else tree._node_random
        points = rebuilt.collect_points()
        points.append(point)
        insertions.append(
            _Insertion(
                path=[
                    (branch, moments[offset + index].copy())
                    for index, branch in enumerate(path[:kept])
                ],
                subtree=tree._build_node(points, rebuilt.depth, rebuilt.node_id, draws),
                draws=draws,
            )
        )
        offset += len(path)
    return insertions


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
        self.root = self._build_node(points, 0, 0, self._node_random)

    def _random_values(
        self,
        node_id: int,
        draws: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[float, float]:
        """Return fixed feature/split quantiles for one visited node.

        Values that are not cached yet are stored in ``draws``, which is the
        cache itself unless a preview collects them separately.
        """
        cached = self._node_random.get(node_id)
        if cached is None and draws is not None:
            cached = draws.get(node_id)
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
            (self._node_random if draws is None else draws)[node_id] = cached
        return cached

    def _build_node(
        self,
        points: list[np.ndarray],
        depth: int,
        node_id: int,
        draws: dict[int, tuple[float, float]],
    ) -> _RHFNode:
        leaf = _RHFLeaf(depth=depth, node_id=node_id, points=list(points))
        if len(points) <= 1 or depth >= self.max_depth:
            return leaf

        matrix = np.array(points)
        moments = np.array([_fold_moments(column) for column in matrix.T.tolist()])
        weights = _kurtosis_weights(moments)
        total = float(np.sum(weights))
        if total <= 0.0:
            return leaf
        feature_random, _ = self._random_values(node_id, draws)
        feature = _selected_feature(weights, total, feature_random)

        values = matrix[:, feature]
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if minimum == maximum:
            return leaf
        _, split_random = self._random_values(node_id, draws)
        split_value = minimum + split_random * (maximum - minimum)
        # Values are finite and the split value is not NaN, so every point
        # goes to exactly one side.
        goes_left = (values <= split_value).tolist()
        left_points = [p for p, left in zip(points, goes_left, strict=True) if left]
        right_points = [
            p for p, left in zip(points, goes_left, strict=True) if not left
        ]
        if not left_points or not right_points:
            return leaf

        return _RHFBranch(
            depth=depth,
            node_id=node_id,
            moments=moments,
            split_feature=feature,
            split_value=split_value,
            left=self._build_node(left_points, depth + 1, 2 * node_id + 1, draws),
            right=self._build_node(right_points, depth + 1, 2 * node_id + 2, draws),
        )

    def apply(self, insertion: _Insertion, point: np.ndarray) -> None:
        """Apply a planned insertion of ``point``."""
        if insertion.draws is not self._node_random:
            self._node_random.update(insertion.draws)
        for branch, moments in insertion.path:
            branch.moments = moments
        if not insertion.path:
            self.root = insertion.subtree
            return
        parent = insertion.path[-1][0]
        if point[parent.split_feature] <= parent.split_value:
            parent.left = insertion.subtree
        else:
            parent.right = insertion.subtree

    def insert(self, point: np.ndarray) -> int:
        """Insert one point, rebuilding a subtree if its split feature changes."""
        if self.root is None:
            self.build([point])
        else:
            self.apply(_plan_insertions([self], point, preview=False)[0], point)
        return self.leaf_size(point)

    @staticmethod
    def _leaf_size_from(node: _RHFNode, point: np.ndarray) -> int:
        """Return the leaf size reached by a point from a supplied subtree."""
        while isinstance(node, _RHFBranch):
            node = node.child(point)
        return node.size

    def leaf_size(self, point: np.ndarray) -> int:
        """Return the size of the leaf reached by point."""
        node = self.root
        if node is None:
            return 0
        return self._leaf_size_from(node, point)


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

    ``score_one`` previews insertion along the affected tree paths so it
    reproduces the candidate-inclusive score without mutating learned state.
    The preview is kept, so learning the same point next applies it instead
    of planning the insertion again.

    Args:
        n_estimators: Number of independently seeded random histogram trees.
        max_depth: Maximum tree depth.
        window_size: Number of samples in the initial reference window and in
            each subsequent replacement window. It must exceed one.
        seed: Root seed from which independent per-tree random streams are
            derived.

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
        window_size = coerce_integer_count(window_size, label="window_size", minimum=2)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.window_size = window_size
        self.seed = seed
        self._schema = FeatureSchema()

        self._tree_seed_sequences = np.random.SeedSequence(seed).spawn(n_estimators)
        self._initial_window: list[np.ndarray] = []
        self._current_window: list[np.ndarray] = []
        self._trees: list[_RandomHistogramTree] = []
        self._forest_size = 0
        # The point that was scored last and the insertions planned for it,
        # applied if the same point is learned next.
        self._previewed: tuple[np.ndarray, list[_Insertion]] | None = None

    def __getstate__(self) -> dict[str, object]:
        # Previewed insertions only cache work for the next learn, so pickles
        # leave them out and scoring never changes a pickled model.
        state = self.__dict__.copy()
        state["_previewed"] = None
        return state

    def _build_forest(self, points: list[np.ndarray]) -> None:
        names = self._schema.names
        if names is None:
            raise RuntimeError("Forest schema is not initialized")
        self._trees = []
        for tree_index in range(self.n_estimators):
            tree = _RandomHistogramTree(
                self.max_depth,
                len(names),
                self._tree_seed_sequences[tree_index],
            )
            tree.build(points)
            self._trees.append(tree)
        self._forest_size = len(points)

    def learn_one(self, x: dict[str, float]) -> None:
        """Insert one sample and replace the forest at window boundaries."""
        prepared = self._schema.preview(x)
        point = prepared.values.copy()

        if not self._trees:
            self._initial_window.append(point)
            if len(self._initial_window) == self.window_size:
                self._build_forest(self._initial_window)
                self._initial_window = []
            self._schema.commit(prepared)
            return

        previewed, self._previewed = self._previewed, None
        if previewed is not None and previewed[0].tobytes() == point.tobytes():
            point, insertions = previewed
        else:
            insertions = _plan_insertions(self._trees, point, preview=False)
        for tree, insertion in zip(self._trees, insertions, strict=True):
            tree.apply(insertion, point)
        self._forest_size += 1
        self._current_window.append(point)
        if len(self._current_window) == self.window_size:
            self._build_forest(self._current_window)
            self._current_window = []
        self._schema.commit(prepared)

    def score_one(self, x: dict[str, float]) -> float:
        """Return the candidate-inclusive STREamRHF leaf-mass score."""
        prepared = self._schema.preview(x)
        if not self._trees:
            return 0.0
        point = prepared.values
        candidate_size = self._forest_size + 1

        insertions = _plan_insertions(self._trees, point, preview=True)
        self._previewed = (point, insertions)
        score = 0.0
        for insertion in insertions:
            leaf_size = _RandomHistogramTree._leaf_size_from(insertion.subtree, point)
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
