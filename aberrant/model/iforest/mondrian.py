"""Mondrian-tree isolation forest for streaming anomaly detection."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema, coerce_integer_count


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
    """Float64 bounds and population shared by every non-empty block."""

    min: np.ndarray
    max: np.ndarray
    count: int
    split_time: float


class MondrianLeaf(_MondrianBlock):
    """An occupied terminal block, with no split or child metadata."""

    __slots__ = ()

    @classmethod
    def from_point(cls, point: np.ndarray, split_time: float) -> MondrianLeaf:
        return cls(
            np.array(point, dtype=np.float64),
            np.array(point, dtype=np.float64),
            1,
            split_time,
        )

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


# Bottom levels of each walk whose extension rates one batch computes. Points
# usually leave the nested boxes only a few levels above their leaf.
_RATE_WINDOW = 12


@dataclass(slots=True)
class _Rates:
    """Extension weights and rates of the bottom levels of one walk.

    Entry ``level + offset`` of each field belongs to ``level`` of the walk, for
    every level from ``low`` down. Boxes nest along a walk, so the point lies
    outside exactly the boxes from ``first`` down, and only those have a
    positive rate. ``below`` and ``above`` tell whether the point lies below a
    box's lower or above its upper bound in some feature.
    """

    weights: np.ndarray
    sums: list[float]
    below: list[bool]
    above: list[bool]
    offset: int
    low: int
    first: int


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
        points = np.asarray(x_projected, dtype=np.float64).reshape(1, -1)
        rows = points.tolist()
        _extend_all((self,), points, rows, [self._walk(rows[0])])

    def learn_one_from_global(self, global_features: np.ndarray) -> None:
        """Project one global feature vector and update this tree."""
        np.take(global_features, self.selected_indices, out=self._projected_buffer)
        self.learn_one(self._projected_buffer)

    def _walk(self, x_values: Sequence[float] | np.ndarray) -> list[MondrianNode]:
        """Return the blocks from the root down to the leaf that holds a point."""
        node = self.root
        if node is None:
            return []
        # Expanding ranges can create arbitrarily deep trees. Keep the walk
        # explicit rather than consuming one Python frame per branch.
        path: list[MondrianNode] = [node]
        while isinstance(node, MondrianBranch):
            node = (
                node.left_child
                if x_values[node.split_feature] <= node.split_threshold
                else node.right_child
            )
            path.append(node)
        return path

    def _extend_block(
        self,
        point: np.ndarray,
        x_values: list[float],
        path: list[MondrianNode],
        rates: _Rates,
    ) -> None:
        """
        Extend the blocks along the walk of a point.

        Top-down, every block that the point lies outside samples a split time.
        The first one that occurs before the block's own split time creates a
        new parent above that block; otherwise the leaf absorbs the point. A
        block that already bounds the point has a zero extension rate and
        samples nothing.
        """
        if not path:
            self.root = MondrianLeaf.from_point(point, self.lambda_)
            self.n_samples += 1
            return

        rng = self.rng
        sums = rates.sums
        offset = rates.offset
        first = rates.first
        stop = len(path) - 1
        parent_split_time = path[first - 1].split_time if first else 0.0
        for level in range(first, len(path)):
            node = path[level]
            extension_rate = sums[level + offset]
            # Generator.exponential(scale) returns scale times one standard
            # exponential draw, and these rates are all positive.
            split_time = (
                parent_split_time + (1.0 / extension_rate) * rng.standard_exponential()
            )
            if split_time < node.split_time:
                split_feature = self._sample_split_feature(
                    rates.weights[level + offset].tolist(), extension_rate
                )
                split_threshold = self._sample_split_threshold(
                    x_values=x_values, node=node, split_feature=split_feature
                )
                new_leaf = MondrianLeaf.from_point(point, self.lambda_)
                left, right = (
                    (new_leaf, node)
                    if x_values[split_feature] <= split_threshold
                    else (node, new_leaf)
                )
                branch = MondrianBranch.join(
                    left, right, split_feature, split_threshold, split_time
                )
                if level == 0:
                    self.root = branch
                else:
                    parent = path[level - 1]
                    assert isinstance(parent, MondrianBranch)
                    if x_values[parent.split_feature] <= parent.split_threshold:
                        parent.left_child = branch
                    else:
                        parent.right_child = branch
                stop = level
                break
            parent_split_time = node.split_time
        else:
            leaf = path[stop]
            assert isinstance(leaf, MondrianLeaf)
            leaf.update_stats(point)

        # Every block above the change gains the point. Only the bounds that
        # the point lies beyond move.
        for node in path[:stop]:
            node.count += 1
        below = rates.below
        above = rates.above
        for level in range(first, stop):
            node = path[level]
            if below[level + offset]:
                np.minimum(node.min, point, out=node.min)
            if above[level + offset]:
                np.maximum(node.max, point, out=node.max)
        self.n_samples += 1

    def _sample_split_feature(
        self, extension_weights: Sequence[float], total: float
    ) -> int:
        """Sample split dimension proportional to extension magnitudes."""
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
        x_values: Sequence[float],
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
        return _path_length(self._walk(x_projected))

    def score_one_from_global(self, global_features: np.ndarray) -> float:
        """Project one global feature vector and score against this tree."""
        np.take(global_features, self.selected_indices, out=self._projected_buffer)
        return self.score_one(self._projected_buffer)


def _path_length(path: list[MondrianNode]) -> float:
    """Return path length plus leaf-size adjustment for one walk."""
    if not path:
        return 0.0
    return float(len(path) - 1 + _average_path_length(path[-1].count))


def _plan(
    trees: Sequence[MondrianTree], values: np.ndarray
) -> tuple[np.ndarray, list[list[float]], list[list[MondrianNode]]]:
    """Project a point into the subspace of each tree and walk each tree."""
    points = values[np.array([tree.selected_indices for tree in trees])]
    rows = points.tolist()
    paths = [tree._walk(row) for tree, row in zip(trees, rows, strict=True)]
    return points, rows, paths


def _extend_all(
    trees: Sequence[MondrianTree],
    points: np.ndarray,
    rows: list[list[float]],
    paths: list[list[MondrianNode]],
) -> None:
    """
    Insert ``points[t]`` into ``trees[t]`` along its walk, for all trees at once.

    The trees share no state and each draws from its own generator in the
    order of a lone insertion, so batching leaves every tree the same.
    """
    for tree, point, row, path, rates in zip(
        trees, points, rows, paths, _walk_rates(paths, points), strict=True
    ):
        tree._extend_block(point, row, path, rates)


def _walk_rates(
    paths: Sequence[list[MondrianNode]], points: np.ndarray
) -> list[_Rates]:
    """Return the extension rates of every box each point lies outside."""
    lows = [max(len(path) - _RATE_WINDOW, 0) for path in paths]
    rates = _bottom_rates(paths, points, lows)
    # A point outside every box of a window may also leave higher boxes.
    again = [t for t, r in enumerate(rates) if 0 < r.low == r.first]
    if again:
        retried = _bottom_rates(
            [paths[t] for t in again], points[again], [0] * len(again)
        )
        for t, r in zip(again, retried, strict=True):
            rates[t] = r
    return rates


def _stack(rows: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    """
    Stack float64 bounds into a new writable array.

    Blocks always hold contiguous float64 bounds. Joining their raw buffers is
    about three times faster than ``np.concatenate`` for this many small rows.
    """
    # Arrays expose their memory through the buffer protocol.
    joined = bytearray().join(rows)  # type: ignore[arg-type]
    return np.frombuffer(joined, dtype=np.float64).reshape(shape)


def _bottom_rates(
    paths: Sequence[list[MondrianNode]], points: np.ndarray, lows: Sequence[int]
) -> list[_Rates]:
    """
    Return the extension weights and rates of each ``path[low:]`` at once.

    Elementwise operations and row sums of the stacked blocks round exactly
    like the same operations on each block alone.
    """
    sizes = [len(path) - low for path, low in zip(paths, lows, strict=True)]
    nodes = [node for path, low in zip(paths, lows, strict=True) for node in path[low:]]
    if not nodes:
        empty = np.empty((0, points.shape[1]))
        return [
            _Rates(empty, [], [], [], 0, low, len(path))
            for path, low in zip(paths, lows, strict=True)
        ]

    x = np.repeat(points, sizes, axis=0)
    lower = _stack([node.min for node in nodes], x.shape)
    upper = _stack([node.max for node in nodes], x.shape)
    np.subtract(lower, x, out=lower)
    np.maximum(lower, 0.0, out=lower)
    np.subtract(x, upper, out=upper)
    np.maximum(upper, 0.0, out=upper)
    below: list[bool] = (lower.max(axis=1) > 0.0).tolist()
    above: list[bool] = (upper.max(axis=1) > 0.0).tolist()
    weights = np.add(lower, upper, out=lower)
    sums = weights.sum(axis=1)
    outside = np.concatenate(([0], np.cumsum(sums > 0.0))).tolist()
    sum_list = sums.tolist()

    result = []
    base = 0
    for path, low, size in zip(paths, lows, sizes, strict=True):
        first = len(path) - (outside[base + size] - outside[base])
        result.append(_Rates(weights, sum_list, below, above, base - low, low, first))
        base += size
    return result


@dataclass(slots=True)
class _Plan:
    """The projections and walks of the last scored point, kept for learning."""

    key: bytes
    points: np.ndarray
    rows: list[list[float]]
    paths: list[list[MondrianNode]]


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
        window_size: Optional bound on recent observations used to replace the
            trees. After the initial window, every ``window_size`` observations
            rebuilds each tree from the latest complete window. Between rebuilds
            each tree contains at most ``2 * window_size - 1`` observations.
            ``None`` preserves lifetime learning, whose tree memory can grow
            without bound on expanding streams.

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
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if subspace_size <= 0:
            raise ValueError("subspace_size must be positive")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        if window_size is not None:
            window_size = coerce_integer_count(
                window_size, label="window_size", minimum=2
            )

        self.n_estimators = n_estimators
        self.subspace_size = subspace_size
        self.lambda_ = lambda_
        self.seed = seed
        self.window_size = window_size

        self.rng = np.random.default_rng(seed)
        self.trees: list[MondrianTree] = []
        self.n_samples = 0
        self._schema = FeatureSchema()
        self._window: deque[np.ndarray] = deque(maxlen=window_size)
        self._planned: _Plan | None = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        # Planned walks are a cache of the trees, not part of the model.
        state["_planned"] = None
        return state

    def learn_one(self, x: dict[str, float]) -> None:
        """Update all trees with one feature dictionary."""
        prepared = self._schema.preview(x)
        planned, self._planned = self._planned, None
        if not self._schema.is_established:
            self._initialize_features(len(prepared.names))

        values = prepared.values
        if planned is not None and planned.key == values.tobytes():
            _extend_all(self.trees, planned.points, planned.rows, planned.paths)
        else:
            _extend_all(self.trees, *_plan(self.trees, values))

        self.n_samples += 1
        if self.window_size is not None:
            self._window.append(values.copy())
            if self.trees[0].n_samples >= 2 * self.window_size:
                self._rebuild()
        self._schema.commit(prepared)

    def _rebuild(self) -> None:
        """
        Rebuild every tree from the latest window.

        Every tree draws from its own generator, so inserting the window point
        by point into all trees matches rebuilding one tree after another. A
        tree that fails stops there, and the trees after the first failing one
        keep their previous state, as they would after a tree-by-tree rebuild.
        """
        kept = [
            (tree.root, tree.n_samples, tree.rng.bit_generator.state)
            for tree in self.trees
        ]
        for tree in self.trees:
            tree.root = None
            tree.n_samples = 0
        building = list(range(len(self.trees)))
        failures: dict[int, Exception] = {}
        for point in self._window:
            if not building:
                break
            trees = [self.trees[t] for t in building]
            points, rows, paths = _plan(trees, point)
            rates = _walk_rates(paths, points)
            for t, tree, x_values, row, path, path_rates in zip(
                list(building), trees, points, rows, paths, rates, strict=True
            ):
                try:
                    tree._extend_block(x_values, row, path, path_rates)
                except Exception as error:
                    failures[t] = error
                    building.remove(t)
        if failures:
            first = min(failures)
            for tree, (root, n_samples, rng_state) in zip(
                self.trees[first + 1 :], kept[first + 1 :], strict=True
            ):
                tree.root = root
                tree.n_samples = n_samples
                tree.rng.bit_generator.state = rng_state
            raise failures[first]

    def score_one(self, x: dict[str, float]) -> float:
        """Compute normalized anomaly score in [0, 1]."""
        prepared = self._schema.preview(x)
        if not self._schema.is_established or self.n_samples <= 1:
            return 0.0
        if not self.trees:
            return 0.0

        values = prepared.values
        points, rows, paths = _plan(self.trees, values)
        self._planned = _Plan(values.tobytes(), points, rows, paths)
        path_length_sum = 0.0
        for path in paths:
            path_length_sum += _path_length(path)
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
        population = (
            self.trees[0].n_samples
            if self.window_size is not None and self.trees
            else self.n_samples
        )
        return _average_path_length(population)

    def __repr__(self) -> str:
        """Return a string representation of the MondrianIsolationForest."""
        return (
            f"MondrianIsolationForest(n_estimators={self.n_estimators}, "
            f"subspace_size={self.subspace_size}, "
            f"lambda_={self.lambda_}, seed={self.seed}, window_size={self.window_size})"
        )
