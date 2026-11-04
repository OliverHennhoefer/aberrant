"""Random Cut Forest (RCF) for streaming anomaly detection."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema


def _point_box(point: np.ndarray) -> np.ndarray:
    """Return the box of one point in the layout that nodes store."""
    return np.stack((point, np.negative(point)))


@dataclass(eq=False)
class _RCFLeaf:
    """Leaf node for one or more identical points.

    Boxes stack the lower corner over the negated upper corner, so that a
    single ``np.minimum`` merges two of them.
    """

    point: np.ndarray
    point_ids: list[int]
    box: np.ndarray
    parent: _RCFBranch | None = None
    size: int = 1


@dataclass(eq=False)
class _RCFBranch:
    """Branch node with random cut split."""

    left: _RCFLeaf | _RCFBranch
    right: _RCFLeaf | _RCFBranch
    cut_dim: int
    cut_value: float
    parent: _RCFBranch | None = None
    box: np.ndarray = field(init=False)
    size: int = 0

    def recompute(self) -> None:
        """Recompute bounding box and subtree size."""
        self.size = self.left.size + self.right.size
        self.box = np.minimum(self.left.box, self.right.box)


_RCFNode = _RCFLeaf | _RCFBranch
# A branch that a walk passes, whether the walk goes left, and the branch box
# grown to hold the walked point when that box changes.
_Step = tuple[_RCFBranch, bool, np.ndarray | None]


@dataclass(slots=True)
class _Descent:
    """Where inserting a point into one tree would attach it.

    The walk passes the branches in ``path`` and stops at ``node``: either a
    node that a sampled ``cut`` separates from the point, given as the cut
    dimension, the cut value and the box holding both, or, when ``cut`` is
    ``None``, the leaf the point reaches. ``skipped`` counts passed branches
    whose box had no span to cut.
    """

    path: list[_Step]
    node: _RCFNode
    cut: tuple[int, float, np.ndarray] | None
    skipped: int

    @property
    def draws(self) -> int:
        """Return how many uniforms the walk drew from its tree's generator."""
        return len(self.path) - self.skipped + (self.cut is not None)

    def first_changed(self, changed: set[int]) -> int | None:
        """Return the first step whose branch ``changed`` names, if any.

        A changed node where the walk stops counts as the step after the
        last one.
        """
        for step, (branch, _, _) in enumerate(self.path):
            if id(branch) in changed:
                return step
        if id(self.node) in changed:
            return len(self.path)
        return None


def _descend(
    trees: list[_RandomCutTree],
    point: np.ndarray,
    starts: list[tuple[list[_Step], _RCFNode | None]] | None = None,
) -> list[_Descent]:
    """Walk ``point`` down every tree as an insertion would, drawing cuts.

    All trees descend together one level at a time. At a branch, each tree
    samples a cut from the branch's box expanded by the point, drawing from
    its own generator exactly as a single tree would. Each walk begins at the
    root, or at the node of its ``starts`` entry after the steps given there.
    """
    if starts is None:
        starts = [([], tree.root) for tree in trees]
    values = point.tolist()
    point_box = _point_box(point)
    draws = [tree.rng.random for tree in trees]
    paths = [list(steps) for steps, _ in starts]
    nodes: list[_RCFNode] = []
    active: list[int] = []
    branches: list[_RCFBranch] = []
    for index, (_, node) in enumerate(starts):
        if node is None:
            raise RuntimeError("Tree is empty")
        nodes.append(node)
        if isinstance(node, _RCFBranch):
            active.append(index)
            branches.append(node)
    cuts: list[tuple[int, float, np.ndarray] | None] = [None] * len(trees)
    skipped = [0] * len(trees)
    scratch = np.zeros((len(trees), point.size + 1))
    while active:
        boxes = np.array([branch.box for branch in branches])
        expanded = np.minimum(boxes, point_box)
        low = expanded[:, 0]
        spans = np.negative(expanded[:, 1]) - low
        totals = spans.sum(axis=1)
        # A box without span samples no cut and draws nothing.
        drawing = totals > 0.0
        randoms = _draw_uniforms(draws, active, drawing, skipped)
        dims, cut_values = _place_cuts(low, spans, randoms * totals, scratch)
        rows = np.arange(len(active))
        hits = (cut_values <= boxes[rows, 0, dims]) | (
            np.negative(cut_values) <= boxes[rows, 1, dims]
        )
        separated = (hits & drawing).tolist()
        growing = (expanded != boxes).reshape(len(active), -1).any(axis=1).tolist()

        still_active: list[int] = []
        still_branches: list[_RCFBranch] = []
        for row, (index, branch) in enumerate(zip(active, branches, strict=True)):
            if separated[row]:
                cuts[index] = (int(dims[row]), float(cut_values[row]), expanded[row])
                continue
            go_left = values[branch.cut_dim] <= branch.cut_value
            grown = expanded[row] if growing[row] else None
            paths[index].append((branch, go_left, grown))
            child = branch.left if go_left else branch.right
            nodes[index] = child
            if isinstance(child, _RCFBranch):
                still_active.append(index)
                still_branches.append(child)
        active = still_active
        branches = still_branches
    return [
        _Descent(paths[i], nodes[i], cuts[i], skipped[i]) for i in range(len(trees))
    ]


def _draw_uniforms(
    draws: Sequence[Callable[[], float]],
    active: list[int],
    drawing: np.ndarray,
    skipped: list[int],
) -> np.ndarray:
    """Draw a uniform for each active walk whose box has span to cut.

    The other walks draw nothing, get zero, and count a skipped step.
    """
    if drawing.all():
        return np.array([draws[i]() for i in active])
    flags = drawing.tolist()
    for i, flag in zip(active, flags, strict=True):
        skipped[i] += not flag
    return np.array(
        [draws[i]() if flag else 0.0 for i, flag in zip(active, flags, strict=True)]
    )


def _place_cuts(
    low: np.ndarray, spans: np.ndarray, offsets: np.ndarray, scratch: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the dimension and value of the cut at each row's offset.

    The offset runs along the spans laid end to end from the lower corner,
    as in reference cut sampling. ``scratch`` has a row per box and a zero
    first column, so that a dimension's column receives the spans before it.
    """
    before = scratch[: len(offsets)]
    np.add.accumulate(spans, axis=1, out=before[:, 1:])
    dims = np.minimum(_search_right(before[:, 1:], offsets), spans.shape[1] - 1)
    rows = np.arange(len(offsets))
    return dims, low[rows, dims] + (offsets - before[rows, dims])


def _search_right(cumulative: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return ``np.searchsorted(row, target, side="right")`` for each row.

    Spans are not negative, so each cumulative row is sorted and counting the
    entries at or below a target finds the same index. A NaN target, from a
    zero draw on an infinite span, uses ``np.searchsorted`` for its ordering.
    """
    counts: np.ndarray = (cumulative <= targets[:, np.newaxis]).sum(axis=1)
    if np.isnan(targets).any():
        for row in np.flatnonzero(np.isnan(targets)).tolist():
            counts[row] = np.searchsorted(cumulative[row], targets[row], side="right")
    return counts


def _weighted_choice(rng: np.random.Generator, probs: np.ndarray) -> int:
    """Return ``rng.choice(probs.size, p=probs)`` for probabilities from spans.

    Probabilities from finite, non-negative spans over their finite total
    always pass ``choice``'s checks, so this draws the same uniform and
    searches the same normalized cumulative sum without them.
    """
    cdf = np.cumsum(probs)
    cdf /= cdf[-1]
    return int(np.searchsorted(cdf, rng.random(), side="right"))


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
        point = point.copy()
        descent = None if self.root is None else _descend([self], point)[0]
        self._attach(point_id, point, point.tolist(), _point_box(point), descent)

    def remove(self, point_id: int) -> set[int]:
        """Remove one point ID from the tree.

        Returns the ids of the branches that the removal took out or whose
        box shrank. Walks that avoid them are unaffected by the removal.
        """
        leaf = self._id_to_leaf.pop(point_id, None)
        if leaf is None:
            return set()

        if len(leaf.point_ids) > 1:
            leaf.point_ids.remove(point_id)
            leaf.size -= 1
            # The leaf keeps its point, so no box changes.
            ancestor = leaf.parent
            while ancestor is not None:
                ancestor.size -= 1
                ancestor = ancestor.parent
            return set()

        parent = leaf.parent
        if parent is None:
            self.root = None
            return set()

        sibling = parent.right if parent.left is leaf else parent.left
        grandparent = parent.parent
        sibling.parent = grandparent

        changed = {id(parent)}
        if grandparent is None:
            self.root = sibling
            return changed

        if grandparent.left is parent:
            grandparent.left = sibling
        else:
            grandparent.right = sibling

        # Every ancestor loses one point. A box merges its children's boxes,
        # so boxes shrink only up to the first one that stays the same.
        node: _RCFBranch | None = grandparent
        settled = False
        while node is not None:
            node.size -= 1
            if not settled:
                box = np.minimum(node.left.box, node.right.box)
                settled = box.tobytes() == node.box.tobytes()
                if not settled:
                    node.box = box
                    changed.add(id(node))
            node = node.parent
        return changed

    def score_point(self, point: np.ndarray) -> float:
        """
        Preview the candidate's collusive displacement.

        Reference RRCF scoring inserts the candidate, computes its CoDisp, and
        then removes it. This walks the candidate down as that insertion would,
        derives the CoDisp it would have, and restores the generator state, so
        future learned inserts see the same random cuts.
        """
        if self.root is None:
            return 0.0
        return _codisp(_plan([self], point)[0], point.tolist())

    def codisp(self, point_id: int) -> float:
        """Return collusive displacement for a stored point ID."""
        leaf = self._id_to_leaf.get(point_id)
        if leaf is None:
            raise KeyError(f"Point ID {point_id} does not exist in tree")
        return self._codisp_from_leaf(leaf)

    def _attach(
        self,
        point_id: int,
        point: np.ndarray,
        values: list[float],
        box: np.ndarray,
        descent: _Descent | None,
    ) -> None:
        """Insert ``point`` where ``descent`` says, or as the root when empty.

        ``values`` lists the point's coordinates. The point and its ``box``
        are stored without copying, so that trees can share them.
        """
        if point_id in self._id_to_leaf:
            raise ValueError(f"Point ID {point_id} already exists in tree")
        new_leaf = _RCFLeaf(point=point, point_ids=[point_id], box=box)
        if descent is None:
            self._id_to_leaf[point_id] = new_leaf
            self.root = new_leaf
            return

        reached = descent.node
        node: _RCFNode
        if descent.cut is not None:
            cut_dim, cut_value, cut_box = descent.cut
            left, right = (
                (new_leaf, reached)
                if values[cut_dim] <= cut_value
                else (reached, new_leaf)
            )
            node = _RCFBranch(
                left=left, right=right, cut_dim=cut_dim, cut_value=cut_value
            )
            node.size = reached.size + 1
            node.box = cut_box.copy()
            left.parent = node
            right.parent = node
            self._id_to_leaf[point_id] = new_leaf
        elif not isinstance(reached, _RCFLeaf):
            raise RuntimeError("A walk without a cut must end at a leaf")
        elif reached.point.tolist() == values:
            node = reached
            reached.point_ids.append(point_id)
            reached.size += 1
            self._id_to_leaf[point_id] = reached
        else:
            node = self._split_leaf(reached, new_leaf)
            self._id_to_leaf[point_id] = new_leaf

        if not descent.path:
            self.root = node
            node.parent = None
            return
        parent, go_left, _ = descent.path[-1]
        if go_left:
            parent.left = node
        else:
            parent.right = node
        node.parent = parent
        # Each ancestor gains the point, and boxes grow where the walk said.
        for parent, _, grown in descent.path:
            parent.size += 1
            if grown is not None:
                parent.box = grown.copy()

    def _split_leaf(
        self,
        existing_leaf: _RCFLeaf,
        new_leaf: _RCFLeaf,
    ) -> _RCFBranch:
        """Create a separating branch between two non-identical leaves."""
        diffs = np.abs(existing_leaf.point - new_leaf.point)
        total_diff = float(diffs.sum())

        if total_diff <= 0.0:
            cut_dim = 0
            cut_value = float(existing_leaf.point[0])
        else:
            probs = diffs / total_diff
            if math.isfinite(total_diff):
                cut_dim = _weighted_choice(self.rng, probs)
            else:
                cut_dim = int(self.rng.choice(diffs.size, p=probs))
            existing_value = float(existing_leaf.point[cut_dim])
            new_value = float(new_leaf.point[cut_dim])
            low = min(existing_value, new_value)
            high = max(existing_value, new_value)
            cut_value = float(self.rng.uniform(low, high))
            if (existing_value <= cut_value) == (new_value <= cut_value):
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


def _plan(trees: list[_RandomCutTree], point: np.ndarray) -> list[_Descent]:
    """Walk ``point`` down every tree, leaving the generators where they were."""
    states = [tree.rng.bit_generator.state for tree in trees]
    try:
        return _descend(trees, point)
    finally:
        for tree, state in zip(trees, states, strict=True):
            tree.rng.bit_generator.state = state


def _codisp(descent: _Descent, values: list[float]) -> float:
    """Return the CoDisp the point would have after the planned insertion.

    The CoDisp is the largest ratio of a sibling's size to the size of the
    subtree that holds the point, over the point's ancestors after the
    insertion.
    """
    node = descent.node
    if isinstance(node, _RCFLeaf) and node.point.tolist() == values:
        # A duplicate joins the leaf it reaches.
        largest = 0.0
    else:
        # The point becomes a new leaf whose sibling is ``node``.
        largest = float(node.size)
    size = node.size + 1
    for parent, go_left, _ in reversed(descent.path):
        sibling = parent.right if go_left else parent.left
        largest = max(largest, float(sibling.size) / float(size))
        size = parent.size + 1
    return largest


class RandomCutForest(BaseModel):
    """
    Random Cut Forest for online anomaly detection.

    This model keeps an ensemble of random cut trees over a bounded sample of
    recent shingled points. `learn_one` performs one-sample updates and
    forgetting, while `score_one` returns an anomaly score for one sample.

    Tree insertion/removal and raw anomaly scoring follow Random Cut Tree
    mechanics. ``score_one`` gives each query the collusive displacement
    (CoDisp) it would have if it were inserted, which reference scoring gets by
    inserting the query, measuring it and removing it again. Each tree walks
    the query down as that insertion would and then restores its RNG state, so
    scoring changes no learned state. Learning the point that was scored last
    reuses those walks wherever removing the oldest point left them intact.
    Optional exponential scaling is a monotonic presentation transform over
    raw CoDisp.

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
        # The walks planned while scoring a point, kept until the next learn.
        self._planned: tuple[bytes, list[_Descent]] | None = None

        seed_sequence = np.random.SeedSequence(self.seed)
        child_sequences = seed_sequence.spawn(self.n_trees)
        self._trees = [
            _RandomCutTree(np.random.default_rng(seq)) for seq in child_sequences
        ]

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        # Planned walks are a cache of the trees, not part of the model.
        state["_planned"] = None
        return state

    def learn_one(self, x: dict[str, float]) -> None:
        """Update forest state with one sample."""
        prepared = self._schema.preview(x)
        planned, self._planned = self._planned, None
        vector = prepared.values.copy()
        self._history.append(vector)

        shingle = self._current_shingle()
        if shingle is None:
            self._schema.commit(prepared)
            return

        changed: list[set[int]] = []
        if len(self._id_window) >= self.sample_size:
            oldest = self._id_window.popleft()
            changed = [tree.remove(oldest) for tree in self._trees]

        point_id = self._next_point_id
        self._next_point_id += 1
        self._id_window.append(point_id)

        # Every tree stores the same point, so the trees share one copy.
        values = shingle.tolist()
        box = _point_box(shingle)
        if self._trees[0].root is None:
            for tree in self._trees:
                tree._attach(point_id, shingle, values, box, None)
        else:
            descents = self._learn_descents(shingle, planned, changed)
            for tree, descent in zip(self._trees, descents, strict=True):
                tree._attach(point_id, shingle, values, box, descent)

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

        descents = _plan(self._trees, query)
        self._planned = (query.tobytes(), descents)
        values = query.tolist()
        raw = float(np.mean([_codisp(descent, values) for descent in descents]))
        if not self.normalize_score:
            return raw

        bounded = 1.0 - float(np.exp(-raw / self.score_scale))
        return float(np.clip(bounded, 0.0, 1.0))

    def _learn_descents(
        self,
        shingle: np.ndarray,
        planned: tuple[bytes, list[_Descent]] | None,
        changed: list[set[int]],
    ) -> list[_Descent]:
        """Return the walks that inserting ``shingle`` takes, drawing cuts.

        Scoring the same point planned these walks before the oldest point
        was removed, and left each generator where its walk began. A walk is
        exact above the first branch that the removal changed, so its tree
        draws again what it drew there and walks on from that branch's
        current place. Walks that the removal did not touch are kept whole.
        """
        if planned is None or planned[0] != shingle.tobytes():
            return _descend(self._trees, shingle)
        descents = planned[1]
        resumed: list[int] = []
        starts: list[tuple[list[_Step], _RCFNode | None]] = []
        for index, (tree, descent) in enumerate(
            zip(self._trees, descents, strict=True)
        ):
            step = descent.first_changed(changed[index]) if changed else None
            if step is None:
                if descent.draws:
                    tree.rng.random(descent.draws)
                continue
            if descent.skipped:
                # Steps without a draw make the count above ``step`` unknown.
                step = 0
            elif step:
                tree.rng.random(step)
            node: _RCFNode | None = tree.root
            if step:
                parent, go_left, _ = descent.path[step - 1]
                node = parent.left if go_left else parent.right
            resumed.append(index)
            starts.append((descent.path[:step], node))
        if resumed:
            walks = _descend([self._trees[i] for i in resumed], shingle, starts)
            for index, descent in zip(resumed, walks, strict=True):
                descents[index] = descent
        return descents

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
