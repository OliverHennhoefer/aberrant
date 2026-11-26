"""Online regression trees: the Hoeffding tree and the adaptive random forest."""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple

import numpy as np

from aberrant._utils.validation import coerce_finite_number, coerce_integer_count
from aberrant.drift import ADWIN
from aberrant.forecast._regression import ScaledRegressor, seeded_rng
from aberrant.forecast.linear import rls_update

LeafPrediction = Literal["mean", "linear", "adaptive"]

# Covariates are standardized, so the root's split grid spans three standard
# deviations either side of the mean.
_SPAN = 3.0
# A split leaves at least this share of a leaf's events on either side.
_MIN_SHARE = 0.05
# Weight of each new error when an adaptive leaf compares its two models.
_LEAF_ALPHA = 0.05
# Initial variance of a linear leaf's weights, in standardized units.
_LEAF_DELTA = 1.0


class _Growth(NamedTuple):
    """How the trees of one regressor grow."""

    grace_period: int
    delta: float
    tau: float
    n_bins: int
    max_depth: int
    leaf_prediction: str
    max_features: int


class _Split(NamedTuple):
    feature: int
    threshold: float
    left: int
    right: int


class _Leaf(NamedTuple):
    depth: int
    # The leaf's region, one interval per covariate.
    lower: np.ndarray
    upper: np.ndarray
    # Smallest and largest covariates seen, +inf and -inf before any.
    seen_low: np.ndarray
    seen_high: np.ndarray
    # Split candidates: covariates, the grid each is binned on and the
    # event weight that fell outside it.
    features: np.ndarray
    low: np.ndarray
    width: np.ndarray
    outside: np.ndarray
    # Per statistic, candidate and grid cell: the event weight and the
    # weighted sums of y and y^2, then for linear leaves of x, x^2 and x y.
    stats: np.ndarray
    # Target moments of every event the leaf and its ancestors' split saw.
    weight: float
    total: float
    square: float
    # Event weight since the last split attempt.
    pending: float
    weights: np.ndarray
    covariance: np.ndarray
    mean_error: float
    linear_error: float


_Node = _Split | _Leaf


def _check_growth(
    *,
    grace_period: object,
    delta: object,
    tau: object,
    n_bins: object,
    max_depth: object,
    leaf_prediction: object,
) -> tuple[int, float, float, int, int, str]:
    grace = coerce_integer_count(grace_period, label="grace_period")
    confidence = coerce_finite_number(delta, label="delta")
    if not 0 < confidence < 1:
        raise ValueError("delta must be in (0, 1)")
    tie = coerce_finite_number(tau, label="tau")
    if tie < 0:
        raise ValueError("tau must be non-negative")
    bins = coerce_integer_count(n_bins, label="n_bins", minimum=2)
    depth = coerce_integer_count(max_depth, label="max_depth", minimum=0)
    if leaf_prediction not in ("mean", "linear", "adaptive"):
        raise ValueError("leaf_prediction must be 'mean', 'linear' or 'adaptive'")
    return grace, confidence, tie, bins, depth, str(leaf_prediction)


def _grid(
    lower: np.ndarray,
    upper: np.ndarray,
    seen_low: np.ndarray,
    seen_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Start and width of the grids over a region and the values seen in it."""
    seen = seen_low <= seen_high
    with np.errstate(invalid="ignore"):
        low = np.where(seen, np.maximum(lower, seen_low), lower)
        high = np.where(seen, np.minimum(upper, seen_high), upper)
        low = np.where(np.isfinite(low), low, np.minimum(-_SPAN, high - 1.0))
        high = np.where(np.isfinite(high), high, np.maximum(_SPAN, low + 1.0))
    narrow = ~(high > low)
    low, high = np.where(narrow, low - 0.5, low), np.where(narrow, high + 0.5, high)
    return low, high - low


def _new_leaf(
    growth: _Growth,
    depth: int,
    *,
    region: tuple[np.ndarray, np.ndarray],
    seen: tuple[np.ndarray, np.ndarray],
    moments: tuple[float, ...],
    model: tuple[np.ndarray, float, float],
    rng: np.random.Generator | None,
) -> _Leaf:
    """A leaf of ``region``, binning the covariates its parent ``seen`` there."""
    lower, upper = region
    d = lower.size
    if depth >= growth.max_depth:
        features = np.zeros(0, dtype=np.intp)
    elif growth.max_features >= d or rng is None:
        features = np.arange(d)
    else:
        features = np.sort(rng.choice(d, size=growth.max_features, replace=False))
    low, width = _grid(
        lower[features], upper[features], seen[0][features], seen[1][features]
    )
    statistics = 3 if growth.leaf_prediction == "mean" else 6
    weights, mean_error, linear_error = model
    return _Leaf(
        depth=depth,
        lower=lower,
        upper=upper,
        seen_low=np.full(d, math.inf),
        seen_high=np.full(d, -math.inf),
        features=features,
        low=low,
        width=width,
        outside=np.zeros(features.size),
        stats=np.zeros((statistics, features.size, growth.n_bins)),
        weight=moments[0],
        total=moments[1],
        square=moments[2],
        pending=0.0,
        weights=weights,
        covariance=np.eye(weights.size) * _LEAF_DELTA,
        mean_error=mean_error,
        linear_error=linear_error,
    )


def _root(growth: _Growth, features: int, rng: np.random.Generator | None) -> _Leaf:
    size = features + 1 if growth.leaf_prediction != "mean" else 0
    return _new_leaf(
        growth,
        0,
        region=(np.full(features, -math.inf), np.full(features, math.inf)),
        seen=(np.full(features, math.inf), np.full(features, -math.inf)),
        moments=(0.0, 0.0, 0.0),
        model=(np.zeros(size), 0.0, 0.0),
        rng=rng,
    )


def _route(nodes: tuple[_Node, ...], z: np.ndarray) -> int:
    index = 0
    node = nodes[0]
    while isinstance(node, _Split):
        index = node.left if z[node.feature] < node.threshold else node.right
        node = nodes[index]
    return index


def _mean(leaf: _Leaf) -> float:
    return leaf.total / leaf.weight if leaf.weight > 0 else 0.0


def _linear(leaf: _Leaf, z: np.ndarray) -> float:
    return float(leaf.weights[:-1] @ z + leaf.weights[-1])


def _leaf_predict(leaf: _Leaf, z: np.ndarray, mode: str) -> float:
    if mode == "mean" or (mode == "adaptive" and leaf.linear_error >= leaf.mean_error):
        return _mean(leaf)
    return _linear(leaf, z)


def _tree_predict(nodes: tuple[_Node, ...], z: np.ndarray, mode: str) -> float:
    leaf = nodes[_route(nodes, z)]
    assert isinstance(leaf, _Leaf)
    return _leaf_predict(leaf, z, mode)


def _finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise OverflowError("Regression arithmetic exceeds the float range")


def _leaf_learned(
    leaf: _Leaf, z: np.ndarray, target: float, weight: float, growth: _Growth
) -> _Leaf:
    stats = leaf.stats
    mass, square = weight * target, weight * target * target
    mean_error = leaf.mean_error + _LEAF_ALPHA * (
        abs(target - _mean(leaf)) - leaf.mean_error
    )
    linear_error = 0.0
    if leaf.weights.size:
        with np.errstate(over="ignore", invalid="ignore"):
            linear_error = leaf.linear_error + _LEAF_ALPHA * (
                abs(target - _linear(leaf, z)) - leaf.linear_error
            )
    moments = (leaf.weight + weight, leaf.total + mass, leaf.square + square)
    _finite(square, mean_error, linear_error, *moments)
    outside = leaf.outside
    if leaf.features.size:
        with np.errstate(over="ignore", invalid="ignore"):
            position = (z[leaf.features] - leaf.low) / leaf.width * growth.n_bins
        outside = outside + weight * ((position < 0) | (position > growth.n_bins))
        bins = np.clip(np.floor(position), 0, growth.n_bins - 1).astype(np.intp)
        rows = np.arange(leaf.features.size)
        increments = np.empty((stats.shape[0], rows.size))
        increments[:3] = ((weight,), (mass,), (square,))
        if stats.shape[0] > 3:
            x = z[leaf.features]
            with np.errstate(over="ignore", invalid="ignore"):
                increments[3:] = (weight * x, weight * x * x, mass * x)
        stats = stats.copy()
        stats[:, rows, bins] += increments
        if not np.isfinite(stats[:, rows, bins]).all():
            raise OverflowError("Regression arithmetic exceeds the float range")
    weights, covariance = leaf.weights, leaf.covariance
    if weights.size:
        root = math.sqrt(weight)
        _, weights, covariance = rls_update(
            weights, covariance, root * np.append(z, 1.0), root * target, 1.0
        )
    return leaf._replace(
        seen_low=np.minimum(leaf.seen_low, z),
        seen_high=np.maximum(leaf.seen_high, z),
        outside=outside,
        stats=stats,
        weight=moments[0],
        total=moments[1],
        square=moments[2],
        pending=leaf.pending + weight,
        weights=weights,
        covariance=covariance,
        mean_error=mean_error,
        linear_error=linear_error,
    )


def _remaining(stats: np.ndarray) -> np.ndarray:
    """Squared error left by the leaf model, per candidate and cell.

    That model is the mean, or with covariate moments a line in the
    candidate covariate.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        n, sy, syy = stats[0], stats[1], stats[2]
        error = syy - sy * sy / n
        if stats.shape[0] > 3:
            sx, sxx, sxy = stats[3], stats[4], stats[5]
            spread = sxx - sx * sx / n
            covariance = sxy - sx * sy / n
            error = error - np.where(spread > 0, covariance**2 / spread, 0.0)
    remaining: np.ndarray = np.maximum(error, 0.0)
    return remaining


def _split(
    leaf: _Leaf, growth: _Growth, rng: np.random.Generator | None
) -> tuple[_Split, _Leaf, _Leaf] | None:
    """The split a Hoeffding bound supports, as the node and its two leaves."""
    totals = leaf.stats.sum(axis=2)
    n = float(totals[0, 0])
    total = float(totals[2, 0] - totals[1, 0] ** 2 / n)
    if not total > 0:
        return None
    lefts = np.cumsum(leaf.stats, axis=2)[:, :, :-1]
    rights = totals[:, :, None] - lefts
    minimum = max(_MIN_SHARE * n, 1e-12)
    valid = (lefts[0] >= minimum) & (rights[0] >= minimum)
    parent = _remaining(totals)[:, None]
    gain = (parent - _remaining(lefts) - _remaining(rights)) / total
    merit = np.where(valid, gain, -math.inf)
    choice = merit.argmax(axis=1)
    best = merit[np.arange(merit.shape[0]), choice]
    row = int(best.argmax())
    rivals = np.delete(best, row)
    second = max(0.0, float(rivals.max())) if rivals.size else 0.0
    epsilon = math.sqrt(math.log(1 / growth.delta) / (2 * n))
    if not (best[row] > 0 and (best[row] - second > epsilon or epsilon < growth.tau)):
        return None
    k = int(choice[row])
    feature = int(leaf.features[row])
    threshold = float(leaf.low[row] + leaf.width[row] * (k + 1) / growth.n_bins)
    left_upper, right_lower = leaf.upper.copy(), leaf.lower.copy()
    left_upper[feature] = right_lower[feature] = threshold
    model = (leaf.weights, leaf.mean_error, leaf.linear_error)
    depth = leaf.depth + 1
    seen = (leaf.seen_low, leaf.seen_high)
    left = _new_leaf(
        growth,
        depth,
        region=(leaf.lower, left_upper),
        seen=seen,
        moments=tuple(float(v) for v in lefts[:3, row, k]),
        model=model,
        rng=rng,
    )
    right = _new_leaf(
        growth,
        depth,
        region=(right_lower, leaf.upper),
        seen=seen,
        moments=tuple(float(v) for v in rights[:3, row, k]),
        model=model,
        rng=rng,
    )
    return _Split(feature, threshold, -1, -1), left, right


def _regridded(leaf: _Leaf) -> _Leaf:
    """The leaf binning afresh over the range of the covariates it saw."""
    f = leaf.features
    low, width = _grid(
        leaf.lower[f], leaf.upper[f], leaf.seen_low[f], leaf.seen_high[f]
    )
    return leaf._replace(
        low=low,
        width=width,
        outside=np.zeros(f.size),
        stats=np.zeros_like(leaf.stats),
        pending=0.0,
    )


def _tree_learned(
    nodes: tuple[_Node, ...],
    z: np.ndarray,
    target: float,
    *,
    weight: float,
    growth: _Growth,
    rng: np.random.Generator | None,
) -> tuple[_Node, ...]:
    """The tree after learning one weighted event, as new nodes."""
    index = _route(nodes, z)
    leaf = nodes[index]
    assert isinstance(leaf, _Leaf)
    leaf = _leaf_learned(leaf, z, target, weight, growth)
    replacement: tuple[_Node, ...] = (leaf,)
    if leaf.features.size and leaf.pending >= growth.grace_period:
        if (leaf.outside > _MIN_SHARE * leaf.stats[0, 0].sum()).any():
            return (*nodes[:index], _regridded(leaf), *nodes[index + 1 :])
        split = _split(leaf, growth, rng)
        if split is None:
            replacement = (leaf._replace(pending=0.0),)
        else:
            node, left, right = split
            size = len(nodes)
            node = node._replace(left=size, right=size + 1)
            return (*nodes[:index], node, *nodes[index + 1 :], left, right)
    return (*nodes[:index], *replacement, *nodes[index + 1 :])


class HoeffdingTreeRegressor(ScaledRegressor):
    """An online regression tree that splits once the evidence suffices.

    Each leaf bins the standardized covariates of its events on a grid of
    ``n_bins`` over the range its parent saw in its region, and keeps the
    moments of the target per cell. Every ``grace_period`` events it scores
    each grid edge by the share of target variance a split there explains
    beyond the leaf's model, the mean or, for linear leaves, a line in that
    covariate. It splits at the best edge when the Hoeffding bound shows it
    beats the best split of every other covariate, and not splitting, at
    confidence ``1 - delta``, or when the bound falls below ``tau``
    (Domingos and Hulten, 2000; Ikonomovska et al., 2011). A split leaves at
    least 5% of the leaf's events on either side, and a leaf with more than
    5% outside its grid bins afresh over the range it saw.

    Leaves predict the mean target of their events, a linear model of the
    covariates fit by recursive least squares, or, with ``"adaptive"``,
    whichever of the two has the smaller faded absolute error so far. With
    linear leaves over lagged values the tree is a threshold autoregression
    that finds its own regimes.

    Args:
        grace_period: Events a leaf learns between split attempts, at
            least 1.
        delta: Probability in (0, 1) that a split is not the best one.
        tau: Bound, at least 0, below which near ties are split anyway.
        n_bins: Grid cells per covariate and leaf, at least 2.
        max_depth: Largest depth of a leaf, at least 0.
        leaf_prediction: ``"mean"``, ``"linear"`` or ``"adaptive"``.
        warm_up: Events whose means and standard deviations fix the scaling
            of the covariates and the target, at least 1.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error that gives the spread, in (0, 1].
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        Ready after ``warm_up + 1`` events. A tree of ``L`` leaves over
        ``d`` covariates keeps O(L d (n_bins + d)) numbers, at most
        ``2 ** max_depth`` leaves, and each event costs O(depth + d n_bins +
        d^2). Unrepresentable arithmetic raises OverflowError without
        changing state.

    Examples:
        ```python
        from aberrant.forecast import HoeffdingTreeRegressor

        model = HoeffdingTreeRegressor(grace_period=50)
        for i in range(2000):
            a = (i % 100) / 50 - 1
            model.learn_one({"a": a}, 1.0 if a > 0.2 else -1.0)
        model.predict_one({"a": 0.6}).mean  # close to (1.0,)
        ```

    References:
        Domingos and Hulten, Mining High-Speed Data Streams, KDD 2000.
        Ikonomovska, Gama and Dzeroski, Learning Model Trees from Evolving
        Data Streams, Data Mining and Knowledge Discovery 23, 2011.
    """

    def __init__(
        self,
        *,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        n_bins: int = 32,
        max_depth: int = 8,
        leaf_prediction: LeafPrediction = "adaptive",
        warm_up: int = 32,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        (
            self.grace_period,
            self.delta,
            self.tau,
            self.n_bins,
            self.max_depth,
            self.leaf_prediction,
        ) = _check_growth(
            grace_period=grace_period,
            delta=delta,
            tau=tau,
            n_bins=n_bins,
            max_depth=max_depth,
            leaf_prediction=leaf_prediction,
        )
        self._init_scaling(warm_up, residual_alpha, min_scale)
        self._reset_state()

    @property
    def n_leaves(self) -> int:
        """Number of leaves of the tree, 0 before the warm-up ends."""
        return sum(isinstance(node, _Leaf) for node in self._nodes)

    def _growth(self, features: int) -> _Growth:
        return _Growth(
            self.grace_period,
            self.delta,
            self.tau,
            self.n_bins,
            self.max_depth,
            self.leaf_prediction,
            features,
        )

    def _reset_model(self) -> None:
        self._nodes: tuple[_Node, ...] = ()

    def _start(self, features: int) -> None:
        self._nodes = (_root(self._growth(features), features, None),)

    def _predict(self, z: np.ndarray) -> float:
        return _tree_predict(self._nodes, z, self.leaf_prediction)

    def _learned(self, z: np.ndarray, target: float) -> dict[str, Any]:
        growth = self._growth(z.size)
        return {
            "_nodes": _tree_learned(
                self._nodes, z, target, weight=1.0, growth=growth, rng=None
            )
        }


class AdaptiveRandomForestRegressor(ScaledRegressor):
    """Online bagging of Hoeffding trees that replaces the trees that drift.

    Each of ``n_models`` trees learns every event with a Poisson
    ``lambda_value`` weight, as online bagging does, and each new leaf
    considers ``max_features`` covariates drawn at random. Two ADWIN
    detectors watch every tree's bounded error ``|e| / (1 + |e|)``, ``e`` in
    target standard deviations: a warning starts a background tree that
    learns alongside, and a drift replaces the tree by it (Gomes et al.,
    2017 and 2018). The forest predicts the mean of its trees.

    Args:
        n_models: Number of trees, at least 1.
        max_features: Covariates each leaf considers, at least 1, or
            ``None`` for the rounded square root of their number.
        lambda_value: Positive mean of the Poisson event weights.
        drift_delta: ADWIN confidence in (0, 1) that replaces a tree.
        warning_delta: ADWIN confidence in (0, 1) that starts a background
            tree; larger than ``drift_delta`` warns earlier.
        grace_period: Events a leaf learns between split attempts, at
            least 1.
        delta: Probability in (0, 1) that a split is not the best one.
        tau: Bound, at least 0, below which near ties are split anyway.
        n_bins: Grid cells per covariate and leaf, at least 2.
        max_depth: Largest depth of a leaf, at least 0.
        leaf_prediction: ``"mean"``, ``"linear"`` or ``"adaptive"``, as in
            :class:`HoeffdingTreeRegressor`.
        warm_up: Events whose means and standard deviations fix the scaling
            of the covariates and the target, at least 1.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error that gives the spread, in (0, 1].
        min_scale: Positive floor for the reported spread, in target units.
        seed: Seed of the event weights and covariate draws, or ``None``.

    Notes:
        Ready after ``warm_up + 1`` events. State and cost are at most twice
        ``n_models`` times a :class:`HoeffdingTreeRegressor`'s, plus the
        detectors. Unrepresentable arithmetic raises OverflowError without
        changing state.

    References:
        Gomes, Bifet, Read, Barddal, Enembreck, Pfharinger, Holmes and
        Abdessalem, Adaptive Random Forests for Evolving Data Stream
        Classification, Machine Learning 106, 2017. Gomes, Barddal, Ferreira
        and Bifet, Adaptive Random Forests for Data Stream Regression,
        ESANN 2018.
    """

    def __init__(
        self,
        *,
        n_models: int = 10,
        max_features: int | None = None,
        lambda_value: float = 6.0,
        drift_delta: float = 0.001,
        warning_delta: float = 0.01,
        grace_period: int = 50,
        delta: float = 0.01,
        tau: float = 0.05,
        n_bins: int = 32,
        max_depth: int = 8,
        leaf_prediction: LeafPrediction = "adaptive",
        warm_up: int = 32,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        self.n_models = coerce_integer_count(n_models, label="n_models")
        self.max_features = (
            None
            if max_features is None
            else coerce_integer_count(max_features, label="max_features")
        )
        self.lambda_value = coerce_finite_number(lambda_value, label="lambda_value")
        if self.lambda_value <= 0:
            raise ValueError("lambda_value must be positive")
        self.drift_delta = coerce_finite_number(drift_delta, label="drift_delta")
        self.warning_delta = coerce_finite_number(warning_delta, label="warning_delta")
        for label, value in (
            ("drift_delta", self.drift_delta),
            ("warning_delta", self.warning_delta),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{label} must be in (0, 1)")
        (
            self.grace_period,
            self.delta,
            self.tau,
            self.n_bins,
            self.max_depth,
            self.leaf_prediction,
        ) = _check_growth(
            grace_period=grace_period,
            delta=delta,
            tau=tau,
            n_bins=n_bins,
            max_depth=max_depth,
            leaf_prediction=leaf_prediction,
        )
        self._init_scaling(warm_up, residual_alpha, min_scale)
        self.seed = seed
        seeded_rng(seed)
        self._reset_state()

    @property
    def n_drifts(self) -> int:
        """Number of trees replaced after a drift since the last reset."""
        return self._drifted

    def _growth(self, features: int) -> _Growth:
        count = self.max_features or max(1, round(math.sqrt(features)))
        return _Growth(
            self.grace_period,
            self.delta,
            self.tau,
            self.n_bins,
            self.max_depth,
            self.leaf_prediction,
            min(count, features),
        )

    def _reset_model(self) -> None:
        self._rng: np.random.Generator | None = None
        self._trees: tuple[tuple[_Node, ...], ...] = ()
        self._backgrounds: tuple[tuple[_Node, ...] | None, ...] = ()
        self._warnings: list[ADWIN] = []
        self._drifts: list[ADWIN] = []
        self._drifted = 0

    def _tree(self, features: int) -> tuple[_Node, ...]:
        return (_root(self._growth(features), features, self._rng),)

    def _start(self, features: int) -> None:
        self._rng = seeded_rng(self.seed)
        self._trees = tuple(self._tree(features) for _ in range(self.n_models))
        self._backgrounds = (None,) * self.n_models
        self._warnings = [ADWIN(delta=self.warning_delta) for _ in self._trees]
        self._drifts = [ADWIN(delta=self.drift_delta) for _ in self._trees]

    def _predict(self, z: np.ndarray) -> float:
        mode = self.leaf_prediction
        total = sum(_tree_predict(tree, z, mode) for tree in self._trees)
        return total / len(self._trees)

    def _learned(self, z: np.ndarray, target: float) -> dict[str, Any]:
        assert self._rng is not None
        growth = self._growth(z.size)
        mode = self.leaf_prediction
        state = self._rng.bit_generator.state
        try:
            counts = self._rng.poisson(self.lambda_value, size=self.n_models)
            trees: list[tuple[_Node, ...]] = []
            backgrounds: list[tuple[_Node, ...] | None] = []
            errors: list[float] = []
            for current, pending, count in zip(
                self._trees, self._backgrounds, counts, strict=True
            ):
                with np.errstate(over="ignore", invalid="ignore"):
                    error = abs(target - _tree_predict(current, z, mode))
                _finite(error)
                errors.append(error / (1 + error))
                tree, background = current, pending
                if count:
                    weight = float(count)
                    tree = _tree_learned(
                        tree, z, target, weight=weight, growth=growth, rng=self._rng
                    )
                    if background is not None:
                        background = _tree_learned(
                            background,
                            z,
                            target,
                            weight=weight,
                            growth=growth,
                            rng=self._rng,
                        )
                trees.append(tree)
                backgrounds.append(background)
        except BaseException:
            self._rng.bit_generator.state = state
            raise
        drifted = self._drifted
        for index, error in enumerate(errors):
            if self._warnings[index].update(error):
                backgrounds[index] = self._tree(z.size)
            if self._drifts[index].update(error):
                trees[index] = backgrounds[index] or self._tree(z.size)
                backgrounds[index] = None
                self._warnings[index] = ADWIN(delta=self.warning_delta)
                drifted += 1
        return {
            "_trees": tuple(trees),
            "_backgrounds": tuple(backgrounds),
            "_drifted": drifted,
        }
