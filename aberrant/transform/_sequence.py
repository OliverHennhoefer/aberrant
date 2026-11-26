"""Sequence transformers: each feature's output follows its own recent values."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from aberrant._utils.validation import coerce_feature_values, coerce_finite_number
from aberrant.base.readiness import Readiness
from aberrant.base.transformer import BaseTransformer
from aberrant.exceptions import ValidationError

State = tuple[Any, ...]
Window = tuple[float, ...]
ChangeKind = Literal["difference", "percent"]
Statistic = Literal[
    "sum",
    "mean",
    "std",
    "min",
    "max",
    "range",
    "median",
    "quantile",
    "iqr",
    "mad",
    "slope",
    "zscore",
    "robust_zscore",
    "rank",
    "rms",
    "skewness",
    "kurtosis",
]
# Scales a median absolute deviation to the standard deviation of normal data.
_MAD_SCALE = 1.4826
# Scales a mean absolute deviation to the standard deviation of normal data.
_MEAN_DEVIATION_SCALE = math.sqrt(math.pi / 2)


def _positive_int(value: object, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _positive_number(value: object, name: str) -> float:
    number = coerce_finite_number(value, label=name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _features(features: Sequence[str] | None) -> list[str] | None:
    if features is None:
        return None
    if isinstance(features, str | bytes):
        raise ValueError("features must be a sequence of names, not a string")
    names = list(features)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("features must contain non-empty feature names")
    if len(names) != len(set(names)):
        raise ValueError("features cannot contain duplicates")
    return names


def _floats(items: object, limit: int, exact: bool = False) -> bool:
    """Whether items is a tuple of at most (or exactly) limit finite floats."""
    return (
        isinstance(items, tuple)
        and (len(items) == limit if exact else len(items) <= limit)
        and all(type(item) is float and math.isfinite(item) for item in items)
    )


def _window(state: object, size: int, ordered: bool) -> bool:
    """Whether state is a window, with its sorted copy when ordered."""
    if not isinstance(state, tuple) or len(state) != 1 + ordered:
        return False
    return _floats(state[0], size) and (
        not ordered or state[1] == tuple(sorted(state[0]))
    )


def _pushed(window: Window, value: float, size: int) -> Window:
    """The window with value appended, dropping the oldest value once full."""
    return (*window, value)[-size:]


def _reordered(ordered: Window, leaving: float | None, value: float) -> Window:
    """The sorted window after leaving goes and value arrives."""
    items = list(ordered)
    if leaving is not None:
        del items[bisect_left(items, leaving)]
    insort(items, value)
    return tuple(items)


class _SequenceTransformer(BaseTransformer):
    """Shared state handling of the sequence transformers.

    Each selected feature keeps its own immutable state, which ``step``
    advances by one value. ``learn_one`` computes the event's output from the
    state before the event and then advances it. ``transform_one`` of the
    event just learned returns that output, as a pipeline's post-update pass
    needs; any other event gets the output it would get as the next one,
    without learning.

    ``initial_state``, ``step`` and ``valid_state`` let an application keep
    the states itself, one per feature; they never change the transformer.
    """

    _keeps_input = False

    def __init__(self, features: Sequence[str] | None) -> None:
        self.features = _features(features)
        self._states: dict[str, State] = {}
        self._seen: dict[str, int] = {}
        self._learned: tuple[dict[str, float], dict[str, float]] | None = None

    @property
    def lead(self) -> int:
        """Earlier values a feature needs before its output is final."""
        raise NotImplementedError

    def initial_state(self) -> State:
        """The state of a feature that has seen no value."""
        return ()

    def step(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        """Advance one feature's state by a finite float.

        Returns the next state and the value's outputs, one per output name;
        ``state`` itself never changes. Raises ValidationError when an output
        is undefined or not finite.
        """
        try:
            following, outputs = self._advance(state, value)
        except OverflowError:
            raise ValidationError("The result must be finite") from None
        if not all(math.isfinite(output) for output in outputs):
            raise ValidationError("The result must be finite")
        return following, outputs

    def valid_state(self, state: object) -> bool:
        """Whether a restored state has the shape this transformer's states have."""
        return state == ()

    def _advance(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        raise NotImplementedError

    def _names(self, feature: str) -> tuple[str, ...]:
        return (feature,)

    @property
    def readiness(self) -> Readiness:
        """Ready once every learned feature has seen ``lead + 1`` values."""
        required = self.lead + 1
        observed = min(self._seen.values(), default=0)
        return Readiness(bool(self._seen) and observed >= required, observed, required)

    def _selected(self, values: dict[str, float]) -> list[str]:
        if self.features is None:
            return list(values)
        missing = [name for name in self.features if name not in values]
        if missing:
            raise ValidationError(f"Missing selected features: {sorted(missing)}")
        return self.features

    def _outputs(
        self, values: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, State]]:
        selected = set(self._selected(values))
        result: dict[str, float] = {}
        states: dict[str, State] = {}
        appended: list[tuple[str, float]] = []
        for key, value in values.items():
            if key not in selected:
                result[key] = value
                continue
            state = self._states.get(key)
            try:
                states[key], outputs = self.step(
                    self.initial_state() if state is None else state, value
                )
            except ValidationError as error:
                raise ValidationError(f"Feature '{key}': {error}") from None
            named = list(zip(self._names(key), outputs, strict=True))
            if self._keeps_input:
                result[key] = value
                appended.extend(named)
            else:
                result.update(named)
        for name, output in appended:
            if name in result:
                raise ValidationError(f"Feature '{name}' already exists")
            result[name] = output
        return result, states

    @contextmanager
    def learning_transaction(self, x: dict[str, float]) -> Iterator[None]:
        """Restore every feature's state if a downstream update fails."""
        states, seen, learned = dict(self._states), dict(self._seen), self._learned
        try:
            yield
        except BaseException:
            self._states, self._seen, self._learned = states, seen, learned
            raise

    def learn_one(self, x: dict[str, float]) -> None:
        """Compute the event's output, then advance each selected feature."""
        values = coerce_feature_values(x)
        result, states = self._outputs(values)
        self._states.update(states)
        for name in states:
            self._seen[name] = self._seen.get(name, 0) + 1
        self._learned = (values, result)

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """The event just learned's output, or the next event's, without learning."""
        values = coerce_feature_values(x)
        if self._learned is not None and self._learned[0] == values:
            result = self._learned[1]
            self._learned = None
            return dict(result)
        return self._outputs(values)[0]


class Lags(_SequenceTransformer):
    """Add each selected feature's values a few events back.

    The output keeps every input feature and appends ``{feature}_lag{k}`` for
    each lag ``k``, ascending. Until a feature has ``k`` earlier values, its
    lag ``k`` repeats the oldest one, or the value itself at the start.

    Each event costs O(F L) for F selected features and L lags, and each
    feature keeps its largest lag's worth of values.

    Args:
        lags: Distinct positive lags, in events.
        features: Features to lag; ``None`` lags every feature of each event.
    """

    _keeps_input = True

    def __init__(
        self, *, lags: Sequence[int] = (1,), features: Sequence[str] | None = None
    ) -> None:
        if isinstance(lags, str | bytes) or not isinstance(lags, Sequence):
            raise ValueError("lags must be a sequence of positive integers")
        checked = [_positive_int(lag, "lags") for lag in lags]
        if not checked or len(checked) != len(set(checked)):
            raise ValueError("lags must be distinct and non-empty")
        self.lags = sorted(checked)
        super().__init__(features)

    @property
    def lead(self) -> int:
        return self.lags[-1]

    def _names(self, feature: str) -> tuple[str, ...]:
        return tuple(f"{feature}_lag{lag}" for lag in self.lags)

    def valid_state(self, state: object) -> bool:
        return _floats(state, self.lags[-1])

    def _advance(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        outputs = tuple(
            state[-lag] if len(state) >= lag else state[0] if state else value
            for lag in self.lags
        )
        return _pushed(state, value, self.lags[-1]), outputs


class Difference(_SequenceTransformer):
    """Replace each selected feature by its change over ``lag`` events.

    ``kind="difference"`` gives ``x[t] - x[t - lag]`` and ``"percent"`` gives
    ``100 (x[t] - x[t - lag]) / |x[t - lag]|``, which rejects a reference of
    0. Until a feature has ``lag`` earlier values, its change is 0. ``Lags``
    gives the value ``lag`` events back itself.

    Args:
        lag: Events between the value and its reference.
        kind: ``"difference"`` or ``"percent"``.
        features: Features to difference; ``None`` uses every feature.
    """

    def __init__(
        self,
        *,
        lag: int = 1,
        kind: ChangeKind = "difference",
        features: Sequence[str] | None = None,
    ) -> None:
        self.lag = _positive_int(lag, "lag")
        if kind not in ("difference", "percent"):
            raise ValueError("kind must be 'difference' or 'percent'")
        self.kind = kind
        super().__init__(features)

    @property
    def lead(self) -> int:
        return self.lag

    def valid_state(self, state: object) -> bool:
        return _floats(state, self.lag)

    def _advance(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        following = _pushed(state, value, self.lag)
        if len(state) < self.lag:
            return following, (0.0,)
        reference = state[-self.lag]
        if self.kind == "difference":
            return following, (value - reference,)
        if reference == 0:
            raise ValidationError("A percent change from 0 is undefined")
        return following, (100 * (value - reference) / abs(reference),)


class ExponentialSmoother(_SequenceTransformer):
    """Replace each selected feature by its exponentially weighted mean.

    ``y[t] = y[t - 1] + alpha (x[t] - y[t - 1])`` with ``alpha = 2 /
    (span + 1)``, starting from the first value. Readiness waits for
    ``span`` values, when the start no longer dominates.

    Args:
        span: Events a value's weight spans; 1 copies the input.
        features: Features to smooth; ``None`` uses every feature.
    """

    def __init__(
        self, *, span: int = 10, features: Sequence[str] | None = None
    ) -> None:
        self.span = _positive_int(span, "span")
        super().__init__(features)

    @property
    def lead(self) -> int:
        return self.span - 1

    def valid_state(self, state: object) -> bool:
        return _floats(state, 1)

    def _advance(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        if not state:
            return (value,), (value,)
        level = state[0] + 2 / (self.span + 1) * (value - state[0])
        return (level,), (level,)


class Despike(_SequenceTransformer):
    """Replace a value far from its recent median by the median (Hampel).

    The window holds the last ``window_size`` values, the current one
    included. A value more than ``threshold`` scaled median absolute
    deviations (``1.4826 MAD``) from the median is a spike. A flat window
    replaces nothing, and an excursion longer than half the window moves the
    median and passes. A window of fewer than 3 values replaces nothing.

    Each event costs O(F W) for F selected features and window W.

    Args:
        window_size: Values in the window, at least 3.
        threshold: Scaled deviations beyond which a value is a spike.
        features: Features to despike; ``None`` uses every feature.
    """

    def __init__(
        self,
        *,
        window_size: int = 7,
        threshold: float = 3.0,
        features: Sequence[str] | None = None,
    ) -> None:
        self.window_size = _positive_int(window_size, "window_size", 3)
        self.threshold = _positive_number(threshold, "threshold")
        super().__init__(features)

    @property
    def lead(self) -> int:
        return self.window_size - 1

    def initial_state(self) -> State:
        return ((), ())

    def valid_state(self, state: object) -> bool:
        return _window(state, self.window_size, ordered=True)

    def _advance(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        values, ordered = state
        leaving = values[0] if len(values) == self.window_size else None
        window = _pushed(values, value, self.window_size)
        ordered = _reordered(ordered, leaving, value)
        if len(window) < 3:
            return (window, ordered), (value,)
        median = _quantile(ordered, 0.5)
        spread = _median_deviation(ordered, median)
        spike = (
            spread > 0 and abs(value - median) > self.threshold * _MAD_SCALE * spread
        )
        return (window, ordered), (median if spike else value,)


class RollingStatistic(_SequenceTransformer):
    """Replace each selected feature by a statistic of its recent values.

    The window holds the last ``window_size`` values, the current one
    included, and starts shorter. Quantiles interpolate linearly.

    - ``sum``, ``mean``, ``min``, ``max``, ``range`` and ``median``.
    - ``std``: the sample deviation, 0 for one value.
    - ``quantile`` at probability ``quantile``; ``iqr`` the interquartile
      range; ``mad`` the unscaled median absolute deviation.
    - ``slope``: the least-squares change per event.
    - ``zscore``: the value's distance from the window mean in sample
      deviations. ``robust_zscore``: its distance from the median in scaled
      MADs, or in scaled mean absolute deviations when more than half the
      window holds one value. Both are 0 in a flat window.
    - ``rank``: the value's place among the window's values, from 0 for the
      lowest to 1 for the highest, ties counting half; 0.5 for one value.
    - ``rms``: the root mean square.
    - ``skewness`` and ``kurtosis`` (excess over a normal distribution), 0 in
      a flat window.

    Deviations are taken without squares that overflow or vanish, so a window
    of large values that differ little keeps its spread. Each event costs
    O(F W) for F selected features and window W.

    Args:
        window_size: Values in the window.
        statistic: The statistic, as listed above.
        quantile: Probability in [0, 1] for ``statistic="quantile"``.
        features: Features to replace; ``None`` uses every feature.
    """

    def __init__(
        self,
        *,
        window_size: int = 10,
        statistic: Statistic = "mean",
        quantile: float = 0.5,
        features: Sequence[str] | None = None,
    ) -> None:
        self.window_size = _positive_int(window_size, "window_size")
        if statistic not in _STATISTIC:
            raise ValueError(f"statistic must be one of {', '.join(_STATISTIC)}")
        self.statistic = statistic
        probability = coerce_finite_number(quantile, label="quantile")
        if not 0 <= probability <= 1:
            raise ValueError("quantile must be in [0, 1]")
        self.quantile = probability
        super().__init__(features)

    @property
    def lead(self) -> int:
        return self.window_size - 1

    @property
    def _ordered(self) -> bool:
        return self.statistic in _ORDERED

    def initial_state(self) -> State:
        return ((), ()) if self._ordered else ((),)

    def valid_state(self, state: object) -> bool:
        return _window(state, self.window_size, self._ordered)

    def _advance(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        values = state[0]
        window = _pushed(values, value, self.window_size)
        if not self._ordered:
            return (window,), (_STATISTIC[self.statistic](window, (), self.quantile),)
        leaving = values[0] if len(values) == self.window_size else None
        ordered = _reordered(state[1], leaving, value)
        result = _STATISTIC[self.statistic](window, ordered, self.quantile)
        return (window, ordered), (result,)


def _quantile(ordered: Window, probability: float) -> float:
    position = (len(ordered) - 1) * probability
    low = int(position)
    weight = position - low
    if weight == 0:
        return ordered[low]
    return (1 - weight) * ordered[low] + weight * ordered[low + 1]


def _deviation(ordered: Window, centre: float, rank: int) -> float:
    """The rank-th smallest distance from centre, counted from 1.

    In a sorted window the rank values nearest to centre lie side by side, so
    a binary search finds where they start and nothing is sorted again.
    """
    low, high = 0, len(ordered) - rank
    while low < high:
        middle = (low + high) // 2
        if centre - ordered[middle] > ordered[middle + rank] - centre:
            low = middle + 1
        else:
            high = middle
    return max(centre - ordered[low], ordered[low + rank - 1] - centre)


def _median_deviation(ordered: Window, centre: float) -> float:
    """The median distance from centre."""
    size = len(ordered)
    middle = _deviation(ordered, centre, size // 2 + 1)
    if size % 2:
        return middle
    return 0.5 * _deviation(ordered, centre, size // 2) + 0.5 * middle


def _deviations(window: Window) -> list[float]:
    """Each value's distance from the mean of the window; 0 in a flat window.

    The mean of large values that differ little rounds to a value next to
    them, which would move every distance by as much as the values differ.
    Values on one side of 0 are therefore measured from the smallest one
    first, and the mean is taken of what remains.
    """
    low, high = min(window), max(window)
    if low == high:
        return [0.0] * len(window)
    start = low if low > 0 or high < 0 else 0.0
    offsets = [item - start for item in window]
    mean = math.fsum(offsets) / len(offsets)
    return [item - mean for item in offsets]


def _squares(values: Sequence[float]) -> float:
    return math.fsum(item * item for item in values)


def _root_mean_square(values: Sequence[float], divisor: int) -> float:
    """The root of the sum of squares over divisor, without overflow."""
    largest = max(map(abs, values))
    if largest == 0:
        return 0.0
    return largest * math.sqrt(_squares([item / largest for item in values]) / divisor)


def _std(window: Window, _ordered: Window, _q: float) -> float:
    if len(window) < 2:
        return 0.0
    return _root_mean_square(_deviations(window), len(window) - 1)


def _rms(window: Window, _ordered: Window, _q: float) -> float:
    # The root lies between the smallest and the largest magnitude, which
    # rounding alone may not keep.
    return max(_root_mean_square(window, len(window)), min(map(abs, window)))


def _slope(window: Window, _ordered: Window, _q: float) -> float:
    size = len(window)
    if size < 2:
        return 0.0
    # Positions centred on the window sum to zero, so the mean drops out and
    # the deviations from it keep the products small.
    centre = (size - 1) / 2
    spread = size * (size * size - 1) / 12
    deviations = _deviations(window)
    return math.fsum((i - centre) * item for i, item in enumerate(deviations)) / spread


def _rank(window: Window, ordered: Window, _q: float) -> float:
    if len(window) < 2:
        return 0.5
    below = bisect_left(ordered, window[-1])
    equal = bisect_right(ordered, window[-1]) - below
    return (below + (equal - 1) / 2) / (len(window) - 1)


def _robust_zscore(window: Window, ordered: Window, _q: float) -> float:
    median = _quantile(ordered, 0.5)
    spread = _MAD_SCALE * _median_deviation(ordered, median)
    if spread == 0:
        mean_deviation = math.fsum(abs(item - median) for item in window)
        spread = _MEAN_DEVIATION_SCALE * mean_deviation / len(window)
    return (window[-1] - median) / spread if spread else 0.0


def _moment(statistic: str) -> Callable[[Window, Window, float], float]:
    """zscore, skewness or excess kurtosis, within the values they can take."""

    def moment(window: Window, _ordered: Window, _q: float) -> float:
        size = len(window)
        deviations = _deviations(window)
        largest = max(map(abs, deviations))
        if largest == 0:
            return 0.0
        unit = [item / largest for item in deviations]
        second = _squares(unit) / size
        if statistic == "zscore":
            # Samuelson: no value lies more than (n - 1) / sqrt(n) deviations out.
            score = unit[-1] / math.sqrt(second * size / (size - 1))
            limit = (size - 1) / math.sqrt(size)
            return min(max(score, -limit), limit)
        if statistic == "skewness":
            third = math.fsum(item * item * item for item in unit) / size
            limit = (size - 2) / math.sqrt(size - 1)
            return min(max(third / math.pow(second, 1.5), -limit), limit)
        fourth = _squares([item * item for item in unit]) / size
        high = max(size - 5 + 1 / (size - 1), 0.0)
        return min(max(fourth / (second * second) - 3, -2.0), high)

    return moment


_ORDERED = frozenset(
    {"median", "min", "max", "range", "quantile", "iqr", "mad", "robust_zscore"}
    | {"rank"}
)
_STATISTIC: dict[str, Callable[[Window, Window, float], float]] = {
    "sum": lambda window, _ordered, _q: math.fsum(window),
    "mean": lambda window, _ordered, _q: math.fsum(window) / len(window),
    "std": _std,
    "min": lambda _window, ordered, _q: ordered[0],
    "max": lambda _window, ordered, _q: ordered[-1],
    "range": lambda _window, ordered, _q: ordered[-1] - ordered[0],
    "median": lambda _window, ordered, _q: _quantile(ordered, 0.5),
    "quantile": lambda _window, ordered, q: _quantile(ordered, q),
    "iqr": lambda _window, ordered, _q: (
        _quantile(ordered, 0.75) - _quantile(ordered, 0.25)
    ),
    "mad": lambda _window, ordered, _q: _median_deviation(
        ordered, _quantile(ordered, 0.5)
    ),
    "slope": _slope,
    "zscore": _moment("zscore"),
    "robust_zscore": _robust_zscore,
    "rank": _rank,
    "rms": _rms,
    "skewness": _moment("skewness"),
    "kurtosis": _moment("kurtosis"),
}


class Deseasonalize(_SequenceTransformer):
    """Subtract from each value the usual level of its position in the cycle.

    Event ``t`` sits at position ``t mod season_length`` of its feature's
    cycle. Each position keeps a level and a spread. They average the
    position's values until ``memory`` cycles are counted, then forget the
    oldest at that pace. A value further than four spreads from its level
    moves them as if it lay at four, so one unusual cycle leaves the profile
    where it was. The level is read before the value updates it, and a
    position without a level gives 0.

    Each event costs O(F m) for F selected features and season length m.

    Args:
        season_length: Events in one cycle, at least 2.
        memory: Cycles the levels average over.
        features: Features to deseasonalize; ``None`` uses every feature.
    """

    def __init__(
        self,
        *,
        season_length: int,
        memory: int = 10,
        features: Sequence[str] | None = None,
    ) -> None:
        self.season_length = _positive_int(season_length, "season_length", 2)
        self.memory = _positive_int(memory, "memory")
        super().__init__(features)

    @property
    def lead(self) -> int:
        return self.season_length

    def initial_state(self) -> State:
        empty = (0.0,) * self.season_length
        return (0, empty, empty, (0,) * self.season_length)

    def valid_state(self, state: object) -> bool:
        m = self.season_length
        if not isinstance(state, tuple) or len(state) != 4:
            return False
        count, levels, spreads, counts = state
        return (
            type(count) is int
            and count >= 0
            and _floats(levels, m, exact=True)
            and _floats(spreads, m, exact=True)
            and all(spread >= 0 for spread in spreads)
            and counts == tuple((count - p + m - 1) // m for p in range(m))
        )

    def _advance(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        count, levels, spreads, counts = state
        position = count % self.season_length
        level, spread, seen = levels[position], spreads[position], counts[position]
        output = value - level if seen else 0.0
        if seen:
            limit = 4 * spread if seen >= 5 and spread > 0 else math.inf
            deviation = min(max(value - level, -limit), limit)
            weight = 1.0 / min(seen + 1, self.memory)
            level += weight * deviation
            spread += weight * (abs(deviation) - spread)
        else:
            level, spread = value, 0.0
        following = (
            count + 1,
            (*levels[:position], level, *levels[position + 1 :]),
            (*spreads[:position], spread, *spreads[position + 1 :]),
            (*counts[:position], seen + 1, *counts[position + 1 :]),
        )
        return following, (output,)
