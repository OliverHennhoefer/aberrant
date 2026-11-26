"""Sequence transformers: each feature's output follows its own recent values."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from aberrant._utils.statistics import linear_quantile
from aberrant._utils.validation import coerce_feature_values, coerce_finite_number
from aberrant.base.readiness import Readiness
from aberrant.base.transformer import BaseTransformer
from aberrant.exceptions import ValidationError

State = tuple[Any, ...]
ChangeKind = Literal["difference", "percent"]
Statistic = Literal[
    "sum", "mean", "std", "min", "max", "range", "median", "quantile", "mad", "zscore"
]
_MAD_SCALE = 1.4826


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


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValidationError(f"Feature '{name}' result must be finite")
    return value


class _SequenceTransformer(BaseTransformer):
    """Shared state handling of the sequence transformers.

    Each selected feature keeps its own immutable state. ``learn_one``
    computes the event's output from the state before the event and then
    advances it. ``transform_one`` of the event just learned returns that
    output, as a pipeline's post-update pass needs; any other event gets the
    output it would get as the next one, without learning.
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

    def _initial(self) -> State:
        return ()

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
            state = self._states.get(key, self._initial())
            states[key], outputs = self._advance(state, value)
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
        for name in states:
            for output_name in self._names(name):
                _finite(result[output_name], output_name)
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

    def _advance(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        outputs = tuple(
            state[-lag] if len(state) >= lag else state[0] if state else value
            for lag in self.lags
        )
        return (*state, value)[-self.lags[-1] :], outputs


class Difference(_SequenceTransformer):
    """Replace each selected feature by its change over ``lag`` events.

    ``kind="difference"`` gives ``x[t] - x[t - lag]`` and ``"percent"`` gives
    ``100 (x[t] - x[t - lag]) / |x[t - lag]|``, which rejects a reference of
    0. Until a feature has ``lag`` earlier values, its change is 0.

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

    def _advance(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        following = (*state, value)[-self.lag :]
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

    Each event costs O(F W log W) for F selected features and window W.

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

    def _advance(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        window = (*state, value)[-self.window_size :]
        if len(window) < 3:
            return window, (value,)
        ordered = sorted(window)
        median = linear_quantile(ordered, 0.5)
        spread = linear_quantile(sorted(abs(item - median) for item in window), 0.5)
        spike = (
            spread > 0 and abs(value - median) > self.threshold * _MAD_SCALE * spread
        )
        return window, (median if spike else value,)


class RollingStatistic(_SequenceTransformer):
    """Replace each selected feature by a statistic of its recent values.

    The window holds the last ``window_size`` values, the current one
    included, and starts shorter. ``std`` is the sample deviation (0 for one
    value), ``mad`` the unscaled median absolute deviation, ``quantile``
    interpolates linearly at ``quantile`` and ``zscore`` is the value's
    distance from the window mean in deviations (0 in a flat window).

    Each event costs O(F W log W) for F selected features and window W.

    Args:
        window_size: Values in the window.
        statistic: One of ``sum``, ``mean``, ``std``, ``min``, ``max``,
            ``range``, ``median``, ``quantile``, ``mad`` and ``zscore``.
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

    def _advance(self, state: State, value: float) -> tuple[State, tuple[float, ...]]:
        window: tuple[float, ...] = (*state, value)[-self.window_size :]
        return window, (_STATISTIC[self.statistic](window, self.quantile),)


def _mean(window: tuple[float, ...], _quantile: float) -> float:
    return math.fsum(window) / len(window)


def _std(window: tuple[float, ...], _quantile: float) -> float:
    if len(window) < 2:
        return 0.0
    mean = _mean(window, _quantile)
    return math.sqrt(
        math.fsum((item - mean) ** 2 for item in window) / (len(window) - 1)
    )


def _zscore(window: tuple[float, ...], _quantile: float) -> float:
    deviation = _std(window, _quantile)
    return (window[-1] - _mean(window, _quantile)) / deviation if deviation > 0 else 0.0


def _median(window: tuple[float, ...], _quantile: float) -> float:
    return linear_quantile(sorted(window), 0.5)


def _mad(window: tuple[float, ...], _quantile: float) -> float:
    median = _median(window, _quantile)
    return linear_quantile(sorted(abs(item - median) for item in window), 0.5)


_STATISTIC: dict[str, Callable[[tuple[float, ...], float], float]] = {
    "sum": lambda window, _quantile: math.fsum(window),
    "mean": _mean,
    "std": _std,
    "min": lambda window, _quantile: min(window),
    "max": lambda window, _quantile: max(window),
    "range": lambda window, _quantile: max(window) - min(window),
    "median": _median,
    "quantile": lambda window, quantile: linear_quantile(sorted(window), quantile),
    "mad": _mad,
    "zscore": _zscore,
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

    def _initial(self) -> State:
        empty = (0.0,) * self.season_length
        return (0, empty, empty, (0,) * self.season_length)

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
