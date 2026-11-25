"""Lifecycle scaffolding for univariate moving-window detectors."""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections import deque
from collections.abc import Callable, Sequence

from aberrant.base.exceptions import ValidationError
from aberrant.base.model import BaseModel
from aberrant.utils.validation import coerce_event_number

_Statistic = Callable[[Sequence[float]], float]


def _extract_univariate_value(
    x: dict[str, float], feature_name: str | None
) -> tuple[str, float]:
    """Validate a univariate sample and return its stable feature and value."""
    if len(x) != 1:
        raise ValidationError("Input must contain exactly one key-value pair.")

    sample_feature = next(iter(x))
    if feature_name is not None and sample_feature != feature_name:
        raise ValidationError(
            f"Input feature must be {feature_name!r}, got {sample_feature!r}."
        )
    return feature_name or sample_feature, coerce_event_number(
        x[sample_feature],
        label=f"Feature '{sample_feature}'",
    )


class _BaseMovingUnivariate(BaseModel):
    """Own window creation, feature locking, learning, and score differencing."""

    def __init__(
        self,
        *,
        window_size: int,
        key: str | None = None,
        abs_diff: bool = True,
    ) -> None:
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        self.window_size = window_size
        self.window: deque[float] = deque(maxlen=window_size)
        self.feature_name = key
        self.abs_diff = abs_diff

    def learn_one(self, x: dict[str, float]) -> None:
        """Append one validated value and lock the feature name on first use."""
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        self.window.append(value)

    def _extract_value(self, x: dict[str, float]) -> float:
        _, value = _extract_univariate_value(x, self.feature_name)
        return value

    def _difference(self, new_value: float, current_value: float) -> float:
        difference = new_value - current_value
        return abs(difference) if self.abs_diff else difference

    def _score_statistic_change(
        self,
        x: dict[str, float],
        statistic: _Statistic,
    ) -> float:
        """Score the change in a statistic after provisionally adding ``x``."""
        if not self.window:
            return 0.0

        current = list(self.window)
        candidate = [*current, self._extract_value(x)]
        return self._difference(statistic(candidate), statistic(current))


class _WithValue(Sequence[float]):
    """A sorted sequence read as if ``value`` were inserted, without copying."""

    __slots__ = ("_sorted", "_value", "_position")

    def __init__(self, ordered: list[float], value: float) -> None:
        self._sorted = ordered
        self._value = value
        self._position = bisect_right(ordered, value)

    def __len__(self) -> int:
        return len(self._sorted) + 1

    def __getitem__(self, index: int) -> float:  # type: ignore[override]
        if index < self._position:
            return self._sorted[index]
        if index == self._position:
            return self._value
        return self._sorted[index - 1]


class _BaseMovingOrderStatistic(_BaseMovingUnivariate):
    """Keep the window sorted so order statistics need no sort per score."""

    def __init__(
        self,
        *,
        window_size: int,
        key: str | None = None,
        abs_diff: bool = True,
    ) -> None:
        super().__init__(window_size=window_size, key=key, abs_diff=abs_diff)
        self._sorted: list[float] = []

    def learn_one(self, x: dict[str, float]) -> None:
        """Append one validated value and keep the sorted window in step."""
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        if len(self.window) == self.window_size:
            del self._sorted[bisect_left(self._sorted, self.window[0])]
        self.window.append(value)
        insort(self._sorted, value)

    def _statistic(self, ordered: Sequence[float]) -> float:
        raise NotImplementedError

    def score_one(self, x: dict[str, float]) -> float:
        """Score the change in the statistic after provisionally adding ``x``."""
        if not self.window:
            return 0.0
        candidate = _WithValue(self._sorted, self._extract_value(x))
        return self._difference(
            self._statistic(candidate), self._statistic(self._sorted)
        )
