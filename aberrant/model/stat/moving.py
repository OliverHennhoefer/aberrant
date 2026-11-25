"""Univariate z-scores over a sliding window."""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import deque

import numpy as np

from aberrant._utils.validation import (
    coerce_event_number,
    coerce_finite_number,
    coerce_integer_count,
)
from aberrant.base.model import BaseModel
from aberrant.exceptions import ValidationError

# Scales the MAD to the standard deviation of a normal distribution.
_MAD_TO_STD = 1.482602218505602


def _window_parameters(window_size: object, warm_up: object) -> tuple[int, int]:
    size = coerce_integer_count(window_size, label="window_size", minimum=2)
    if warm_up is None:
        return size, size
    count = coerce_integer_count(warm_up, label="warm_up", minimum=2)
    if count > size:
        raise ValueError("warm_up must not exceed window_size")
    return size, count


def _floor(value: object, label: str) -> float:
    floor = coerce_finite_number(value, label=label)
    if floor <= 0.0:
        raise ValueError(f"{label} must be positive")
    return floor


class _UnivariateWindow(BaseModel):
    """Validate one-feature events and keep the last ``window_size`` values."""

    def __init__(self, *, window_size: int, warm_up: int | None) -> None:
        self.window_size, self.warm_up = _window_parameters(window_size, warm_up)
        self._feature: str | None = None
        self._window: deque[float] = deque()

    @property
    def n_samples_seen(self) -> int:
        """Number of values in the window."""
        return len(self._window)

    def _value(self, x: dict[str, float]) -> tuple[str, float]:
        if len(x) != 1:
            raise ValidationError("Input must contain exactly one key-value pair.")
        feature = next(iter(x))
        if self._feature is not None and feature != self._feature:
            raise ValidationError(
                f"Input feature must be {self._feature!r}, got {feature!r}."
            )
        return feature, coerce_event_number(x[feature], label=f"Feature '{feature}'")

    def _push(self, x: dict[str, float]) -> tuple[float, float | None]:
        """Append the event's value and return it with the evicted value."""
        feature, value = self._value(x)
        self._feature = feature
        evicted = (
            self._window.popleft() if len(self._window) == self.window_size else None
        )
        self._window.append(value)
        return value, evicted


class MovingZScore(_UnivariateWindow):
    """Absolute z-score of a value against a sliding window.

    The score is ``|x - mean| / max(std, min_std)``, where ``mean`` and the
    population ``std`` describe the last ``window_size`` learned values. It is
    ``0.0`` until ``warm_up`` values have been learned. Mean and variance are
    updated in constant time and recomputed exactly once per window to bound
    rounding drift, so scoring and learning cost ``O(1)``.

    Args:
        window_size: Number of most recent learned values in the reference.
        warm_up: Values to learn before scoring; defaults to ``window_size``.
            It must be at least two and at most ``window_size``.
        min_std: Smallest standard deviation used as the denominator, so a
            constant window does not divide by zero.
    """

    def __init__(
        self,
        *,
        window_size: int = 256,
        warm_up: int | None = None,
        min_std: float = 1e-12,
    ) -> None:
        """Initialize a sliding-window z-score."""
        super().__init__(window_size=window_size, warm_up=warm_up)
        self.min_std = _floor(min_std, "min_std")
        self._mean = 0.0
        self._m2 = 0.0
        self._updates = 0

    def learn_one(self, x: dict[str, float]) -> None:
        """Add one value to the window, evicting the oldest when full."""
        value, evicted = self._push(x)
        self._updates += 1
        if self._updates >= self.window_size:
            values = np.fromiter(self._window, dtype=np.float64)
            self._mean = float(values.mean())
            self._m2 = float(((values - self._mean) ** 2).sum())
            self._updates = 0
            return
        if evicted is not None:
            count = len(self._window) - 1
            delta = evicted - self._mean
            self._mean -= delta / count
            self._m2 -= delta * (evicted - self._mean)
        count = len(self._window)
        delta = value - self._mean
        self._mean += delta / count
        self._m2 += delta * (value - self._mean)

    def score_one(self, x: dict[str, float]) -> float:
        """Return the absolute z-score of the event's value."""
        _, value = self._value(x)
        count = len(self._window)
        if count < self.warm_up:
            return 0.0
        std = math.sqrt(max(self._m2, 0.0) / count)
        return abs(value - self._mean) / max(std, self.min_std)


class MovingMAD(_UnivariateWindow):
    """Robust z-score of a value against the median and MAD of a window.

    The score is ``|x - median| / max(1.4826 * MAD, min_mad)``, the modified
    z-score of Iglewicz and Hoaglin, where ``median`` and the median absolute
    deviation ``MAD`` describe the last ``window_size`` learned values. Unlike
    a z-score, a few extreme values in the window barely move it. It is ``0.0``
    until ``warm_up`` values have been learned. Learning keeps the window
    sorted and recomputes the MAD in ``O(window_size)``; scoring is ``O(1)``.

    Args:
        window_size: Number of most recent learned values in the reference.
        warm_up: Values to learn before scoring; defaults to ``window_size``.
            It must be at least two and at most ``window_size``.
        min_mad: Smallest scaled MAD used as the denominator, so a window in
            which more than half the values are equal does not divide by zero.

    References:
        Iglewicz, B., & Hoaglin, D. C. (1993). How to Detect and Handle
        Outliers. ASQC Quality Press.
    """

    def __init__(
        self,
        *,
        window_size: int = 256,
        warm_up: int | None = None,
        min_mad: float = 1e-12,
    ) -> None:
        """Initialize a sliding-window robust z-score."""
        super().__init__(window_size=window_size, warm_up=warm_up)
        self.min_mad = _floor(min_mad, "min_mad")
        self._sorted: list[float] = []
        self._median = 0.0
        self._scale = 0.0

    def learn_one(self, x: dict[str, float]) -> None:
        """Add one value to the window and refresh its median and MAD."""
        value, evicted = self._push(x)
        if evicted is not None:
            del self._sorted[bisect_left(self._sorted, evicted)]
        insort(self._sorted, value)
        ordered = np.asarray(self._sorted, dtype=np.float64)
        self._median = float(np.median(ordered))
        self._scale = _MAD_TO_STD * float(np.median(np.abs(ordered - self._median)))

    def score_one(self, x: dict[str, float]) -> float:
        """Return the robust z-score of the event's value."""
        _, value = self._value(x)
        if len(self._window) < self.warm_up:
            return 0.0
        return abs(value - self._median) / max(self._scale, self.min_mad)
