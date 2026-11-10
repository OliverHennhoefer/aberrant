"""Overflow-checked arithmetic and per-lead error tracking shared by forecasters."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from statistics import mean


def finite(value: float) -> float:
    """Return ``value`` or raise when arithmetic left the float range."""
    if not math.isfinite(value):
        raise OverflowError("Forecast arithmetic exceeds the float range")
    return value


def fsum(values: Iterable[float]) -> float:
    """Correctly rounded, overflow-checked sum."""
    return finite(math.fsum(finite(value) for value in values))


def blend(previous: float, observed: float, weight: float) -> float:
    """Exponentially faded update of ``previous`` towards ``observed``."""
    return fsum(((1.0 - weight) * previous, weight * observed))


def holt_winters_initialize(
    values: list[float], season_length: int
) -> tuple[float, float, list[float], float]:
    """Initialize additive Holt-Winters from exactly two cycles.

    Returns level, per-observation trend, seasonal components and the mean
    absolute error of the fitted warm-up model.
    """
    m = season_length
    first_mean = mean(values[:m])
    second_mean = mean(values[m:])
    trend = fsum((second_mean / m, -first_mean / m))
    midpoint = (m - 1) / 2
    seasonal = [
        fsum(
            (
                values[index] / 2,
                values[m + index] / 2,
                -first_mean / 2,
                -second_mean / 2,
                -trend * (index - midpoint),
            )
        )
        for index in range(m)
    ]
    level = fsum((second_mean, trend * midpoint))
    errors = (
        abs(
            fsum(
                (
                    value,
                    -(first_mean if index < m else second_mean),
                    -trend * (index % m - midpoint),
                    -seasonal[index % m],
                )
            )
        )
        / (2 * m)
        for index, value in enumerate(values)
    )
    return level, trend, seasonal, fsum(errors)


class LeadErrors:
    """The last ``horizon`` issued forecasts and a faded MAE for each lead.

    Every value that arrives was forecast once at each lead ``h``, by the
    forecast issued ``h`` events earlier. Each lead keeps its own faded mean
    absolute error, since errors grow with the lead.
    """

    def __init__(self, horizon: int, residual_alpha: float, min_scale: float) -> None:
        self.horizon = horizon
        self.residual_alpha = residual_alpha
        self.min_scale = min_scale
        self.issued: deque[tuple[float, ...]] = deque(maxlen=horizon)
        self.mae: list[float] = []

    @property
    def scales(self) -> tuple[float, ...]:
        """Faded MAE per lead, floored at ``min_scale``."""
        return tuple(max(self.min_scale, value) for value in self.mae)

    def seeded(self, value: float) -> list[float]:
        """Every lead's MAE set to one fitted warm-up error."""
        return [value] * self.horizon

    def updated(self, y: float) -> list[float]:
        """MAE per lead after learning ``y``, without changing state."""
        mae = list(self.mae)
        for lead in range(1, min(self.horizon, len(self.issued)) + 1):
            forecast = self.issued[-lead][lead - 1]
            error = abs(fsum((y, -forecast)))
            mae[lead - 1] = blend(mae[lead - 1], error, self.residual_alpha)
        return mae

    def publish(self, mae: list[float], forecast: tuple[float, ...]) -> None:
        """Commit new errors and the forecast issued at the new origin."""
        self.mae = mae
        self.issued.append(forecast)
