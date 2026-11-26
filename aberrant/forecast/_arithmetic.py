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
    values: list[float], season_length: int, observed: list[bool]
) -> tuple[float, float, list[float], float]:
    """Initialize additive Holt-Winters from exactly two cycles.

    Returns level, per-observation trend, seasonal components and the mean
    absolute error of the fitted warm-up model at the observed values.
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
    count = sum(observed)
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
        / count
        for index, value in enumerate(values)
        if observed[index]
    )
    return level, trend, seasonal, fsum(errors)


class CycleWarmUp:
    """The first two cycles of a seasonal forecaster, with gaps filled.

    A missing value takes the value one cycle earlier, or in the first cycle
    the last observed value. Missing values before the first observed one do
    not start the series.
    """

    def __init__(self, season_length: int) -> None:
        self.season_length = season_length
        self.values: list[float] = []
        self.observed: list[bool] = []

    @property
    def started(self) -> bool:
        """Whether an observed value has started the series."""
        return bool(self.values)

    @property
    def complete(self) -> bool:
        """Whether both cycles are filled."""
        return len(self.values) == 2 * self.season_length

    @property
    def stale(self) -> int:
        """Missing values since the last observed one."""
        count = 0
        for observed in reversed(self.observed):
            if observed:
                break
            count += 1
        return count

    def discard_last(self) -> None:
        """Undo the last ``add``."""
        self.values.pop()
        self.observed.pop()

    def add(self, value: float | None) -> None:
        """Add the next value of a started series, or fill a missing one."""
        if value is None:
            m = self.season_length
            fill = self.values[-m] if len(self.values) >= m else self.values[-1]
            self.values.append(fill)
            self.observed.append(False)
        else:
            self.values.append(value)
            self.observed.append(True)


class LeadErrors:
    """The last ``horizon`` issued forecasts and a faded MAE for each lead.

    Every value that arrives was forecast once at each lead ``h``, by the
    forecast issued ``h`` events earlier. Each lead keeps its own faded mean
    absolute error, since errors grow with the lead. A forecast issued after
    a missing value repeats the one issued after the last observed value, so
    only forecasts issued after observed values are scored, and a lead ``h``
    forecast ``s`` missing values on reports the spread of lead ``s + h``,
    or of the longest lead beyond the horizon.

    Without a seeded warm-up error (``seeded=False``) each lead starts at 0
    and takes the plain mean of its errors until ``1 / residual_alpha`` are
    counted.
    """

    def __init__(
        self,
        horizon: int,
        residual_alpha: float,
        min_scale: float,
        *,
        seeded: bool = True,
    ) -> None:
        self.horizon = horizon
        self.residual_alpha = residual_alpha
        self.min_scale = min_scale
        self.issued: deque[tuple[float, ...]] = deque(maxlen=horizon)
        self.observed: deque[bool] = deque(maxlen=horizon)
        self.mae: list[float] = [] if seeded else [0.0] * horizon
        self.counts: list[int] | None = None if seeded else [0] * horizon
        self.stale = 0

    def _scored(self) -> range:
        """Leads with an issued forecast for the next value."""
        return range(1, min(self.horizon, len(self.issued)) + 1)

    @property
    def scales(self) -> tuple[float, ...]:
        """Faded MAE per lead, floored at ``min_scale``."""
        last = self.horizon - 1
        return tuple(
            max(self.min_scale, self.mae[min(self.stale + lead, last)])
            for lead in range(self.horizon)
        )

    def seeded(self, value: float) -> list[float]:
        """Every lead's MAE set to one fitted warm-up error."""
        return [value] * self.horizon

    def updated(self, y: float) -> list[float]:
        """MAE per lead after learning ``y``, without changing state."""
        mae = list(self.mae)
        for lead in self._scored():
            if not self.observed[-lead]:
                continue
            forecast = self.issued[-lead][lead - 1]
            error = abs(fsum((y, -forecast)))
            weight = self.residual_alpha
            if self.counts is not None:
                weight = max(weight, 1 / (self.counts[lead - 1] + 1))
            mae[lead - 1] = blend(mae[lead - 1], error, weight)
        return mae

    def publish(
        self, mae: list[float], forecast: tuple[float, ...], stale: int
    ) -> None:
        """Commit new errors and the forecast issued at the new origin.

        ``stale`` counts the missing values up to the new origin, 0 when its
        value was observed.
        """
        if self.counts is not None and stale == 0:
            for lead in self._scored():
                self.counts[lead - 1] += self.observed[-lead]
        self.mae = mae
        self.issued.append(forecast)
        self.observed.append(stale == 0)
        self.stale = stale
