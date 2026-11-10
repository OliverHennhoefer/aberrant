"""Seasonal-naive forecasts: repeat the last observed cycle."""

from __future__ import annotations

from aberrant.base.forecast import Forecast
from aberrant.forecast._arithmetic import LeadErrors, fsum
from aberrant.forecast.base import BaseForecaster, check_spread_parameters
from aberrant.utils.validation import coerce_finite_number, coerce_integer_count


class SeasonalNaive(BaseForecaster):
    """Forecast each lead with the value one cycle earlier.

    Lead ``h`` repeats the value at the same position of the last observed
    cycle. With ``season_length=1`` this is the naive forecast: every lead
    repeats the last value. It is the usual baseline a forecaster must beat.

    Args:
        season_length: Observations per cycle; 1 for the naive forecast.
        horizon: Largest lead to forecast.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error kept for every lead, in (0, 1].
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        Two cycles warm the forecaster up. The spread of every lead starts at
        the mean absolute seasonal-naive error within the second cycle, then
        each lead fades towards its own errors. State is
        O(season_length + horizon^2) and each event after warm-up costs
        O(horizon).

    Examples:
        ```python
        from aberrant.forecast import SeasonalNaive

        model = SeasonalNaive(season_length=3, horizon=2)
        for y in [1.0, 5.0, 3.0, 1.5, 5.5, 3.5]:
            model.learn_one(y)
        model.forecast_one().mean  # (1.5, 5.5)
        ```
    """

    def __init__(
        self,
        season_length: int = 1,
        *,
        horizon: int = 1,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        super().__init__(horizon)
        self.season_length = coerce_integer_count(season_length, label="season_length")
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )
        self.reset()

    @property
    def is_ready(self) -> bool:
        """Whether two complete cycles have been learned."""
        return self.n_samples_seen >= 2 * self.season_length

    def reset(self) -> None:
        """Forget every learned value and error."""
        self.n_samples_seen = 0
        # Warm-up values, then the last cycle as a ring whose oldest value
        # sits at ``_oldest``.
        self._values: list[float] = []
        self._oldest = 0
        self._errors = LeadErrors(self.horizon, self.residual_alpha, self.min_scale)

    def _means(self, oldest: int, replaced: float) -> tuple[float, ...]:
        """Forecasts once ``replaced`` overwrites the value at ``oldest``."""
        m = self.season_length
        start = (oldest + 1) % m
        means = []
        for lead in range(1, self.horizon + 1):
            position = (start + (lead - 1) % m) % m
            means.append(replaced if position == oldest else self._values[position])
        return tuple(means)

    def learn_one(self, y: float) -> None:
        """Learn the next value of the series."""
        value = coerce_finite_number(y, label="y")
        m = self.season_length
        if self.n_samples_seen + 1 < 2 * m:
            self._values.append(value)
            self.n_samples_seen += 1
            return
        if self.is_ready:
            mae = self._errors.updated(value)
            means = self._means(self._oldest, value)
            self._errors.publish(mae, means)
            self._values[self._oldest] = value
            self._oldest = (self._oldest + 1) % m
        else:
            values = [*self._values, value]
            warmup = fsum(abs(fsum((values[m + i], -values[i]))) / m for i in range(m))
            self._errors.publish(
                self._errors.seeded(warmup),
                tuple(
                    values[m + (lead - 1) % m] for lead in range(1, self.horizon + 1)
                ),
            )
            self._values = values[m:]
            self._oldest = 0
        self.n_samples_seen += 1

    def forecast_one(self) -> Forecast | None:
        """Forecast leads ``1..horizon``, or ``None`` before two cycles."""
        if not self.is_ready:
            return None
        return Forecast(
            origin=self.n_samples_seen,
            leads=tuple(range(1, self.horizon + 1)),
            mean=self._errors.issued[-1],
            scale=self._errors.scales,
        )

    def __repr__(self) -> str:
        return (
            f"SeasonalNaive(season_length={self.season_length}, horizon={self.horizon})"
        )
