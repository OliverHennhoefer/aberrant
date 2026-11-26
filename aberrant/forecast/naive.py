"""Seasonal-naive forecasts: repeat the last observed cycle."""

from __future__ import annotations

from aberrant._utils.validation import coerce_event_number, coerce_integer_count
from aberrant.base.forecast import Forecast
from aberrant.base.readiness import Readiness
from aberrant.forecast._arithmetic import CycleWarmUp, LeadErrors, fsum
from aberrant.forecast.base import BaseForecaster, check_spread_parameters


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
        each lead fades towards its own errors. ``learn_one(None)`` learns
        that a value is missing: once warm, the forecaster repeats the value
        one cycle earlier and reports the spreads of the longer leads its
        forecasts now are; during warm-up see :class:`HoltWinters`. State is
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
        *,
        season_length: int = 1,
        horizon: int = 1,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        super().__init__(horizon=horizon)
        self.season_length = coerce_integer_count(season_length, label="season_length")
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )
        self._reset_state()

    @property
    def readiness(self) -> Readiness:
        """Ready once two complete cycles have been learned."""
        return Readiness.after(self._learned, 2 * self.season_length)

    def _reset_state(self) -> None:
        """Forget every learned value and error."""
        self.n_samples_seen = 0
        # Events since the first observed value.
        self._learned = 0
        self._warmup = CycleWarmUp(self.season_length)
        # The last cycle as a ring whose oldest value sits at ``_oldest``.
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

    def learn_one(self, y: float | None) -> None:
        """Learn the next value of the series, or ``None`` for a missing one."""
        value = None if y is None else coerce_event_number(y, label="y")
        m = self.season_length
        if self._learned >= 2 * m:
            if value is None:
                kept = self._values[self._oldest]
                self._errors.publish(
                    list(self._errors.mae),
                    self._means(self._oldest, kept),
                    self._errors.stale + 1,
                )
            else:
                mae = self._errors.updated(value)
                means = self._means(self._oldest, value)
                self._errors.publish(mae, means, 0)
                self._values[self._oldest] = value
            self._oldest = (self._oldest + 1) % m
            self._learned += 1
        elif value is not None or self._warmup.started:
            self._warmup.add(value)
            if self._warmup.complete:
                try:
                    self._start()
                except OverflowError:
                    self._warmup.discard_last()
                    raise
            self._learned += 1
        self.n_samples_seen += 1

    def _start(self) -> None:
        """Seed the spreads and the first forecast from the two cycles."""
        m = self.season_length
        values, observed = self._warmup.values, self._warmup.observed
        pairs = [i for i in range(m) if observed[i] and observed[m + i]]
        warmup = fsum(abs(fsum((values[m + i], -values[i]))) for i in pairs)
        self._errors.publish(
            self._errors.seeded(warmup / len(pairs) if pairs else 0.0),
            tuple(values[m + (lead - 1) % m] for lead in range(1, self.horizon + 1)),
            self._warmup.stale,
        )
        self._values = values[m:]
        self._oldest = 0
        self._warmup = CycleWarmUp(m)

    def forecast_one(self) -> Forecast | None:
        """Forecast leads ``1..horizon``, or ``None`` before two cycles."""
        if self._learned < 2 * self.season_length:
            return None
        return Forecast(
            origin=self.n_samples_seen,
            leads=tuple(range(1, self.horizon + 1)),
            mean=self._errors.issued[-1],
            scale=self._errors.scales,
        )
