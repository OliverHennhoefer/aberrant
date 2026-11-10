"""Additive Holt-Winters forecasts updated one observation at a time."""

from __future__ import annotations

from aberrant.base.forecast import Forecast
from aberrant.forecast._arithmetic import (
    LeadErrors,
    fsum,
    holt_winters_initialize,
)
from aberrant.forecast.base import BaseForecaster, check_spread_parameters
from aberrant.utils.validation import coerce_finite_number, coerce_integer_count


class HoltWinters(BaseForecaster):
    """Forecast level, trend and one season with additive Holt-Winters.

    Lead ``h`` forecasts ``level + h * trend + season`` at the matching seasonal
    position. Every learned value advances one seasonal position.

    Args:
        season_length: Known number of observations in a cycle, at least two.
        horizon: Largest lead to forecast.
        alpha: Level smoothing weight in [0, 1].
        beta: Trend smoothing weight in [0, 1] (the usual beta-star parameter).
        gamma: Seasonal smoothing weight in [0, 1-alpha], using prior level
            and trend in the seasonal update.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error kept for every lead, in (0, 1].
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        Two cycles initialize the model as in
        :class:`~aberrant.model.timeseries.SeasonalResidualDetector`, whose
        scores equal this forecaster's lead-1 residual divided by its lead-1
        spread. Every lead's spread starts at the mean absolute error of the
        fitted warm-up model, then fades towards that lead's own errors. State
        is O(season_length + horizon^2) and each event costs O(horizon).
        Unrepresentable arithmetic raises OverflowError without changing state.

    References:
        Hyndman and Athanasopoulos, Forecasting: Principles and Practice, 3rd ed.,
        additive Holt-Winters component equations:
        https://otexts.com/fpp3/holt-winters.html
    """

    def __init__(
        self,
        season_length: int,
        *,
        horizon: int = 1,
        alpha: float = 0.2,
        beta: float = 0.05,
        gamma: float = 0.1,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        super().__init__(horizon)
        self.season_length = coerce_integer_count(
            season_length, label="season_length", minimum=2
        )
        self.alpha = coerce_finite_number(alpha, label="alpha")
        self.beta = coerce_finite_number(beta, label="beta")
        self.gamma = coerce_finite_number(gamma, label="gamma")
        if not 0 <= self.alpha <= 1 or not 0 <= self.beta <= 1:
            raise ValueError("alpha and beta must be in [0, 1]")
        if not 0 <= self.gamma <= 1 - self.alpha:
            raise ValueError("gamma must be in [0, 1-alpha]")
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )
        self.reset()

    @property
    def is_ready(self) -> bool:
        """Whether two complete cycles have been learned."""
        return self.n_samples_seen >= 2 * self.season_length

    def reset(self) -> None:
        """Forget learned components and errors."""
        self.n_samples_seen = 0
        self._warmup: list[float] = []
        self._seasonal: list[float] = []
        self._level = 0.0
        self._trend = 0.0
        self._errors = LeadErrors(self.horizon, self.residual_alpha, self.min_scale)

    def _means(
        self,
        level: float,
        trend: float,
        seasonal: list[float],
        origin: int,
        updated: tuple[int, float] | None = None,
    ) -> tuple[float, ...]:
        """Forecasts at ``origin``; ``updated`` overrides one seasonal position."""
        m = self.season_length
        means = []
        for lead in range(1, self.horizon + 1):
            position = (origin + lead - 1) % m
            season = seasonal[position]
            if updated is not None and position == updated[0]:
                season = updated[1]
            means.append(fsum((fsum((level, lead * trend)), season)))
        return tuple(means)

    def learn_one(self, y: float) -> None:
        """Learn the next value of the series."""
        value = coerce_finite_number(y, label="y")
        origin = self.n_samples_seen + 1
        if not self.is_ready:
            if origin < 2 * self.season_length:
                self._warmup.append(value)
                self.n_samples_seen = origin
                return
            level, trend, seasonal, scale = holt_winters_initialize(
                [*self._warmup, value], self.season_length
            )
            mae = self._errors.seeded(scale)
            means = self._means(level, trend, seasonal, origin)
            self._seasonal = seasonal
        else:
            phase = self.n_samples_seen % self.season_length
            base = fsum((self._level, self._trend))
            residual = fsum((value, -fsum((base, self._seasonal[phase]))))
            # Residual corrections avoid overflowing unweighted intermediates.
            level_correction = self.alpha * residual
            level = fsum((base, level_correction))
            trend = fsum((self._trend, self.beta * level_correction))
            season = fsum((self._seasonal[phase], self.gamma * residual))
            mae = self._errors.updated(value)
            means = self._means(level, trend, self._seasonal, origin, (phase, season))
            # Publish only after all arithmetic succeeds; no per-event copy.
            self._seasonal[phase] = season
        self._level, self._trend = level, trend
        self._errors.publish(mae, means)
        self._warmup = []
        self.n_samples_seen = origin

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
            f"HoltWinters(season_length={self.season_length}, horizon={self.horizon})"
        )
