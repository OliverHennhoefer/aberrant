"""Seasonal autoregressive forecasts fitted online by recursive least squares."""

from __future__ import annotations

from collections import deque

import numpy as np

from aberrant._utils.validation import (
    coerce_event_number,
    coerce_finite_number,
    coerce_integer_count,
)
from aberrant.base.forecast import Forecast
from aberrant.base.readiness import Readiness
from aberrant.forecast._arithmetic import LeadErrors, finite, fsum
from aberrant.forecast.base import BaseForecaster, check_spread_parameters
from aberrant.forecast.linear import rls_update


def _differencing(
    difference: int, seasonal_difference: int, season_length: int
) -> tuple[tuple[int, float], ...]:
    """Lags and weights of ``(1 - B)^d (1 - B^m)^D`` past lag 0."""
    weights = {0: 1.0}
    factors = [1] * difference + [season_length] * seasonal_difference
    for step in factors:
        product: dict[int, float] = {}
        for lag, weight in weights.items():
            product[lag] = product.get(lag, 0.0) + weight
            product[lag + step] = product.get(lag + step, 0.0) - weight
        weights = product
    return tuple(
        (lag, weight) for lag, weight in sorted(weights.items()) if lag and weight
    )


class AutoRegressive(BaseForecaster):
    """Forecast with a seasonal autoregression whose weights adapt online.

    The series is differenced ``difference`` times and seasonally
    ``seasonal_difference`` times, and each differenced value ``w_t`` is
    predicted linearly from its own past::

        w_t = c + sum of a_k w_{t-k} over the lags k = i + j m
              with 0 <= i <= p, 0 <= j <= P and k > 0

    with ``p = order``, ``P = seasonal_order`` and ``m = season_length``.
    These lags hold the seasonal autoregression of SARIMA, whose factors
    ``(1 - phi(B))(1 - Phi(B^m))`` multiply to exactly these lags; each lag
    keeps its own weight, so the fit stays linear. Every observed value
    updates the weights by recursive least squares with geometric
    ``forgetting``, so they follow a slowly changing process. Leads beyond 1
    feed the forecasts back in as lags and undo the differencing. This is
    SARIMA without moving-average terms; it captures the autocorrelation
    that smoothing methods leave in their errors.

    Args:
        horizon: Largest lead to forecast.
        order: Number of non-seasonal lags ``p``, at least 0.
        season_length: Observations per cycle ``m``; 1 for none.
        seasonal_order: Number of seasonal cycles ``P`` to look back, at
            least 0. Needs
            ``season_length`` of at least 2.
        difference: Non-seasonal differences ``d``, 0, 1 or 2. 1 suits a
            series that wanders or trends.
        seasonal_difference: Seasonal differences ``D``, 0 or 1. Needs
            ``season_length`` of at least 2.
        forgetting: Weight factor in (0, 1] applied to past events at every
            update. 1 weighs all events equally; 0.999 remembers roughly the
            last 1,000.
        delta: Positive initial variance of the weights.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error kept for every lead, in (0, 1]. Until
            ``1 / residual_alpha`` errors are seen at a lead, the plain mean
            is used.
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        The spread of lead ``h`` is the faded mean absolute error of the lead
        ``h`` forecasts. The forecaster is ready once the longest lead has
        been scored once: after ``d + D m`` values to difference, the largest
        lag ``p + P m`` of differenced values, one weight update per weight
        (``k + 1`` for the ``k`` lags, at most ``(p + 1)(P + 1) - 1``) and
        ``horizon`` more values.

        ``learn_one(None)`` learns that a value is missing: once forecasting,
        the lead-1 forecast takes its place without updating the weights,
        and each forecast reports the spread of the longer lead it now is.
        Before that, the last observed value takes its place. Missing values
        before the first observed one do not start the series.

        State is O(d + D m + p + P m + k^2 + horizon^2), and each event costs
        O(k^2 + horizon k) plus copying the lag buffers. Unrepresentable
        arithmetic raises OverflowError without changing state.

    Examples:
        ```python
        from aberrant.forecast import AutoRegressive

        model = AutoRegressive(horizon=2, order=1)
        y = 0.0
        for _ in range(200):
            y = 0.8 * y + 1.0
            model.learn_one(y)
        model.forecast_one().mean  # close to (5.0, 5.0)
        ```

    References:
        Box, Jenkins, Reinsel and Ljung, Time Series Analysis: Forecasting
        and Control, 5th ed., chapters 3, 4 and 9 (seasonal ARIMA models).
        Haykin, Adaptive Filter Theory, 5th ed., chapter 10 (recursive least
        squares).
    """

    def __init__(
        self,
        *,
        horizon: int = 1,
        order: int = 2,
        season_length: int = 1,
        seasonal_order: int = 0,
        difference: int = 0,
        seasonal_difference: int = 0,
        forgetting: float = 0.999,
        delta: float = 1000.0,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        super().__init__(horizon=horizon)
        self.order = coerce_integer_count(order, label="order", minimum=0)
        self.season_length = coerce_integer_count(season_length, label="season_length")
        self.seasonal_order = coerce_integer_count(
            seasonal_order, label="seasonal_order", minimum=0
        )
        self.difference = coerce_integer_count(
            difference, label="difference", minimum=0
        )
        if self.difference > 2:
            raise ValueError("difference must be 0, 1 or 2")
        self.seasonal_difference = coerce_integer_count(
            seasonal_difference, label="seasonal_difference", minimum=0
        )
        if self.seasonal_difference > 1:
            raise ValueError("seasonal_difference must be 0 or 1")
        if (self.seasonal_order or self.seasonal_difference) and self.season_length < 2:
            raise ValueError(
                "seasonal_order and seasonal_difference need season_length >= 2"
            )
        self.forgetting = coerce_finite_number(forgetting, label="forgetting")
        if not 0 < self.forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1]")
        self.delta = coerce_finite_number(delta, label="delta")
        if self.delta <= 0:
            raise ValueError("delta must be positive")
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )
        m = self.season_length
        lags = {
            i + j * m
            for i in range(self.order + 1)
            for j in range(self.seasonal_order + 1)
        }
        # Lags of the differenced series the weights multiply, in order.
        self._lags = tuple(sorted(lags - {0}))
        self._differencing = _differencing(self.difference, self.seasonal_difference, m)
        self._reset_state()

    @property
    def _undone(self) -> int:
        """Past values that differencing needs: ``d + D m``."""
        return self.difference + self.seasonal_difference * self.season_length

    @property
    def _required(self) -> int:
        lags = self._lags
        return self._undone + max(lags, default=0) + len(lags) + 1 + self.horizon

    def _reset_state(self) -> None:
        """Forget the weights, the lag buffers and the errors."""
        lags = self._lags
        size = len(lags) + 1
        self.n_samples_seen = 0
        # Events since the first observed value.
        self._learned = 0
        self._values: deque[float] = deque(maxlen=self._undone or None)
        self._differenced: deque[float] = deque(maxlen=max(lags, default=0) or None)
        self._weights = np.zeros(size)
        self._covariance = np.eye(size) * self.delta
        self._updates = 0
        self._stale = 0
        self._last = 0.0
        self._errors = LeadErrors(
            self.horizon, self.residual_alpha, self.min_scale, seeded=False
        )

    @property
    def readiness(self) -> Readiness:
        """Ready once a forecast at the longest lead has been scored."""
        assert self._errors.counts is not None
        return Readiness(self._errors.counts[-1] > 0, self._learned, self._required)

    def _windows_full(self) -> tuple[bool, bool]:
        undone = self._undone
        values_full = undone == 0 or len(self._values) == undone
        lag = max(self._lags, default=0)
        return values_full, lag == 0 or len(self._differenced) == lag

    def _design(self, differenced: list[float] | deque[float]) -> np.ndarray:
        return np.array([*(differenced[-lag] for lag in self._lags), 1.0], dtype=float)

    def learn_one(self, y: float | None) -> None:
        """Learn the next value of the series, or ``None`` for a missing one."""
        value = None if y is None else coerce_event_number(y, label="y")
        if value is None and self._learned == 0:
            self.n_samples_seen += 1
            return
        observed = value is not None
        if value is None:
            issued = self._errors.issued
            value = issued[-1][0] if issued else self._last
        values_full, lags_full = self._windows_full()
        differencing = self._differencing
        weights, covariance, updates = self._weights, self._covariance, self._updates
        differenced = None
        if values_full:
            differenced = fsum(
                (value, *(weight * self._values[-lag] for lag, weight in differencing))
            )
            if lags_full and observed:
                _, weights, covariance = rls_update(
                    weights,
                    covariance,
                    self._design(self._differenced),
                    differenced,
                    self.forgetting,
                )
                updates += 1
        mae = (
            self._errors.updated(value)
            if observed and self._errors.issued
            else list(self._errors.mae)
        )
        stale = 0 if observed else self._stale + 1
        values = [*self._values, value]
        lagged = [*self._differenced] + ([] if differenced is None else [differenced])
        means = None
        if updates >= len(weights):
            means = self._means(weights, values, lagged, differencing)
        # Publish only after all arithmetic succeeds.
        if self._undone:
            self._values.append(value)
        if differenced is not None and self._differenced.maxlen is not None:
            self._differenced.append(differenced)
        self._weights, self._covariance, self._updates = weights, covariance, updates
        if means is not None:
            self._errors.publish(mae, means, stale)
        self._stale = stale
        if observed:
            self._last = value
        self._learned += 1
        self.n_samples_seen += 1

    def _means(
        self,
        weights: np.ndarray,
        values: list[float],
        lagged: list[float],
        differencing: tuple[tuple[int, float], ...],
    ) -> tuple[float, ...]:
        """Recursive forecasts of leads ``1..horizon`` from the new state."""
        means = []
        for _ in range(self.horizon):
            with np.errstate(over="ignore", invalid="ignore"):
                step = finite(float(weights @ self._design(lagged)))
            mean = fsum(
                (step, *(-weight * values[-lag] for lag, weight in differencing))
            )
            means.append(mean)
            values.append(mean)
            lagged.append(step)
        return tuple(means)

    def forecast_one(self) -> Forecast | None:
        """Forecast leads ``1..horizon``, or ``None`` before readiness."""
        if not self.readiness.ready:
            return None
        return Forecast(
            origin=self.n_samples_seen,
            leads=tuple(range(1, self.horizon + 1)),
            mean=self._errors.issued[-1],
            scale=self._errors.scales,
        )
