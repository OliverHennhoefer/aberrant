"""Forecasts from any online regressor of the series' own past."""

from __future__ import annotations

import math
from collections import deque

from aberrant._utils.validation import coerce_event_number, coerce_integer_count
from aberrant.base.forecast import Forecast
from aberrant.base.protocols import ForecasterProtocol, RegressorProtocol
from aberrant.base.readiness import Readiness
from aberrant.forecast._arithmetic import LeadErrors, finite, fsum
from aberrant.forecast.autoregressive import _differencing
from aberrant.forecast.base import BaseForecaster, check_spread_parameters


class LagForecaster(BaseForecaster):
    """Forecast a series with any online regressor of its own past values.

    The series is differenced ``difference`` times and seasonally
    ``seasonal_difference`` times. Each differenced value is a regression
    target, and its covariates are the ``order`` values before it, the
    values ``season_length``, ``2 * season_length``, ... up to
    ``seasonal_order`` cycles before it, and ``harmonics`` sine and cosine
    pairs of its phase in the cycle::

        {"lag_1": w_{t-1}, ..., "lag_12": w_{t-12},
         "phase_sin_1": sin(2 pi t / m), "phase_cos_1": cos(2 pi t / m)}

    The regressor learns every observed value. Leads beyond 1 feed its
    predictions back in as lags and undo the differencing. With a linear
    regressor this is a seasonal autoregression; with a tree ensemble, a
    kernel or neighbours regressor it captures nonlinear dynamics.

    Args:
        regressor: The regressor to train, owned by the forecaster. Its
            spread is not used.
        horizon: Largest lead to forecast.
        order: Number of recent values ``p`` used, at least 0.
        season_length: Observations per cycle ``m``; 1 for none.
        seasonal_order: Number of earlier cycles ``P`` whose value at the same
            phase is used, at least 0. Needs ``season_length`` of at least 2.
        harmonics: Number of sine and cosine pairs of the phase, at least 0.
            Needs ``season_length`` of at least 2.
        difference: Non-seasonal differences ``d``, 0, 1 or 2.
        seasonal_difference: Seasonal differences ``D``, 0 or 1. Needs
            ``season_length`` of at least 2.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error kept for every lead, in (0, 1]. Until
            ``1 / residual_alpha`` errors are seen at a lead, the plain mean
            is used.
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        The spread of lead ``h`` is the faded mean absolute error of the lead
        ``h`` forecasts. The forecaster is ready once the longest lead has
        been scored once: after ``d + D m`` values to difference, the largest
        lag ``max(p, P m)``, the events the regressor needs and ``horizon``
        more values. ``readiness.required`` counts them once the regressor
        reports its own requirement.

        ``learn_one(None)`` learns that a value is missing: once forecasting,
        the lead-1 forecast takes its place and the regressor learns nothing;
        each forecast reports the spread of the longer lead it now is. Before
        that, the last observed value takes its place. Missing values before
        the first observed one do not start the series.

        Each event costs one regressor update and ``horizon`` predictions. An
        OverflowError from the regressor's predictions propagates after it
        learned the value, so reset the forecaster after one.

    Examples:
        ```python
        from aberrant.forecast import LagForecaster, RLSRegressor

        model = LagForecaster(RLSRegressor(), horizon=2, order=1)
        y = 0.0
        for _ in range(200):
            y = 0.8 * y + 1.0
            model.learn_one(y)
        model.forecast_one().mean  # close to (5.0, 5.0)
        ```
    """

    def __init__(
        self,
        regressor: RegressorProtocol,
        *,
        horizon: int = 1,
        order: int = 3,
        season_length: int = 1,
        seasonal_order: int = 0,
        harmonics: int = 0,
        difference: int = 0,
        seasonal_difference: int = 0,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        super().__init__(horizon=horizon)
        if not isinstance(regressor, RegressorProtocol) or isinstance(
            regressor, ForecasterProtocol
        ):
            raise ValueError("regressor must be a regressor")
        self.regressor = regressor
        self.order = coerce_integer_count(order, label="order", minimum=0)
        self.season_length = coerce_integer_count(season_length, label="season_length")
        self.seasonal_order = coerce_integer_count(
            seasonal_order, label="seasonal_order", minimum=0
        )
        self.harmonics = coerce_integer_count(harmonics, label="harmonics", minimum=0)
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
        seasonal = self.seasonal_order or self.seasonal_difference or self.harmonics
        if seasonal and self.season_length < 2:
            raise ValueError(
                "seasonal_order, harmonics and seasonal_difference need "
                "season_length >= 2"
            )
        if not self.order and not self.seasonal_order and not self.harmonics:
            raise ValueError("order, seasonal_order or harmonics must be positive")
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )
        m = self.season_length
        self._lags = tuple(
            sorted(
                {*range(1, self.order + 1)}
                | {j * m for j in range(1, self.seasonal_order + 1)}
            )
        )
        self._differencing = _differencing(self.difference, self.seasonal_difference, m)
        self._reset_state()

    @property
    def _undone(self) -> int:
        return self.difference + self.seasonal_difference * self.season_length

    def _reset_state(self) -> None:
        """Forget the lag buffers and the errors."""
        self.n_samples_seen = 0
        # Events since the first observed value.
        self._learned = 0
        self._values: deque[float] = deque(maxlen=self._undone or None)
        self._differenced: deque[float] = deque(maxlen=max(self._lags, default=1))
        self._stale = 0
        self._last = 0.0
        self._errors = LeadErrors(
            self.horizon, self.residual_alpha, self.min_scale, seeded=False
        )

    @property
    def readiness(self) -> Readiness:
        """Ready once a forecast at the longest lead has been scored."""
        assert self._errors.counts is not None
        needed = self.regressor.readiness.required
        required = (
            None
            if needed is None
            else self._undone + max(self._lags, default=0) + needed + self.horizon
        )
        return Readiness(self._errors.counts[-1] > 0, self._learned, required)

    def _features(self, lagged: list[float], index: int) -> dict[str, float]:
        """Covariates of the differenced value at event ``index``."""
        x = {f"lag_{lag}": lagged[-lag] for lag in self._lags}
        for k in range(1, self.harmonics + 1):
            angle = 2 * math.pi * k * (index % self.season_length) / self.season_length
            x[f"phase_sin_{k}"] = math.sin(angle)
            x[f"phase_cos_{k}"] = math.cos(angle)
        return x

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
        undone = self._undone
        values_full = undone == 0 or len(self._values) == undone
        lags_full = len(self._differenced) >= max(self._lags, default=0)
        differenced = None
        if values_full:
            differenced = fsum(
                (
                    value,
                    *(
                        weight * self._values[-lag]
                        for lag, weight in self._differencing
                    ),
                )
            )
        mae = (
            self._errors.updated(value)
            if observed and self._errors.issued
            else list(self._errors.mae)
        )
        if differenced is not None and lags_full and observed:
            x = self._features(list(self._differenced), self._learned)
            self.regressor.learn_one(x, differenced)
        stale = 0 if observed else self._stale + 1
        values = [*self._values, value]
        lagged = [*self._differenced] + ([] if differenced is None else [differenced])
        means = None
        if len(lagged) >= max(self._lags, default=0) and differenced is not None:
            means = self._means(values, lagged)
        if undone:
            self._values.append(value)
        if differenced is not None:
            self._differenced.append(differenced)
        if means is not None:
            self._errors.publish(mae, means, stale)
        self._stale = stale
        if observed:
            self._last = value
        self._learned += 1
        self.n_samples_seen += 1

    def _means(
        self, values: list[float], lagged: list[float]
    ) -> tuple[float, ...] | None:
        """Recursive forecasts of leads ``1..horizon``, or ``None`` before ready."""
        means = []
        for lead in range(1, self.horizon + 1):
            prediction = self.regressor.predict_one(
                self._features(lagged, self._learned + lead)
            )
            if prediction is None:
                return None
            step = finite(prediction.mean[0])
            mean = fsum(
                (step, *(-weight * values[-lag] for lag, weight in self._differencing))
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
