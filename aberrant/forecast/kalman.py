"""Local level and local linear trend forecasts from an exact Kalman filter."""

from __future__ import annotations

import math

from aberrant._utils.validation import coerce_event_number, coerce_finite_number
from aberrant.base.forecast import Forecast
from aberrant.base.readiness import Readiness
from aberrant.forecast._arithmetic import finite
from aberrant.forecast.base import BaseForecaster, check_spread_parameters

# A symmetric 2x2 matrix as (top left, off diagonal, bottom right).
_Symmetric = tuple[float, float, float]


def _ratio(value: object, label: str) -> float:
    ratio = coerce_finite_number(value, label=label)
    if ratio < 0:
        raise ValueError(f"{label} must be non-negative")
    return ratio


class KalmanForecaster(BaseForecaster):
    """Forecast a local level, or a local linear trend, with a Kalman filter.

    The series is a level observed with noise, ``y_t = level_t + e_t``, and
    the level moves by a random step each event. With ``slope_ratio`` it also
    moves by a slope, which itself drifts::

        level_{t+1} = level_t + slope_t + u_t
        slope_{t+1} = slope_t + w_t

    ``level_ratio`` and ``slope_ratio`` are the variances of ``u`` and ``w``
    over that of ``e``. They set how fast the forecast follows the series: a
    local level with ratio ``q`` is exponential smoothing with weight
    ``(sqrt(q^2 + 4q) - q) / 2``, so the default 0.05 corresponds to about
    0.2. The noise variance itself is estimated online, as the faded mean of
    the squared standardized one-step errors.

    The spread of lead ``h`` is the standard deviation of the model's
    predictive distribution, which grows with the lead as the filter's
    uncertainty about the level and slope does. With ``score="scaled"`` in a
    :class:`~aberrant.forecast.ResidualDetector`, a score is the error in
    predictive standard deviations.

    ``learn_one(None)`` learns that a value is missing: the series advances,
    the state is predicted without a correction, and the spreads widen.

    Args:
        horizon: Largest lead to forecast.
        level_ratio: Variance of the level's step over the noise variance,
            at least 0; 0 fits one constant level.
        slope_ratio: Variance of the slope's step over the noise variance,
            at least 0, or ``None`` for no slope. 0 fits one constant slope.
        residual_alpha: Weight of each new squared standardized error in the
            faded noise variance, in (0, 1]. Until ``1 / residual_alpha``
            errors are seen, the plain mean is used.
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        The initial level and slope are diffuse: the exact diffuse filter
        fits them to the first one or two observed values, which therefore
        give no error. The forecaster is ready after two observed values, or
        three with a slope. State is O(horizon) and each event costs
        O(horizon). Unrepresentable arithmetic raises
        OverflowError without changing state.

    Examples:
        ```python
        from aberrant.forecast import KalmanForecaster

        model = KalmanForecaster(horizon=3, slope_ratio=0.0)
        for y in [1.0, 2.0, 3.0, None, 5.0, 6.0]:
            model.learn_one(y)
        forecast = model.forecast_one()
        forecast.mean  # (7.0, 8.0, 9.0)
        ```

    References:
        Durbin and Koopman, Time Series Analysis by State Space Methods,
        2nd ed., chapters 2, 3 and 5 (the exact diffuse initialization).
        Harvey, Forecasting, Structural Time Series Models and the Kalman
        Filter, 1989.
    """

    def __init__(
        self,
        *,
        horizon: int = 1,
        level_ratio: float = 0.05,
        slope_ratio: float | None = None,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        super().__init__(horizon=horizon)
        self.level_ratio = _ratio(level_ratio, "level_ratio")
        self.slope_ratio = (
            None if slope_ratio is None else _ratio(slope_ratio, "slope_ratio")
        )
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )
        self._reset_state()

    @property
    def _order(self) -> int:
        return 1 if self.slope_ratio is None else 2

    def _reset_state(self) -> None:
        """Forget the filtered state and the noise variance."""
        self.n_samples_seen = 0
        self.n_observed = 0
        self._level = 0.0
        self._slope = 0.0
        # The state covariance, in units of the noise variance, splits into
        # a diffuse part, scaled by an infinite prior variance, and the rest.
        self._diffuse: _Symmetric = (1.0, 0.0, 0.0 if self.slope_ratio is None else 1.0)
        self._covariance: _Symmetric = (0.0, 0.0, 0.0)
        self._variance = 0.0
        self._errors = 0
        self._forecast: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())

    @property
    def readiness(self) -> Readiness:
        """Ready once the state is fitted and one error estimates the noise."""
        return Readiness.after(self.n_observed, self._order + 1)

    def learn_one(self, y: float | None) -> None:
        """Learn the next value of the series, or ``None`` for a missing one."""
        level, slope = self._level, self._slope
        diffuse, covariance = self._diffuse, self._covariance
        variance, errors = self._variance, self._errors
        if y is not None:
            value = coerce_event_number(y, label="y")
            error = finite(value - level)
            d00, d01, d11 = diffuse
            p00, p01, p11 = covariance
            spread = p00 + 1
            if d00 > 0:
                level = finite(level + error)
                slope = finite(slope + d01 / d00 * error)
                diffuse = (0.0, 0.0, d11 - d01 * d01 / d00)
                covariance = (
                    1.0,
                    d01 / d00,
                    finite(p11 + (d01 / d00) ** 2 * spread - 2 * p01 * d01 / d00),
                )
            else:
                gain = p01 / spread
                level = finite(level + p00 / spread * error)
                slope = finite(slope + gain * error)
                covariance = (p00 / spread, gain, p11 - p01 * gain)
                errors += 1
                weight = max(self.residual_alpha, 1 / errors)
                variance = finite(
                    (1 - weight) * variance + weight * (error * error / spread)
                )
        level = finite(level + slope)
        diffuse, covariance = self._predicted(diffuse, covariance)
        issued = self._issued(level, slope, covariance, variance)
        self._level, self._slope = level, slope
        self._diffuse, self._covariance = diffuse, covariance
        self._variance, self._errors = variance, errors
        self._forecast = issued
        self.n_samples_seen += 1
        self.n_observed += y is not None

    def _predicted(
        self, diffuse: _Symmetric, covariance: _Symmetric
    ) -> tuple[_Symmetric, _Symmetric]:
        """Both covariance parts one event later."""
        d00, d01, d11 = diffuse
        p00, p01, p11 = covariance
        slope_ratio = self.slope_ratio or 0.0
        return (d00 + 2 * d01 + d11, d01 + d11, d11), (
            finite(p00 + 2 * p01 + p11 + self.level_ratio),
            finite(p01 + p11),
            finite(p11 + slope_ratio),
        )

    def _issued(
        self, level: float, slope: float, covariance: _Symmetric, variance: float
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Mean and predictive standard deviation of every lead."""
        p00, p01, p11 = covariance
        slope_ratio = self.slope_ratio or 0.0
        means, scales = [], []
        steps = 0.0
        for ahead in range(self.horizon):
            means.append(finite(level + ahead * slope))
            spread = p00 + 2 * ahead * p01 + ahead * ahead * p11 + steps + 1
            scale = finite(math.sqrt(finite(variance * spread)))
            scales.append(max(self.min_scale, scale))
            steps += self.level_ratio + ahead * ahead * slope_ratio
        return tuple(means), tuple(scales)

    def forecast_one(self) -> Forecast | None:
        """Forecast leads ``1..horizon``, or ``None`` before readiness."""
        if not self.readiness.ready:
            return None
        means, scales = self._forecast
        return Forecast(
            origin=self.n_samples_seen,
            leads=tuple(range(1, self.horizon + 1)),
            mean=means,
            scale=scales,
        )
