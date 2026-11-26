"""Structural time series forecasts from an exact diffuse Kalman filter."""

from __future__ import annotations

import math

import numpy as np

from aberrant._utils.validation import (
    coerce_event_number,
    coerce_finite_number,
    coerce_integer_count,
)
from aberrant.base.forecast import Forecast
from aberrant.base.readiness import Readiness
from aberrant.forecast.base import BaseForecaster, check_spread_parameters

# Diffuse variances at or below this are zero: the filter has fitted them.
_TOLERANCE = 1e-9


def _ratio(value: object, label: str) -> float:
    ratio = coerce_finite_number(value, label=label)
    if ratio < 0:
        raise ValueError(f"{label} must be non-negative")
    return ratio


class KalmanForecaster(BaseForecaster):
    """Forecast a structural time series model with a Kalman filter.

    The series is a level, optionally a slope, and optionally a season,
    observed with noise. Each component moves by a random step every event::

        y_t = level_t + season_t + e_t
        level_{t+1} = level_t + slope_t + u_t
        slope_{t+1} = slope_t + w_t

    The season is a sum of ``harmonics`` sine waves with periods
    ``season_length / j``, each of whose amplitude and phase drift by a
    random step. All harmonics reproduce any repeating pattern of
    ``season_length`` values; fewer give a smoother season with fewer states,
    which suits long cycles such as a day of minutes.

    ``level_ratio``, ``slope_ratio`` and ``season_ratio`` are the variances of
    the steps over that of ``e``. They set how fast the forecast follows the
    series: a local level with ratio ``q`` is exponential smoothing with
    weight ``(sqrt(q^2 + 4q) - q) / 2``, so the default 0.05 corresponds to
    about 0.2. The noise variance itself is estimated online, as the faded
    mean of the squared standardized one-step errors.

    The spread of lead ``h`` is the standard deviation of the model's
    predictive distribution, which grows with the lead as the filter's
    uncertainty about the components does. With ``score="scaled"`` in a
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
        season_length: Observations per cycle; 1 for no season.
        season_ratio: Variance of each harmonic's step over the noise
            variance, at least 0; 0 fits one fixed season.
        harmonics: Number of harmonics, from 1 to ``season_length // 2``, or
            ``None`` for all of them. Needs ``season_length`` of at least 2.
        residual_alpha: Weight of each new squared standardized error in the
            faded noise variance, in (0, 1]. Until ``1 / residual_alpha``
            errors are seen, the plain mean is used.
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        The initial state is diffuse: the exact diffuse filter fits it to the
        first observed values, one per state, which therefore give no error.
        There are ``k`` states: 1 for the level, 1 for a slope, and 2 per
        harmonic, except 1 for the harmonic of period 2. The forecaster is
        ready after ``k + 1`` observed values, or later when missing values
        leave part of the state unfitted. State is O(k^2 + horizon k) and each
        event costs O(k^3 + horizon k^2). Unrepresentable arithmetic raises
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
        Filter, 1989, chapter 2 (the trigonometric season).
    """

    def __init__(
        self,
        *,
        horizon: int = 1,
        level_ratio: float = 0.05,
        slope_ratio: float | None = None,
        season_length: int = 1,
        season_ratio: float = 0.001,
        harmonics: int | None = None,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        super().__init__(horizon=horizon)
        self.level_ratio = _ratio(level_ratio, "level_ratio")
        self.slope_ratio = (
            None if slope_ratio is None else _ratio(slope_ratio, "slope_ratio")
        )
        self.season_length = coerce_integer_count(season_length, label="season_length")
        self.season_ratio = _ratio(season_ratio, "season_ratio")
        largest = self.season_length // 2
        if harmonics is None:
            self.harmonics = None
        else:
            if largest == 0:
                raise ValueError("harmonics needs season_length >= 2")
            self.harmonics = coerce_integer_count(harmonics, label="harmonics")
            if self.harmonics > largest:
                raise ValueError("harmonics must be at most season_length // 2")
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )
        self._build(largest if self.harmonics is None else self.harmonics)
        self._reset_state()

    def _build(self, harmonics: int) -> None:
        """Transition, observation and step variances of the state."""
        blocks: list[np.ndarray] = [np.ones((1, 1))]
        observe, noise = [1.0], [self.level_ratio]
        if self.slope_ratio is not None:
            blocks[0] = np.array([[1.0, 1.0], [0.0, 1.0]])
            observe.append(0.0)
            noise.append(self.slope_ratio)
        for j in range(1, harmonics + 1):
            if 2 * j == self.season_length:
                blocks.append(-np.ones((1, 1)))
                observe.append(1.0)
                noise.append(self.season_ratio)
                continue
            angle = 2 * math.pi * j / self.season_length
            cos, sin = math.cos(angle), math.sin(angle)
            blocks.append(np.array([[cos, sin], [-sin, cos]]))
            observe += [1.0, 0.0]
            noise += [self.season_ratio] * 2
        size = len(observe)
        transition = np.zeros((size, size))
        start = 0
        for block in blocks:
            end = start + len(block)
            transition[start:end, start:end] = block
            start = end
        self._transition = transition
        self._observe = np.array(observe)
        self._noise = np.diag(noise)
        # Row h of _leads maps the next state to lead h + 1; _steps adds the
        # variance of the steps taken before that lead.
        leads, steps = [self._observe], [0.0]
        for _ in range(1, self.horizon):
            steps.append(steps[-1] + float(leads[-1] ** 2 @ np.array(noise)))
            leads.append(leads[-1] @ transition)
        self._leads = np.array(leads)
        self._steps = np.array(steps)

    def _reset_state(self) -> None:
        """Forget the filtered state and the noise variance."""
        size = self._observe.size
        self.n_samples_seen = 0
        self.n_observed = 0
        self._state = np.zeros(size)
        # The state covariance, in units of the noise variance, splits into
        # a diffuse part, scaled by an infinite prior variance, and the rest.
        self._diffuse = np.eye(size)
        self._covariance = np.zeros((size, size))
        self._variance = 0.0
        self._errors = 0
        self._forecast: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())

    @property
    def readiness(self) -> Readiness:
        """Ready once the state is fitted and one error estimates the noise."""
        fitted = self._errors > 0 and not self._diffuse.any()
        return Readiness(fitted, self.n_observed, self._observe.size + 1)

    def learn_one(self, y: float | None) -> None:
        """Learn the next value of the series, or ``None`` for a missing one."""
        value = None if y is None else coerce_event_number(y, label="y")
        state, diffuse, covariance = self._state, self._diffuse, self._covariance
        variance, errors = self._variance, self._errors
        transition = self._transition
        fitting = bool(diffuse.any())
        with np.errstate(over="ignore", invalid="ignore"):
            if value is not None:
                z = self._observe
                error = value - float(z @ state)
                m_star = covariance @ z
                f_star = float(z @ m_star) + 1.0
                m_inf = diffuse @ z if fitting else m_star
                f_inf = float(z @ m_inf) if fitting else 0.0
                if f_inf > _TOLERANCE:
                    state = state + m_inf * (error / f_inf)
                    outer = np.outer(m_inf, m_inf)
                    diffuse = diffuse - outer / f_inf
                    cross = np.outer(m_star, m_inf)
                    covariance = (
                        covariance
                        + outer * (f_star / f_inf**2)
                        - (cross + cross.T) / f_inf
                    )
                else:
                    state = state + m_star * (error / f_star)
                    covariance = covariance - np.outer(m_star, m_star) / f_star
                    errors += 1
                    weight = max(self.residual_alpha, 1 / errors)
                    variance = (1 - weight) * variance + weight * (
                        error * error / f_star
                    )
            state = transition @ state
            if fitting:
                diffuse = transition @ diffuse @ transition.T
                if np.abs(diffuse).max() <= _TOLERANCE:
                    diffuse = np.zeros_like(diffuse)
            covariance = transition @ covariance @ transition.T + self._noise
            covariance = (covariance + covariance.T) / 2
            means, scales = self._issued(state, covariance, variance)
        if not (
            math.isfinite(variance)
            and np.isfinite(diffuse).all()
            and np.isfinite(covariance).all()
            and np.isfinite(means).all()
            and np.isfinite(scales).all()
        ):
            raise OverflowError("Forecast arithmetic exceeds the float range")
        self._state, self._diffuse, self._covariance = state, diffuse, covariance
        self._variance, self._errors = variance, errors
        self._forecast = tuple(means.tolist()), tuple(scales.tolist())
        self.n_samples_seen += 1
        self.n_observed += value is not None

    def _issued(
        self, state: np.ndarray, covariance: np.ndarray, variance: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and predictive standard deviation of every lead."""
        leads = self._leads
        spreads = ((leads @ covariance) * leads).sum(axis=1) + self._steps + 1
        return leads @ state, np.maximum(self.min_scale, np.sqrt(variance * spreads))

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
