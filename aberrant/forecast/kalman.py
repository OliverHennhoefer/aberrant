"""Structural time series forecasts from an exact diffuse Kalman filter."""

from __future__ import annotations

import math
from typing import NamedTuple

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
# Bounds of the ratios that are learned, and the fading of their squared
# gradients.
_RATIO_FLOOR = 1e-8
_RATIO_CEILING = 1e3
_GRADIENT_FADING = 0.99


def _ratio(value: object, label: str) -> float:
    ratio = coerce_finite_number(value, label=label)
    if ratio < 0:
        raise ValueError(f"{label} must be non-negative")
    return ratio


class _Filter(NamedTuple):
    """Everything one event changes, so that it changes all or nothing."""

    state: np.ndarray
    diffuse: np.ndarray
    covariance: np.ndarray
    variance: float
    errors: int
    log_ratios: np.ndarray
    squares: np.ndarray
    tangent_state: np.ndarray
    tangent_covariance: np.ndarray


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
    weight ``(sqrt(q^2 + 4q) - q) / 2``, so 0.05 corresponds to about 0.2.
    By default the filter learns them from the series, starting at the
    given values; ``learn_ratios=False`` keeps them fixed. The noise
    variance itself is estimated online, as the faded mean of the squared
    standardized one-step errors.

    The spread of lead ``h`` is the standard deviation of the model's
    predictive distribution, which grows with the lead as the filter's
    uncertainty about the components does. With ``score="scaled"`` in a
    :class:`~aberrant.forecast.ResidualDetector`, a score is the error in
    predictive standard deviations.

    The ratios are learned by recursive maximum likelihood: a tangent filter carries the
    derivatives of the state and its covariance with respect to the log
    ratios, and every scored value moves each log ratio along the gradient
    of its log-likelihood, in steps of about ``ratio_rate`` normalized by
    the faded root mean square of the gradient (Ljung and Soderstrom,
    1983). The harmonics share one season ratio. Learned ratios stay within
    1e-8 and 1e3.

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
        learn_ratios: Whether to learn the ratios from the series, starting
            at the values given.
        ratio_rate: Positive step size of the learned log ratios.
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
        event costs O(k^3 + horizon k^2); learning ``r`` ratios multiplies
        both state and cost by ``r + 1``. Unrepresentable arithmetic raises
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
        Filter, 1989, chapter 2 (the trigonometric season). Ljung and
        Soderstrom, Theory and Practice of Recursive Identification, 1983
        (recursive prediction error methods).
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
        learn_ratios: bool = True,
        ratio_rate: float = 0.01,
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
        if not isinstance(learn_ratios, bool):
            raise ValueError("learn_ratios must be a boolean")
        self.learn_ratios = learn_ratios
        self.ratio_rate = coerce_finite_number(ratio_rate, label="ratio_rate")
        if self.ratio_rate <= 0:
            raise ValueError("ratio_rate must be positive")
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )
        self._build(largest if self.harmonics is None else self.harmonics)
        self._reset_state()

    def _build(self, harmonics: int) -> None:
        """Transition, observation and step variances of the state."""
        blocks: list[np.ndarray] = [np.ones((1, 1))]
        observe, groups = [1.0], [0]
        ratios = [self.level_ratio]
        if self.slope_ratio is not None:
            blocks[0] = np.array([[1.0, 1.0], [0.0, 1.0]])
            observe.append(0.0)
            groups.append(1)
            ratios.append(self.slope_ratio)
        if harmonics:
            ratios.append(self.season_ratio)
        for j in range(1, harmonics + 1):
            if 2 * j == self.season_length:
                blocks.append(-np.ones((1, 1)))
                observe.append(1.0)
                groups.append(len(ratios) - 1)
                continue
            angle = 2 * math.pi * j / self.season_length
            cos, sin = math.cos(angle), math.sin(angle)
            blocks.append(np.array([[cos, sin], [-sin, cos]]))
            observe += [1.0, 0.0]
            groups += [len(ratios) - 1] * 2
        size = len(observe)
        transition = np.zeros((size, size))
        start = 0
        for block in blocks:
            end = start + len(block)
            transition[start:end, start:end] = block
            start = end
        self._transition = transition
        self._observe = np.array(observe)
        # The ratio of each state's step, and the ratios as given.
        self._groups = np.array(groups)
        self._ratios = np.array(ratios)
        # Row h of _leads maps the next state to lead h + 1; row h of _paths
        # weighs the state steps taken before that lead.
        leads = [self._observe]
        paths = [np.zeros(size)]
        for _ in range(1, self.horizon):
            paths.append(paths[-1] + leads[-1] ** 2)
            leads.append(leads[-1] @ transition)
        self._leads = np.array(leads)
        self._paths = np.array(paths)

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
        count = self._ratios.size if self.learn_ratios else 0
        self._log_ratios = np.log(np.maximum(self._ratios, _RATIO_FLOOR))
        self._squares = np.zeros(count)
        self._tangent_state = np.zeros((count, size))
        self._tangent_covariance = np.zeros((count, size, size))
        self._forecast: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())

    @property
    def readiness(self) -> Readiness:
        """Ready once the state is fitted and one error estimates the noise."""
        fitted = self._errors > 0 and not self._diffuse.any()
        return Readiness(fitted, self.n_observed, self._observe.size + 1)

    @property
    def ratios(self) -> dict[str, float]:
        """The step variance ratios in use, by component."""
        names = ["level"]
        if self.slope_ratio is not None:
            names.append("slope")
        if self._ratios.size > len(names):
            names.append("season")
        values = np.exp(self._log_ratios) if self.learn_ratios else self._ratios
        return dict(zip(names, values.tolist(), strict=True))

    def _noise(self, log_ratios: np.ndarray) -> np.ndarray:
        """Step variance ratio of every state."""
        ratios = np.exp(log_ratios) if self.learn_ratios else self._ratios
        noise: np.ndarray = ratios[self._groups]
        return noise

    def learn_one(self, y: float | None) -> None:
        """Learn the next value of the series, or ``None`` for a missing one."""
        value = None if y is None else coerce_event_number(y, label="y")
        filtered = _Filter(
            self._state,
            self._diffuse,
            self._covariance,
            self._variance,
            self._errors,
            self._log_ratios,
            self._squares,
            self._tangent_state,
            self._tangent_covariance,
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if value is not None:
                filtered = self._corrected(filtered, value)
            predicted = self._predicted(filtered)
            noise = self._noise(predicted.log_ratios)
            means, scales = self._issued(
                predicted.state, predicted.covariance, predicted.variance, noise
            )
        arrays = (*predicted[:3], *predicted[5:], means, scales)
        if not (
            math.isfinite(predicted.variance)
            and all(np.isfinite(array).all() for array in arrays)
        ):
            raise OverflowError("Forecast arithmetic exceeds the float range")
        self._state, self._diffuse, self._covariance = predicted[:3]
        self._variance, self._errors = predicted.variance, predicted.errors
        self._log_ratios, self._squares = predicted.log_ratios, predicted.squares
        self._tangent_state = predicted.tangent_state
        self._tangent_covariance = predicted.tangent_covariance
        self._forecast = tuple(means.tolist()), tuple(scales.tolist())
        self.n_samples_seen += 1
        self.n_observed += value is not None

    def _corrected(self, filtered: _Filter, value: float) -> _Filter:
        """The filter after observing ``value``: diffuse or standard update."""
        state, diffuse, covariance = (
            filtered.state,
            filtered.diffuse,
            filtered.covariance,
        )
        z = self._observe
        error = value - float(z @ state)
        m_star = covariance @ z
        f_star = float(z @ m_star) + 1.0
        fitting = bool(diffuse.any())
        m_inf = diffuse @ z if fitting else m_star
        f_inf = float(z @ m_inf) if fitting else 0.0
        if f_inf > _TOLERANCE:
            outer = np.outer(m_inf, m_inf)
            cross = np.outer(m_star, m_inf)
            return filtered._replace(
                state=state + m_inf * (error / f_inf),
                diffuse=diffuse - outer / f_inf,
                covariance=covariance
                + outer * (f_star / f_inf**2)
                - (cross + cross.T) / f_inf,
                tangent_state=np.zeros_like(filtered.tangent_state),
                tangent_covariance=np.zeros_like(filtered.tangent_covariance),
            )
        log_ratios, squares = filtered.log_ratios, filtered.squares
        tangents = (filtered.tangent_state, filtered.tangent_covariance)
        variance = filtered.variance
        if self.learn_ratios:
            if variance > 0:
                gradient = _gradient(tangents, z, error, f_star, variance)
                log_ratios, squares = self._stepped(log_ratios, squares, gradient)
            tangents = _filtered(tangents, z, error, m_star, f_star)
        errors = filtered.errors + 1
        weight = max(self.residual_alpha, 1 / errors)
        return filtered._replace(
            state=state + m_star * (error / f_star),
            covariance=covariance - np.outer(m_star, m_star) / f_star,
            variance=(1 - weight) * variance + weight * (error * error / f_star),
            errors=errors,
            log_ratios=log_ratios,
            squares=squares,
            tangent_state=tangents[0],
            tangent_covariance=tangents[1],
        )

    def _predicted(self, filtered: _Filter) -> _Filter:
        """The filter one event on."""
        transition = self._transition
        diffuse = filtered.diffuse
        if diffuse.any():
            diffuse = transition @ diffuse @ transition.T
            if np.abs(diffuse).max() <= _TOLERANCE:
                diffuse = np.zeros_like(diffuse)
        noise = self._noise(filtered.log_ratios)
        covariance = transition @ filtered.covariance @ transition.T + np.diag(noise)
        tangent_state, tangent_covariance = (
            filtered.tangent_state,
            filtered.tangent_covariance,
        )
        if self.learn_ratios:
            tangent_state = tangent_state @ transition.T
            tangent_covariance = transition @ tangent_covariance @ transition.T
            indices = np.arange(noise.size)
            tangent_covariance[self._groups, indices, indices] += noise
            tangent_covariance = (
                tangent_covariance + tangent_covariance.transpose(0, 2, 1)
            ) / 2
        return filtered._replace(
            state=transition @ filtered.state,
            diffuse=diffuse,
            covariance=(covariance + covariance.T) / 2,
            tangent_state=tangent_state,
            tangent_covariance=tangent_covariance,
        )

    def _stepped(
        self, log_ratios: np.ndarray, squares: np.ndarray, gradient: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Log ratios and faded squared gradients after one gradient step.

        A gradient is clipped at ten root mean squares, so an outlier moves
        the ratios no more than a few ordinary values do.
        """
        if squares.any():
            bound = 10 * np.sqrt(squares)
            gradient = np.clip(gradient, -bound, bound)
            squares = _GRADIENT_FADING * squares + (1 - _GRADIENT_FADING) * gradient**2
        else:
            squares = gradient**2
        step = np.where(squares > 0, gradient / np.sqrt(squares), 0.0)
        log_ratios = np.clip(
            log_ratios + self.ratio_rate * step,
            math.log(_RATIO_FLOOR),
            math.log(_RATIO_CEILING),
        )
        return log_ratios, squares

    def _issued(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        variance: float,
        noise: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and predictive standard deviation of every lead."""
        leads = self._leads
        spreads = ((leads @ covariance) * leads).sum(axis=1) + self._paths @ noise + 1
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


def _gradient(
    tangents: tuple[np.ndarray, np.ndarray],
    z: np.ndarray,
    error: float,
    f: float,
    variance: float,
) -> np.ndarray:
    """Gradient of one value's Gaussian log-likelihood in the log ratios."""
    d_state, d_covariance = tangents
    d_error = -(d_state @ z)
    d_f = (d_covariance @ z) @ z
    standardized = error * error / (variance * f)
    gradient: np.ndarray = -0.5 * (
        d_f / f * (1 - standardized) + 2 * error * d_error / (variance * f)
    )
    return gradient


def _filtered(
    tangents: tuple[np.ndarray, np.ndarray],
    z: np.ndarray,
    error: float,
    m: np.ndarray,
    f: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of the filtered state and covariance after one value."""
    d_state, d_covariance = tangents
    d_m = d_covariance @ z
    d_f = d_m @ z
    d_error = -(d_state @ z)
    gain = m / f
    d_gain = d_m / f - np.outer(d_f, m) / f**2
    d_state = d_state + d_gain * error + np.outer(d_error, gain)
    outer = np.einsum("pi,j->pij", d_m, m)
    d_covariance = (
        d_covariance
        - (outer + outer.transpose(0, 2, 1)) / f
        + np.einsum("p,i,j->pij", d_f / f**2, m, m)
    )
    return d_state, d_covariance
