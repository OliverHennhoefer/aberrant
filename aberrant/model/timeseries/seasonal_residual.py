"""Causal anomaly scores from additive Holt-Winters forecast residuals."""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import mean

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema, coerce_finite_number


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise OverflowError("Seasonal forecast arithmetic exceeds the float range")
    return value


def _sum(values: Iterable[float]) -> float:
    return _finite(math.fsum(_finite(value) for value in values))


def _blend(previous: float, observed: float, weight: float) -> float:
    return _sum(((1.0 - weight) * previous, weight * observed))


class SeasonalResidualDetector(BaseModel):
    """Score departures from a seasonal one-step-ahead forecast.

    The model learns additive Holt-Winters level, trend, and seasonal components
    for one finite numeric feature sampled at a fixed cadence. Every successful
    learning call advances one seasonal position. There is no timestamp or gap
    inference: preprocess missing observations before using this detector.

    Args:
        season_length: Known number of observations in a cycle, at least two.
        key: Fixed feature name, or infer it on the first successful learning.
        alpha: Level smoothing weight in [0, 1].
        beta: Trend smoothing weight in [0, 1] (the usual beta-star parameter).
        gamma: Seasonal smoothing weight in [0, 1-alpha], using prior level
            and trend in the seasonal update.
        residual_alpha: Absolute forecast-error smoothing weight in (0, 1].
        min_scale: Positive, finite floor for the residual scale, in input units.
        normalize: Divide absolute residual by the previously learned residual
            scale when true; otherwise return absolute residual in input units.

    Notes:
        Exactly two cycles initialize the model. Their mean difference gives
        the per-observation trend. Detrended observations at each seasonal
        position are averaged to initialize seasonality, and the second cycle
        mean is advanced to its final position to initialize level. The initial
        residual scale is the mean absolute error of that fitted warm-up model;
        subsequent errors are genuinely one-step-ahead and exponentially faded.
        Warm-up scoring returns zero with no forecast or residual.

        Scores use only previously learned state and are non-negative, without
        a probability interpretation or fixed upper bound. Learning every event
        adapts both the forecast and scale to anomalies. Skipping an update also
        skips its seasonal position; this API requires every time step to be
        learned. State is O(season_length), with O(1) work per event after the
        O(season_length) initialization. Unrepresentable arithmetic raises
        OverflowError without changing learned state.

    References:
        Hyndman and Athanasopoulos, Forecasting: Principles and Practice, 3rd ed.,
        additive Holt-Winters component equations:
        https://otexts.com/fpp3/holt-winters.html
    """

    def __init__(
        self,
        season_length: int,
        *,
        key: str | None = None,
        alpha: float = 0.2,
        beta: float = 0.05,
        gamma: float = 0.1,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
        normalize: bool = True,
    ) -> None:
        if isinstance(season_length, bool) or not isinstance(season_length, int):
            raise ValueError("season_length must be an integer of at least two")
        if season_length < 2:
            raise ValueError("season_length must be an integer of at least two")
        self.season_length = season_length
        self.key = key
        self.alpha = coerce_finite_number(alpha, label="alpha")
        self.beta = coerce_finite_number(beta, label="beta")
        self.gamma = coerce_finite_number(gamma, label="gamma")
        self.residual_alpha = coerce_finite_number(
            residual_alpha, label="residual_alpha"
        )
        self.min_scale = coerce_finite_number(min_scale, label="min_scale")
        if not 0 <= self.alpha <= 1 or not 0 <= self.beta <= 1:
            raise ValueError("alpha and beta must be in [0, 1]")
        if not 0 <= self.gamma <= 1 - self.alpha:
            raise ValueError("gamma must be in [0, 1-alpha]")
        if not 0 < self.residual_alpha <= 1:
            raise ValueError("residual_alpha must be in (0, 1]")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive")
        if not isinstance(normalize, bool):
            raise ValueError("normalize must be a boolean")
        self.normalize = normalize
        self.reset()

    @property
    def is_ready(self) -> bool:
        """Whether two complete cycles have been learned."""
        return self.n_samples_seen >= 2 * self.season_length

    @property
    def residual_scale(self) -> float:
        """Previously learned mean absolute error, including the configured floor."""
        return max(self.min_scale, self._residual_scale)

    def reset(self) -> None:
        """Clear learned components, readiness, and any inferred feature name."""
        self._schema = FeatureSchema(
            names=None if self.key is None else [self.key], expected_size=1
        )
        self.n_samples_seen = 0
        self._warmup: list[float] = []
        self._seasonal: list[float] = []
        self._level = 0.0
        self._trend = 0.0
        self._residual_scale = 0.0

    def _initialize(
        self, values: list[float]
    ) -> tuple[float, float, list[float], float]:
        m = self.season_length
        first_mean = mean(values[:m])
        second_mean = mean(values[m:])
        trend = _sum((second_mean / m, -first_mean / m))
        midpoint = (m - 1) / 2
        seasonal = [
            _sum(
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
        level = _sum((second_mean, trend * midpoint))
        errors = (
            abs(
                _sum(
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
        return level, trend, seasonal, _sum(errors)

    def explain_one(
        self, x: dict[str, float]
    ) -> tuple[float, float | None, float | None]:
        """Return (score, forecast, signed residual) without learning.

        Before readiness return (0.0, None, None). A positive residual means the
        candidate exceeds its prior forecast; a negative one means it falls
        below. Repeated calls neither advance seasonal phase nor lock a name.
        """
        prepared = self._schema.preview(x)
        if not self.is_ready:
            return 0.0, None, None
        base = _sum((self._level, self._trend))
        forecast = _sum(
            (base, self._seasonal[self.n_samples_seen % self.season_length])
        )
        residual = _sum((float(prepared.values[0]), -forecast))
        score = abs(residual)
        if self.normalize:
            score = _finite(score / self.residual_scale)
        return score, forecast, residual

    def score_one(self, x: dict[str, float]) -> float:
        """Return the absolute, optionally scaled prior forecast residual."""
        return self.explain_one(x)[0]

    def learn_one(self, x: dict[str, float]) -> None:
        """Learn one fixed-cadence observation, independently of scoring calls."""
        prepared = self._schema.preview(x)
        value = float(prepared.values[0])
        if not self.is_ready:
            if self.n_samples_seen + 1 == 2 * self.season_length:
                level, trend, seasonal, scale = self._initialize([*self._warmup, value])
                self._level, self._trend = level, trend
                self._seasonal, self._residual_scale = seasonal, scale
                self._warmup = []
            else:
                self._warmup.append(value)
        else:
            phase = self.n_samples_seen % self.season_length
            base = _sum((self._level, self._trend))
            forecast = _sum((base, self._seasonal[phase]))
            residual = _sum((value, -forecast))
            # Residual corrections avoid overflowing unweighted intermediates.
            level_correction = self.alpha * residual
            level = _sum((base, level_correction))
            trend = _sum((self._trend, self.beta * level_correction))
            seasonal_value = _sum((self._seasonal[phase], self.gamma * residual))
            scale = _blend(self._residual_scale, abs(residual), self.residual_alpha)
            # Publish only after all arithmetic succeeds. No per-event season copy.
            self._level, self._trend = level, trend
            self._seasonal[phase] = seasonal_value
            self._residual_scale = scale
        self.n_samples_seen += 1
        self._schema.commit(prepared)

    def __repr__(self) -> str:
        return f"SeasonalResidualDetector(season_length={self.season_length}, normalize={self.normalize})"
