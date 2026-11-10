"""Base classes for online forecasters and regressors."""

from __future__ import annotations

import abc

from aberrant.base.forecast import Forecast
from aberrant.base.protocols import FeatureMap
from aberrant.utils.validation import coerce_finite_number, coerce_integer_count


def check_spread_parameters(
    residual_alpha: object, min_scale: object
) -> tuple[float, float]:
    """Validate the faded-error weight and the spread floor."""
    alpha = coerce_finite_number(residual_alpha, label="residual_alpha")
    if not 0 < alpha <= 1:
        raise ValueError("residual_alpha must be in (0, 1]")
    floor = coerce_finite_number(min_scale, label="min_scale")
    if floor <= 0:
        raise ValueError("min_scale must be positive")
    return alpha, floor


class BaseForecaster(abc.ABC):
    """An online forecaster of one series sampled at a fixed cadence.

    ``forecast_one()`` forecasts leads ``1..horizon`` from the values learned
    so far, and ``learn_one(y)`` learns the next value. Each learned value
    advances the series by one step: there are no timestamps, so resample
    irregular or missing readings before learning them.
    """

    def __init__(self, horizon: int) -> None:
        self.horizon = coerce_integer_count(horizon, label="horizon")

    @property
    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Whether ``forecast_one`` returns a forecast."""

    @abc.abstractmethod
    def learn_one(self, y: float) -> None:
        """Learn the next value of the series."""

    @abc.abstractmethod
    def forecast_one(self) -> Forecast | None:
        """Forecast leads ``1..horizon``, or ``None`` before readiness."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Forget everything learned."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(horizon={self.horizon})"


class BaseRegressor(abc.ABC):
    """An online regressor of one target from the covariates of each event.

    ``predict_one(x)`` predicts the target of the event with covariates ``x``
    at lead 0, and ``learn_one(x, y)`` learns that event's target once known.
    """

    @property
    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Whether ``predict_one`` returns a prediction."""

    @abc.abstractmethod
    def learn_one(self, x: FeatureMap, y: float) -> None:
        """Learn the target ``y`` of the event with covariates ``x``."""

    @abc.abstractmethod
    def predict_one(self, x: FeatureMap) -> Forecast | None:
        """Predict the target at lead 0, or ``None`` before readiness."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Forget everything learned."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
