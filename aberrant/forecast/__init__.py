"""Online forecasters and regressors for forecast-based anomaly detection.

A forecaster learns one series with ``learn_one(y)`` and forecasts its next
values with ``forecast_one()``. A regressor learns a target from the covariates
of each event with ``learn_one(x, y)`` and predicts it with ``predict_one(x)``.
Both issue a :class:`~aberrant.forecast.Forecast` with a spread per lead.

:class:`ResidualDetector` turns either into an anomaly detector, and
:func:`nonconformity` turns a forecast and the value that arrived into a score
for :mod:`aberrant.calibration`.
"""

from aberrant.base.forecast import Forecast
from aberrant.forecast.autoregressive import AutoRegressive
from aberrant.forecast.base import BaseForecaster, BaseRegressor
from aberrant.forecast.echo_state import EchoStateForecaster
from aberrant.forecast.ensemble import EnsembleForecaster
from aberrant.forecast.holt_winters import HoltWinters
from aberrant.forecast.kalman import KalmanForecaster
from aberrant.forecast.lagged import LagForecaster
from aberrant.forecast.linear import RLSRegressor
from aberrant.forecast.naive import SeasonalNaive
from aberrant.forecast.regressors import (
    KNNRegressor,
    PassiveAggressiveRegressor,
    RandomFourierRegressor,
    SGDRegressor,
)
from aberrant.forecast.residual import ResidualDetector
from aberrant.forecast.scores import NonconformityScore, interval, nonconformity
from aberrant.forecast.trees import (
    AdaptiveRandomForestRegressor,
    HoeffdingTreeRegressor,
)

__all__ = [
    "AdaptiveRandomForestRegressor",
    "AutoRegressive",
    "BaseForecaster",
    "BaseRegressor",
    "EchoStateForecaster",
    "EnsembleForecaster",
    "Forecast",
    "HoeffdingTreeRegressor",
    "HoltWinters",
    "KNNRegressor",
    "KalmanForecaster",
    "LagForecaster",
    "NonconformityScore",
    "PassiveAggressiveRegressor",
    "RLSRegressor",
    "RandomFourierRegressor",
    "ResidualDetector",
    "SGDRegressor",
    "SeasonalNaive",
    "interval",
    "nonconformity",
]
