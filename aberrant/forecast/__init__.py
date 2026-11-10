"""Online forecasters and regressors for forecast-based anomaly detection.

A forecaster learns one series with ``learn_one(y)`` and forecasts its next
values with ``forecast_one()``. A regressor learns a target from the covariates
of each event with ``learn_one(x, y)`` and predicts it with ``predict_one(x)``.
Both issue a :class:`~aberrant.base.Forecast` with a spread per lead.

:class:`ResidualDetector` turns either into an anomaly detector, and
:func:`nonconformity` turns a forecast and the value that arrived into a score
for :mod:`aberrant.calibration`.
"""

from aberrant.base.forecast import Forecast
from aberrant.forecast.base import BaseForecaster, BaseRegressor
from aberrant.forecast.holt_winters import HoltWinters
from aberrant.forecast.linear import RLSRegressor
from aberrant.forecast.naive import SeasonalNaive
from aberrant.forecast.residual import ResidualDetector
from aberrant.forecast.scores import NonconformityScore, interval, nonconformity

__all__ = [
    "BaseForecaster",
    "BaseRegressor",
    "Forecast",
    "HoltWinters",
    "NonconformityScore",
    "RLSRegressor",
    "ResidualDetector",
    "SeasonalNaive",
    "interval",
    "nonconformity",
]
