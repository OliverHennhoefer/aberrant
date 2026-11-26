# Forecasting API

Forecasters learn one series with `learn_one(y)` and forecast it with
`forecast_one()`. Regressors learn a target from covariates with
`learn_one(x, y)` and predict it with `predict_one(x)`. Both issue a
[`Forecast`](#aberrant.forecast.Forecast).

::: aberrant.forecast.Forecast

::: aberrant.forecast.SeasonalNaive

::: aberrant.forecast.HoltWinters

::: aberrant.forecast.KalmanForecaster

::: aberrant.forecast.AutoRegressive

::: aberrant.forecast.EnsembleForecaster

::: aberrant.forecast.LagForecaster

::: aberrant.forecast.RLSRegressor

::: aberrant.forecast.RandomFourierRegressor

::: aberrant.forecast.PassiveAggressiveRegressor

::: aberrant.forecast.SGDRegressor

::: aberrant.forecast.KNNRegressor

::: aberrant.forecast.HoeffdingTreeRegressor

::: aberrant.forecast.AdaptiveRandomForestRegressor

::: aberrant.forecast.ResidualDetector
    options:
      inherited_members: true
      show_bases: false

::: aberrant.forecast.nonconformity

::: aberrant.forecast.NonconformityScore

::: aberrant.forecast.interval

::: aberrant.forecast.BaseForecaster

::: aberrant.forecast.BaseRegressor
