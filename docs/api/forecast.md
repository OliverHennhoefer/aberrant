# Forecasting API

Forecasters learn one series with `learn_one(y)` and forecast it with
`forecast_one()`. Regressors learn a target from covariates with
`learn_one(x, y)` and predict it with `predict_one(x)`. Both issue a
[`Forecast`](base.md#aberrant.base.Forecast).

::: aberrant.forecast.SeasonalNaive

::: aberrant.forecast.HoltWinters

::: aberrant.forecast.RLSRegressor

::: aberrant.forecast.ResidualDetector
    options:
      inherited_members: true
      show_bases: false

::: aberrant.forecast.nonconformity

::: aberrant.forecast.interval

::: aberrant.forecast.BaseForecaster

::: aberrant.forecast.BaseRegressor
