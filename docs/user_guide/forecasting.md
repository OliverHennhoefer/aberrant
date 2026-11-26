# Forecast-Based Detection

A detector scores how unusual an event looks. Forecast-based detection asks a
narrower question instead: how far does the target miss what was expected
for it? `aberrant.forecast` holds the online models that produce those
expectations, and the pieces that turn their misses into anomaly scores.

## Forecasters and regressors

Both learn one numeric target online and predict it before it is known. They
differ in what they predict from, which shows in how they are called:

| | Forecaster | Regressor |
| --- | --- | --- |
| Predicts from | The target's own past | The covariates of the same event |
| Learn | `learn_one(y)` | `learn_one(x, y)` |
| Predict | `forecast_one()` | `predict_one(x)` |
| Leads | `1..horizon` | `0` |
| Included | `SeasonalNaive`, `HoltWinters`, `KalmanForecaster`, `AutoRegressive`, `EnsembleForecaster`, `LagForecaster`, `EchoStateForecaster` | `RLSRegressor`, `RandomFourierRegressor`, `PassiveAggressiveRegressor`, `SGDRegressor`, `KNNRegressor`, `HoeffdingTreeRegressor`, `AdaptiveRandomForestRegressor` |

The target is its own argument, never a key of `x`, so a regressor cannot see
the value it predicts and transformers in front of it never touch the target.
A regressor's target may arrive later than its covariates: predict when the
covariates arrive and call `learn_one(x, y)` once `y` is known.

Every model in this family predicts first and learns afterwards, and reports
a spread with every forecast. The spread is what lets a residual be compared
across leads and series, and calibrated.

## The Forecast value

`forecast_one()` and `predict_one(x)` return a `Forecast`, or `None` until
the model is ready:

```python
from aberrant.forecast import HoltWinters

model = HoltWinters(season_length=24, horizon=6)
for y in series:
    forecast = model.forecast_one()  # issued before y arrives
    model.learn_one(y)

forecast.origin  # events learned when it was issued
forecast.leads  # (1, 2, 3, 4, 5, 6)
forecast.mean  # point forecast per lead
forecast.scale  # spread per lead, in target units
```

Lead `k` forecasts the value learned `k` events after the origin. A
`Forecast` may also carry quantiles on any grid of levels
(`levels`, `quantiles`), which `quantile(lead, level)` interpolates.

Forecasters count events instead of reading timestamps: every learned value
advances the series by one step. Resample irregular readings before learning
them, and learn `None` for a step whose value is missing. Every forecaster
then advances the series without a correction. Its next forecast is the one it
would have issued before the gap, so a lead-1 forecast after two missing
values is really a lead-3 forecast, and reports lead 3's spread (beyond the
horizon, at least the longest lead's). Errors are measured only against
forecasts issued right after an observed value, so a gap never teaches a
lead an error that belongs to a longer one.

## Spreads

`SeasonalNaive`, `HoltWinters` and `AutoRegressive` report each lead's faded
mean absolute error as its spread. `KalmanForecaster` reports the standard deviation of its
predictive distribution, from a structural model of a level, optionally a
slope (`slope_ratio`) and a season (`season_length`), whose noise variance it
estimates online. That spread
grows with the lead as the model says it should, and on Gaussian noise
`|y - mean| / spread` stays within 1.645 for 90% of values at every lead.
`level_ratio`, `slope_ratio` and `season_ratio` set how quickly each
component follows the series. The filter learns them online by recursive
maximum likelihood, starting from the values given, so a series that wanders
faster or slower than they say is still forecast well; `learn_ratios=False`
keeps them fixed, and the fixed default level ratio matches exponential
smoothing with weight 0.2.

## Choosing a forecaster

`HoltWinters` and `KalmanForecaster` with `season_length` suit a series whose
season repeats with noise around it. The Kalman filter's spreads are
predictive standard deviations that grow with the lead as its model says, and
`harmonics` keeps a long cycle, such as a day of minutes, to a few states.
Without a season, `KalmanForecaster` suits a level, or a trend, that wanders.
`AutoRegressive` suits a series whose values depend on the recent past and on
the same phase of earlier cycles, which is common in telemetry from
controlled processes: it fits a seasonal autoregression online, differences a
series that wanders, and leaves less correlation in its errors than the
smoothing methods do. Start with `order=2`, and add `season_length` with
`seasonal_order=1` for a seasonal series.

When no single choice is clear, or the series changes character over time,
`EnsembleForecaster` runs several forecasters side by side and learns, for
every lead, how much to trust each one. Its weights follow the member that
does best lately, and it often beats every member, since a weighted mean of
different models cancels part of their errors. Its spread is its own faded
mean absolute error.

`LagForecaster` turns any regressor into a forecaster: it regresses each
value on the values before it, the same phase of earlier cycles and the
phase itself, and forecasts longer leads by feeding its predictions back.
With `RLSRegressor` it is the autoregression above; with a nonlinear
regressor it follows dynamics no linear model can. `RandomFourierRegressor`
is kernel ridge regression online and fits smooth bends; `KNNRegressor`
forecasts the future of the most similar recent situations;
`SGDRegressor` uses the Huber loss, so an anomaly in the target barely moves
it; and `PassiveAggressiveRegressor` is the cheapest linear learner.
`HoeffdingTreeRegressor` splits the covariates into regions once the
evidence for a split suffices and fits a line in each, so over lags it is a
threshold autoregression that finds its own regimes.
`AdaptiveRandomForestRegressor` bags such trees and replaces a tree whose
errors drift, so it recovers after the dynamics change. All of them
standardize their covariates and target over their first `warm_up` events.

`EchoStateForecaster` needs no choice of lags: a fixed random recurrent
reservoir keeps a fading nonlinear memory of the series, and only a linear
readout from it learns, by recursive least squares. A small `input_scale`
keeps the reservoir nearly linear and its memory long, which suits seasonal
series; a larger one makes it more nonlinear.

## How they compare

`benchmarks/compare_forecasters.py` runs every forecaster on eight kinds of
synthetic series, one value at a time, and scores the forecasts after the
first 1,000 of 4,000 values. The series are a stationary autoregression, a
seasonal autoregression, a random walk, a noisy trend with a fixed season, a
series wandering in both level and season, a threshold autoregression
(SETAR), a series that switches from an autoregression to a noisy season
halfway, and a noisy logistic map. Regressors run inside `LagForecaster`
with three lags, and every season is 12 values long. The tables are the mean
over three series of each kind, from
`uv run python benchmarks/compare_forecasters.py --seeds 3 --markdown`.

The relative error is the geometric mean, over the series, of a
forecaster's lead-1 mean absolute error divided by the best forecaster's on
that series, so 1.00 would be best everywhere. "Near best" counts the series
where it is within 2% of the best.

| Forecaster | Relative MAE@1 | Near best (of 8) | µs per event |
| --- | ---: | ---: | ---: |
| `Ensemble(mixed)` | 1.335 | 7 | 1134 |
| `Ensemble(classic)` | 1.576 | 6 | 516 |
| `AR(p=2, P=1)` | 1.636 | 4 | 83 |
| `Lag(RandomFourier)` | 1.653 | 1 | 270 |
| `EchoState(D=1)` | 1.817 | 1 | 223 |
| `Kalman(trend, season)` | 1.821 | 1 | 148 |
| `Lag(AdaptiveForest)` | 1.862 | 2 | 1265 |
| `Lag(KNN)` | 1.991 | 0 | 338 |
| `AR(p=2, P=1, d=1, D=1)` | 2.028 | 1 | 93 |
| `EchoState` | 2.058 | 2 | 208 |
| `Kalman(season)` | 2.201 | 1 | 144 |
| `HoltWinters` | 2.269 | 0 | 23 |
| `Lag(HoeffdingTree)` | 2.489 | 3 | 213 |
| `Lag(RLS)` | 2.547 | 2 | 145 |
| `AR(p=2)` | 2.616 | 2 | 77 |
| `Lag(PassiveAggressive)` | 2.752 | 1 | 123 |
| `Kalman(trend)` | 2.947 | 0 | 148 |
| `Kalman(level)` | 3.016 | 1 | 132 |
| `Kalman(season, fixed)` | 3.075 | 0 | 75 |
| `Lag(SGD Huber)` | 3.159 | 1 | 134 |
| `Kalman(level, fixed)` | 3.710 | 0 | 63 |
| `SeasonalNaive` | 4.146 | 0 | 16 |

Where the model class contains the process, the best lead-1 error is about
0.80, the mean absolute value of the standard normal innovations:

| Series | Best MAE@1 | Best forecaster | Best MAE@6 |
| --- | ---: | --- | ---: |
| ar2 | 0.805 | `AR(p=2)` | 0.944 |
| seasonal_ar | 0.806 | `AR(p=2, P=1)` | 0.924 |
| random_walk | 0.805 | `Lag(HoeffdingTree)` | 1.956 |
| trend_season | 0.834 | `Kalman(season)` | 0.842 |
| wandering_season | 0.806 | `AR(p=2, P=1, d=1, D=1)` | 1.960 |
| setar | 0.841 | `Lag(AdaptiveForest)` | 1.029 |
| regime_switch | 0.855 | `Ensemble(mixed)` | 0.913 |
| logistic | 0.004 | `Lag(RandomFourier)` | 0.051 |

Three patterns stand out. A model whose structure matches the series wins
it, often by a wide margin: the autoregressions on autoregressive series,
`KalmanForecaster` on a noisy season, the nonlinear regressors on SETAR and
the logistic map. `EnsembleForecaster` over models of different kinds is
within 2% of the best on seven of the eight kinds and best after the
regime switch, at the sum of its members' cost. And learning the variance
ratios makes `KalmanForecaster` markedly better than its fixed ratios on
every series that moves faster or slower than they assume. When the series
is unknown, start with an ensemble of a seasonal `AutoRegressive`, a
seasonal `KalmanForecaster` and one nonlinear member.

## One value, many forecasts

A forecaster with horizon `H` forecasts every value `H` times: at lead 1 one
event before it arrives, at lead 2 two events before, and so on. Errors grow
with the lead, so each lead is its own score stream with its own spread and
its own calibrator. Never pool leads.

Leads see anomalies differently. A forecaster learns every event, so after
a level shift its lead-1 forecast catches up within a few events. A lead-`k`
forecast was issued `k` events before the shift and keeps flagging it for up
to `k` events. Short leads suit spikes; longer leads suit shifts and drifts.

## Scores from residuals

`ResidualDetector` makes any forecaster or regressor an anomaly detector
with the usual `score_one(x)` and `learn_one(x)`. It takes the target from
each event by name:

```python
from aberrant.forecast import HoltWinters, ResidualDetector, RLSRegressor

seasonal = ResidualDetector(
    HoltWinters(season_length=24, horizon=3), target="load", lead=3
)
virtual_sensor = ResidualDetector(RLSRegressor(), target="output")

for x in stream:  # {"output": 4.2, "input_a": 1.1, "input_b": 0.3}
    score = virtual_sensor.score_one(x)
    virtual_sensor.learn_one(x)
```

For a forecaster, the score compares the target with the forecast issued
`lead` events earlier, and other keys are ignored. For a regressor, every
key except the target is a covariate. `score="scaled"` (the default) divides
the absolute residual by the spread at that lead; `score="absolute"` keeps
target units. The detector works anywhere a model does: in pipelines, graph
pipelines and `PrequentialEvaluator`.

`forecast_for(x)` returns the forecast an event is scored against, or `None`
before one exists. It does not read the target, so the expected value and its
band are known before the value arrives:

```python
forecast = seasonal.forecast_for({})
if forecast is not None:
    expected = forecast.point(seasonal.lead)
    spread = forecast.spread(seasonal.lead)
```


### Own past and covariates together

A forecaster learns only its target, and a regressor only the covariates of
the same event. To model both, the target's own past and other features, put
`Lags` in front of a regressor: the lagged target becomes a covariate like
any other. With a linear regressor this is an autoregression with exogenous
inputs (ARX), and with a tree or neighbours regressor a nonlinear one:

```python
from aberrant.forecast import ResidualDetector, RLSRegressor
from aberrant.transform import Lags

detector = Lags(lags=[1, 2], apply_to=["load"]) | ResidualDetector(
    RLSRegressor(), target="load"
)
for x in stream:  # {"load": 41.0, "temperature": 18.5}
    score = detector.score_one(x)
    detector.learn_one(x)
```

`Difference` in front of the lags models changes instead of levels. This
scores each event against a prediction from its own covariates (lead 0);
forecasts several leads ahead use only the series' own past.

## Calibrating forecast scores

`nonconformity(forecast, lead, y, score)` scores one value against one
forecast, and `interval(forecast, lead, threshold, score)` gives the values a
threshold accepts:

| Score | Value | Needs |
| --- | --- | --- |
| `absolute` | `abs(y - mean)` | A point forecast |
| `scaled` | `abs(y - mean) / spread` | A spread |
| `cqr` | `max(lo - y, y - hi)` at levels `alpha / 2` and `1 - alpha / 2` | Quantiles |

A [`ScoreCalibrator`](calibration.md) takes the score with delayed feedback.
Key each threshold by the event it targets:

```python
from aberrant.calibration import ScoreCalibrator
from aberrant.forecast import HoltWinters, interval, nonconformity

lead = 2
model = HoltWinters(season_length=12, horizon=lead)
calibrator = ScoreCalibrator("aci", alpha=0.1, window_size=200)
pending = {}

for step, y in enumerate(series):
    if step in pending:  # the forecast issued `lead` events ago
        forecast, threshold = pending.pop(step)
        score = nonconformity(forecast, lead, y, "scaled")
        alarm = score > threshold
        calibrator.update(step, score)
    model.learn_one(y)
    forecast = model.forecast_one()
    if forecast is not None:
        target = step + lead
        threshold = calibrator.predict(target)
        band = interval(forecast, lead, threshold, "scaled")
        pending[target] = (forecast, threshold)
```

`band` is the calibrated prediction interval for the value at `target`,
known `lead` events in advance.
