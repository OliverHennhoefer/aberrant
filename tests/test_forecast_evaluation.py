"""ForecastEvaluator: forecast before learning, and score every lead."""

import math
import pickle
from dataclasses import FrozenInstanceError

import pytest

from aberrant.base import Readiness
from aberrant.base.forecast import Forecast
from aberrant.evaluate import ForecastEvaluator, ForecastResult, LeadScores
from aberrant.forecast import AutoRegressive, SeasonalNaive


class LastValue:
    """Forecasts the last learned value at every lead, with spread 1."""

    def __init__(self, horizon: int = 2) -> None:
        self.horizon = horizon
        self.calls: list[tuple[str, float | None]] = []
        self.values: list[float | None] = []

    @property
    def readiness(self) -> Readiness:
        return Readiness(bool(self.values))

    def learn_one(self, y):
        self.calls.append(("learn", y))
        self.values.append(y)

    def forecast_one(self):
        self.calls.append(("forecast", None))
        known = [value for value in self.values if value is not None]
        if not known:
            return None
        return Forecast(
            origin=len(self.values),
            leads=tuple(range(1, self.horizon + 1)),
            mean=(known[-1],) * self.horizon,
            scale=(1.0,) * self.horizon,
        )


def test_forecasts_before_learning_and_scores_each_lead_once() -> None:
    model = LastValue(horizon=2)
    evaluator = ForecastEvaluator(model, measure_time=False)
    records = list(evaluator.iter_evaluate([1.0, 3.0, 4.0, 8.0]))
    assert model.calls[:4] == [
        ("forecast", None),
        ("learn", 1.0),
        ("forecast", None),
        ("learn", 3.0),
    ]
    assert [record.errors for record in records] == [
        (None, None),
        (2.0, None),
        (1.0, 3.0),
        (4.0, 5.0),
    ]
    result = evaluator.result()
    assert result.lead(1) == LeadScores(
        lead=1,
        n_scored=3,
        mae=7 / 3,
        rmse=math.sqrt(21 / 3),
        bias=7 / 3,
        mase=(7 / 3) / (7 / 3),
        spread_ratio=7 / 3,
    )
    assert result.lead(2).n_scored == 2
    assert result.lead(2).mae == pytest.approx(4.0)
    assert (result.n_seen, result.n_observed, result.n_forecasts) == (4, 4, 3)


def test_warmup_skips_forecasts_issued_before_it() -> None:
    evaluator = ForecastEvaluator(LastValue(horizon=1), warm_up=2, measure_time=False)
    records = list(evaluator.iter_evaluate([1.0, 3.0, 4.0, 8.0]))
    assert [record.forecast is not None for record in records] == [
        False,
        False,
        True,
        True,
    ]
    assert evaluator.result().lead(1).mae == pytest.approx(2.5)


def test_a_missing_value_is_learned_but_never_scored() -> None:
    model = LastValue(horizon=2)
    evaluator = ForecastEvaluator(model, measure_time=False)
    records = list(evaluator.iter_evaluate([1.0, None, 4.0]))
    assert model.values == [1.0, None, 4.0]
    assert records[1].errors == (None, None)
    assert records[2].errors == (3.0, 3.0)
    result = evaluator.result()
    assert (result.n_seen, result.n_observed) == (3, 2)


def test_mase_scales_by_the_seasonal_naive_error() -> None:
    values = [float(t % 4) for t in range(400)]
    exact = ForecastEvaluator(
        SeasonalNaive(season_length=4, horizon=4), season_length=4
    )
    assert exact.evaluate(values).lead(4).mae == pytest.approx(0.0)
    assert exact.result().lead(4).mase is None
    last = ForecastEvaluator(LastValue(horizon=1), season_length=1)
    assert last.evaluate(values).lead(1).mase == pytest.approx(1.0)


def test_matches_a_direct_loop_on_a_real_forecaster() -> None:
    values = [math.sin(t / 3) + 0.01 * ((t * 7919) % 11) for t in range(600)]
    result = ForecastEvaluator(
        AutoRegressive(order=2, horizon=3), warm_up=100
    ).evaluate(values)
    model = AutoRegressive(order=2, horizon=3)
    errors: list[float] = []
    for index, y in enumerate(values):
        forecast = model.forecast_one()
        if index >= 100 and forecast is not None:
            target = index + 2
            if target < len(values):
                errors.append(abs(values[target] - forecast.point(3)))
        model.learn_one(y)
    assert result.lead(3).n_scored == len(errors)
    assert result.lead(3).mae == pytest.approx(sum(errors) / len(errors))


def test_unscored_leads_and_spreadless_forecasts_report_none() -> None:
    class NoSpread(LastValue):
        def forecast_one(self):
            forecast = super().forecast_one()
            if forecast is None:
                return None
            return Forecast(forecast.origin, forecast.leads, forecast.mean)

    empty = ForecastEvaluator(LastValue(horizon=2)).result()
    assert empty.lead(1) == LeadScores(1, 0, None, None, None, None, None)
    result = ForecastEvaluator(NoSpread(horizon=1)).evaluate([1.0, 2.0, 4.0])
    assert result.lead(1).spread_ratio is None
    assert result.lead(1).mae == pytest.approx(1.5)


def test_a_failed_learn_leaves_the_evaluator_unchanged() -> None:
    class Failing(LastValue):
        def learn_one(self, y):
            if y == 99.0:
                raise RuntimeError("refused")
            super().learn_one(y)

    evaluator = ForecastEvaluator(Failing(horizon=2), measure_time=False)
    evaluator.evaluate([1.0, 2.0, 3.0])
    before = pickle.dumps(evaluator.result())
    with pytest.raises(RuntimeError):
        evaluator.update(99.0)
    assert pickle.dumps(evaluator.result()) == before
    record = evaluator.update(5.0)
    assert record.index == 3
    assert record.errors == (2.0, 3.0)


def test_results_are_frozen_and_timed() -> None:
    result = ForecastEvaluator(LastValue()).evaluate([1.0, 2.0, 3.0])
    assert isinstance(result, ForecastResult)
    assert result.forecast_time_ns is not None and result.learn_time_ns is not None
    with pytest.raises(FrozenInstanceError):
        result.n_seen = 0  # type: ignore[misc]
    for lead in (0, 3, True, 1.0):
        with pytest.raises(KeyError):
            result.lead(lead)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "params",
    [
        {"warm_up": -1},
        {"warm_up": True},
        {"season_length": 0},
        {"season_length": 1.0},
        {"measure_time": 1},
    ],
)
def test_rejects_invalid_configuration(params) -> None:
    with pytest.raises(ValueError):
        ForecastEvaluator(LastValue(), **params)


def test_rejects_non_forecasters_and_bad_values() -> None:
    with pytest.raises(TypeError):
        ForecastEvaluator(object())  # type: ignore[arg-type]
    evaluator = ForecastEvaluator(LastValue())
    for value in (math.nan, math.inf, "1"):
        with pytest.raises(ValueError):
            evaluator.update(value)  # type: ignore[arg-type]
    assert evaluator.result().n_seen == 0


def test_rejects_a_forecast_of_the_wrong_type() -> None:
    class Wrong(LastValue):
        def forecast_one(self):
            return (1.0,)

    evaluator = ForecastEvaluator(Wrong())
    with pytest.raises(TypeError):
        evaluator.update(1.0)
