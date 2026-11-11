"""ResidualDetector and the forecast-to-calibrator path with delayed feedback."""

import numpy as np
import pytest

from aberrant.base import BaseModel, Forecast
from aberrant.calibration import ScoreCalibrator
from aberrant.evaluate import PrequentialEvaluator
from aberrant.forecast import (
    HoltWinters,
    ResidualDetector,
    RLSRegressor,
    SeasonalNaive,
    interval,
    nonconformity,
)


class _Recorder:
    """A forecaster whose lead-h forecast at origin o is 100 * o + h."""

    horizon = 3

    def __init__(self) -> None:
        self.n = 0

    def learn_one(self, y: float) -> None:
        self.n += 1

    def forecast_one(self) -> Forecast | None:
        if self.n == 0:
            return None
        leads = (1, 2, 3)
        return Forecast(
            origin=self.n,
            leads=leads,
            mean=tuple(100.0 * self.n + h for h in leads),
            scale=(1.0, 1.0, 1.0),
        )


@pytest.mark.parametrize("lead", [1, 2, 3])
def test_scores_against_the_forecast_issued_lead_events_earlier(lead: int) -> None:
    detector = ResidualDetector(_Recorder(), "y", lead=lead, score="absolute")
    for event in range(1, 8):
        _, expected, _ = detector.explain_one({"y": 0.0})
        origin = event - lead
        assert expected == (None if origin < 1 else 100.0 * origin + lead)
        detector.learn_one({"y": 0.0})


@pytest.mark.parametrize("lead", [1, 3])
def test_forecast_for_gives_the_forecast_an_event_is_scored_against(lead: int) -> None:
    detector = ResidualDetector(_Recorder(), "y", lead=lead, score="scaled")
    for event in range(1, 8):
        # The target is not read: the forecast is known before the value.
        forecast = detector.forecast_for({})
        assert forecast is detector.forecast_for({"y": 5.0})
        score, expected, _ = detector.explain_one({"y": 5.0})
        if event <= lead:
            assert forecast is None and expected is None
        else:
            assert forecast.origin == event - lead
            assert forecast.point(lead) == expected
            assert score == nonconformity(forecast, lead, 5.0, "scaled")
            low, high = interval(forecast, lead, 2.0, "scaled")
            assert (low, high) == (expected - 2.0, expected + 2.0)
        detector.learn_one({"y": 5.0})


def test_forecast_for_a_regressor_is_its_prediction_from_the_covariates() -> None:
    detector = ResidualDetector(RLSRegressor(forgetting=1.0, delta=1e8), "power")
    assert detector.forecast_for({"a": 1.0, "b": 1.0}) is None
    rng = np.random.default_rng(0)
    for _ in range(100):
        a, b = rng.uniform(1, 2, size=2)
        detector.learn_one({"a": float(a), "b": float(b), "power": float(2 * a + b)})
    forecast = detector.forecast_for({"a": 1.0, "b": 1.0})
    assert forecast is not None and forecast.point(0) == pytest.approx(3.0)
    _, expected, _ = detector.explain_one({"a": 1.0, "b": 1.0, "power": 5.0})
    assert forecast.point(0) == expected


def test_regressor_scores_its_own_prediction_from_covariates() -> None:
    detector = ResidualDetector(RLSRegressor(forgetting=1.0, delta=1e8), "power")
    rng = np.random.default_rng(0)
    for _ in range(100):
        a, b = rng.uniform(1, 2, size=2)
        x = {"a": float(a), "b": float(b), "power": float(2 * a + b)}
        detector.learn_one(x)
    score, expected, residual = detector.explain_one({"a": 1.0, "b": 1.0, "power": 5.0})
    assert expected == pytest.approx(3.0)
    assert residual == pytest.approx(2.0)
    assert score > 1e3  # a noise-free fit leaves a tiny spread


def test_construction_checks() -> None:
    assert isinstance(ResidualDetector(SeasonalNaive(), "y"), BaseModel)
    with pytest.raises(ValueError, match="horizon"):
        ResidualDetector(HoltWinters(2, horizon=2), "y", lead=3)
    with pytest.raises(ValueError):
        ResidualDetector(HoltWinters(2), "y", lead=0)
    with pytest.raises(ValueError, match="lead 0"):
        ResidualDetector(RLSRegressor(), "y", lead=1)
    with pytest.raises(ValueError):
        ResidualDetector(SeasonalNaive(), "")
    with pytest.raises(ValueError):
        ResidualDetector(SeasonalNaive(), "y", score="squared")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ResidualDetector(object(), "y")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing the target"):
        ResidualDetector(SeasonalNaive(), "y").score_one({"z": 1.0})


def test_runs_in_prequential_evaluation() -> None:
    detector = ResidualDetector(SeasonalNaive(season_length=4), "y")
    events = [({"y": float(t % 4)}, 0) for t in range(40)] + [({"y": 9.0}, 1)]
    evaluator = PrequentialEvaluator(detector)
    scores = [evaluator.update(x, label).score or 0.0 for x, label in events]
    assert scores[-1] > 10 * max(scores[:-1])
    assert evaluator.result().n_learned == len(events)


def test_lead_two_scores_feed_a_calibrator_with_delayed_feedback() -> None:
    rng = np.random.default_rng(4)
    model = HoltWinters(12, horizon=2)
    calibrator = ScoreCalibrator("aci", alpha=0.1, window=200, gamma=0.005)
    pending: dict[int, tuple[Forecast, float]] = {}
    misses = []
    t = np.arange(6000)
    series = 5 * np.sin(2 * np.pi * t / 12) + rng.standard_t(5, size=t.size)
    for step, y in enumerate(series):
        if step in pending:
            forecast, threshold = pending.pop(step)
            score = nonconformity(forecast, 2, float(y), "scaled")
            misses.append(score > threshold)
            calibrator.update(step, score)
        model.learn_one(float(y))
        forecast = model.forecast_one()
        if forecast is not None:
            target = step + 2  # lead 2 targets the second event from now
            pending[target] = (forecast, calibrator.predict(target))
    assert len(pending) == 2
    assert np.mean(misses[500:]) == pytest.approx(0.1, abs=0.02)
