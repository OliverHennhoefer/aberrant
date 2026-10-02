"""Independent component equations, causal scores, and finite-state boundaries."""

import math
import pickle
import sys

import numpy as np
import pytest

from aberrant.catalog import (
    DetectorConfig,
    EventKind,
    ScoreKind,
    StateKind,
    get_model_spec,
)
from aberrant.model.timeseries import SeasonalResidualDetector
from aberrant.transform import FeatureSchemaGuard


def _learn(model, values, key="value"):
    for value in values:
        model.learn_one({key: float(value)})


@pytest.mark.parametrize(
    "params",
    [{"season_length": value} for value in (True, 1, 0, -1, 2.0, "2", None)]
    + [{"season_length": 2, "key": value} for value in ("", 3)]
    + [
        {"season_length": 2, name: value}
        for name in ("alpha", "beta", "gamma", "residual_alpha", "min_scale")
        for value in (float("nan"), float("inf"), -0.1, "0.5")
    ]
    + [
        {"season_length": 2, "alpha": 1.1},
        {"season_length": 2, "beta": 1.1},
        {"season_length": 2, "alpha": 0.9, "gamma": 0.2},
        {"season_length": 2, "residual_alpha": 0.0},
        {"season_length": 2, "residual_alpha": 1.1},
        {"season_length": 2, "min_scale": 0.0},
        {"season_length": 2, "normalize": 1},
    ],
)
def test_constructor_rejects_invalid_configuration(params):
    with pytest.raises(ValueError):
        SeasonalResidualDetector(**params)


def test_two_cycles_readiness_and_explicit_initialization():
    model = SeasonalResidualDetector(2, normalize=False)
    for value in (9.0, 13.0, 11.0, 15.0):
        assert model.explain_one({"value": value}) == (0.0, None, None)
        model.learn_one({"value": value})
    assert model.is_ready
    assert model.n_samples_seen == 4
    assert model.explain_one({"value": 17.0}) == (4.0, 13.0, 4.0)
    assert model.residual_scale == 1e-6


def test_update_follows_independent_holt_winters_equations():
    model = SeasonalResidualDetector(
        2, alpha=0.25, beta=0.5, gamma=0.125, residual_alpha=0.4, normalize=False
    )
    _learn(model, (9, 13, 11, 15))
    # Independent initialized components: level 13.5, trend 1, seasons [-1.5, 1.5].
    level, trend, seasons, scale = 13.5, 1.0, [-1.5, 1.5], 0.0
    for index, value in enumerate((17.0, 20.0, 18.0, 24.0, 22.0, 30.0)):
        phase = index % 2
        expected = level + trend + seasons[phase]
        error = value - expected
        assert model.explain_one({"value": value}) == pytest.approx(
            (abs(error), expected, error)
        )
        next_level = 0.25 * (value - seasons[phase]) + 0.75 * (level + trend)
        next_trend = 0.5 * (next_level - level) + 0.5 * trend
        next_season = 0.125 * (value - level - trend) + 0.875 * seasons[phase]
        scale = 0.4 * abs(error) + 0.6 * scale
        model.learn_one({"value": value})
        level, trend, seasons[phase] = next_level, next_trend, next_season
        assert model.residual_scale == pytest.approx(scale)


def test_initial_scale_is_mean_absolute_warmup_fit_error():
    model = SeasonalResidualDetector(2, min_scale=0.01)
    _learn(model, (9, 13, 12, 14))
    # Cycle means 11 and 13, trend 1; seasons [-1, 1]. Fit errors [-.5,.5,.5,-.5].
    assert model.residual_scale == pytest.approx(0.5)
    assert model.explain_one({"value": 15.5}) == pytest.approx((4.0, 13.5, 2.0))


def test_normalized_score_uses_prior_scale_and_candidate_never_updates_it():
    model = SeasonalResidualDetector(2, min_scale=2.0, residual_alpha=0.5)
    _learn(model, (9, 13, 11, 15))
    before = pickle.dumps(model)
    assert model.explain_one({"value": 23.0}) == (5.0, 13.0, 10.0)
    assert model.score_one({"value": 3.0}) == 5.0
    assert pickle.dumps(model) == before
    model.learn_one({"value": 23.0})
    assert model.residual_scale == 5.0


@pytest.mark.parametrize("trend", [0.0, 0.5, -0.5])
def test_detrended_initialization_and_continued_seasonal_forecasts(trend):
    model = SeasonalResidualDetector(3, normalize=False)
    seasonal = (-2.0, 3.0, -1.0)
    for step in range(500):
        value = 10 + trend * step + seasonal[step % 3]
        if model.is_ready:
            score, forecast, residual = model.explain_one({"value": value})
            assert forecast == pytest.approx(value, abs=1e-10)
            assert score == pytest.approx(0.0, abs=1e-10)
            assert residual == pytest.approx(0.0, abs=1e-10)
        model.learn_one({"value": value})
    assert len(model._seasonal) == 3
    assert model._warmup == []


def test_constant_series_zero_scale_floor_and_negative_residual():
    model = SeasonalResidualDetector(4, min_scale=0.5)
    _learn(model, [10] * 8)
    assert model.score_one({"value": 10.0}) == 0.0
    assert model.explain_one({"value": 8.0}) == (4.0, 10.0, -2.0)


def test_scoring_does_not_establish_name_and_reset_clears_inferred_schema():
    model = SeasonalResidualDetector(2)
    before = pickle.dumps(model)
    assert model.score_one({"unlearned": 1.0}) == 0.0
    assert pickle.dumps(model) == before
    _learn(model, (1, 2, 1, 2), key="first")
    model.reset()
    assert not model.is_ready and model.n_samples_seen == 0
    assert model.residual_scale == model.min_scale
    _learn(model, (5, 6, 5, 6), key="second")
    assert model.explain_one({"second": 5.0}) == (0.0, 5.0, 0.0)


def test_reset_keeps_explicit_name_and_pickle_continues_identically():
    model = SeasonalResidualDetector(2, key="sensor")
    _learn(model, (1, 2, 1, 2), key="sensor")
    restored = pickle.loads(pickle.dumps(model))
    for value in (3, 5, 1, 4):
        event = {"sensor": float(value)}
        assert restored.explain_one(event) == model.explain_one(event)
        restored.learn_one(event)
        model.learn_one(event)
    model.reset()
    with pytest.raises(ValueError, match="feature keys"):
        model.learn_one({"other": 1.0})


@pytest.mark.parametrize("warmup", [0, 1, 4])
@pytest.mark.parametrize("method", ["score_one", "explain_one", "learn_one"])
@pytest.mark.parametrize(
    "event",
    [
        {},
        {"value": math.nan},
        {"value": math.inf},
        {"value": "bad"},
        {"value": 1.0, "extra": 2.0},
        {"other": 1.0},
        {1: 1.0},
    ],
)
def test_invalid_inputs_never_change_state(warmup, method, event):
    model = SeasonalResidualDetector(2, key="value")
    _learn(model, [1.0] * warmup)
    before = pickle.dumps(model)
    with pytest.raises(ValueError):
        getattr(model, method)(event)
    assert pickle.dumps(model) == before


def test_initialization_overflow_rolls_back_candidate_and_readiness():
    model = SeasonalResidualDetector(2)
    limit = sys.float_info.max
    _learn(model, (limit, limit, -limit))
    before = pickle.dumps(model)
    with pytest.raises(OverflowError):
        model.learn_one({"value": -limit})
    assert pickle.dumps(model) == before
    assert not model.is_ready


@pytest.mark.parametrize("season_length", [3, 12])
@pytest.mark.parametrize(
    "value",
    [sys.float_info.max, -sys.float_info.max, math.ulp(0.0), -math.ulp(0.0)],
)
def test_constant_float_extremes_initialize_and_continue(season_length, value):
    model = SeasonalResidualDetector(season_length, normalize=False)
    _learn(model, [value] * (2 * season_length))
    assert model.is_ready
    for _ in range(season_length):
        assert model.explain_one({"value": value}) == (0.0, value, 0.0)
        model.learn_one({"value": value})
    assert model.n_samples_seen == 3 * season_length
    assert model.residual_scale == model.min_scale


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("alpha", [0.0, 0.2])
def test_representable_update_avoids_unweighted_overflow(sign, alpha):
    model = SeasonalResidualDetector(2, alpha=alpha, normalize=False)
    _learn(model, [0.0, sign * 1e308, 0.0, sign * 1e308])
    event = {"value": sign * 1.5e308}
    assert model.explain_one(event) == (1.5e308, 0.0, event["value"])
    model.learn_one(event)
    # Weighted components fit in a float even though value minus season does not.
    expected_level = sign * (5e307 + alpha * 1.5e308)
    expected_trend = sign * (0.05 * (alpha * 1.5e308))
    expected_season = sign * -3.5e307
    expected_forecast = expected_level + expected_trend + sign * 5e307
    assert model.explain_one({"value": expected_forecast}) == pytest.approx(
        (0.0, expected_forecast, 0.0), abs=1e293
    )
    assert model.residual_scale == pytest.approx(7.5e306)
    model.learn_one({"value": expected_forecast})
    following_forecast = expected_level + 2 * expected_trend + expected_season
    assert model.explain_one({"value": following_forecast}) == pytest.approx(
        (0.0, following_forecast, 0.0), abs=1e293
    )
    assert model.n_samples_seen == 6


def test_unrepresentable_level_update_is_atomic():
    model = SeasonalResidualDetector(2, alpha=1.0, gamma=0.0, normalize=False)
    _learn(model, [0.0, 1e308, 0.0, 1e308])
    before = pickle.dumps(model)
    with pytest.raises(OverflowError):
        model.learn_one({"value": 1.5e308})
    assert pickle.dumps(model) == before


@pytest.mark.parametrize("method", ["score_one", "explain_one", "learn_one"])
def test_ready_residual_overflow_is_atomic(method):
    model = SeasonalResidualDetector(2)
    _learn(model, [sys.float_info.max] * 4)
    before = pickle.dumps(model)
    with pytest.raises(OverflowError):
        getattr(model, method)({"value": -sys.float_info.max})
    assert pickle.dumps(model) == before


def test_normalized_score_overflow_leaves_model_learnable():
    model = SeasonalResidualDetector(2, min_scale=1e-300)
    _learn(model, [0] * 4)
    before = pickle.dumps(model)
    with pytest.raises(OverflowError):
        model.score_one({"value": 1e100})
    assert pickle.dumps(model) == before
    model.learn_one({"value": 1e100})
    assert model.n_samples_seen == 5


def test_catalog_contract_declarative_build_and_pipeline_scoring():
    spec = get_model_spec("seasonal_residual_detector")
    assert spec.parameter_schema()["required"] == ["season_length"]
    caps = spec.capabilities({"season_length": 7})
    assert caps.event_kind == EventKind.UNIVARIATE
    assert caps.score_kind == ScoreKind.NON_NEGATIVE
    assert caps.state == StateKind.BOUNDED and caps.resettable
    assert caps.warmup.minimum == 14
    model = DetectorConfig.from_mapping(
        {"model": {"id": "seasonal_residual_detector", "params": {"season_length": 2}}}
    ).build()
    pipeline = FeatureSchemaGuard(features=["value"]) | model
    _learn(pipeline, (1, 2, 1, 2))
    assert pipeline.score_one({"value": 1.0}) == 0.0


def test_seasonal_deviation_exceeds_seeded_normal_errors():
    model = SeasonalResidualDetector(12, min_scale=0.05)
    rng = np.random.default_rng(42)
    scores = []
    for step in range(120):
        value = 10 + 2 * math.sin(2 * math.pi * step / 12) + rng.normal(0, 0.05)
        if model.is_ready:
            scores.append(model.score_one({"value": value}))
        model.learn_one({"value": value})
    assert model.score_one({"value": 6.0}) > 5 * max(scores)
