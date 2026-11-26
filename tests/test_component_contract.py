"""Every catalog component meets the shared parameter, clone and reset contract."""

import inspect
import pickle
from collections.abc import Callable

import pytest

from aberrant.base import ReadinessUnit
from aberrant.calibration import ScoreCalibrator
from aberrant.catalog import create_model, create_transformer, model_capabilities
from aberrant.compose import GraphNode, GraphPipeline, MergeNode
from aberrant.drift import ADWIN, CUSUM, KSWIN, PageHinkley
from aberrant.forecast import (
    AutoRegressive,
    EchoStateForecaster,
    EnsembleForecaster,
    HoltWinters,
    KalmanForecaster,
    LagForecaster,
    RLSRegressor,
    SeasonalNaive,
)
from aberrant.model.forest import HalfSpaceTrees
from aberrant.model.stat import MovingCovariance
from aberrant.transform import MinMaxScaler, Select, StandardScaler
from tests.catalog_cases import (
    MODEL_IDS,
    TRANSFORMER_IDS,
    TRANSFORMER_PARAMS,
    event_factory,
    model_params,
    quick_params,
)

# The random baseline draws a new score on every call by design.
_STATEFUL_SCORES = {"random"}


def _seeded(factory: Callable[..., object], params: dict[str, object]):
    built = factory(params)
    if "seed" in inspect.signature(type(built)).parameters and "seed" not in params:
        return factory({**params, "seed": 7})
    return built


def _model(component_id: str):
    return _seeded(lambda p: create_model(component_id, p), model_params(component_id))


def _transformer(component_id: str):
    params = TRANSFORMER_PARAMS.get(component_id, {})
    return _seeded(lambda p: create_transformer(component_id, p), params)


def _events(component_id: str, count: int = 120) -> list[dict[str, object]]:
    make_event = event_factory(component_id)
    return [make_event(index) for index in range(count)]


def _scores(model, events) -> list[float]:
    scores = []
    for event in events:
        scores.append(model.score_one(event))
        model.learn_one(event)
    return scores


def _outputs(transformer, events) -> list[dict[str, float]]:
    outputs = []
    for event in events:
        transformer.learn_one(event)
        outputs.append(transformer.transform_one(event))
    return outputs


def _signature_names(component: object) -> set[str]:
    return set(inspect.signature(type(component)).parameters)


@pytest.mark.parametrize("component_id", MODEL_IDS)
def test_model_parameters_rebuild_the_model(component_id):
    model = _model(component_id)
    params = model.get_params()
    assert set(params) == _signature_names(model)
    rebuilt = type(model)(**params)
    events = _events(component_id)
    assert _scores(rebuilt, events) == _scores(_model(component_id), events)


@pytest.mark.parametrize("component_id", MODEL_IDS)
def test_model_scores_zero_until_ready_and_judges_soon_after(component_id):
    model = create_model(component_id, quick_params(component_id))
    ready_at = first_score = None
    for index, event in enumerate(_events(component_id, 200)):
        ready = model.readiness.ready
        score = model.score_one(event)
        if not ready:
            assert score == 0.0, f"scored {score} while not ready at event {index}"
        elif ready_at is None:
            ready_at = index
        if score != 0.0 and first_score is None:
            first_score = index
        model.learn_one(event)
    if first_score is not None:
        # A model that forgot to declare its warm-up would sit at zero long
        # after claiming to be ready.
        assert ready_at is not None
        assert first_score - ready_at <= 5


@pytest.mark.parametrize("component_id", MODEL_IDS)
def test_catalog_warmup_is_the_learned_events_readiness_needs(component_id):
    params = quick_params(component_id)
    model = create_model(component_id, params)
    capabilities = model_capabilities(component_id, params)
    assert capabilities.warm_up == model.readiness.required
    assert capabilities.warm_up_unit is model.readiness.unit
    if capabilities.warm_up_unit is not ReadinessUnit.EVENTS:
        pytest.skip("only an event count maps onto learned events")
    assert capabilities.warm_up is not None
    for index, event in enumerate(_events(component_id, capabilities.warm_up)):
        assert not model.readiness.ready, f"ready after {index} events"
        model.learn_one(event)
    assert model.readiness.ready


@pytest.mark.parametrize("component_id", MODEL_IDS)
def test_model_clone_is_untrained_and_independent(component_id):
    events = _events(component_id)
    trained = _model(component_id)
    _scores(trained, events[:60])
    probe = events[60:]
    before = [trained.score_one(event) for event in probe]

    clone = trained.clone()
    assert _scores(clone, events) == _scores(_model(component_id), events)
    if component_id not in _STATEFUL_SCORES:
        assert [trained.score_one(event) for event in probe] == before


@pytest.mark.parametrize("component_id", MODEL_IDS)
def test_model_reset_matches_a_fresh_model(component_id):
    events = _events(component_id)
    model = _model(component_id)
    _scores(model, events)
    model.reset()
    assert _scores(model, events) == _scores(_model(component_id), events)


@pytest.mark.parametrize("component_id", MODEL_IDS)
def test_model_pickle_keeps_learned_state(component_id):
    events = _events(component_id)
    model = _model(component_id)
    _scores(model, events[:60])
    restored = pickle.loads(pickle.dumps(model))
    assert _scores(restored, events[60:]) == _scores(model, events[60:])


@pytest.mark.parametrize("component_id", MODEL_IDS)
def test_model_batches_match_single_events(component_id):
    events = _events(component_id)
    by_event = _model(component_id)
    by_batch = _model(component_id)
    for event in events[:60]:
        by_event.learn_one(event)
    by_batch.learn_many(events[:60])
    probe = events[60:]
    assert by_batch.score_many(probe) == [by_event.score_one(e) for e in probe]


@pytest.mark.parametrize("component_id", MODEL_IDS)
def test_model_repr_names_the_class(component_id):
    model = _model(component_id)
    assert repr(model).startswith(f"{type(model).__name__}(")
    assert str(model) == repr(model)


@pytest.mark.parametrize("component_id", TRANSFORMER_IDS)
def test_transformer_contract(component_id):
    events = _events(component_id)
    transformer = _transformer(component_id)
    assert set(transformer.get_params()) == _signature_names(transformer)
    expected = _outputs(_transformer(component_id), events)

    assert _outputs(type(transformer)(**transformer.get_params()), events) == expected
    _outputs(transformer, events)
    assert _outputs(transformer.clone(), events) == expected
    restored = pickle.loads(pickle.dumps(transformer))
    assert _outputs(restored, events) == _outputs(transformer, events)
    transformer.reset()
    assert _outputs(transformer, events) == expected
    assert repr(transformer).startswith(f"{type(transformer).__name__}(")


_SERIES = [float(index % 7) + 0.1 * (index % 3) for index in range(80)]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SeasonalNaive(season_length=4),
        lambda: HoltWinters(season_length=4),
        lambda: KalmanForecaster(horizon=3, slope_ratio=0.01),
        lambda: AutoRegressive(horizon=3, season_length=7, seasonal_order=1),
        lambda: LagForecaster(
            RLSRegressor(), horizon=2, order=2, season_length=7, harmonics=1
        ),
        lambda: EnsembleForecaster(
            [SeasonalNaive(season_length=4, horizon=2), KalmanForecaster(horizon=2)]
        ),
        lambda: EchoStateForecaster(horizon=2, n_reservoir=20, warm_up=8, seed=0),
    ],
)
def test_forecaster_contract(factory):
    def forecasts(forecaster):
        out = []
        for value in _SERIES:
            out.append(forecaster.forecast_one())
            forecaster.learn_one(value)
        return out

    forecaster = factory()
    expected = forecasts(factory())
    assert set(forecaster.get_params()) == _signature_names(forecaster)
    assert forecasts(type(forecaster)(**forecaster.get_params())) == expected
    forecasts(forecaster)
    assert forecasts(forecaster.clone()) == expected
    forecaster.reset()
    assert forecasts(forecaster) == expected


def test_regressor_contract():
    def predictions(regressor):
        out = []
        for index, value in enumerate(_SERIES):
            x = {"a": float(index % 5)}
            out.append(regressor.predict_one(x))
            regressor.learn_one(x, value)
        return out

    regressor = RLSRegressor()
    expected = predictions(RLSRegressor())
    predictions(regressor)
    assert predictions(regressor.clone()) == expected
    regressor.reset()
    assert predictions(regressor) == expected


@pytest.mark.parametrize("detector", [ADWIN, CUSUM, KSWIN, PageHinkley])
def test_drift_detector_contract(detector):
    signal = [0.0] * 200 + [5.0] * 200
    built = detector()
    fresh = detector()
    expected = [fresh.update(value) for value in signal]

    for value in signal:
        built.update(value)
    clone = built.clone()
    assert [clone.update(value) for value in signal] == expected
    built.reset()
    assert [built.update(value) for value in signal] == expected
    assert set(built.get_params()) == _signature_names(built)


_PAIRS = [{"x": float(index % 9), "y": float(index % 4)} for index in range(120)]


def _linear():
    return StandardScaler() | MinMaxScaler() | HalfSpaceTrees(seed=7)


def _graph(observer=None):
    return GraphPipeline(
        [
            GraphNode("x", Select(features=["x"])),
            GraphNode("y", Select(features=["y"])),
            GraphNode("scaled", StandardScaler(), input="x"),
            MergeNode("features", ("scaled", "y")),
        ],
        MovingCovariance(window_size=16),
        output="features",
        observer=observer,
    )


@pytest.mark.parametrize("factory", [_linear, _graph], ids=["linear", "graph"])
def test_pipeline_contract(factory):
    expected = _scores(factory(), _PAIRS)
    pipeline = factory()
    _scores(pipeline, _PAIRS[:60])
    probe = _PAIRS[60:]
    before = [pipeline.score_one(event) for event in probe]

    clone = pipeline.clone()
    assert not clone.readiness.ready
    assert _scores(clone, _PAIRS) == expected
    assert [pipeline.score_one(event) for event in probe] == before
    restored = pickle.loads(pickle.dumps(pipeline))
    assert _scores(restored, probe) == _scores(pipeline, probe)
    pipeline.reset()
    assert _scores(pipeline, _PAIRS) == expected
    assert repr(pipeline) == repr(factory())
    assert str(pipeline) != ""


def test_linear_pipeline_parameters_and_batches():
    pipeline = _linear()
    rebuilt = type(pipeline)(**pipeline.get_params())
    single = _linear()
    for event in _PAIRS[:60]:
        single.learn_one(event)
    rebuilt.learn_many(_PAIRS[:60])

    assert rebuilt.score_many(_PAIRS[60:]) == [
        single.score_one(event) for event in _PAIRS[60:]
    ]


def test_graph_pipeline_parameters_rebuild_it_and_share_the_observer():
    seen = []
    graph = _graph(seen.append)

    assert set(graph.get_params()) == _signature_names(graph)
    assert repr(graph).startswith("GraphPipeline(nodes=(GraphNode(id='x', ")
    clone = graph.clone()
    assert clone.observer is graph.observer
    clone.learn_one(_PAIRS[0])
    assert seen
    assert pickle.loads(pickle.dumps(graph)).observer is None


def test_calibrator_contract():
    scores = [float(index % 11) / 10.0 for index in range(80)]

    def thresholds(calibrator):
        out = []
        for key, score in enumerate(scores):
            out.append(calibrator.predict(key))
            calibrator.update(key, score)
        return out

    def build():
        return ScoreCalibrator("aci", alpha=0.1, window_size=40, gamma=0.02)

    calibrator = build()
    expected = thresholds(build())
    params = calibrator.get_params()
    assert params == {"method": "aci", "alpha": 0.1, "window_size": 40, "gamma": 0.02}
    assert thresholds(ScoreCalibrator(params.pop("method"), **params)) == expected
    thresholds(calibrator)
    assert thresholds(calibrator.clone()) == expected
    calibrator.reset()
    assert thresholds(calibrator) == expected
    assert repr(calibrator) == (
        "ScoreCalibrator('aci', alpha=0.1, window_size=40, gamma=0.02)"
    )
    assert (
        repr(ScoreCalibrator("aci", alpha=0.1)) == "ScoreCalibrator('aci', alpha=0.1)"
    )
