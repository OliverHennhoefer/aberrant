"""Prequential order, policy, retention, catalog and metric contracts."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from aberrant.base import MissingOptionalDependencyError
from aberrant.catalog import ComponentConfig, DetectorConfig
from aberrant.evaluate import EvaluationRecord, EvaluationResult, PrequentialEvaluator
from aberrant.model import NullModel
from aberrant.transform import StandardScaler


class RecordingModel:
    def __init__(self):
        self.calls = []
        self.values = []

    def score_one(self, x):
        self.calls.append(("score", x["x"]))
        return float(len(self.values))

    def learn_one(self, x):
        self.calls.append(("learn", x["x"]))
        self.values.append(x["x"])


class ValueModel(RecordingModel):
    def score_one(self, x):
        super().score_one(x)
        return x["x"]


def test_score_before_learn_and_warmup_boundary():
    model = RecordingModel()
    evaluator = PrequentialEvaluator(model, warmup=2, measure_time=False)
    records = list(
        evaluator.iter_evaluate([({"x": float(i)}, i % 2) for i in range(4)])
    )
    assert [r.score for r in records] == [None, None, 2.0, 3.0]
    assert [r.exclusion for r in records] == ["warmup", "warmup", None, None]
    assert model.calls == [
        ("learn", 0.0),
        ("learn", 1.0),
        ("score", 2.0),
        ("learn", 2.0),
        ("score", 3.0),
        ("learn", 3.0),
    ]
    result = evaluator.result()
    assert (result.n_seen, result.n_scored, result.n_learned, result.n_warmup) == (
        4,
        2,
        4,
        2,
    )
    assert result.n_labeled == 2 and result.prevalence == 0.5
    assert result.score_time_ns is result.learn_time_ns is None
    assert result.metric_samples == 0


def test_readiness_uses_prior_model_state():
    model = RecordingModel()
    evaluator = PrequentialEvaluator(model, readiness=lambda m: len(m.values) >= 2)
    records = list(evaluator.iter_evaluate([({"x": 1.0}, None)] * 3))
    assert [r.exclusion for r in records] == ["not_ready", "not_ready", None]
    assert records[-1].score == 2
    assert evaluator.result().n_not_ready == 2


def test_learning_filter_counts_only_successful_learns_for_warmup():
    evaluator = PrequentialEvaluator(
        RecordingModel(),
        warmup=2,
        learn_filter=lambda x, _s: x["x"] > 0,
        learn_policy_name="positive_features",
    )
    records = list(
        evaluator.iter_evaluate([({"x": v}, 0) for v in [-1.0, 1.0, 2.0, 3.0]])
    )
    assert [r.score for r in records] == [None, None, None, 2]
    assert [r.learned for r in records] == [False, True, True, True]
    assert evaluator.result().learn_policy_name == "positive_features"


def test_filter_can_skip_scored_events_without_hiding_their_metrics():
    evaluator = PrequentialEvaluator(
        ValueModel(),
        learn_filter=lambda _x, score: score is None or score < 2,
        learn_policy_name="score_under_two",
    )
    record = evaluator.update({"x": 3.0}, 1)
    assert record.score == 3 and not record.learned
    assert record.learn_time_ns is None
    assert (
        evaluator.result().n_scored,
        evaluator.result().n_labeled,
        evaluator.result().n_learned,
    ) == (1, 1, 0)


def test_pipeline_preserves_post_update_learning_and_prior_state_scoring():
    model = ValueModel()
    pipeline = StandardScaler() | model
    evaluator = PrequentialEvaluator(pipeline, warmup=2)
    evaluator.update({"x": 0.0})
    evaluator.update({"x": 2.0})
    record = evaluator.update({"x": 3.0})
    assert record.score == 2.0
    assert model.values[:2] == [0.0, 1.0]
    assert model.values[2] == pytest.approx((3 - 5 / 3) / np.sqrt(14 / 9))


def test_lazy_streaming_and_continuation():
    evaluator = PrequentialEvaluator(RecordingModel(), measure_time=False)
    stream = evaluator.iter_evaluate([({"x": 1.0}, 0), ({"x": 2.0}, 1)])
    assert evaluator.result().n_seen == 0
    assert next(stream).index == 0
    assert evaluator.result().n_seen == 1
    assert next(stream).index == 1
    result = evaluator.evaluate([({"x": 3.0}, None)])
    assert (result.n_seen, result.n_scored, result.n_learned) == (3, 3, 3)


def test_separate_call_timings(monkeypatch):
    times = iter([10, 20, 30, 60])
    monkeypatch.setattr(
        "aberrant.evaluate._prequential.perf_counter_ns", lambda: next(times)
    )
    evaluator = PrequentialEvaluator(NullModel())
    record = evaluator.update({"x": 1.0})
    assert (record.score_time_ns, record.learn_time_ns) == (10, 30)
    assert (evaluator.result().score_time_ns, evaluator.result().learn_time_ns) == (
        10,
        30,
    )


def test_records_and_results_are_detached_and_immutable():
    evaluator = PrequentialEvaluator(NullModel())
    record = evaluator.update({"x": 1.0}, 1)
    result = evaluator.result()
    assert isinstance(record, EvaluationRecord) and isinstance(result, EvaluationResult)
    with pytest.raises(FrozenInstanceError):
        record.score = 3
    with pytest.raises(FrozenInstanceError):
        result.n_seen = 10
    evaluator.update({"x": 2.0})
    assert result.n_seen == 1 and evaluator.result().n_seen == 2


@pytest.mark.parametrize(
    "label",
    [False, True, 0, 1, 0.0, 1.0, np.int64(1), np.float32(0), np.bool_(True), None],
)
def test_binary_labels(label):
    record = PrequentialEvaluator(NullModel()).update({"x": 1.0}, label)
    assert record.label == (None if label is None else int(label))


@pytest.mark.parametrize(
    "label", [2, -1, 0.5, float("nan"), float("inf"), "1", object()]
)
def test_invalid_labels_fail_before_model_calls(label):
    model = RecordingModel()
    evaluator = PrequentialEvaluator(model)
    with pytest.raises(ValueError):
        evaluator.update({"x": 1.0}, label)
    assert not model.calls and evaluator.result().n_seen == 0


@pytest.mark.parametrize(
    "event",
    [{}, {"": 1.0}, {1: 1.0}, {"x": float("nan")}, {"x": float("inf")}, {"x": "1"}],
)
def test_invalid_events_fail_before_model_calls(event):
    model = RecordingModel()
    evaluator = PrequentialEvaluator(model)
    with pytest.raises(ValueError):
        evaluator.update(event)
    assert not model.calls and evaluator.result().n_seen == 0


def test_input_is_isolated_from_callback_and_model_mutation():
    class MutatingModel(ValueModel):
        def score_one(self, x):
            value = super().score_one(x)
            x.clear()
            return value

    def policy(x, score):
        assert x == {"x": 3.0} and score == 3
        x.clear()
        return True

    model = MutatingModel()
    evaluator = PrequentialEvaluator(
        model, learn_filter=policy, learn_policy_name="mutating_policy"
    )
    event = {"x": 3.0}
    evaluator.update(event)
    assert event == {"x": 3.0} and model.values == [3.0]


def test_invalid_score_never_learns_or_advances_statistics():
    evaluator = PrequentialEvaluator(ValueModel())
    evaluator.model.score_one = lambda x: float("nan")
    with pytest.raises(ValueError, match="Anomaly score"):
        evaluator.update({"x": 1.0})
    assert evaluator.result().n_seen == 0 and evaluator.model.values == []


def test_learning_failure_does_not_advance_evaluator():
    class RejectingModel(ValueModel):
        def learn_one(self, x):
            if x["x"] < 0:
                raise ValueError("rejected")
            super().learn_one(x)

    evaluator = PrequentialEvaluator(RejectingModel())
    with pytest.raises(ValueError, match="rejected"):
        evaluator.update({"x": -1.0}, 1)
    assert evaluator.result().n_seen == evaluator.result().n_scored == 0
    assert evaluator.update({"x": 1.0}, 0).index == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup": -1},
        {"warmup": True},
        {"warmup": 1.5},
        {"metric_window_size": 0},
        {"metric_window_size": True},
        {"metric_window_size": -1},
        {"metrics": 1},
        {"measure_time": "yes"},
        {"higher_is_more_anomalous": 1},
        {"readiness": False},
        {"learn_filter": False},
        {"learn_filter": lambda x, s: True},
        {"learn_filter": lambda x, s: True, "learn_policy_name": " "},
        {"learn_policy_name": "orphan"},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        PrequentialEvaluator(NullModel(), **kwargs)


def test_reject_nonmodel():
    with pytest.raises(TypeError, match="ModelProtocol"):
        PrequentialEvaluator(object())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"readiness": lambda m: 1},
        {"learn_filter": lambda x, s: 1, "learn_policy_name": "bad"},
    ],
)
def test_callback_result_requires_boolean(kwargs):
    evaluator = PrequentialEvaluator(NullModel(), **kwargs)
    with pytest.raises(ValueError, match="boolean"):
        evaluator.update({"x": 1.0})
    assert evaluator.result().n_seen == 0


def test_config_constructs_fresh_model_and_resolves_metadata():
    config = DetectorConfig(
        ComponentConfig(
            "online_isolation_forest", {"window_size": 16, "num_trees": 2, "seed": 19}
        )
    )
    first = PrequentialEvaluator(config)
    second = PrequentialEvaluator(config)
    assert first.model is not second.model
    assert first.result().warmup == config.capabilities().warmup.minimum
    assert first.result().config_fingerprint == config.fingerprint()
    assert first.result().config.as_dict()["model"]["params"]["seed"] == 19


@pytest.mark.parametrize(
    "config",
    [
        DetectorConfig(ComponentConfig("sdo_stream")),
        DetectorConfig(ComponentConfig("null"), (ComponentConfig("standard_scaler"),)),
    ],
)
def test_non_event_and_pipeline_readiness_is_explicit(config):
    with pytest.raises(ValueError, match="warmup or readiness"):
        PrequentialEvaluator(config)
    assert (
        PrequentialEvaluator(config, warmup=10, higher_is_more_anomalous=True)
        .result()
        .warmup
        == 10
    )
    assert (
        PrequentialEvaluator(
            config, readiness=lambda _m: False, higher_is_more_anomalous=True
        )
        .result()
        .uses_readiness_callback
    )


def test_unknown_catalog_orientation_is_explicit():
    config = DetectorConfig(
        ComponentConfig("moving_average", {"window_size": 10, "abs_diff": False})
    )
    with pytest.raises(ValueError, match="orientation"):
        PrequentialEvaluator(config)
    assert (
        PrequentialEvaluator(config, higher_is_more_anomalous=False)
        .result()
        .higher_is_more_anomalous
        is False
    )


def test_missing_optional_dependency_is_actionable_and_disabled_mode_works(monkeypatch):
    def missing(_name):
        raise ModuleNotFoundError("sklearn")

    monkeypatch.setattr("aberrant.evaluate._metrics.import_module", missing)
    with pytest.raises(MissingOptionalDependencyError, match="eval"):
        PrequentialEvaluator(NullModel(), metrics=True)
    result = PrequentialEvaluator(NullModel()).evaluate([({"x": 1.0}, 1)])
    assert result.average_precision is result.roc_auc is None


def test_window_metrics_match_reference_and_retain_only_labeled_scored_events():
    metrics = pytest.importorskip("sklearn.metrics")
    evaluator = PrequentialEvaluator(
        ValueModel(), warmup=1, metrics=True, metric_window_size=3
    )
    records = list(
        evaluator.iter_evaluate(
            [
                ({"x": 100.0}, 1),
                ({"x": 3.0}, 1),
                ({"x": 0.0}, 0),
                ({"x": 1.0}, 0),
                ({"x": 2.0}, 1),
                ({"x": 999.0}, None),
            ]
        )
    )
    result = evaluator.result()
    assert records[0].score is None
    assert result.metric_samples == 3 and result.n_labeled == 4
    assert result.prevalence == 0.5 and result.metric_prevalence == pytest.approx(1 / 3)
    assert result.average_precision == pytest.approx(
        metrics.average_precision_score([0, 0, 1], [0, 1, 2])
    )
    assert result.roc_auc == 1.0


def test_full_stream_metrics_and_orientation():
    metrics = pytest.importorskip("sklearn.metrics")
    scores, labels = [3.0, -2.0, 0.0, -1.0], [0, 1, 0, 1]
    evaluator = PrequentialEvaluator(
        ValueModel(),
        metrics=True,
        metric_window_size=None,
        higher_is_more_anomalous=False,
    )
    records = list(
        evaluator.iter_evaluate(
            [({"x": x}, y) for x, y in zip(scores, labels, strict=True)]
        )
    )
    result = evaluator.result()
    assert [r.score for r in records] == scores
    assert result.metric_samples == 4 and result.metric_window_size is None
    assert result.average_precision == pytest.approx(
        metrics.average_precision_score(labels, [-x for x in scores])
    )
    assert result.roc_auc == 1.0


@pytest.mark.parametrize("labels,ap", [([], None), ([0, 0], None), ([1, 1], 1.0)])
def test_degenerate_metrics(labels, ap):
    pytest.importorskip("sklearn.metrics")
    evaluator = PrequentialEvaluator(ValueModel(), metrics=True)
    result = evaluator.evaluate([({"x": float(i)}, y) for i, y in enumerate(labels)])
    assert result.average_precision == ap and result.roc_auc is None


def test_bounded_metric_retention_over_long_stream():
    pytest.importorskip("sklearn.metrics")
    evaluator = PrequentialEvaluator(NullModel(), metrics=True, metric_window_size=7)
    result = evaluator.evaluate(({"x": float(i)}, i % 2) for i in range(10000))
    assert result.n_seen == 10000 and result.metric_samples == 7
