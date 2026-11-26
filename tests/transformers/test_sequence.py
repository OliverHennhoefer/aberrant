"""Sequence transformers: each feature's output follows its own recent values."""

import math
import pickle
from array import array

import numpy as np
import pytest

from aberrant.base import Readiness
from aberrant.base.pipeline import Pipeline
from aberrant.exceptions import ValidationError
from aberrant.transform import (
    Deseasonalize,
    Despike,
    Difference,
    ExponentialSmoother,
    Lags,
    RollingStatistic,
)


def learned(transformer, values, feature="a"):
    """The post-update output of each value, as a pipeline's learning sees it."""
    outputs = []
    for value in values:
        event = {feature: float(value)}
        transformer.learn_one(event)
        outputs.append(transformer.transform_one(event))
    return outputs


class Recorder:
    """A model that keeps every event it scores or learns."""

    def __init__(self) -> None:
        self.scored: list[dict[str, float]] = []
        self.learned: list[dict[str, float]] = []

    @property
    def readiness(self) -> Readiness:
        return Readiness(True)

    def score_one(self, x):
        self.scored.append(dict(x))
        return 0.0

    def learn_one(self, x):
        self.learned.append(dict(x))


def test_lags_append_earlier_values_and_repeat_the_oldest_at_the_start() -> None:
    outputs = learned(Lags(lags=[2, 1]), [1, 2, 3, 4])
    assert outputs == [
        {"a": 1.0, "a_lag1": 1.0, "a_lag2": 1.0},
        {"a": 2.0, "a_lag1": 1.0, "a_lag2": 1.0},
        {"a": 3.0, "a_lag1": 2.0, "a_lag2": 1.0},
        {"a": 4.0, "a_lag1": 3.0, "a_lag2": 2.0},
    ]


def test_scoring_and_learning_see_the_same_features_in_a_pipeline() -> None:
    model = Recorder()
    pipeline = Pipeline(Lags(lags=[1]), model)
    for value in [1.0, 2.0, 2.0, 5.0]:
        pipeline.score_one({"a": value})
        pipeline.learn_one({"a": value})
    # Scoring waits for readiness; from then on both see the same features.
    assert len(model.scored) == 2
    assert model.scored == model.learned[-2:]
    assert [event["a_lag1"] for event in model.learned] == [1.0, 1.0, 2.0, 2.0]


def test_an_event_equal_to_the_last_one_is_still_the_next_event() -> None:
    lags = Lags()
    lags.learn_one({"a": 1.0})
    lags.learn_one({"a": 2.0})
    assert lags.transform_one({"a": 2.0}) == {"a": 2.0, "a_lag1": 1.0}
    assert lags.transform_one({"a": 2.0}) == {"a": 2.0, "a_lag1": 2.0}


def test_difference_and_percent_change() -> None:
    assert learned(Difference(), [1, 3, 6]) == [{"a": 0.0}, {"a": 2.0}, {"a": 3.0}]
    assert learned(Difference(lag=2), [1, 3, 6, 10]) == [
        {"a": 0.0},
        {"a": 0.0},
        {"a": 5.0},
        {"a": 7.0},
    ]
    assert learned(Difference(kind="percent"), [-4, 2])[1] == {"a": 150.0}


def test_a_percent_change_from_zero_changes_nothing() -> None:
    change = Difference(kind="percent")
    change.learn_one({"a": 0.0})
    before = pickle.dumps(change)
    with pytest.raises(ValidationError, match="from 0"):
        change.learn_one({"a": 1.0})
    assert pickle.dumps(change) == before


def test_exponential_smoother_follows_its_span_and_waits_for_it() -> None:
    smoother = ExponentialSmoother(span=3)
    outputs = learned(smoother, [4, 0, 0, 0])
    assert [row["a"] for row in outputs] == [4.0, 2.0, 1.0, 0.5]
    assert ExponentialSmoother(span=1).transform_one({"a": 7.0}) == {"a": 7.0}
    readiness = ExponentialSmoother(span=3)
    learned(readiness, [1, 1])
    assert not readiness.readiness.ready
    learned(readiness, [1])
    assert readiness.readiness == Readiness(True, 3, 3)


def test_despike_replaces_a_spike_by_the_median_and_keeps_steps() -> None:
    values = [1, 2, 1, 2, 1, 50, 2, 1, 2]
    outputs = [row["a"] for row in learned(Despike(window_size=5), values)]
    assert outputs[5] == pytest.approx(2.0)
    assert outputs[:5] == [1.0, 2.0, 1.0, 2.0, 1.0]
    step = [0.0] * 5 + [10.0] * 5
    stepped = [row["a"] for row in learned(Despike(window_size=5), step)]
    assert stepped[-3:] == [10.0, 10.0, 10.0]
    flat = [row["a"] for row in learned(Despike(window_size=3), [5, 5, 5, 9])]
    assert flat[-1] == 9.0


@pytest.mark.parametrize(
    ("statistic", "expected"),
    [
        ("sum", lambda w: float(np.sum(w))),
        ("mean", lambda w: float(np.mean(w))),
        ("std", lambda w: float(np.std(w, ddof=1)) if len(w) > 1 else 0.0),
        ("min", min),
        ("max", max),
        ("range", lambda w: max(w) - min(w)),
        ("median", lambda w: float(np.median(w))),
        ("quantile", lambda w: float(np.quantile(w, 0.25))),
        ("mad", lambda w: float(np.median(np.abs(np.array(w) - np.median(w))))),
        (
            "zscore",
            lambda w: (
                (w[-1] - np.mean(w)) / np.std(w, ddof=1)
                if len(w) > 1 and np.std(w) > 0
                else 0.0
            ),
        ),
        ("iqr", lambda w: float(np.quantile(w, 0.75) - np.quantile(w, 0.25))),
        (
            "slope",
            lambda w: float(np.polyfit(range(len(w)), w, 1)[0]) if len(w) > 1 else 0.0,
        ),
        ("rms", lambda w: float(np.sqrt(np.mean(np.square(w))))),
        (
            "rank",
            lambda w: (
                (sum(item < w[-1] for item in w) + (w.count(w[-1]) - 1) / 2)
                / (len(w) - 1)
                if len(w) > 1
                else 0.5
            ),
        ),
        (
            "robust_zscore",
            lambda w: (
                (w[-1] - np.median(w))
                / (1.4826 * np.median(np.abs(np.array(w) - np.median(w))))
            ),
        ),
        (
            "skewness",
            lambda w: _moment(w, 3) / _moment(w, 2) ** 1.5 if len(w) > 2 else 0.0,
        ),
        (
            "kurtosis",
            lambda w: _moment(w, 4) / _moment(w, 2) ** 2 - 3 if len(w) > 1 else 0.0,
        ),
    ],
)
def test_rolling_statistics_match_numpy(statistic, expected) -> None:
    values = [3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, 6.0, 6.0]
    rolling = RollingStatistic(window_size=4, statistic=statistic, quantile=0.25)
    outputs = [row["a"] for row in learned(rolling, values)]
    windows = [values[max(0, end - 3) : end + 1] for end in range(len(values))]
    start = 2 if statistic == "robust_zscore" else 0
    assert outputs[start:] == pytest.approx(
        [expected(window) for window in windows[start:]]
    )


def _moment(window, power):
    return float(np.mean((np.array(window) - np.mean(window)) ** power))


def test_window_statistics_stay_exact_for_large_values_that_differ_little() -> None:
    values = [1e9 + 0.1 * k for k in (1, 3, 2, 5)]
    for statistic, expected in [
        ("std", float(np.std(np.array(values) - 1e9, ddof=1))),
        ("slope", float(np.polyfit(range(4), np.array(values) - 1e9, 1)[0])),
    ]:
        rolling = RollingStatistic(window_size=4, statistic=statistic)
        assert learned(rolling, values)[-1]["a"] == pytest.approx(expected, rel=1e-6)


def test_shape_statistics_stay_within_what_a_window_allows() -> None:
    two = RollingStatistic(window_size=2, statistic="kurtosis")
    assert learned(two, [0, 1])[-1] == {"a": -2.0}
    skew = RollingStatistic(window_size=2, statistic="skewness")
    assert learned(skew, [0, 1])[-1] == {"a": 0.0}
    for statistic in ("zscore", "robust_zscore", "skewness", "kurtosis"):
        flat = RollingStatistic(window_size=3, statistic=statistic)
        assert learned(flat, [4, 4, 4])[-1] == {"a": 0.0}


def test_robust_zscore_falls_back_to_the_mean_deviation() -> None:
    rolling = RollingStatistic(window_size=5, statistic="robust_zscore")
    # The MAD of [1, 1, 1, 1, 6] is 0; its mean absolute deviation is 1.
    output = learned(rolling, [1, 1, 1, 1, 6])[-1]["a"]
    assert output == pytest.approx(5 / math.sqrt(math.pi / 2))


@pytest.mark.parametrize(
    "transformer",
    [
        Lags(lags=[1, 3]),
        Difference(lag=2, kind="percent"),
        ExponentialSmoother(span=3),
        Despike(window_size=4),
        RollingStatistic(window_size=3, statistic="mean"),
        RollingStatistic(window_size=3, statistic="rank"),
        Deseasonalize(season_length=3, memory=2),
    ],
    ids=lambda transformer: type(transformer).__name__,
)
def test_steps_on_states_an_application_keeps(transformer) -> None:
    values = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0]
    reference = pickle.loads(pickle.dumps(transformer))
    rows = learned(reference, values)
    state = transformer.initial_state()
    assert transformer.valid_state(state)
    for value, row in zip(values, rows, strict=True):
        before = state
        state, outputs = transformer.step(state, value)
        assert transformer.valid_state(state)
        assert before == pickle.loads(pickle.dumps(before))
        assert outputs == tuple(row[name] for name in transformer._names("a"))
    assert transformer.lead + 1 == reference.readiness.required
    assert transformer.readiness == Readiness(False, 0, transformer.lead + 1)
    invalid_states = (
        None,
        [],
        (math.nan,),
        ("1",),
        (1,),
        (0.0,) * 40,
        (array("d", [math.nan]),),
        (array("d", [0.0] * 40),) * 2,
    )
    for invalid in invalid_states:
        assert not transformer.valid_state(invalid)


def test_step_rejects_results_that_are_not_finite() -> None:
    change = Difference()
    state, _ = change.step(change.initial_state(), -1.7e308)
    with pytest.raises(ValidationError, match="finite"):
        change.step(state, 1.7e308)
    rolling = RollingStatistic(window_size=2, statistic="sum")
    state, _ = rolling.step(rolling.initial_state(), 1.7e308)
    with pytest.raises(ValidationError, match="finite"):
        rolling.step(state, 1.7e308)


def test_rolling_states_must_keep_their_sorted_copy() -> None:
    rolling = RollingStatistic(window_size=3, statistic="median")
    assert rolling.valid_state((array("d", [2.0, 1.0]), array("d", [1.0, 2.0])))
    assert not rolling.valid_state((array("d", [2.0, 1.0]), array("d", [2.0, 1.0])))
    assert not rolling.valid_state((array("d", [2.0, 1.0]),))
    assert not rolling.valid_state(((2.0, 1.0), (1.0, 2.0)))
    assert not rolling.valid_state((array("f", [2.0]), array("f", [2.0])))
    deseason = Deseasonalize(season_length=2)
    assert not deseason.valid_state((1, (0.0, 0.0), (0.0, 0.0), (0, 0)))
    assert not deseason.valid_state((0, (0.0, 0.0), (-1.0, 0.0), (0, 0)))


def test_deseasonalize_removes_a_cycle_and_ignores_one_odd_value() -> None:
    noise = np.random.default_rng(0).normal(0.0, 0.1, 120)
    values = [[0.0, 5.0, -2.0, 1.0][t % 4] + noise[t] for t in range(120)]
    values[70] += 100.0
    outputs = [
        row["a"] for row in learned(Deseasonalize(season_length=4, memory=5), values)
    ]
    assert outputs[:4] == [0.0] * 4
    assert outputs[4:8] == pytest.approx([0.0] * 4, abs=0.5)
    assert outputs[70] == pytest.approx(100.0, abs=1.0)
    assert outputs[74] == pytest.approx(0.0, abs=0.5)
    assert outputs[-4:] == pytest.approx([0.0] * 4, abs=0.5)


def test_selected_features_change_and_the_others_pass() -> None:
    change = Difference(apply_to=["a"])
    change.learn_one({"a": 1.0, "b": 1.0})
    change.learn_one({"a": 3.0, "b": 7.0})
    assert change.transform_one({"a": 3.0, "b": 7.0}) == {"a": 2.0, "b": 7.0}
    with pytest.raises(ValidationError, match="apply_to"):
        change.learn_one({"b": 1.0})


def test_every_feature_keeps_its_own_history() -> None:
    change = Difference()
    change.learn_one({"a": 1.0})
    change.learn_one({"a": 2.0, "b": 10.0})
    assert change.transform_one({"a": 4.0, "b": 13.0}) == {"a": 2.0, "b": 3.0}
    assert change.readiness == Readiness(False, 1, 2)


def test_a_lag_name_that_exists_is_rejected() -> None:
    with pytest.raises(ValidationError, match="already exists"):
        Lags(apply_to=["a"]).learn_one({"a": 1.0, "a_lag1": 2.0})


def test_a_failed_downstream_update_restores_the_history() -> None:
    class Failing(Recorder):
        def learn_one(self, x):
            if x["a"] == 99.0:
                raise RuntimeError("refused")
            super().learn_one(x)

    lags = Lags()
    pipeline = Pipeline(lags, Failing())
    pipeline.learn_one({"a": 1.0})
    before = pickle.dumps(lags)
    with pytest.raises(RuntimeError):
        pipeline.learn_one({"a": 99.0})
    assert pickle.dumps(lags) == before
    pipeline.learn_one({"a": 2.0})
    assert lags.transform_one({"a": 3.0}) == {"a": 3.0, "a_lag1": 2.0}


def test_results_must_be_finite() -> None:
    change = Difference()
    change.learn_one({"a": -1.7e308})
    with pytest.raises(ValidationError, match="finite"):
        change.learn_one({"a": 1.7e308})


@pytest.mark.parametrize(
    ("factory", "params"),
    [
        (Lags, {"lags": []}),
        (Lags, {"lags": [1, 1]}),
        (Lags, {"lags": [0]}),
        (Lags, {"lags": "1"}),
        (Lags, {"apply_to": "a"}),
        (Lags, {"apply_to": []}),
        (Lags, {"apply_to": ["a", "a"]}),
        (Difference, {"lag": 0}),
        (Difference, {"kind": "ratio"}),
        (ExponentialSmoother, {"span": 0}),
        (Despike, {"window_size": 2}),
        (Despike, {"threshold": 0.0}),
        (RollingStatistic, {"window_size": 0}),
        (RollingStatistic, {"statistic": "mode"}),
        (RollingStatistic, {"quantile": 1.5}),
        (Deseasonalize, {"season_length": 1}),
        (Deseasonalize, {"season_length": 4, "memory": 0}),
    ],
)
def test_rejects_invalid_configuration(factory, params) -> None:
    with pytest.raises(ValueError):
        factory(**params)


def test_values_before_any_learning_pass_through_cold() -> None:
    assert Lags(lags=[3]).transform_one({"a": 2.0}) == {"a": 2.0, "a_lag3": 2.0}
    assert Difference().transform_one({"a": 2.0}) == {"a": 0.0}
    assert math.isclose(RollingStatistic().transform_one({"a": 2.0})["a"], 2.0)
