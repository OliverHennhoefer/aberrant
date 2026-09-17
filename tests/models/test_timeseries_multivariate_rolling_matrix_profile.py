"""Joint streaming scores checked against independent pairwise distances."""

import math
import pickle

import numpy as np
import pytest

from aberrant.model.timeseries import (
    MultivariateRollingMatrixProfile,
    RollingMatrixProfile,
)


def _event(values):
    return {f"c{index}": float(value) for index, value in enumerate(values)}


def _distance(left, right, normalize):
    if not normalize:
        return math.dist(left, right)
    constant_left = bool(np.all(left == left[0]))
    constant_right = bool(np.all(right == right[0]))
    if constant_left or constant_right:
        return 0.0 if constant_left and constant_right else math.sqrt(len(left))
    normalized = []
    for values in (left, right):
        centered = np.asarray(values, dtype=np.longdouble) - values[0]
        centered /= np.max(np.abs(centered))
        centered -= centered.mean()
        normalized.append(centered / np.sqrt(np.mean(centered**2)))
    return float(np.sqrt(np.sum((normalized[0] - normalized[1]) ** 2)))


def _oracle(model, history, candidate):
    m = model.subsequence_length
    if len(history) < m + model.exclusion_zone:
        return {}
    series = np.asarray([*history, candidate], dtype=np.float64)
    first = max(0, len(series) - model.window_size)
    query_start = len(series) - m
    return {
        index: {
            f"c{channel}": _distance(
                series[index : index + m, channel],
                series[query_start:, channel],
                model.normalize,
            )
            for channel in range(series.shape[1])
        }
        for index in range(first, query_start - model.exclusion_zone)
    }


def _assert_matches_oracle(model, history, candidate):
    distances = _oracle(model, history, candidate)
    result = model.explain_one(_event(candidate))
    if not distances:
        assert result == (0.0, None, {})
        return
    score, start, channels = result
    tolerance = 1e-12 if model.normalize else 0.0
    expected = min(max(values.values()) for values in distances.values())
    assert score == pytest.approx(expected, rel=1e-10, abs=tolerance)
    assert start in distances
    assert max(distances[start].values()) == pytest.approx(
        expected, rel=1e-10, abs=tolerance
    )
    assert channels == pytest.approx(distances[start], rel=1e-10, abs=tolerance)
    assert score == max(channels.values())


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("channels", [1, 2, 5])
@pytest.mark.parametrize(
    "m,window,zone", [(2, 4, 0), (4, 6, 1), (5, 17, 4), (7, 23, 0)]
)
def test_stream_matches_direct_joint_oracle(normalize, channels, m, window, zone):
    model = MultivariateRollingMatrixProfile(
        m, window, normalize=normalize, exclusion_zone=zone
    )
    history = []
    for values in np.random.default_rng(42).normal(size=(120, channels)):
        _assert_matches_oracle(model, history, values)
        model.learn_one(_event(values))
        history.append(values)


def test_joint_winner_need_not_be_any_channels_nearest_match():
    series = np.array(
        [[0, 10], [0, 10], [5, 5], [5, 5], [10, 0], [10, 0], [0, 0], [0, 0]]
    )
    model = MultivariateRollingMatrixProfile(2, 8, normalize=False, exclusion_zone=1)
    for values in series[:-1]:
        model.learn_one(_event(values))
    score, start, channels = model.explain_one(_event(series[-1]))
    assert start == 2
    assert score == math.sqrt(50)
    assert channels == {"c0": score, "c1": score}
    distances = _oracle(model, list(series[:-1]), series[-1])
    assert min(distances, key=lambda index: distances[index]["c0"]) == 0
    assert min(distances, key=lambda index: distances[index]["c1"]) == 4


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "offsets,scales",
    [
        ([0, 0, 0], [1e-300, 1, 1e300]),
        ([1e12, -1e14, 1], [1, 1, 1e-14]),
        ([0, 0, 0], [1e-300, 1e-300, 1e-300]),
        ([0, 0, 0], [1e300, 1e300, 1e300]),
        ([0, 0, 0], [1e-320, 1e-320, 1e-320]),
    ],
)
def test_channel_scales_offsets_and_tiny_variation(normalize, offsets, scales):
    model = MultivariateRollingMatrixProfile(8, 31, normalize=normalize)
    history = []
    series = np.array(offsets) + np.array(scales) * np.random.default_rng(18).normal(
        size=(90, 3)
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for values in series:
            _assert_matches_oracle(model, history, values)
            model.learn_one(_event(values))
            history.append(values)


@pytest.mark.parametrize("normalize", [True, False])
def test_ambiguous_joint_minima_are_all_refined(normalize):
    rng = np.random.default_rng(17)
    pattern = rng.normal(size=(8, 3))
    history = np.vstack(
        (
            [[1e100, 0, -1e100], [-1e100, 0, 1e100]],
            pattern,
            rng.normal(size=(7, 3)),
            pattern + 1e-7 * rng.normal(size=(8, 3)),
            rng.normal(size=(7, 3)),
            pattern[:-1],
        )
    )
    model = MultivariateRollingMatrixProfile(8, 64, normalize=normalize)
    for values in history:
        model.learn_one(_event(values))
    _assert_matches_oracle(model, list(history), pattern[-1])
    assert model.explain_one(_event(pattern[-1])) == (
        0.0,
        2,
        dict.fromkeys(_event(pattern[-1]), 0.0),
    )


def test_huge_query_preserves_small_reference_precision():
    rng = np.random.default_rng(8)
    huge = -math.ldexp(1.0, 900)
    history = np.vstack(
        (
            np.ldexp(rng.normal(size=(20, 2)), -100),
            np.ldexp(rng.normal(size=(30, 2)), -10),
            [huge, 0.0],
        )
    )
    model = MultivariateRollingMatrixProfile(13, 57, exclusion_zone=8)
    for values in history:
        model.learn_one(_event(values))
    _assert_matches_oracle(model, list(history), [huge, 0.0])


@pytest.mark.parametrize("normalize", [True, False])
def test_constant_runs_and_transitions(normalize):
    model = MultivariateRollingMatrixProfile(4, 13, normalize=normalize)
    values = np.array([0.0] * 20 + [1.0, 2.0, 3.0, 4.0] * 4 + [5.0] * 20)
    series = np.column_stack((values, values[::-1], np.ones(len(values))))
    history = []
    for row in series:
        _assert_matches_oracle(model, history, row)
        model.learn_one(_event(row))
        history.append(row)


def test_constant_distances_are_explained_per_channel():
    model = MultivariateRollingMatrixProfile(3, 6, exclusion_zone=2)
    series = np.array(
        [[4, 0, 4], [4, 1, 4], [4, 2, 4], [7, 7, 1], [7, 7, 2], [7, 7, 3]]
    )
    for row in series[:-1]:
        model.learn_one(_event(row))
    assert model.explain_one(_event(series[-1])) == (
        math.sqrt(3),
        0,
        {"c0": 0.0, "c1": math.sqrt(3), "c2": math.sqrt(3)},
    )


@pytest.mark.parametrize("normalize", [True, False])
def test_one_channel_equivalence_and_extra_matching_channels(normalize):
    scalar = RollingMatrixProfile(4, 19, normalize=normalize)
    single = MultivariateRollingMatrixProfile(4, 19, normalize=normalize)
    extended = MultivariateRollingMatrixProfile(4, 19, normalize=normalize)
    values = [1.0] * 8 + list(np.random.default_rng(5).normal(size=90)) + [2.0] * 25
    for value in values:
        event = {"value": float(value)}
        extra = {**event, "duplicate": float(value), "constant": 5.0}
        assert (
            scalar.match_one(event)
            == single.match_one(event)
            == extended.match_one(extra)
        )
        scalar.learn_one(event)
        single.learn_one(event)
        extended.learn_one(extra)


def test_defaults_and_first_valid_score():
    model = MultivariateRollingMatrixProfile(5)
    assert (model.window_size, model.exclusion_zone, model.normalize) == (80, 2, True)
    for index in range(7):
        assert not model.is_ready
        assert model.explain_one({"a": float(index), "b": 0.0}) == (0.0, None, {})
        model.learn_one({"a": float(index), "b": 0.0})
    assert model.is_ready
    assert model.match_one({"a": 7.0, "b": 0.0})[1] == 0
    assert model.n_history < model.window_size


@pytest.mark.parametrize("zone", [0, 1, 3, 8])
def test_minimum_capacity_continues_scoring_during_eviction(zone):
    model = MultivariateRollingMatrixProfile(4, 4 + zone + 1, exclusion_zone=zone)
    for index in range(80):
        event = {"a": float(index % 7), "b": float(index % 5)}
        score, match = model.match_one(event)
        if model.is_ready:
            assert match == max(0, index - model.window_size + 1)
            assert math.isfinite(score)
        model.learn_one(event)


@pytest.mark.parametrize("normalize", [True, False])
def test_ties_choose_earliest_retained_interval(normalize):
    model = MultivariateRollingMatrixProfile(3, 11, normalize=normalize)
    for index in range(35):
        if model.is_ready:
            assert model.match_one({"a": 2.0, "b": 3.0}) == (0.0, max(0, index - 10))
        model.learn_one({"a": 2.0, "b": 3.0})


def test_expired_nearest_match_is_removed_before_scoring():
    model = MultivariateRollingMatrixProfile(2, 5, normalize=False, exclusion_zone=1)
    for value in [1.0, 2.0, 9.0, 1.0]:
        model.learn_one({"a": value, "b": 2 * value})
    assert model.match_one({"a": 2.0, "b": 4.0}) == (0.0, 0)
    model.learn_one({"a": 2.0, "b": 4.0})
    assert model.explain_one({"a": 3.0, "b": 6.0}) == (12.0, 1, {"a": 6.0, "b": 12.0})


def test_queries_are_read_only_and_learning_is_independent():
    queried = MultivariateRollingMatrixProfile(4, 16)
    learn_only = MultivariateRollingMatrixProfile(4, 16)
    assert queried.explain_one({"unused_name": 1.0}) == (0.0, None, {})
    for values in np.random.default_rng(9).normal(size=(70, 3)):
        event = _event(values)
        before = pickle.dumps(queried)
        result = queried.explain_one(event)
        assert queried.score_one(event) == result[0]
        assert queried.match_one(event) == result[:2]
        queried.explain_one(_event([400.0] * 3))
        assert queried.explain_one(dict(reversed(list(event.items())))) == result
        result[2]["altered"] = -1.0
        assert "altered" not in queried.explain_one(event)[2]
        assert list(queried.explain_one(event)[2]) == (
            sorted(event) if queried.is_ready else []
        )
        assert pickle.dumps(queried) == before
        queried.learn_one(event)
        learn_only.learn_one(dict(reversed(list(event.items()))))
    assert pickle.dumps(queried) == pickle.dumps(learn_only)


@pytest.mark.parametrize("learned", [0, 2, 12])
@pytest.mark.parametrize(
    "event",
    [
        {},
        {"a": "bad", "b": 0.0},
        {"a": math.nan, "b": 0.0},
        {"a": 0.0, "b": math.inf},
        {1: 1.0, "b": 0.0},
    ],
)
def test_invalid_input_is_atomic_before_and_after_readiness(learned, event):
    model = MultivariateRollingMatrixProfile(3, 8)
    for index in range(learned):
        model.learn_one({"a": float(index), "b": 0.0})
    before = pickle.dumps(model)
    for method in (
        model.score_one,
        model.match_one,
        model.explain_one,
        model.learn_one,
    ):
        with pytest.raises(ValueError):
            method(event)
        assert pickle.dumps(model) == before


@pytest.mark.parametrize(
    "event", [{"a": 1.0}, {"a": 1.0, "c": 2.0}, {"a": 1.0, "b": 2.0, "c": 3.0}]
)
def test_first_successful_learning_commits_fixed_schema(event):
    model = MultivariateRollingMatrixProfile(3, 8)
    model.learn_one({"a": 1.0, "b": 2.0})
    before = pickle.dumps(model)
    for method in (
        model.score_one,
        model.match_one,
        model.explain_one,
        model.learn_one,
    ):
        with pytest.raises(ValueError, match="feature keys"):
            method(event)
        assert pickle.dumps(model) == before


@pytest.mark.parametrize("normalize", [True, False])
def test_long_stream_stays_ready_bounded_and_forgets_old_prefix(normalize):
    model = MultivariateRollingMatrixProfile(5, 32, normalize=normalize)
    history = []
    for index, values in enumerate(np.random.default_rng(12).normal(size=(2000, 3))):
        score = model.score_one(_event(values))
        assert math.isfinite(score) and score >= 0.0
        assert model.is_ready == (index >= 7)
        if index % 97 == 0:
            _assert_matches_oracle(model, history, values)
        model.learn_one(_event(values))
        history.append(values)
        assert model.n_history == min(index + 1, 32)
        assert len(model._statistics) == max(0, model.n_history - 4)
        assert sum(len(item) for item in model._statistics) <= 3 * 28
    assert model.n_samples_seen == len(history)
    other = MultivariateRollingMatrixProfile(5, 32, normalize=normalize)
    for values in [[-100.0] * 3] * (len(history) - 32) + history[-32:]:
        other.learn_one(_event(values))
    assert model.explain_one(_event([1.5] * 3)) == other.explain_one(_event([1.5] * 3))


def test_reset_clears_schema_indices_and_readiness():
    model = MultivariateRollingMatrixProfile(4, 16, normalize=False, exclusion_zone=3)
    for index in range(30):
        model.learn_one({"old": float(index)})
    model.reset()
    fresh = MultivariateRollingMatrixProfile(4, 16, normalize=False, exclusion_zone=3)
    assert pickle.dumps(model) == pickle.dumps(fresh)
    for index in range(20):
        event = {"a": float(index), "b": 0.0}
        assert model.explain_one(event) == fresh.explain_one(event)
        model.learn_one(event)
        fresh.learn_one(event)


@pytest.mark.parametrize(
    "params",
    [
        {"subsequence_length": 1},
        {"subsequence_length": 2.5},
        {"subsequence_length": True},
        {"subsequence_length": None},
        {"window_size": 4},
        {"window_size": 16.0},
        {"window_size": True},
        {"exclusion_zone": -1},
        {"exclusion_zone": 1.5},
        {"exclusion_zone": False},
        {"normalize": 1},
    ],
)
def test_invalid_configuration(params):
    with pytest.raises(ValueError):
        MultivariateRollingMatrixProfile(**{"subsequence_length": 4, **params})


def test_unrepresentable_joint_raw_distance_does_not_mutate_state():
    model = MultivariateRollingMatrixProfile(2, 4, normalize=False, exclusion_zone=1)
    for value in [-1e308, -1e308, 1e308]:
        model.learn_one({"large": value, "normal": 1.0})
    before = pickle.dumps(model)
    for method in (model.score_one, model.match_one, model.explain_one):
        with pytest.raises(OverflowError, match="float64"):
            method({"large": 1e308, "normal": 1.0})
        assert pickle.dumps(model) == before
    model.learn_one({"large": 1e308, "normal": 1.0})
    assert model.n_samples_seen == 4
