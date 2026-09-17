"""Behavioral and numerical checks against an independent direct-distance oracle."""

import math
import pickle

import numpy as np
import pytest

from aberrant.model.timeseries import RollingMatrixProfile


def _direct_distance(left, right, normalize):
    if not normalize:
        return math.dist(left, right)
    constant_left = bool(np.all(left == left[0]))
    constant_right = bool(np.all(right == right[0]))
    if constant_left or constant_right:
        return 0.0 if constant_left and constant_right else math.sqrt(len(left))
    normalized = []
    for values in (left, right):
        centered = np.asarray(values, dtype=np.longdouble) - values[0]
        centered -= centered.mean()
        centered /= np.max(np.abs(centered))
        normalized.append(centered / np.sqrt(np.mean(centered**2)))
    return float(np.sqrt(np.sum((normalized[0] - normalized[1]) ** 2)))


def _oracle(history, candidate, m, window_size, exclusion_zone, normalize):
    if len(history) < m + exclusion_zone:
        return 0.0, {}
    series = np.asarray([*history, candidate], dtype=np.float64)
    start = max(0, series.size - window_size)
    query_start = series.size - m
    query = series[query_start:]
    distances = {
        index: _direct_distance(series[index : index + m], query, normalize)
        for index in range(start, query_start - exclusion_zone)
    }
    return min(distances.values()), distances


def _assert_matches_oracle(model, history, value):
    expected, distances = _oracle(
        history,
        value,
        model.subsequence_length,
        model.window_size,
        model.exclusion_zone,
        model.normalize,
    )
    score, match = model.match_one({"value": float(value)})
    tolerance = 1e-12 if model.normalize else 0.0
    assert score == pytest.approx(expected, rel=1e-10, abs=tolerance)
    if distances:
        assert match in distances
        assert distances[match] == pytest.approx(expected, rel=1e-10, abs=tolerance)
    else:
        assert match is None


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "m,window,zone", [(2, 4, 0), (4, 6, 1), (5, 17, 4), (7, 23, 0)]
)
def test_stream_matches_direct_oracle(normalize, m, window, zone):
    model = RollingMatrixProfile(m, window, normalize=normalize, exclusion_zone=zone)
    history = []
    for value in np.random.default_rng(42).normal(size=200):
        _assert_matches_oracle(model, history, value)
        model.learn_one({"value": float(value)})
        history.append(value)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "offset,scale", [(1e12, 1.0), (0.0, 1e-300), (0.0, 1e300), (1.0, 1e-14)]
)
def test_extreme_scales_and_small_variations(normalize, offset, scale):
    model = RollingMatrixProfile(8, 31, normalize=normalize)
    history = []
    series = offset + scale * np.random.default_rng(18).normal(size=120)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for value in series:
            _assert_matches_oracle(model, history, value)
            model.learn_one({"value": float(value)})
            history.append(value)


@pytest.mark.parametrize("normalize", [True, False])
def test_ambiguous_fft_minima_are_all_considered(normalize):
    rng = np.random.default_rng(17)
    pattern = rng.normal(size=8)
    # A large outlier makes FFT estimates for small windows ill-conditioned.
    # Several almost-identical matches must all survive candidate screening.
    history = [
        1e100,
        -1e100,
        *pattern,
        *rng.normal(size=7),
        *(pattern + 1e-7 * rng.normal(size=8)),
        *rng.normal(size=7),
        *pattern[:-1],
    ]
    model = RollingMatrixProfile(8, 64, normalize=normalize)
    for value in history:
        model.learn_one({"value": float(value)})
    _assert_matches_oracle(model, history, pattern[-1])
    assert model.match_one({"value": float(pattern[-1])}) == (0.0, 2)


@pytest.mark.parametrize("normalize", [True, False])
def test_constant_runs_and_transitions_keep_scoring(normalize):
    model = RollingMatrixProfile(4, 13, normalize=normalize)
    history = []
    for value in [0.0] * 20 + [1.0, 2.0, 3.0, 4.0] * 4 + [5.0] * 20:
        _assert_matches_oracle(model, history, value)
        model.learn_one({"value": value})
        history.append(value)


def test_huge_query_preserves_reference_fft_precision():
    rng = np.random.default_rng(8)
    value = -math.ldexp(1.0, 900)
    history = [
        *np.ldexp(rng.normal(size=20), -100),
        *np.ldexp(rng.normal(size=30), -10),
        value,
    ]
    model = RollingMatrixProfile(13, 57, exclusion_zone=8)
    for item in history:
        model.learn_one({"value": float(item)})
    _assert_matches_oracle(model, history, value)


@pytest.mark.parametrize(
    "history,candidate,expected",
    [
        ([4.0, 4.0, 4.0, 7.0, 7.0], 7.0, 0.0),
        ([0.0, 1.0, 2.0, 7.0, 7.0], 7.0, math.sqrt(3)),
        ([4.0, 4.0, 4.0, 1.0, 2.0], 3.0, math.sqrt(3)),
    ],
)
def test_normalized_constant_convention(history, candidate, expected):
    model = RollingMatrixProfile(3, 6, exclusion_zone=2)
    for value in history:
        model.learn_one({"value": value})
    assert model.match_one({"value": candidate}) == (expected, 0)


def test_default_parameters_and_warmup_boundary():
    model = RollingMatrixProfile(5)
    assert model.window_size == 80
    assert model.exclusion_zone == 2
    assert model.normalize is True
    for index in range(7):
        assert not model.is_ready
        assert model.match_one({"value": float(index)}) == (0.0, None)
        model.learn_one({"value": float(index)})
    assert model.is_ready
    assert model.match_one({"value": 7.0})[1] == 0
    assert model.n_history < model.window_size


@pytest.mark.parametrize("zone", [0, 1, 3, 8])
def test_minimum_capacity_keeps_one_eligible_reference(zone):
    model = RollingMatrixProfile(4, 4 + zone + 1, exclusion_zone=zone)
    for index in range(80):
        score, match = model.match_one({"value": float(index % 7)})
        if model.is_ready:
            assert match == max(0, index - model.window_size + 1)
            assert math.isfinite(score)
        model.learn_one({"value": float(index % 7)})


def test_expired_nearest_match_is_removed_before_scoring():
    model = RollingMatrixProfile(2, 5, normalize=False, exclusion_zone=1)
    for value in [1.0, 2.0, 9.0, 1.0]:
        model.learn_one({"value": value})
    assert model.match_one({"value": 2.0}) == (0.0, 0)
    model.learn_one({"value": 2.0})
    assert model.match_one({"value": 3.0}) == (6.0, 1)


@pytest.mark.parametrize("normalize", [True, False])
def test_equal_matches_choose_earliest_retained_start(normalize):
    model = RollingMatrixProfile(3, 11, normalize=normalize)
    for index in range(35):
        if model.is_ready:
            assert model.match_one({"value": 2.0}) == (
                0.0,
                max(0, index - model.window_size + 1),
            )
        model.learn_one({"value": 2.0})


def test_scoring_is_read_only_and_learning_does_not_depend_on_it():
    queried = RollingMatrixProfile(4, 16)
    learn_only = RollingMatrixProfile(4, 16)
    assert queried.score_one({"unused_name": 1.0}) == 0.0
    for value in np.random.default_rng(9).normal(size=70):
        event = {"value": float(value)}
        before = pickle.dumps(queried)
        result = queried.match_one(event)
        assert queried.score_one(event) == result[0]
        queried.match_one({"value": 400.0})
        assert queried.match_one(event) == result
        assert pickle.dumps(queried) == before
        queried.learn_one(event)
        learn_only.learn_one(event)
    assert pickle.dumps(queried) == pickle.dumps(learn_only)


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"a": 1.0, "b": 2.0},
        {"value": "bad"},
        {"value": math.nan},
        {"value": math.inf},
        {1: 1.0},
        {"different": 1.0},
    ],
)
def test_invalid_input_leaves_ready_state_unchanged(event):
    model = RollingMatrixProfile(3, 8)
    for index in range(12):
        model.learn_one({"value": float(index)})
    before = pickle.dumps(model)
    for method in (model.score_one, model.match_one, model.learn_one):
        with pytest.raises(ValueError):
            method(event)
        assert pickle.dumps(model) == before


@pytest.mark.parametrize("normalize", [True, False])
def test_long_stream_has_bounded_state_and_forgets_old_prefix(normalize):
    model = RollingMatrixProfile(5, 32, normalize=normalize)
    history = []
    for index, value in enumerate(np.random.default_rng(12).normal(size=3000)):
        score = model.score_one({"value": float(value)})
        assert math.isfinite(score) and score >= 0.0
        assert model.is_ready == (index >= 7)
        if index % 97 == 0:
            _assert_matches_oracle(model, history, value)
        model.learn_one({"value": float(value)})
        history.append(value)
        assert model.n_history == min(index + 1, 32)
        assert len(model._statistics) <= 28
    other = RollingMatrixProfile(5, 32, normalize=normalize)
    for value in [-100.0] * (len(history) - 32) + history[-32:]:
        other.learn_one({"value": float(value)})
    assert model.match_one({"value": 1.5}) == other.match_one({"value": 1.5})


def test_reset_clears_schema_indices_and_readiness():
    model = RollingMatrixProfile(4, 16, normalize=False, exclusion_zone=3)
    for index in range(30):
        model.learn_one({"old": float(index)})
    model.reset()
    fresh = RollingMatrixProfile(4, 16, normalize=False, exclusion_zone=3)
    assert pickle.dumps(model) == pickle.dumps(fresh)
    for index in range(20):
        event = {"new": float(index)}
        assert model.match_one(event) == fresh.match_one(event)
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
    kwargs = {"subsequence_length": 4, **params}
    with pytest.raises(ValueError):
        RollingMatrixProfile(**kwargs)


def test_unrepresentable_raw_distance_does_not_mutate_state():
    model = RollingMatrixProfile(2, 4, normalize=False, exclusion_zone=1)
    for value in [-1e308, -1e308, 1e308]:
        model.learn_one({"value": value})
    before = pickle.dumps(model)
    with pytest.raises(OverflowError, match="float64"):
        model.score_one({"value": 1e308})
    assert pickle.dumps(model) == before
    model.learn_one({"value": 1e308})
    assert model.n_samples_seen == 4
