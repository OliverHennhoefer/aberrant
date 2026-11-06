"""Tests for the online ECOD and COPOD tail-probability detectors."""

from __future__ import annotations

import math
import pickle
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import skew

from aberrant.base.readiness import Readiness
from aberrant.model.stat import OnlineCOPOD, OnlineECOD
from aberrant.model.stat._tail_reference import (
    _SketchReference,
    _WindowReference,
    pareto_log_tail,
)

DETECTORS = [OnlineECOD, OnlineCOPOD]


def _batch_score(reference: np.ndarray, x: np.ndarray, aggregation: str, variant: str):
    """Batch ECOD/COPOD of ``x`` against ``reference`` plus ``x`` itself."""
    data = np.vstack([reference, x])
    n = len(data)
    left = -np.log(np.sum(data <= x, axis=0) / n)
    right = -np.log(np.sum(data >= x, axis=0) / n)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        # scipy warns about the constant feature, whose skewness is undefined.
        warnings.simplefilter("ignore", RuntimeWarning)
        signs = np.sign(np.nan_to_num(skew(data, axis=0)))
    if aggregation == "paper":
        skewed = np.where(signs < 0, left, right)
        return max(left.sum(), right.sum(), skewed.sum())
    # PyOD's expressions, verbatim.
    skewed = left * -1 * np.sign(signs - 1) + right * np.sign(signs + 1)
    if variant == "ecod":
        return np.maximum(skewed, np.maximum(left, right)).sum()
    return np.maximum(skewed, (left + right) / 2).sum()


def _mixed_stream(rng: np.random.Generator, n: int) -> np.ndarray:
    """Symmetric, right- and left-skewed, discrete and constant features."""
    return np.column_stack(
        [
            rng.normal(size=n),
            rng.lognormal(size=n),
            -rng.lognormal(size=n),
            rng.integers(0, 4, n).astype(float),
            np.full(n, 3.0),
        ]
    )


def _event(row: np.ndarray) -> dict[str, float]:
    return {f"f{index}": float(value) for index, value in enumerate(row)}


def _variant(model_type: type) -> str:
    return "ecod" if model_type is OnlineECOD else "copod"


@pytest.mark.parametrize("model_type", DETECTORS)
@pytest.mark.parametrize("aggregation", ["paper", "pyod"])
def test_window_score_equals_batch_method_on_window_plus_candidate(
    model_type: type, aggregation: str
) -> None:
    rng = np.random.default_rng(7)
    data = _mixed_stream(rng, 300)
    window = 60
    model = model_type(
        memory="window", window_size=window, min_samples=20, aggregation=aggregation
    )
    for index, row in enumerate(data):
        score = model.score_one(_event(row))
        if index >= 20:
            reference = data[max(0, index - window) : index]
            expected = _batch_score(reference, row, aggregation, _variant(model_type))
            assert score == pytest.approx(expected, abs=1e-12)
        else:
            assert score == 0.0
        model.learn_one(_event(row))


@pytest.mark.parametrize("model_type", DETECTORS)
def test_uncompressed_sketch_equals_batch_method_on_all_history(
    model_type: type,
) -> None:
    rng = np.random.default_rng(8)
    data = _mixed_stream(rng, 400)
    model = model_type(compression=1e6, min_samples=10)
    for index, row in enumerate(data):
        score = model.score_one(_event(row))
        if index >= 10:
            expected = _batch_score(data[:index], row, "paper", _variant(model_type))
            assert score == pytest.approx(expected, abs=1e-12)
        model.learn_one(_event(row))


@pytest.mark.parametrize("model_type", DETECTORS)
def test_pyod_aggregation_matches_pyod_when_installed(model_type: type) -> None:
    pyod_models = pytest.importorskip("pyod.models")
    pytest.importorskip("pyod.models.copod")
    pytest.importorskip("pyod.models.ecod")

    batch_type = (
        pyod_models.ecod.ECOD if model_type is OnlineECOD else pyod_models.copod.COPOD
    )
    rng = np.random.default_rng(9)
    data = _mixed_stream(rng, 120)[:, :4]
    model = model_type(
        memory="window", window_size=50, min_samples=20, aggregation="pyod"
    )
    for index, row in enumerate(data):
        score = model.score_one(_event(row))
        if index >= model.min_samples:
            reference = batch_type().fit(data[max(0, index - 50) : index])
            expected = reference.decision_function(row[np.newaxis, :])[0]
            assert score == pytest.approx(expected, abs=1e-9)
        model.learn_one(_event(row))


def test_paper_scores_of_ecod_and_copod_coincide() -> None:
    rng = np.random.default_rng(10)
    ecod, copod = OnlineECOD(min_samples=5), OnlineCOPOD(min_samples=5)
    for row in rng.lognormal(size=(200, 3)):
        assert ecod.score_one(_event(row)) == copod.score_one(_event(row))
        ecod.learn_one(_event(row))
        copod.learn_one(_event(row))


@pytest.mark.parametrize(
    "params",
    [
        {"memory": "window", "window_size": 32},
        {"memory": "sketch", "half_life": 50.0},
        {"memory": "hybrid", "window_size": 32, "tail": "pareto"},
    ],
)
def test_scoring_and_explaining_never_change_state(params: dict) -> None:
    rng = np.random.default_rng(11)
    model = OnlineECOD(min_samples=10, **params)
    for row in rng.normal(size=(150, 3)):
        model.learn_one(_event(row))
    before = pickle.dumps(model)
    candidate = _event(np.array([4.0, -3.0, 0.5]))
    first = model.score_one(candidate)
    model.explain_one(candidate)
    assert model.score_one(candidate) == first
    assert pickle.dumps(model) == before


def test_scoring_before_learning_does_not_lock_the_schema() -> None:
    model = OnlineECOD()
    assert model.score_one({"a": 1.0}) == 0.0
    model.learn_one({"b": 1.0, "c": 2.0})
    with pytest.raises(ValueError):
        model.learn_one({"a": 1.0})
    with pytest.raises(ValueError):
        model.score_one({"b": 1.0})


def test_window_forgets_everything_older_than_the_window() -> None:
    rng = np.random.default_rng(12)
    data = rng.normal(size=(500, 2))
    long_run = OnlineECOD(memory="window", window_size=40, min_samples=40)
    fresh = OnlineECOD(memory="window", window_size=40, min_samples=40)
    for row in data:
        long_run.learn_one(_event(row))
    for row in data[-40:]:
        fresh.learn_one(_event(row))
    for probe in rng.normal(scale=2.0, size=(50, 2)):
        assert long_run.score_one(_event(probe)) == pytest.approx(
            fresh.score_one(_event(probe)), abs=1e-12
        )


def _exact_log_tails(
    data: np.ndarray, weights: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = weights.sum()
    at_most = weights @ (data <= x)
    at_least = weights @ (data >= x)
    return (
        np.log((at_most + 1.0) / (total + 1.0)),
        np.log((at_least + 1.0) / (total + 1.0)),
    )


def test_sketch_tail_probabilities_have_small_relative_error() -> None:
    rng = np.random.default_rng(13)
    n = 20_000
    data = np.column_stack(
        [
            rng.normal(size=n),
            rng.lognormal(size=n),
            rng.pareto(2.0, size=n),
            rng.integers(0, 10, n).astype(float),
        ]
    )
    reference = _SketchReference(4, compression=100.0)
    for row in data:
        reference.learn(row)
    probes = np.vstack(
        [
            data[rng.integers(0, n, 400)],
            np.quantile(data, [0.0001, 0.001, 0.01, 0.99, 0.999, 0.9999], axis=0),
            data.min(axis=0),
            data.max(axis=0),
            data.max(axis=0) * 2.0,
        ]
    )
    ones = np.ones(n)
    errors = []
    for probe in probes:
        left, right = reference.tail_log_probabilities(probe)
        exact_left, exact_right = _exact_log_tails(data, ones, probe)
        errors.append(np.abs(np.concatenate([left - exact_left, right - exact_right])))
    errors = np.array(errors)
    # Continuous features: errors in nats stay small even in the far tails.
    continuous = errors[:, [0, 1, 2, 4, 5, 6]]
    assert np.median(continuous) < 0.02
    assert continuous.max() < 0.3
    # Repeated discrete values are held exactly by the endpoint atoms.
    assert errors[:, [3, 7]].max() < 1e-12
    # The digest stays far smaller than the data.
    assert reference.n_centroids < 4 * 150


def test_decayed_sketch_tracks_the_exponentially_weighted_ecdf() -> None:
    rng = np.random.default_rng(14)
    n, half_life = 6_000, 500.0
    data = np.concatenate(
        [rng.normal(size=(n // 2, 1)), rng.normal(loc=3.0, size=(n // 2, 1))]
    )
    reference = _SketchReference(1, compression=100.0, half_life=half_life)
    for row in data:
        reference.learn(row)
    weights = 0.5 ** (np.arange(n)[::-1] / half_life)
    assert reference.size == pytest.approx(weights.sum(), rel=1e-9)
    errors = []
    for probe in np.linspace(-3.0, 7.0, 41):
        left, right = reference.tail_log_probabilities(np.array([probe]))
        exact_left, exact_right = _exact_log_tails(data, weights, np.array([probe]))
        errors.extend([abs(left[0] - exact_left[0]), abs(right[0] - exact_right[0])])
    # Centroids keep the resolution they had when they were merged, so the
    # region that held the old bulk is coarser after the shift.
    assert np.median(errors) < 0.02
    assert max(errors) < 0.35


def test_decayed_sketch_stays_accurate_on_a_long_stationary_stream() -> None:
    # Stored weights inflate as the stream ages; neither the compression nor
    # the centroids' ranges may coarsen because of it.
    rng = np.random.default_rng(21)
    n, half_life = 60_000, 500.0
    data = rng.normal(size=n)
    reference = _SketchReference(1, compression=100.0, half_life=half_life)
    for value in data:
        reference.learn(np.array([value]))
    weights = 0.5 ** (np.arange(n)[::-1] / half_life)
    recent = data[-3_000:]
    for probe in np.quantile(recent, [0.0005, 0.005, 0.05, 0.5, 0.95, 0.995, 0.9995]):
        left, right = reference.tail_log_probabilities(np.array([probe]))
        exact_left, exact_right = _exact_log_tails(
            data[:, np.newaxis], weights, np.array([probe])
        )
        assert left[0] == pytest.approx(exact_left[0], abs=0.1)
        assert right[0] == pytest.approx(exact_right[0], abs=0.1)


def test_weights_that_decay_to_zero_are_dropped() -> None:
    # A tiny half-life underflows old weights to zero after renormalization.
    model = OnlineECOD(half_life=0.1, min_samples=1)
    for _ in range(1_000):
        model.learn_one({"x": 1.0})
    assert model.score_one({"x": 1.0}) == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(model.score_one({"x": 2.0}))


def test_decay_survives_weight_renormalization() -> None:
    rng = np.random.default_rng(15)
    # A half-life of one sample doubles the stored weight per learn, which
    # forces several renormalizations within a thousand samples.
    reference = _SketchReference(2, compression=50.0, half_life=1.0)
    for row in rng.normal(size=(1_000, 2)):
        reference.learn(row)
    assert reference.size == pytest.approx(2.0, rel=1e-6)
    left, right = reference.tail_log_probabilities(np.zeros(2))
    assert np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    assert np.all(left <= 0.0) and np.all(right <= 0.0)


def test_hybrid_uses_window_until_history_is_ready_then_the_smaller_score() -> None:
    rng = np.random.default_rng(16)
    data = rng.normal(size=(400, 2))
    hybrid = OnlineECOD(
        memory="hybrid", window_size=50, compression=1e6, min_samples=20
    )
    window = OnlineECOD(memory="window", window_size=50, min_samples=20)
    history = OnlineECOD(compression=1e6, min_samples=1)
    for index, row in enumerate(data):
        probe = _event(row * 1.5)
        hybrid_score = hybrid.score_one(probe)
        window_score = window.score_one(probe)
        if index < 70:
            assert hybrid_score == window_score
        elif index >= 50:
            assert hybrid_score == pytest.approx(
                min(window_score, history.score_one(probe)), abs=1e-12
            )
        hybrid.learn_one(_event(row))
        window.learn_one(_event(row))
        if index >= 50:
            history.learn_one(_event(data[index - 50]))


@pytest.mark.parametrize("memory", ["window", "sketch"])
def test_pareto_tail_keeps_scoring_past_the_observed_range(memory: str) -> None:
    rng = np.random.default_rng(17)
    empirical = OnlineECOD(memory=memory)
    pareto = OnlineECOD(memory=memory, tail="pareto")
    for value in rng.exponential(size=3_000):
        empirical.learn_one({"x": float(value)})
        pareto.learn_one({"x": float(value)})
    beyond = [12.0, 16.0, 24.0, 40.0]
    empirical_scores = [empirical.score_one({"x": value}) for value in beyond]
    pareto_scores = [pareto.score_one({"x": value}) for value in beyond]
    assert len(set(empirical_scores)) == 1  # saturated at log(n + 1)
    assert all(b > a for a, b in zip(pareto_scores, pareto_scores[1:], strict=False))
    assert pareto_scores[-1] > empirical_scores[-1]
    # Inside the body of the distribution both tails agree.
    for value in (0.2, 0.7, 1.5):
        assert pareto.score_one({"x": value}) == empirical.score_one({"x": value})


def test_pareto_fit_recovers_exponential_and_heavy_tails() -> None:
    rng = np.random.default_rng(18)
    for sample, x, expected in (
        (np.sort(rng.exponential(size=50_000)), 12.0, -12.0),
        (np.sort(rng.pareto(2.0, size=50_000) + 1.0), 200.0, math.log(200.0**-2)),
    ):
        estimate = pareto_log_tail(sample, np.ones(len(sample)), len(sample), x, 0.98)
        assert estimate is not None
        assert estimate == pytest.approx(expected, abs=1.0)
    # Too few exceedances or a candidate inside the body: no estimate.
    small = np.sort(rng.normal(size=100))
    assert pareto_log_tail(small, np.ones(100), 100.0, 10.0, 0.98) is None
    body = np.sort(rng.normal(size=5_000))
    assert pareto_log_tail(body, np.ones(5_000), 5_000.0, 0.0, 0.98) is None


@pytest.mark.parametrize("model_type", DETECTORS)
@pytest.mark.parametrize("aggregation", ["paper", "pyod"])
@pytest.mark.parametrize("memory", ["window", "sketch", "hybrid"])
def test_explanation_sums_to_the_score(
    model_type: type, aggregation: str, memory: str
) -> None:
    rng = np.random.default_rng(19)
    model = model_type(memory=memory, aggregation=aggregation, min_samples=10)
    names = ["a", "b", "c"]
    assert model.explain_one(dict.fromkeys(names, 0.0)) == dict.fromkeys(names, 0.0)
    for row in rng.lognormal(size=(1_200, 3)):
        model.learn_one(dict(zip(names, row.tolist(), strict=True)))
    for row in rng.lognormal(sigma=2.0, size=(20, 3)):
        event = dict(zip(names, row.tolist(), strict=True))
        explanation = model.explain_one(event)
        assert list(explanation) == names
        assert all(value >= 0.0 for value in explanation.values())
        assert sum(explanation.values()) == pytest.approx(model.score_one(event))


@settings(max_examples=40, deadline=None)
@given(
    data=st.lists(
        st.lists(st.integers(-1000, 1000), min_size=3, max_size=3),
        min_size=5,
        max_size=40,
    ),
    scale=st.sampled_from([0.25, 0.5, 2.0, 8.0]),
    shift=st.integers(-100, 100),
)
def test_window_scores_are_bounded_and_affine_invariant(
    data: list[list[int]], scale: float, shift: int
) -> None:
    # Integers with power-of-two scales keep every transform exact.
    rows = np.array(data[:-1], dtype=np.float64)
    candidate = np.array(data[-1], dtype=np.float64)
    plain = OnlineECOD(memory="window", window_size=64, min_samples=1)
    moved = OnlineECOD(memory="window", window_size=64, min_samples=1)
    permuted = OnlineECOD(memory="window", window_size=64, min_samples=1)
    order = [2, 0, 1]
    for row in rows:
        plain.learn_one(_event(row))
        moved.learn_one(_event(row * scale + shift))
        permuted.learn_one(_event(row[order]))
    score = plain.score_one(_event(candidate))
    assert 0.0 <= score <= 3 * math.log(len(rows) + 1) + 1e-12
    assert permuted.score_one(_event(candidate[order])) == pytest.approx(score)
    # Ranks are affine invariant; the skewness sign is too unless rounding
    # makes a symmetric feature's sign flip, so compare the two tail sums.
    assert moved._primary is not None and plain._primary is not None
    left, right = plain._primary.tail_log_probabilities(candidate)
    moved_left, moved_right = moved._primary.tail_log_probabilities(
        candidate * scale + shift
    )
    np.testing.assert_allclose(moved_left, left, atol=1e-12)
    np.testing.assert_allclose(moved_right, right, atol=1e-12)


def test_warm_up_and_readiness() -> None:
    model = OnlineECOD(min_samples=3)
    assert model.readiness == Readiness(False, 0, 3)
    for value in (1.0, 2.0):
        model.learn_one({"x": value})
        assert model.score_one({"x": 100.0}) == 0.0
    assert model.readiness == Readiness(False, 2, 3)
    model.learn_one({"x": 3.0})
    assert model.readiness == Readiness(True, 3, 3)
    assert model.is_ready
    assert model.score_one({"x": 100.0}) > 0.0


def test_constant_feature_scores_departures_at_the_floor() -> None:
    model = OnlineECOD(memory="window", window_size=100, min_samples=10)
    for _ in range(99):
        model.learn_one({"x": 5.0})
    assert model.score_one({"x": 5.0}) == pytest.approx(0.0)
    assert model.score_one({"x": 5.5}) == pytest.approx(math.log(100.0))


def test_time_key_is_excluded_and_must_not_decrease() -> None:
    model = OnlineECOD(time_key="t", min_samples=2)
    for t, value in enumerate([1.0, 2.0, 3.0]):
        model.learn_one({"t": float(t), "x": value})
    assert model.explain_one({"t": 5.0, "x": 9.0}).keys() == {"x"}
    with pytest.raises(ValueError):
        model.learn_one({"t": 0.0, "x": 1.0})
    with pytest.raises(ValueError):
        model.learn_one({"x": 1.0})


def test_non_finite_values_are_rejected() -> None:
    model = OnlineECOD()
    with pytest.raises(ValueError):
        model.learn_one({"x": float("nan")})
    with pytest.raises(ValueError):
        model.score_one({"x": float("inf")})
    with pytest.raises(ValueError):
        model.learn_one({})


@pytest.mark.parametrize(
    "params",
    [
        {"memory": "tree"},
        {"tail": "gumbel"},
        {"aggregation": "max"},
        {"memory": "sketch", "window_size": 10},
        {"memory": "window", "half_life": 10.0},
        {"memory": "window", "compression": 50.0},
        {"tail_quantile": 0.9},
        {"tail": "pareto", "tail_quantile": 1.0},
        {"tail": "pareto", "tail_quantile": 0.0},
        {"memory": "window", "window_size": 1},
        {"memory": "window", "window_size": 10, "min_samples": 11},
        {"half_life": 0.0},
        {"half_life": float("inf")},
        {"compression": 5.0},
        {"min_samples": 0},
        {"min_samples": True},
        {"time_key": ""},
    ],
)
def test_invalid_parameters_are_rejected(params: dict) -> None:
    with pytest.raises(ValueError):
        OnlineECOD(**params)


def test_defaults_resolve_per_memory() -> None:
    sketch = OnlineECOD()
    assert (sketch.memory, sketch.window_size, sketch.half_life) == (
        "sketch",
        None,
        None,
    )
    assert sketch.compression == 100.0
    assert sketch.tail_quantile is None
    window = OnlineCOPOD(memory="window")
    assert (window.window_size, window.compression) == (1024, None)
    hybrid = OnlineECOD(memory="hybrid", tail="pareto")
    assert (hybrid.window_size, hybrid.compression, hybrid.tail_quantile) == (
        1024,
        100.0,
        0.98,
    )


@pytest.mark.parametrize(
    "params",
    [
        {"memory": "window", "window_size": 64},
        {"memory": "sketch", "half_life": 200.0},
        {"memory": "hybrid", "window_size": 64, "half_life": 400.0},
    ],
)
def test_state_stays_bounded_on_long_streams(params: dict) -> None:
    rng = np.random.default_rng(20)
    model = OnlineECOD(min_samples=8, **params)
    for row in rng.standard_t(3, size=(20_000, 2)):
        model.learn_one(_event(row))
    references = [model._primary, model._history]
    for reference in references:
        if isinstance(reference, _WindowReference):
            assert reference.size == 64
        elif isinstance(reference, _SketchReference):
            assert reference.n_centroids <= 2 * 150
            assert reference._buffered < reference.buffer_size
    assert math.isfinite(model.score_one({"f0": 50.0, "f1": -50.0}))
    assert model.n_samples_seen == 20_000


def test_reset_and_repr() -> None:
    model = OnlineCOPOD(memory="hybrid", window_size=16, min_samples=4)
    for value in range(40):
        model.learn_one({"x": float(value)})
    model.reset()
    assert model.n_samples_seen == 0
    assert model.score_one({"y": 1.0}) == 0.0
    model.learn_one({"y": 1.0})
    assert repr(model).startswith("OnlineCOPOD(memory='hybrid', window_size=16")
