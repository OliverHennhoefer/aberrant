"""RLSRegressor: exact least squares, forgetting, readiness and input checks."""

import math
import pickle

import numpy as np
import pytest

from aberrant.forecast import RLSRegressor


def _events(n: int, seed: int = 0, noise: float = 0.0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a, b = rng.normal(size=2)
        yield (
            {"a": float(a), "b": float(b)},
            3.0 * a - 2.0 * b + 0.5 + noise * rng.normal(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forgetting": 0.0},
        {"forgetting": 1.1},
        {"delta": 0.0},
        {"fit_intercept": 1},
        {"residual_alpha": 2.0},
        {"min_scale": -1.0},
    ],
)
def test_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RLSRegressor(**kwargs)  # type: ignore[arg-type]


def test_ready_after_one_measured_error_and_recovers_coefficients() -> None:
    model = RLSRegressor(forgetting=1.0, delta=1e8)
    events = list(_events(50))
    for count, (x, y) in enumerate(events, start=1):
        model.learn_one(x, y)
        assert model.is_ready == (count >= 4)  # three coefficients, then one error
    assert model.feature_names == ("a", "b")
    assert model.coefficients == pytest.approx({"a": 3.0, "b": -2.0, "intercept": 0.5})
    prediction = model.predict_one({"a": 1.0, "b": 1.0})
    assert prediction is not None
    assert prediction.leads == (0,)
    assert prediction.origin == 50
    assert prediction.mean == pytest.approx((1.5,))
    assert prediction.scale is not None and prediction.scale[0] > 0


def test_matches_weighted_least_squares_with_forgetting() -> None:
    forgetting, delta = 0.97, 1e3
    model = RLSRegressor(forgetting=forgetting, delta=delta)
    events = list(_events(80, seed=1, noise=0.3))
    for x, y in events:
        model.learn_one(x, y)
    n = len(events)
    design = np.array([[x["a"], x["b"], 1.0] for x, _ in events])
    target = np.array([y for _, y in events])
    weights = forgetting ** np.arange(n - 1, -1, -1)
    # The initial covariance acts as a ridge prior that fades like old events.
    prior = np.eye(3) * forgetting**n / delta
    gram = design.T @ (weights[:, None] * design) + prior
    expected = np.linalg.solve(gram, design.T @ (weights * target))
    got = model.coefficients
    assert [got["a"], got["b"], got["intercept"]] == pytest.approx(expected, rel=1e-6)


def test_tracks_a_drifting_relation_with_forgetting() -> None:
    model = RLSRegressor(forgetting=0.95)
    rng = np.random.default_rng(2)
    for t in range(600):
        a = float(rng.normal())
        slope = 1.0 if t < 300 else -1.0
        model.learn_one({"a": a}, slope * a)
    assert model.coefficients["a"] == pytest.approx(-1.0, abs=1e-3)


def test_input_checks_and_atomic_failure() -> None:
    model = RLSRegressor(fit_intercept=False)
    assert model.predict_one({"a": 1.0}) is None
    model.learn_one({"a": 1.0}, 2.0)
    with pytest.raises(ValueError, match="Inconsistent feature keys"):
        model.learn_one({"b": 1.0}, 2.0)
    with pytest.raises(ValueError):
        model.learn_one({"a": 1.0}, math.nan)
    state = pickle.dumps(model)
    with pytest.raises(OverflowError):
        model.learn_one({"a": 1e200}, 1e300)
    assert pickle.dumps(model) == state
    model.reset()
    assert model.feature_names is None
    assert model.coefficients == {}
