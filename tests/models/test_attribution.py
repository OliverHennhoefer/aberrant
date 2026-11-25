"""Models that split their score among the event's features."""

import pickle

import numpy as np
import pytest

from aberrant.base import AttributingModelProtocol, ModelProtocol
from aberrant.model import NullModel
from aberrant.model.forest import HalfSpaceTrees
from aberrant.model.sketch import LODA
from aberrant.model.stat import MovingMahalanobisDistance, OnlineCOPOD, OnlineECOD

NAMES = [f"f{index}" for index in range(6)]

# Each model with the score of an event that gives no evidence either way;
# ``None`` where that depends on what the model learned.
MODELS = {
    "ecod": (lambda: OnlineECOD(warm_up=64), 0.0),
    "copod": (lambda: OnlineCOPOD(warm_up=64), 0.0),
    "loda": (lambda: LODA(warm_up=64, n_projections=60, seed=3), None),
    "mahalanobis": (lambda: MovingMahalanobisDistance(window_size=200), 0.0),
    "hst": (
        lambda: HalfSpaceTrees(n_trees=20, height=6, window_size=100, seed=5),
        1 - 2**-6,
    ),
}


def _events(n: int, seed: int = 0) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.08, size=(n, 1))
    values = np.clip(base + rng.normal(0.0, 0.05, size=(n, len(NAMES))), 0, 1)
    return [dict(zip(NAMES, row.tolist(), strict=True)) for row in values]


def _trained(name: str):
    factory, _ = MODELS[name]
    model = factory()
    for event in _events(400):
        model.learn_one(event)
    return model


@pytest.mark.parametrize("name", MODELS)
def test_shares_sum_to_the_score_above_one_baseline(name):
    model = _trained(name)
    assert isinstance(model, AttributingModelProtocol)
    baselines = []
    for event in _events(30, seed=1):
        shares = model.attribute_one(event)
        assert list(shares) == NAMES
        baselines.append(model.score_one(event) - sum(shares.values()))
    expected = MODELS[name][1]
    assert baselines == pytest.approx(
        [baselines[0] if expected is None else expected] * len(baselines),
        rel=1e-9,
        abs=1e-12,
    )


@pytest.mark.parametrize("name", MODELS)
def test_the_feature_that_breaks_the_pattern_mostly_gets_the_largest_share(name):
    model = _trained(name)
    found = 0
    for event in _events(50, seed=9):
        # Far from its usual value and from its correlation with the others.
        event["f3"] = 0.98 if event["f3"] < 0.5 else 0.02
        shares = model.attribute_one(event)
        found += max(shares, key=shares.__getitem__) == "f3"
    assert found >= 35


def test_loda_baseline_is_the_score_of_a_typical_event():
    model = _trained("loda")
    events = _events(2000, seed=4)
    typical = np.mean([model.score_one(event) for event in events])
    event = events[0]
    baseline = model.score_one(event) - sum(model.attribute_one(event).values())
    assert baseline == pytest.approx(typical, rel=0.05)


@pytest.mark.parametrize("name", MODELS)
def test_shares_are_zero_before_readiness_and_never_change_state(name):
    factory, _ = MODELS[name]
    model = factory()
    events = _events(5)
    for event in events[:2]:
        model.learn_one(event)
    assert model.attribute_one(events[3]) == dict.fromkeys(NAMES, 0.0)

    model = _trained(name)
    before = pickle.dumps(model)
    model.attribute_one(events[4])
    assert pickle.dumps(model) == before


def test_models_without_attributions_do_not_claim_the_protocol():
    assert isinstance(NullModel(), ModelProtocol)
    assert not isinstance(NullModel(), AttributingModelProtocol)
