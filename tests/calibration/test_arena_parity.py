"""The calibrators reproduce the Timestral Uncertainty Arena's thresholds.

The fixture holds every arena method's raw thresholds on one score stream
(absolute residuals, a volatility jump, then signed scores) under two
settings, generated from the arena's TypeScript at the commit it names.
Feedback is delayed by the lead, and a method learns a score before the
score joins the shared history, as in the arena's engine.
"""

import json
import math
from pathlib import Path

import pytest

from aberrant.calibration import CALIBRATORS, ScoreHistory

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "arena_thresholds.json").read_text(
        encoding="utf-8"
    )
)
_SPECIAL = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}


def _decode(value):
    return _SPECIAL[value] if isinstance(value, str) else value


def _same(got, want):
    if math.isnan(want):
        return math.isnan(got)
    if math.isinf(want):
        return got == want
    return math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize(
    "run",
    FIXTURE["runs"],
    ids=lambda run: f"{run['method']}-alpha{run['alpha']}-lead{run['lead']}",
)
def test_thresholds_match_the_arena(run):
    spec = CALIBRATORS[run["method"]]
    history = ScoreHistory(run["window"])
    method = spec.factory(history, run["alpha"], spec.resolve(run["params"]))
    scores = FIXTURE["scores"]
    mismatches = []
    for t, expected in enumerate(run["thresholds"]):
        key = t - run["lead"]
        if key >= 0:
            method.update(key, scores[key])
            history.push(scores[key])
        got, want = method.predict(t), _decode(expected)
        if not _same(got, want):
            mismatches.append((t, got, want))
    assert not mismatches, mismatches[:5]


def test_every_arena_method_is_registered_with_its_defaults():
    arena = {spec["id"]: spec for spec in FIXTURE["specs"]}

    assert set(CALIBRATORS) == set(arena)
    for identifier, spec in arena.items():
        ours = CALIBRATORS[identifier]
        assert ours.family == spec["family"]
        assert ours.guarantee == spec["guarantee"]
        assert ours.defaults() == {p["key"]: p["default"] for p in spec["params"]}
