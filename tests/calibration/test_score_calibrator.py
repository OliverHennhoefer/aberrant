"""ScoreCalibrator: issuing thresholds, delayed feedback and validation."""

import math
import pickle
import random

import pytest

from aberrant.calibration import CALIBRATORS, ScoreCalibrator


def _scores(n=400, seed=3):
    rng = random.Random(seed)
    return [abs(rng.gauss(0, 1)) for _ in range(n)]


@pytest.mark.parametrize("method", sorted(CALIBRATORS))
def test_no_threshold_before_the_first_score(method):
    calibrator = ScoreCalibrator(method, alpha=0.1)

    assert calibrator.predict(0) == math.inf


@pytest.mark.parametrize("method", sorted(CALIBRATORS))
def test_thresholds_stay_within_the_scores_seen(method):
    calibrator = ScoreCalibrator(method, alpha=0.1, window=50)
    largest = 0.0
    for t, score in enumerate(_scores(200)):
        threshold = calibrator.predict(t)
        if t:
            assert -largest <= threshold <= largest
        calibrator.update(t, score)
        largest = max(largest, abs(score))


@pytest.mark.parametrize("method", sorted(CALIBRATORS))
def test_a_pickled_calibrator_continues_identically(method):
    scores = _scores(300)
    original = ScoreCalibrator(method, alpha=0.1, window=50)
    for t in range(150):
        original.predict(t)
        if t >= 2:
            original.update(t - 2, scores[t - 2])
    restored = pickle.loads(pickle.dumps(original))
    for t in range(150, 300):
        assert restored.predict(t) == original.predict(t)
        restored.update(t - 2, scores[t - 2])
        original.update(t - 2, scores[t - 2])


@pytest.mark.parametrize("method", ["split", "aci", "dtaci", "qt", "pid", "saocp"])
def test_long_run_coverage_is_near_the_target(method):
    calibrator = ScoreCalibrator(method, alpha=0.1, window=200)
    covered = []
    for t, score in enumerate(_scores(3000)):
        threshold = calibrator.predict(t)
        if t >= 200:
            covered.append(score <= threshold)
        calibrator.update(t, score)

    assert sum(covered) / len(covered) == pytest.approx(0.9, abs=0.03)


def test_feedback_for_an_unknown_key_still_joins_the_history():
    calibrator = ScoreCalibrator("aci", alpha=0.1)
    calibrator.update("never issued", 1.0)

    assert calibrator.history.count == 1
    assert calibrator.predict(0) == 1.0


def test_parameters_default_and_validate():
    calibrator = ScoreCalibrator("eci", alpha=0.1, c=2)

    assert calibrator.params == {"lr": 0.05, "c": 2.0, "h": 0.0}
    with pytest.raises(ValueError, match="has no parameter gamma"):
        ScoreCalibrator("eci", alpha=0.1, gamma=0.1)
    with pytest.raises(ValueError, match=r"rho must be in \(0, 1\)"):
        ScoreCalibrator("nexcp", alpha=0.1, rho=1)
    with pytest.raises(ValueError, match="g must be an integer"):
        ScoreCalibrator("saocp", alpha=0.1, g=2.5)
    with pytest.raises(ValueError, match="alpha must be in"):
        ScoreCalibrator("aci", alpha=1.0)
    with pytest.raises(ValueError, match="Unknown calibration method"):
        ScoreCalibrator("conformal", alpha=0.1)
    with pytest.raises(ValueError, match="score must be finite"):
        calibrator.update(0, math.inf)


def test_every_default_is_accepted_by_its_parameter():
    for spec in CALIBRATORS.values():
        for parameter in spec.parameters:
            assert parameter.coerce(parameter.default) == parameter.default
            assert parameter.minimum <= parameter.low <= parameter.high


@pytest.mark.parametrize("method", ["bocpd", "bocpd_aci", "cpcp"])
def test_a_unit_changepoint_hazard_keeps_calibrating(method):
    calibrator = ScoreCalibrator(method, alpha=0.1, **{"lambda": 1})
    for t, score in enumerate(_scores(50)):
        calibrator.predict(t)
        calibrator.update(t, score)

    assert not math.isnan(calibrator.predict(50))


def test_eci_survives_a_score_far_below_its_threshold():
    calibrator = ScoreCalibrator("eci", alpha=0.1)
    for t, score in enumerate([1.0, 2.0, -1000.0, 1.5]):
        calibrator.predict(t)
        calibrator.update(t, score)

    assert math.isfinite(calibrator.predict(4))


def test_only_the_rolling_split_promises_exchangeable_coverage():
    coverage = {spec.coverage for spec in CALIBRATORS.values()}

    assert coverage == {"exchangeable", "long_run", "none"}
    assert [m for m, s in CALIBRATORS.items() if s.coverage == "exchangeable"] == [
        "split"
    ]
