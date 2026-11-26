"""Tests for the tabular CUSUM drift detector."""

import math
import pickle

import numpy as np
import pytest

from aberrant.drift import CUSUM


def _first_signal(detector, values):
    for index, value in enumerate(values):
        if detector.update(value):
            return index
    return None


def test_sums_follow_the_tabular_recursion():
    detector = CUSUM(k=0.5, h=100.0, target=10.0, scale=2.0)
    upper = lower = 0.0
    for value in [12.0, 14.0, 9.0, 4.0, 4.0, 10.0, 16.0]:
        z = (value - 10.0) / 2.0
        upper = max(0.0, upper + z - 0.5)
        lower = max(0.0, lower - z - 0.5)
        detector.update(value)
        assert detector.statistic == pytest.approx(max(upper, lower), abs=1e-15)


def test_upward_shift_is_signalled_with_its_onset():
    detector = CUSUM(target=0.0, scale=1.0)
    index = _first_signal(detector, [0.0] * 64 + [2.0] * 64)

    # Each shifted value adds 2 - 0.5 to the upper sum, which passes 10 on
    # the seventh.
    assert index == 70
    assert detector.direction == "up"
    assert index - detector.onset + 1 == 64
    assert detector.n_detections == 1


@pytest.mark.parametrize(
    ("mode", "expected"), [("down", "down"), ("both", "down"), ("up", None)]
)
def test_mode_selects_the_direction(mode, expected):
    detector = CUSUM(target=0.0, scale=1.0, mode=mode)
    index = _first_signal(detector, [0.0] * 20 + [-3.0] * 20)
    if expected is None:
        assert index is None
    else:
        assert detector.direction == expected


def test_the_signal_lasts_one_update():
    detector = CUSUM(target=0.0, scale=1.0)
    _first_signal(detector, [0.0] * 5 + [4.0] * 10)
    detector.update(0.0)
    assert not detector.drift_detected
    assert detector.direction is None
    assert detector.onset == 0
    assert detector.statistic == 0.0
    assert detector.n_detections == 1


def test_estimated_reference_is_invariant_to_the_signal_scale():
    rng = np.random.default_rng(3)
    base = np.concatenate([rng.standard_normal(150), rng.standard_normal(100) + 1.5])

    plain = _first_signal(CUSUM(), base.tolist())
    scaled = _first_signal(CUSUM(), (5000.0 + 1e3 * base).tolist())
    assert plain is not None and plain >= 150
    assert scaled == plain


def test_warm_up_never_signals():
    detector = CUSUM(warm_up=10)
    assert not any(detector.update(value) for value in [0.0] * 9 + [1e9])


def test_an_estimated_reference_learns_the_new_level_after_a_signal():
    rng = np.random.default_rng(4)
    stream = np.concatenate([rng.standard_normal(300), rng.standard_normal(3000) + 4])

    estimated = CUSUM()
    for value in stream:
        estimated.update(value)
    fixed = CUSUM(target=0.0, scale=1.0)
    for value in stream:
        fixed.update(value)

    assert estimated.n_detections == 1
    # A fixed reference stays out of control and keeps signalling.
    assert fixed.n_detections > 100


def test_in_control_noise_rarely_alarms():
    rng = np.random.default_rng(5)
    detector = CUSUM(h=5.0, target=0.0, scale=1.0)
    for value in rng.standard_normal(50_000):
        detector.update(value)

    # About 470 observations pass between false alarms at h = 5.
    assert 70 <= detector.n_detections <= 145

    detector = CUSUM()
    for value in rng.standard_normal(50_000):
        detector.update(value)
    assert detector.n_detections <= 8


def test_any_change_from_a_constant_warm_up_signals():
    detector = CUSUM(warm_up=5)
    stream = [3.0] * 20
    assert _first_signal(detector, stream) is None
    assert detector.update(3.0 + 1e-9)
    assert detector.direction == "up"


def test_extreme_values_keep_the_state_finite():
    detector = CUSUM(warm_up=4)
    for value in [1e308, -1e308] * 50:
        detector.update(value)
        assert not math.isnan(detector.statistic)

    detector = CUSUM(warm_up=4)
    for value in [1e308] * 4:
        detector.update(value)
    assert detector.update(-1e308)
    assert detector.direction == "down"


def test_pickled_detector_continues_identically():
    rng = np.random.default_rng(6)
    stream = np.concatenate([rng.standard_normal(200), rng.standard_normal(200) + 2])
    detector = CUSUM(warm_up=50)
    for value in stream[:220]:
        detector.update(value)
    restored = pickle.loads(pickle.dumps(detector))
    rest = stream[220:]
    assert [restored.update(v) for v in rest] == [detector.update(v) for v in rest]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": -0.1}, "k must be"),
        ({"k": math.inf}, "k must be"),
        ({"h": 0.0}, "h must be"),
        ({"h": math.nan}, "h must be"),
        ({"warm_up": 1}, "warm_up"),
        ({"target": math.inf}, "target must be finite"),
        ({"scale": 0.0}, "scale must be"),
        ({"scale": -1.0}, "scale must be"),
        ({"mode": "sideways"}, "mode must be"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CUSUM(**kwargs)


def test_repr_shows_changed_parameters():
    assert repr(CUSUM(h=4.0, mode="up")) == "CUSUM(h=4.0, mode='up')"
