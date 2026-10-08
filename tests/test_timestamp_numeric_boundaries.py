"""Timestamp representation and storage limits at model input boundaries."""

import math
import pickle
from fractions import Fraction

import numpy as np
import pytest

from aberrant.model.distance import SDOStream
from aberrant.model.sketch import StreamingRSHash
from aberrant.model.sketch.rshash import _time_difference
from aberrant.utils.validation import coerce_finite_number


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_finite_float_coercion_rejects_unrepresentable_integer(value: int) -> None:
    with pytest.raises(ValueError, match="Timestamp value must be finite"):
        coerce_finite_number(value, label="Timestamp value")


@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("method", ["learn_one", "score_one"])
def test_sdostream_rejects_timestamp_storage_overflow_without_state_changes(
    learned: bool, method: str
) -> None:
    model = SDOStream(k=2, x_neighbors=1, time_key="t", seed=3)
    if learned:
        model.learn_one({"x": 1.0, "t": 0})
    before = pickle.dumps(model)

    with pytest.raises(ValueError, match="Timestamp value must be finite"):
        getattr(model, method)({"x": 2.0, "t": 10**400})

    assert pickle.dumps(model) == before
    # Rejected initial input must not establish the schema or array dimensions.
    sample = {"x": 2.0, "t": 1} if learned else {"y": 1.0, "z": 2.0, "t": 0}
    model.learn_one(sample)
    assert model.n_observers >= 1


@pytest.mark.parametrize("decay", [0.01, 5e-324])
@pytest.mark.parametrize(
    ("previous", "current"),
    [
        (0.25, 10**400),
        (-(10**400), 0.25),
        (0.25, 10**322),
        (-(10**322), 0.25),
        (-1e308, 1e308),
        (-(10**309), -1e308),
    ],
)
def test_rshash_fading_supports_mixed_or_overflowing_timestamp_differences(
    previous: int | float, current: int | float, decay: float
) -> None:
    model = StreamingRSHash(
        components_num=2,
        hash_num=2,
        bins=8,
        warm_up_samples=1,
        time_key="t",
        decay=decay,
        seed=3,
    )
    model.learn_one({"x": 1.0, "t": previous})
    # Fraction supplies an independent reference for the true elapsed time.
    elapsed_decay = (Fraction(current) - Fraction(previous)) * Fraction(decay)
    expected = 0.0 if elapsed_decay > 745 else math.exp(-float(elapsed_decay))
    assert model._preview_scale(current) == pytest.approx(expected, rel=1e-13, abs=0.0)

    before = pickle.dumps(model)
    assert math.isfinite(model.score_one({"x": 1.0, "t": current}))
    assert pickle.dumps(model) == before
    model.learn_one({"x": 1.0, "t": current})
    assert model.n_samples_seen == 2
    assert model._boundary.clock.max_time == current


@pytest.mark.parametrize("base", [2**53, 2**64])
@pytest.mark.parametrize("gap", [1, 3])
@pytest.mark.parametrize("float_first", [False, True])
def test_rshash_mixed_timestamps_preserve_small_gaps_above_float_precision(
    base: int, gap: int, float_first: bool
) -> None:
    previous, current = (
        (float(base), base + gap) if float_first else (base - gap, float(base))
    )
    model = StreamingRSHash(time_key="t", warm_up_samples=1, seed=3)
    reference = StreamingRSHash(time_key="t", warm_up_samples=1, seed=3)
    model.learn_one({"x": 1.0, "t": previous})
    reference.learn_one({"x": 1.0, "t": 0})

    assert model._preview_scale(current) == reference._preview_scale(gap)
    assert model.score_one({"x": 1.0, "t": current}) == reference.score_one(
        {"x": 1.0, "t": gap}
    )
    model.learn_one({"x": 1.0, "t": current})
    reference.learn_one({"x": 1.0, "t": gap})
    np.testing.assert_array_equal(model._counts, reference._counts)
    assert model._scale == reference._scale


@pytest.mark.parametrize(
    ("previous", "current"), [(-0.25, 2), (-2, 0.25), (0.25, 2), (-2, -0.25)]
)
def test_rshash_mixed_timestamps_preserve_fractional_elapsed_time(
    previous: int | float, current: int | float
) -> None:
    model = StreamingRSHash(time_key="t", decay=0.01)
    model.learn_one({"x": 1.0, "t": previous})
    elapsed = float(Fraction(current) - Fraction(previous))
    assert model._preview_scale(current) == float(np.exp(-0.01 * elapsed))


@pytest.mark.parametrize("float_first", [False, True])
def test_mixed_elapsed_time_rounds_once_after_exact_subtraction(
    float_first: bool,
) -> None:
    previous, current = (-0.25, 2**53 + 1) if float_first else (-(2**53 + 1), 0.25)
    assert _time_difference(current, previous) == float(2**53 + 2)


@pytest.mark.parametrize("previous", [0.25, -123.5, 2.0])
@pytest.mark.parametrize("delta", [0.0, 0.25, 10.0, 745.0, 1000.0])
@pytest.mark.parametrize("decay", [0.0, 0.01, 5e-324])
def test_rshash_ordinary_float_fading_is_unchanged(
    previous: float, delta: float, decay: float
) -> None:
    model = StreamingRSHash(time_key="t", decay=decay)
    model.learn_one({"x": 1.0, "t": previous})
    current = previous + delta
    if decay == 0.0:
        expected = model._scale
    elif delta > 745.0:
        log_elapsed_decay = math.log(delta) + math.log(decay)
        expected = (
            0.0
            if log_elapsed_decay > math.log(745.0)
            else model._scale * math.exp(-math.exp(log_elapsed_decay))
        )
    else:
        expected = model._scale * float(np.exp(-decay * delta))

    assert model._preview_scale(current) == expected
