"""Strict, consistent count parameters across bounded streaming components."""

import math
import warnings
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from aberrant.drift import ADWIN
from aberrant.model.graph import SignedGraphSketchDetector
from aberrant.model.iforest import (
    MondrianIsolationForest,
    StreamRandomHistogramForest,
    XStream,
)
from aberrant.model.sketch import StreamingLODA
from aberrant.model.svm import IncrementalOneClassSVMAdaptiveKernel
from aberrant.stream.dataset.streamers import BatchStreamer, Sample
from aberrant.transform import IncrementalPCA
from aberrant.utils.validation import coerce_integer_count


class _EmptyStream:
    def stream(self) -> Iterator[Sample]:
        yield from ()

    def get_metadata(self) -> None:
        return None


def _batch_streamer(*, batch_size: int = 1000) -> BatchStreamer:
    return BatchStreamer(_EmptyStream(), batch_size=batch_size)


_CAPACITY_CASES = [
    (ADWIN, {}, "clock", 1),
    (ADWIN, {}, "max_buckets", 1),
    (ADWIN, {}, "min_window_length", 1),
    (ADWIN, {}, "grace_period", 0),
    (ADWIN, {"min_window_length": 1}, "max_window_size", 2),
    (IncrementalPCA, {"n_components": 1, "n0": 4}, "n_components", 1),
    (IncrementalPCA, {"n_components": 1, "n0": 4}, "n0", 1),
    (_batch_streamer, {}, "batch_size", 1),
    (
        SignedGraphSketchDetector,
        {"num_clusters": 1, "warm_up_graphs": 1},
        "max_graphs",
        1,
    ),
    (StreamingLODA, {}, "warm_up_samples", 1),
    (StreamRandomHistogramForest, {"n_estimators": 1}, "window_size", 2),
    (XStream, {}, "window_size", 1),
    (XStream, {}, "init_sample_size", 1),
    (XStream, {}, "max_feature_cache_size", 1),
    (IncrementalOneClassSVMAdaptiveKernel, {}, "buffer_size", 1),
    (IncrementalOneClassSVMAdaptiveKernel, {}, "sv_budget", 1),
    (MondrianIsolationForest, {"n_estimators": 1}, "window_size", 2),
]


@pytest.mark.parametrize(
    ("factory", "defaults", "parameter", "minimum"), _CAPACITY_CASES
)
@pytest.mark.parametrize(
    "value",
    [True, False, np.bool_(True), 3.0, 2.5, math.nan, math.inf, -math.inf, "4"],
)
def test_count_parameters_reject_values_that_can_bypass_retention(
    factory: Callable[..., object],
    defaults: dict[str, object],
    parameter: str,
    minimum: int,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=parameter):
        factory(**{**defaults, parameter: value})


@pytest.mark.parametrize(
    ("factory", "defaults", "parameter", "minimum"), _CAPACITY_CASES
)
def test_count_parameters_enforce_their_minimum(
    factory: Callable[..., object],
    defaults: dict[str, object],
    parameter: str,
    minimum: int,
) -> None:
    with pytest.raises(ValueError, match=parameter):
        factory(**{**defaults, parameter: minimum - 1})
    model = factory(**{**defaults, parameter: minimum})
    assert getattr(model, parameter) == minimum


@pytest.mark.parametrize(
    ("factory", "defaults", "parameter", "minimum"), _CAPACITY_CASES
)
@pytest.mark.parametrize("integer_type", [int, np.int8, np.int64, np.uint64])
def test_count_parameters_normalize_integer_scalars_before_storage(
    factory: Callable[..., object],
    defaults: dict[str, object],
    parameter: str,
    minimum: int,
    integer_type: Callable[[int], object],
) -> None:
    model = factory(**{**defaults, parameter: integer_type(4)})
    assert getattr(model, parameter) == 4
    assert type(getattr(model, parameter)) is int


def test_optional_capacity_bounds_still_accept_none() -> None:
    assert ADWIN(max_window_size=None).max_window_size is None
    assert XStream(max_feature_cache_size=None).max_feature_cache_size is None
    assert MondrianIsolationForest(window_size=None).window_size is None


@pytest.mark.parametrize(
    ("minimum", "invalid_bound"),
    [(np.int64(2**62), 10), (np.uint64(2**63), np.uint64(2**64 - 1))],
)
def test_adwin_checks_window_relationship_using_python_integer_arithmetic(
    minimum: np.integer,
    invalid_bound: int | np.integer,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match=r"2 \* min_window_length"):
            ADWIN(min_window_length=minimum, max_window_size=invalid_bound)
        detector = ADWIN(min_window_length=minimum, max_window_size=2 * int(minimum))
    assert detector.min_window_length == int(minimum)
    assert detector.max_window_size == 2 * int(minimum)
    assert type(detector.min_window_length) is int


@pytest.mark.parametrize("value", [np.uint64(2**64 - 1), 10**400])
def test_integer_count_coercion_does_not_pass_through_float(value: object) -> None:
    count = coerce_integer_count(value, label="capacity")
    assert count == value
    assert type(count) is int


@pytest.mark.parametrize("window_size", [np.int64(4), np.uint64(4)])
def test_mondrian_numpy_window_capacity_survives_repeated_rebuilds(
    window_size: np.integer,
) -> None:
    model = MondrianIsolationForest(
        n_estimators=2, subspace_size=1, window_size=window_size, seed=3
    )
    for index in range(32):
        model.learn_one({"x": float(index)})
        assert len(model._window) <= 4
        assert all(tree.n_samples <= 7 for tree in model.trees)
