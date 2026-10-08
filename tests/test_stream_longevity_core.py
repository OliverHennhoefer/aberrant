"""Duration, retention and lazy consumption contracts for streaming infrastructure."""

import gc
import math
import tracemalloc
from collections.abc import Iterator
from itertools import count, islice

import numpy as np
import pytest

from aberrant.base import ValidationError
from aberrant.catalog import StateKind, get_model_spec
from aberrant.drift import ADWIN, KSWIN, PageHinkley
from aberrant.evaluate import PrequentialEvaluator
from aberrant.model import NullModel
from aberrant.model.stat import MovingAverage
from aberrant.stream.dataset.streamers import BatchStreamer, Sample
from aberrant.transform import (
    FeatureSchemaGuard,
    IncrementalPCA,
    MinMaxScaler,
    RandomProjection,
    RollingRobustScaler,
    StandardScaler,
)
from aberrant.utils.validation import EdgeEventBoundary, MonotonicClock


class _InfiniteStream:
    def __init__(self) -> None:
        self.consumed = 0
        self.closed = False

    def stream(self) -> Iterator[Sample]:
        try:
            for index in count():
                self.consumed += 1
                yield {"x": float(index % 31)}, index % 2
        finally:
            self.closed = True

    def get_metadata(self) -> None:
        return None


def test_catalog_distinguishes_mondrian_bounded_and_lifetime_modes() -> None:
    spec = get_model_spec("mondrian_isolation_forest")
    assert spec.capabilities().state is StateKind.GROWING
    assert spec.capabilities({"window_size": 32}).state is StateKind.BOUNDED


@pytest.mark.parametrize("size", [True, False, 0, -1, 2.5, math.inf, math.nan])
def test_batch_size_cannot_disable_yielding(size: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        BatchStreamer(_InfiniteStream(), batch_size=size)


def test_batching_an_infinite_source_is_lazy_bounded_and_closable() -> None:
    source = _InfiniteStream()
    batches = BatchStreamer(source, batch_size=7).stream()
    for batch_number, (features, labels) in enumerate(islice(batches, 10_000), 1):
        assert len(features) == len(labels) == 7
        assert source.consumed == batch_number * 7
    batches.close()
    assert source.closed


def test_numpy_integer_capacities_remain_supported() -> None:
    batches = BatchStreamer(_InfiniteStream(), batch_size=np.int64(4)).stream()
    assert len(next(batches)[0]) == 4
    batches.close()
    pca = IncrementalPCA(n_components=np.int64(1), n0=np.int64(2))
    pca.learn_one({"x": 1.0})
    pca.learn_one({"x": 2.0})
    assert pca.n0_reached and not pca.window


@pytest.mark.parametrize("integer_like", [True, False])
def test_implicit_clock_advances_exactly_beyond_float_integer_precision(
    integer_like: bool,
) -> None:
    clock = MonotonicClock(integer_like=integer_like)
    clock._arrival_index = 2**53
    for step in range(1, 10):
        event = clock.preview(implicit=True)
        assert event.value == 2**53 + step
        clock.commit(event)
    assert clock.max_time == 2**53 + 9


def test_edge_ids_and_integer_timestamps_remain_distinct() -> None:
    boundary = EdgeEventBoundary(source_key="src", destination_key="dst", time_key="t")
    for offset in range(4):
        identity = 2**64 + offset
        event = boundary.preview(
            {"src": identity, "dst": np.uint64(2**63 + offset), "t": identity}
        )
        assert event.source == identity
        assert event.destination == 2**63 + offset
        assert event.bucket == identity
        boundary.commit(event)
    with pytest.raises(ValueError, match="Non-monotonic"):
        boundary.preview({"src": 1, "dst": 2, "t": 2**64 + 2})


def test_clock_rejects_decreasing_integers_beyond_float_range() -> None:
    clock = MonotonicClock(integer_like=True)
    clock.commit(clock.preview(10**400, implicit=False))
    with pytest.raises(ValueError, match="Non-monotonic"):
        clock.preview(10**400 - 1, implicit=False)


@pytest.mark.parametrize(
    "parameter", ["clock", "max_buckets", "min_window_length", "grace_period"]
)
@pytest.mark.parametrize("value", [True, 1.5, math.inf, math.nan])
def test_adwin_rejects_parameters_that_disable_compression_or_detection(
    parameter: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        ADWIN(**{parameter: value})


@pytest.mark.parametrize("limit", [0, 9, True, 12.5, math.inf])
def test_adwin_rejects_invalid_hard_window_bound(limit: object) -> None:
    with pytest.raises(ValueError, match="max_window_size"):
        ADWIN(max_window_size=limit)


def test_adwin_stationary_storage_is_logarithmic_without_a_bound() -> None:
    detector = ADWIN()
    for _ in range(100_000):
        detector.update(0.25)
    assert detector.width == 100_000
    assert sum(detector._bucket_count) <= detector.max_buckets * (100_000).bit_length()
    assert len(detector._bucket_count) > 10  # The default is not constant memory.
    assert detector.estimation == 0.25
    assert detector.variance == 0.0


@pytest.mark.parametrize("max_buckets", [1, 2, 5])
@pytest.mark.parametrize("limit", [127, 128, 129])
def test_adwin_hard_bound_survives_repeated_bucket_expiry(
    max_buckets: int, limit: int
) -> None:
    detector = ADWIN(max_buckets=max_buckets, max_window_size=limit)
    values = [0.25 + 0.01 * (index % 7) for index in range(20_000)]
    for index, value in enumerate(values):
        detector.update(value)
        assert 0 < detector.width <= limit
        assert len(detector._bucket_count) <= limit.bit_length()
        assert all(size <= max_buckets for size in detector._bucket_count)
        if index % 127 == 0:
            recent = values[index + 1 - detector.width : index + 1]
            assert detector.estimation == pytest.approx(np.mean(recent), abs=1e-12)
            assert detector.variance == pytest.approx(np.var(recent), abs=1e-12)
    detector.reset()
    assert detector.width == 0 and detector.max_window_size == limit


def test_bounded_adwin_still_detects_drift() -> None:
    detector = ADWIN(clock=1, max_window_size=128)
    for _ in range(5_000):
        detector.update(0.0)
    assert any(detector.update(10.0).drift_detected for _ in range(128))


def test_kswin_window_stays_bounded_through_repeated_drift() -> None:
    detector = KSWIN(window_size=32, stat_size=10, seed=4)
    for index in range(5_000):
        detector.update(float((index // 100) % 2))
        assert len(detector._window) <= 32
    assert detector.n_detections > 10
    detector.reset()
    assert not detector._window and detector.n_detections == 0


def test_page_hinkley_does_not_overflow_a_constant_signals_lifetime_sum() -> None:
    detector = PageHinkley()
    for _ in range(1_000):
        detector.update(1e308)
        assert detector.mean == 1e308
        assert math.isfinite(detector._sum_up)
        assert math.isfinite(detector._sum_down)
        assert not detector.drift_detected


@pytest.mark.parametrize(
    "factory",
    [MinMaxScaler, StandardScaler, lambda: RollingRobustScaler(window_size=17)],
)
def test_schema_guard_bounds_feature_state_through_novel_feature_rejections(
    factory,
) -> None:
    scaler = factory()
    guarded = FeatureSchemaGuard(features=["x", "y"]) | scaler
    for index in range(5_000):
        sample = {"x": float(index % 19), "y": float(index % 23)}
        guarded.learn_one(sample)
        assert all(
            math.isfinite(value) for value in guarded.transform_one(sample).values()
        )
        with pytest.raises(ValidationError):
            guarded.learn_one({f"novel_{index}": 1.0})
    if isinstance(scaler, RollingRobustScaler):
        assert scaler.sample_counts == {"x": 17, "y": 17}
    elif isinstance(scaler, StandardScaler):
        assert (
            set(scaler.counts)
            == set(scaler.means)
            == set(scaler.sum_sq_diffs)
            == {"x", "y"}
        )
    else:
        assert set(scaler.min) == set(scaler.max) == {"x", "y"}


@pytest.mark.parametrize("forgetting_factor", [None, 0.05])
def test_pca_storage_and_components_stay_stable_after_long_warmup(
    forgetting_factor,
) -> None:
    pca = IncrementalPCA(n_components=2, n0=8, forgetting_factor=forgetting_factor)
    projection = RandomProjection(n_components=2, seed=4)
    rng = np.random.default_rng(4)
    for _ in range(10_000):
        sample = dict(zip(("x", "y", "z"), rng.normal(size=3), strict=True))
        pca.learn_one(sample)
        projection.learn_one(sample)
        assert all(math.isfinite(value) for value in pca.transform_one(sample).values())
    assert pca.window == []
    assert pca.values.shape == (2,) and pca.vectors.shape == (3, 2)
    np.testing.assert_allclose(pca.vectors.T @ pca.vectors, np.eye(2), atol=1e-8)
    assert projection.random_matrix.shape == (3, 2)


@pytest.mark.parametrize("parameter", ["n0", "n_components"])
@pytest.mark.parametrize("value", [True, 2.5, math.inf, math.nan])
def test_pca_capacity_parameters_cannot_disable_warmup_completion(
    parameter: str,
    value: object,
) -> None:
    params = {"n_components": 1, parameter: value}
    with pytest.raises(ValueError, match="positive integer"):
        IncrementalPCA(**params)


@pytest.mark.parametrize("metrics", [False, True])
def test_evaluator_and_pipeline_retained_memory_plateau(metrics: bool) -> None:
    if metrics:
        pytest.importorskip("sklearn")
    scaler = RollingRobustScaler(window_size=17)
    model = (
        FeatureSchemaGuard(features=["x"])
        | scaler
        | MovingAverage(window_size=17, key="x")
    )
    evaluator = PrequentialEvaluator(
        model, warmup=17, metrics=metrics, metric_window_size=31, measure_time=False
    )
    # Exclude imports and initial allocations; compare live traced bytes rather
    # than RSS, whose allocator high-water marks do not measure retained state.
    for index in range(1_000):
        evaluator.update({"x": float(index % 19)}, index % 2)
    tracemalloc.start()
    try:
        for index in range(5_000):
            evaluator.update({"x": float(index % 19)}, index % 2)
        gc.collect()
        earlier = tracemalloc.get_traced_memory()[0]
        for index in range(20_000):
            evaluator.update({"x": float(index % 19)}, index % 2)
        gc.collect()
        later = tracemalloc.get_traced_memory()[0]
        assert later - earlier < 32_768
    finally:
        tracemalloc.stop()
    assert len(evaluator._ranking.pairs) == (31 if metrics else 0)
    assert evaluator.result().n_seen == 26_000
    assert scaler.sample_counts == {"x": 17}


def test_evaluator_can_stop_and_resume_an_infinite_stream() -> None:
    source = _InfiniteStream()
    evaluator = PrequentialEvaluator(NullModel(), measure_time=False)
    records = evaluator.iter_evaluate(source.stream())
    for record in islice(records, 10_000):
        assert record.index == source.consumed - 1
    records.close()
    assert source.closed
    evaluator.evaluate(islice(_InfiniteStream().stream(), 1_000))
    assert evaluator.result().n_seen == 11_000
