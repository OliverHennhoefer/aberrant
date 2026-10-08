"""Retained-state and accelerated numeric-aging regressions for stream models."""

from __future__ import annotations

import gc
import math
import pickle
import sys
from collections import deque
from collections.abc import Callable

import numpy as np
import pytest

from aberrant.base.model import BaseModel
from aberrant.model.graph import ISCONNA, MIDAS, AnoEdgeL, SignedGraphSketchDetector
from aberrant.model.iforest import StreamRandomHistogramForest, XStream
from aberrant.model.sketch import MStream, StreamingLODA, StreamingRSHash
from aberrant.model.svm import (
    GraphGatedOneClassSVM,
    IncrementalOneClassSVMAdaptiveKernel,
)


def _retained_bytes(value: object, seen: set[int] | None = None) -> int:
    """Measure live model-owned Python containers and NumPy buffers once."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + (
            _retained_bytes(value.base, seen) if value.base is not None else 0
        )
    if isinstance(value, memoryview):
        return size + _retained_bytes(value.obj, seen)
    if isinstance(value, dict):
        return size + sum(
            _retained_bytes(key, seen) + _retained_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple | set | deque):
        return size + sum(_retained_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        size += _retained_bytes(vars(value), seen)
    for slot in getattr(type(value), "__slots__", ()):
        if hasattr(value, slot):
            size += _retained_bytes(getattr(value, slot), seen)
    return size


@pytest.mark.parametrize("owner_kind", ["ndarray", "buffer"])
def test_retained_bytes_counts_storage_pinned_by_tiny_numpy_views(owner_kind):
    if owner_kind == "buffer":
        owner = bytearray(1_000_000)
        view = np.frombuffer(owner, dtype=np.uint8)[:1]
    else:
        owner = np.zeros(1_000_000, dtype=np.uint8)
        view = owner[:1]

    assert view.nbytes == 1
    assert _retained_bytes(view) >= 1_000_000


def test_retained_bytes_counts_shared_numpy_storage_only_once():
    owner = np.zeros(1_000_000, dtype=np.uint8)
    first, second = owner[:1], owner[-1:]
    views = [first, second]

    expected = sum(sys.getsizeof(value) for value in (views, first, second, owner))
    assert _retained_bytes(views) == expected


_FACTORIES: list[tuple[str, Callable[[], BaseModel], str]] = [
    (
        "midas-relational",
        lambda: MIDAS(count_min_cols=32, time_key=None, seed=5),
        "edge",
    ),
    (
        "midas-edge",
        lambda: MIDAS(count_min_cols=32, time_key=None, use_relational=False, seed=5),
        "edge",
    ),
    (
        "isconna-endpoints",
        lambda: ISCONNA(count_min_cols=32, time_key=None, seed=5),
        "edge",
    ),
    (
        "isconna-edge",
        lambda: ISCONNA(
            count_min_cols=32, time_key=None, include_endpoints=False, seed=5
        ),
        "edge",
    ),
    (
        "anoedge",
        lambda: AnoEdgeL(
            count_min_rows=8,
            count_min_cols=8,
            num_hashes=2,
            num_dense_submatrices=2,
            time_key=None,
            seed=5,
        ),
        "edge",
    ),
    (
        "signed-graph-churn",
        lambda: SignedGraphSketchDetector(
            sketch_dim=16,
            shingle_size=3,
            max_graphs=8,
            num_clusters=2,
            warm_up_graphs=2,
            time_key=None,
            seed=5,
        ),
        "graph",
    ),
    (
        "mstream",
        lambda: MStream(rows=2, buckets=32, categorical_features=("category",), seed=5),
        "categorical",
    ),
    (
        "rshash-decay",
        lambda: StreamingRSHash(
            components_num=3, hash_num=2, bins=32, warm_up_samples=4, seed=5
        ),
        "numeric",
    ),
    (
        "rshash-cumulative",
        lambda: StreamingRSHash(
            components_num=3, hash_num=2, bins=32, warm_up_samples=4, decay=0.0, seed=5
        ),
        "numeric",
    ),
    (
        "loda-decay",
        lambda: StreamingLODA(
            n_projections=3, n_bins=8, warm_up_samples=8, decay=0.9, seed=5
        ),
        "numeric",
    ),
    (
        "loda-cumulative",
        lambda: StreamingLODA(n_projections=3, n_bins=8, warm_up_samples=8, seed=5),
        "numeric",
    ),
    (
        "graph-gated-svm-cycle",
        lambda: GraphGatedOneClassSVM(graph={0: [1], 1: [2], 2: [0]}, threshold=-1.0),
        "numeric",
    ),
    (
        "adaptive-svm",
        lambda: IncrementalOneClassSVMAdaptiveKernel(
            buffer_size=16, sv_budget=8, seed=5
        ),
        "numeric",
    ),
]


def _sample(index: int, kind: str) -> dict[str, float]:
    if kind == "edge":
        # Every event has novel endpoints; sketches must never retain ID maps.
        return {"src": float(index), "dst": float(index + 1)}
    if kind == "graph":
        # Some graph IDs recur and others churn beyond the eviction budget.
        return {
            "graph": float(index // 4),
            "src": float(index),
            "dst": float(index + 1),
        }
    sample = {"x": float(index % 29), "y": math.sin(index / 7)}
    if kind == "categorical":
        sample["y"] += 1.1
        sample["category"] = float(index)
    return sample


@pytest.mark.parametrize(
    ("factory", "kind"),
    [(factory, kind) for _name, factory, kind in _FACTORIES],
    ids=[name for name, _factory, _kind in _FACTORIES],
)
def test_long_stream_retained_state_and_nonmutating_scoring(factory, kind):
    model = factory()
    for index in range(1000):
        sample = _sample(index, kind)
        assert math.isfinite(model.score_one(sample))
        model.learn_one(sample)
    gc.collect()
    initial_bytes = _retained_bytes(model)

    for index in range(1000, 5000):
        sample = _sample(index, kind)
        assert math.isfinite(model.score_one(sample))
        model.learn_one(sample)
    gc.collect()
    # Allow bounded dictionary/deque allocation changes and larger scalar ints.
    # Retaining the extra 4,000 event dictionaries would far exceed this limit.
    assert _retained_bytes(model) <= initial_bytes + 8192
    query = _sample(5000, kind)
    before = pickle.dumps(model)
    first = model.score_one(query)
    assert model.score_one(query) == first
    assert pickle.dumps(model) == before


@pytest.mark.parametrize("use_relational", [False, True])
def test_midas_handles_large_time_span_without_intermediate_overflow(use_relational):
    model = MIDAS(count_min_cols=8, use_relational=use_relational, seed=3)
    model.learn_one({"src": 1, "dst": 2, "t": 0})
    score = model.score_one({"src": 1, "dst": 2, "t": 10**200})
    assert math.isfinite(score)
    expected = (1.5 if use_relational else 1.0) ** 2 / 2.0 * 1e200
    assert score == pytest.approx(expected)


def test_midas_can_hash_integer_identifiers_beyond_signed_int64():
    model = MIDAS(count_min_cols=8, time_key=None, seed=3)
    for source in (2**63 - 1, 2**63, 2**64, -(2**63) - 1, 2**200):
        sample = {"src": source, "dst": -source}
        assert math.isfinite(model.score_one(sample))
        model.learn_one(sample)
    assert model.n_samples_seen == 5
    assert model._edge_payload(2**64, 3) != model._edge_payload(2**64 + 1, 3)
    # Arbitrary variable-width payload bytes must not alias the old int64
    # encoding, even when their total payload length happens to match.
    assert model._edge_payload(1, 2**96).startswith(b"E")
    assert model._edge_payload(1, 2).startswith(b"e")


def test_midas_large_cumulative_counts_do_not_overflow_time_scaled_square():
    score = MIDAS._compute_score(1e10, 1e11, 10**149)
    assert score == pytest.approx(1e158)


def test_mstream_log_score_survives_unrepresentable_raw_anomaly():
    model = MStream(rows=1, buckets=8, time_key="t", seed=3)
    model.learn_one({"x": 1.0, "y": 1.0, "t": 0})
    # The sum of the three raw statistics exceeds float64 even though the
    # returned logarithm is a perfectly representable finite number.
    score = model.score_one({"x": 1.0, "y": 1.0, "t": 10**400})
    expected = 400.0 * math.log(10.0) + math.log(3 * 1.6**2 / 2.0)
    assert score == pytest.approx(expected)


@pytest.mark.parametrize("category", [1e20, -1e20, 1e100, -1e100])
def test_mstream_large_categorical_ids_do_not_overflow_int64(category):
    model = MStream(rows=2, buckets=31, categorical_features=("category",), seed=3)
    sample = {"x": 1.0, "category": category}
    with np.errstate(all="raise"):
        model.learn_one(sample)
        assert math.isfinite(model.score_one(sample))
        model.learn_one(sample)


def test_isconna_g_test_handles_large_time_span():
    actual = ISCONNA._g_test(4.0, 8.0, 10**400)
    expected = 8.0 * (math.log(4.0) + 400 * math.log(10.0) - math.log(8.0))
    assert actual == pytest.approx(expected)


def test_isconna_pattern_counters_never_wrap_negative():
    model = ISCONNA(count_min_rows=1, count_min_cols=2, include_endpoints=False)
    group = model._edge
    largest = np.iinfo(np.int64).max
    group.width_time.fill(largest)
    group.gap_time.fill(largest)
    model._record_observation(group)
    assert np.all(group.width_time == largest)
    group.busy_current.fill(False)
    group.busy_previous.fill(True)
    model._reset_group(group)
    assert np.all(group.gap_time == largest)


def test_rshash_sample_normalization_accepts_counters_beyond_uint64():
    model = StreamingRSHash(
        components_num=2, hash_num=2, bins=8, warm_up_samples=1, seed=3
    )
    model.learn_one({"x": 1.0})
    model._samples_seen = 2**64
    assert math.isfinite(model.score_one({"x": 1.0}))
    model.learn_one({"x": 1.0})
    assert model.n_samples_seen == 2**64 + 1


@pytest.mark.parametrize("decay", [0.01, 5e-324])
def test_rshash_large_elapsed_time_forgets_old_counts_without_overflow(decay):
    model = StreamingRSHash(
        components_num=2,
        hash_num=2,
        bins=8,
        warm_up_samples=1,
        time_key="t",
        decay=decay,
        seed=3,
    )
    model.learn_one({"x": 1.0, "t": 0})
    assert math.isfinite(model.score_one({"x": 1.0, "t": 10**400}))
    model.learn_one({"x": 1.0, "t": 10**400})
    assert model._scale == 1.0
    assert np.sum(model._counts) == 4.0


@pytest.mark.parametrize(
    "capacity", [math.inf, -math.inf, math.nan, 2.5, 3.0, True, False]
)
@pytest.mark.parametrize(
    ("model_type", "parameter"),
    [
        (StreamingLODA, "warm_up_samples"),
        (XStream, "window_size"),
        (XStream, "init_sample_size"),
        (XStream, "max_feature_cache_size"),
        (StreamRandomHistogramForest, "window_size"),
        (SignedGraphSketchDetector, "max_graphs"),
        (IncrementalOneClassSVMAdaptiveKernel, "sv_budget"),
        (IncrementalOneClassSVMAdaptiveKernel, "buffer_size"),
    ],
)
def test_invalid_capacities_cannot_disable_retention_bounds(
    model_type, parameter, capacity
):
    with pytest.raises(ValueError, match=parameter):
        model_type(**{parameter: capacity})


def test_numpy_integer_capacities_keep_bounded_models_working():
    capacity = np.int64(4)
    models = [
        StreamingLODA(n_projections=2, warm_up_samples=capacity),
        XStream(
            k=2,
            n_chains=2,
            depth=2,
            cms_width=8,
            window_size=capacity,
            init_sample_size=capacity,
            max_feature_cache_size=capacity,
        ),
        StreamRandomHistogramForest(n_estimators=2, max_depth=2, window_size=capacity),
        IncrementalOneClassSVMAdaptiveKernel(buffer_size=capacity, sv_budget=capacity),
    ]
    for model in models:
        for index in range(20):
            sample = {"x": float(index % 3)}
            model.learn_one(sample)
            assert math.isfinite(model.score_one(sample))
