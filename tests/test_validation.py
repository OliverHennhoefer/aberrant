"""Tests for stateful feature and event boundaries."""

from collections import defaultdict
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from aberrant.utils.validation import (
    EdgeEventBoundary,
    FeatureSchema,
    MonotonicClock,
    NumericEventBoundary,
    PreparedFeatures,
    PreparedNumericEvent,
)


def test_feature_schema_preview_is_non_mutating_and_commit_locks_order() -> None:
    schema = FeatureSchema()

    candidate = schema.preview({"z": 3.0, "a": 1.0})

    assert schema.names is None
    assert candidate.names == ("a", "z")
    assert candidate.values.tolist() == [1.0, 3.0]

    schema.commit(candidate)
    assert schema.names == ("a", "z")
    reordered = schema.preview({"a": 4.0, "z": 5.0})
    assert reordered.values.tolist() == [4.0, 5.0]


def test_feature_schema_rejects_invalid_and_stale_candidates_atomically() -> None:
    schema = FeatureSchema(expected_size=2)

    with pytest.raises(ValueError, match="must be finite"):
        schema.preview({"a": 1.0, "b": float("nan")})
    assert schema.names is None

    first = schema.preview({"a": 1.0, "b": 2.0})
    stale = schema.preview({"a": 3.0, "b": 4.0})
    schema.commit(first)
    with pytest.raises(RuntimeError, match="stale"):
        schema.commit(stale)


def test_monotonic_clock_only_advances_on_commit() -> None:
    clock = MonotonicClock()

    first_preview = clock.preview(implicit=True)
    repeated_preview = clock.preview(implicit=True)
    assert first_preview.value == repeated_preview.value == 1.0
    assert clock.arrival_index == 0

    clock.commit(first_preview)
    assert clock.arrival_index == 1
    assert clock.preview(implicit=True).value == 2.0


def test_explicit_clock_allows_future_preview_without_committing_it() -> None:
    clock = MonotonicClock(integer_like=True)
    first = clock.preview(2.0, implicit=False)
    clock.commit(first)

    assert clock.preview(10.0, implicit=False).value == 10.0
    assert clock.max_time == 2.0
    assert clock.preview(3.0, implicit=False).value == 3.0
    with pytest.raises(ValueError, match="Non-monotonic"):
        clock.preview(1.0, implicit=False)
    with pytest.raises(ValueError, match="integer-like"):
        clock.preview(2.5, implicit=False)


def test_numeric_event_boundary_commits_schema_and_clock_together() -> None:
    boundary = NumericEventBoundary(time_key="t", integer_time=True)

    event = boundary.preview({"t": 4.0, "b": 2.0, "a": 1.0})
    assert boundary.schema.names is None
    assert boundary.clock.max_time == float("-inf")

    boundary.commit(event)
    assert boundary.schema.names == ("a", "b")
    assert boundary.clock.max_time == 4.0


def test_edge_event_boundary_validates_without_advancing_on_preview() -> None:
    boundary = EdgeEventBoundary(
        source_key="src",
        destination_key="dst",
        time_key=None,
    )

    event = boundary.preview({"src": 1.0, "dst": 2.0})
    assert (event.bucket, event.source, event.destination) == (1, 1, 2)
    assert boundary.clock.arrival_index == 0

    boundary.commit(event)
    assert boundary.clock.arrival_index == 1
    with pytest.raises(ValueError, match="integer-like"):
        boundary.preview({"src": 1.5, "dst": 2.0})


_SCHEMA_SAMPLES = [
    {"a": 1.0, "b": 2.0},
    {"b": -0.0, "a": 1e308},
    {"a": np.float64(1.5), "b": 2.0},
    {"a": 1, "b": 2.0},
    {"a": True, "b": 2.0},
    {"a": np.float32(0.1), "b": 2.0},
    {"a": np.int64(3), "b": 2.0},
    {"a": "1.0", "b": 2.0},
    {"a": None, "b": 2.0},
    {"a": float("nan"), "b": 2.0},
    {"a": 1.0, "b": float("inf")},
    {"a": np.float64("-inf"), "b": 2.0},
    {"a": 10**400, "b": 2.0},
    {"a": 1.0},
    {"a": 1.0, "b": 2.0, "c": 3.0},
    {"a": 1.0, "c": 3.0},
    {"a": "bad", "c": float("nan")},
    {1: 1.0, "b": 2.0},
    {},
]


def _outcome(preview: Callable[[Any], object], sample: Any) -> tuple[object, ...]:
    """Summarize a preview result or the exception it raised."""
    try:
        prepared = preview(sample)
    except Exception as exc:
        return type(exc), str(exc)
    if isinstance(prepared, PreparedNumericEvent):
        return (
            prepared.timestamp,
            prepared.features.names,
            prepared.features.values.dtype,
            prepared.features.values.tolist(),
            prepared.features.revision,
        )
    assert isinstance(prepared, PreparedFeatures)
    return (
        prepared.names,
        prepared.values.dtype,
        prepared.values.tolist(),
        prepared.revision,
    )


@pytest.mark.parametrize("sample", _SCHEMA_SAMPLES)
def test_feature_schema_fast_path_matches_complete_validation(
    sample: dict[Any, Any],
) -> None:
    schema = FeatureSchema()
    schema.commit(schema.preview({"b": 0.0, "a": 0.0}))

    # A read-only proxy is not a plain dict, so it always takes the complete path.
    fast = _outcome(schema.preview, sample)
    complete = _outcome(schema.preview, MappingProxyType(sample))

    assert fast == complete


@pytest.mark.parametrize("timestamp", [None, 6.0, 4.0, float("nan"), "x"])
@pytest.mark.parametrize("sample", _SCHEMA_SAMPLES)
def test_numeric_boundary_fast_path_matches_complete_validation(
    sample: dict[Any, Any], timestamp: object
) -> None:
    time_key = None if timestamp is None else "t"
    boundary = NumericEventBoundary(time_key=time_key)
    first = {"t": 5.0, "a": 0.0, "b": 0.0} if time_key else {"b": 0.0, "a": 0.0}
    boundary.commit(boundary.preview(first))
    event = dict(sample) if time_key is None else {**sample, time_key: timestamp}

    fast = _outcome(boundary.preview, event)
    complete = _outcome(boundary.preview, MappingProxyType(event))

    assert fast == complete


def test_feature_schema_fast_path_never_mutates_missing_key_mappings() -> None:
    schema = FeatureSchema()
    schema.commit(schema.preview({"a": 0.0, "b": 0.0}))
    sample: defaultdict[str, float] = defaultdict(float, {"a": 1.0, "c": 2.0})

    with pytest.raises(ValueError, match="Inconsistent feature keys"):
        schema.preview(sample)
    assert dict(sample) == {"a": 1.0, "c": 2.0}


def test_feature_schema_fast_path_covers_plain_float_dicts_only() -> None:
    schema = FeatureSchema()
    assert schema._well_formed_values({"a": 1.0}) is None
    schema.commit(schema.preview({"b": 0.0, "a": 0.0}))

    assert schema._well_formed_values({"b": 2.0, "a": np.float64(1.0)}) == [1.0, 2.0]
    assert schema._well_formed_values(MappingProxyType({"a": 1.0, "b": 2.0})) is None
    assert schema._well_formed_values({"a": 1, "b": 2.0}) is None
