"""Tests for stateful feature and event boundaries."""

import pytest

from aberrant.utils.validation import (
    EdgeEventBoundary,
    FeatureSchema,
    MonotonicClock,
    NumericEventBoundary,
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
