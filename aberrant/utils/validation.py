"""Stateful input-boundary validation for streaming models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

NumericScalar: TypeAlias = int | float | np.number


def coerce_finite_number(value: object, *, label: str) -> float:
    """Return a finite float or raise a consistently worded error."""
    if not isinstance(value, int | float | np.number):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def coerce_integer_feature(value: object, key: str) -> int:
    """Return an integer-like finite feature value."""
    as_float = coerce_finite_number(value, label=f"Feature '{key}'")
    as_int = int(round(as_float))
    if not np.isclose(as_float, float(as_int), rtol=0.0, atol=1e-9):
        raise ValueError(f"Feature '{key}' must be integer-like")
    return as_int


@dataclass(frozen=True, slots=True)
class PreparedFeatures:
    """Validated feature vector awaiting an optional schema commit."""

    names: tuple[str, ...]
    values: np.ndarray
    revision: int


class FeatureSchema:
    """Own a stable feature order with explicit preview and commit operations."""

    def __init__(
        self,
        names: Sequence[str] | None = None,
        *,
        expected_size: int | None = None,
    ) -> None:
        if expected_size is not None and expected_size <= 0:
            raise ValueError("expected_size must be positive or None")
        resolved_names = None if names is None else tuple(names)
        if resolved_names is not None:
            if not resolved_names:
                raise ValueError("Feature schema cannot be empty")
            if any(not isinstance(name, str) or not name for name in resolved_names):
                raise ValueError("Feature names must be non-empty strings")
            if len(set(resolved_names)) != len(resolved_names):
                raise ValueError("Feature names cannot contain duplicates")
            if expected_size is not None and len(resolved_names) != expected_size:
                raise ValueError(
                    f"Expected {expected_size} feature names, got {len(resolved_names)}"
                )

        self._names = resolved_names
        self._expected_size = expected_size
        self._revision = 0

    @property
    def names(self) -> tuple[str, ...] | None:
        """Return the committed feature order, if one exists."""
        return self._names

    @property
    def is_established(self) -> bool:
        """Return whether learning has committed a schema."""
        return self._names is not None

    def preview(self, features: Mapping[str, object]) -> PreparedFeatures:
        """Validate and vectorize features without mutating schema state."""
        if not features:
            raise ValueError("Input dictionary cannot be empty")

        validated: dict[str, float] = {}
        for key, value in features.items():
            if not isinstance(key, str):
                raise ValueError("All feature keys must be strings")
            validated[key] = coerce_finite_number(
                value,
                label=f"Feature '{key}'",
            )

        names = self._names or tuple(sorted(features))
        if self._expected_size is not None and len(features) != self._expected_size:
            raise ValueError(
                f"Expected {self._expected_size} features, got {len(features)}"
            )
        if set(features) != set(names):
            expected_keys = ", ".join(names)
            received_keys = ", ".join(sorted(features))
            raise ValueError(
                "Inconsistent feature keys. "
                f"Expected [{expected_keys}], received [{received_keys}]."
            )

        values = np.fromiter(
            (validated[name] for name in names),
            dtype=np.float64,
            count=len(names),
        )
        return PreparedFeatures(names=names, values=values, revision=self._revision)

    def commit(self, prepared: PreparedFeatures) -> None:
        """Commit a previously previewed schema after successful learning."""
        if prepared.revision != self._revision:
            raise RuntimeError("Cannot commit a stale feature preview")
        if self._names is None:
            self._names = prepared.names
        elif self._names != prepared.names:
            raise RuntimeError("Prepared feature schema no longer matches model state")
        self._revision += 1


@dataclass(frozen=True, slots=True)
class PreparedTimestamp:
    """Validated timestamp awaiting an optional clock commit."""

    value: float
    implicit: bool
    arrival_index: int
    revision: int


class MonotonicClock:
    """Own implicit arrivals and explicit monotonic timestamp state."""

    def __init__(self, *, integer_like: bool = False) -> None:
        self._integer_like = integer_like
        self._arrival_index = 0
        self._max_time = float("-inf")
        self._revision = 0

    @property
    def arrival_index(self) -> int:
        """Return the number of committed implicit arrivals."""
        return self._arrival_index

    @property
    def max_time(self) -> float:
        """Return the greatest committed timestamp."""
        return self._max_time

    def preview(
        self,
        value: object | None = None,
        *,
        implicit: bool,
    ) -> PreparedTimestamp:
        """Resolve and validate a timestamp without advancing the clock."""
        arrival_index = self._arrival_index + 1 if implicit else self._arrival_index
        if implicit:
            current_time = float(arrival_index)
        else:
            current_time = coerce_finite_number(value, label="Timestamp value")
            if self._integer_like:
                rounded = int(round(current_time))
                if not np.isclose(
                    current_time,
                    float(rounded),
                    rtol=0.0,
                    atol=1e-9,
                ):
                    raise ValueError("Timestamp value must be integer-like")
                current_time = float(rounded)

        if current_time < self._max_time:
            raise ValueError(
                f"Non-monotonic timestamp: received {current_time:g}, "
                f"current {self._max_time:g}"
            )
        return PreparedTimestamp(
            value=current_time,
            implicit=implicit,
            arrival_index=arrival_index,
            revision=self._revision,
        )

    def commit(self, prepared: PreparedTimestamp) -> None:
        """Advance the clock using a previously previewed timestamp."""
        if prepared.revision != self._revision:
            raise RuntimeError("Cannot commit a stale timestamp preview")
        if prepared.value < self._max_time:
            raise RuntimeError("Prepared timestamp no longer matches clock state")
        if prepared.implicit:
            self._arrival_index = prepared.arrival_index
        self._max_time = prepared.value
        self._revision += 1


@dataclass(frozen=True, slots=True)
class PreparedNumericEvent:
    """A validated timestamped feature vector awaiting commit."""

    timestamp: PreparedTimestamp
    features: PreparedFeatures


class NumericEventBoundary:
    """Compose a feature schema and clock for numeric stream samples."""

    def __init__(
        self,
        *,
        time_key: str | None = None,
        integer_time: bool = False,
        expected_size: int | None = None,
    ) -> None:
        if time_key is not None and (not isinstance(time_key, str) or not time_key):
            raise ValueError("time_key must be a non-empty string or None")
        self.time_key = time_key
        self.schema = FeatureSchema(expected_size=expected_size)
        self.clock = MonotonicClock(integer_like=integer_time)

    def preview(self, sample: Mapping[str, object]) -> PreparedNumericEvent:
        """Validate and vectorize a sample without mutating boundary state."""
        if not sample:
            raise ValueError("Input dictionary cannot be empty")
        if any(not isinstance(key, str) for key in sample):
            raise ValueError("All feature keys must be strings")

        if self.time_key is None:
            timestamp = self.clock.preview(implicit=True)
            features = sample
        else:
            if self.time_key not in sample:
                raise ValueError(f"Missing time_key '{self.time_key}' in input sample")
            timestamp = self.clock.preview(sample[self.time_key], implicit=False)
            features = {
                name: value for name, value in sample.items() if name != self.time_key
            }
            if not features:
                raise ValueError("Input must contain at least one non-time feature")

        return PreparedNumericEvent(
            timestamp=timestamp,
            features=self.schema.preview(features),
        )

    def commit(self, prepared: PreparedNumericEvent) -> None:
        """Commit schema and clock state after successful model learning."""
        self.schema.commit(prepared.features)
        self.clock.commit(prepared.timestamp)


@dataclass(frozen=True, slots=True)
class PreparedEdgeEvent:
    """Validated integer edge event awaiting a clock commit."""

    timestamp: PreparedTimestamp
    source: int
    destination: int

    @property
    def bucket(self) -> int:
        """Return the event timestamp as an integer bucket."""
        return int(self.timestamp.value)


class EdgeEventBoundary:
    """Own field validation and monotonic time for integer edge streams."""

    def __init__(
        self,
        *,
        source_key: str,
        destination_key: str,
        time_key: str | None,
    ) -> None:
        if not isinstance(source_key, str) or not source_key:
            raise ValueError("source_key must be a non-empty string")
        if not isinstance(destination_key, str) or not destination_key:
            raise ValueError("destination_key must be a non-empty string")
        if source_key == destination_key:
            raise ValueError("source_key and destination_key must be different")
        if time_key is not None and (not isinstance(time_key, str) or not time_key):
            raise ValueError("time_key must be a non-empty string or None")
        if time_key is not None and time_key in (source_key, destination_key):
            raise ValueError(
                "time_key must be different from source_key and destination_key"
            )

        self.source_key = source_key
        self.destination_key = destination_key
        self.time_key = time_key
        self.clock = MonotonicClock(integer_like=True)

    def preview(self, sample: Mapping[str, object]) -> PreparedEdgeEvent:
        """Validate an edge event without advancing its clock."""
        if not sample:
            raise ValueError("Input dictionary cannot be empty")
        if self.source_key not in sample:
            raise ValueError(f"Missing source_key '{self.source_key}' in input sample")
        if self.destination_key not in sample:
            raise ValueError(
                f"Missing destination_key '{self.destination_key}' in input sample"
            )

        source = coerce_integer_feature(sample[self.source_key], self.source_key)
        destination = coerce_integer_feature(
            sample[self.destination_key], self.destination_key
        )
        if self.time_key is None:
            timestamp = self.clock.preview(implicit=True)
        else:
            if self.time_key not in sample:
                raise ValueError(f"Missing time_key '{self.time_key}' in input sample")
            timestamp = self.clock.preview(sample[self.time_key], implicit=False)
        return PreparedEdgeEvent(
            timestamp=timestamp,
            source=source,
            destination=destination,
        )

    def commit(self, prepared: PreparedEdgeEvent) -> None:
        """Advance event time after successful model learning."""
        self.clock.commit(prepared.timestamp)
