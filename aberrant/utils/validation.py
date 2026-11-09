"""Stateful input-boundary validation for streaming models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

# Built once: an ``int | float | np.number`` expression creates a union per call.
_NUMERIC_TYPES = (int, float, np.number)
# Exact value types whose float value is the value itself, so a well-formed
# sample made of them needs only a finiteness check.
_PLAIN_FLOAT_TYPES = (float, np.float64)
# Largest distance to the nearest integer that still counts as integer-like.
_INTEGER_TOLERANCE = 1e-9


def coerce_integer_count(value: object, *, label: str, minimum: int = 1) -> int:
    """Validate a strict integer count and normalize it before bound arithmetic."""
    if minimum == 0:
        requirement = "non-negative integer"
    elif minimum == 1:
        requirement = "positive integer"
    else:
        requirement = f"an integer greater than {minimum - 1}"
    message = f"{label} must be {requirement}"
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(message)
    result = int(value)
    if result < minimum:
        raise ValueError(message)
    return result


def coerce_finite_number(value: object, *, label: str) -> float:
    """Return a finite float or raise a consistently worded error."""
    if not isinstance(value, _NUMERIC_TYPES):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def coerce_integer_feature(value: object, key: str) -> int:
    """Return an integer-like finite feature value."""
    if isinstance(value, int | np.integer):
        return int(value)
    as_float = coerce_finite_number(value, label=f"Feature '{key}'")
    as_int = int(round(as_float))
    if abs(as_float - float(as_int)) > _INTEGER_TOLERANCE:
        raise ValueError(f"Feature '{key}' must be integer-like")
    return as_int


def coerce_feature_values(features: Mapping[str, float]) -> dict[str, float]:
    """Validate a complete transformer sample, preserving float coercion."""
    validated = {}
    for key, value in features.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{key}' must be numeric") from exc
        validated[key] = coerce_finite_number(numeric, label=f"Feature '{key}'")
    return validated


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
        sort_names: bool = True,
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
        self._sort_names = sort_names
        self._revision = 0

    @property
    def names(self) -> tuple[str, ...] | None:
        """Return the committed feature order, if one exists."""
        return self._names

    @property
    def is_established(self) -> bool:
        """Return whether learning has committed a schema."""
        return self._names is not None

    def _well_formed_values(
        self,
        sample: Mapping[str, object],
        time_key: str | None = None,
    ) -> list[float] | None:
        """Return feature values in schema order if ``sample`` needs no coercion.

        This is the fast path for the common case once a schema is committed: a
        plain ``dict`` holding exactly the committed names, plus ``time_key``
        when one is given, whose feature values are finite built-in or NumPy
        floats. Any other input returns ``None``, and the caller then runs the
        complete validation, which coerces the values or raises the precise
        error. ``time_key`` is never a feature name because callers remove it
        from the features before preview.
        """
        names = self._names
        if names is None or type(sample) is not dict:
            return None
        if time_key is None:
            if len(sample) != len(names):
                return None
        elif len(sample) != len(names) + 1 or time_key not in sample:
            return None
        try:
            values = [sample[name] for name in names]
        except KeyError:
            return None
        for value in values:
            if type(value) not in _PLAIN_FLOAT_TYPES or not math.isfinite(value):
                return None
        return values

    def _prepare(self, values: list[float]) -> PreparedFeatures:
        """Wrap values from ``_well_formed_values`` as a committable preview."""
        assert self._names is not None
        return PreparedFeatures(
            self._names, np.array(values, dtype=np.float64), self._revision
        )

    def preview_values(self, features: Mapping[str, object]) -> list[float]:
        """Validate features like ``preview`` and return their values as a list.

        This suits callers that only read the values, such as scoring in
        pure-Python tree models, and skips building an array they would convert
        back to a list.
        """
        plain_values = self._well_formed_values(features)
        if plain_values is not None:
            return plain_values
        return cast(list[float], self.preview(features).values.tolist())

    def preview(self, features: Mapping[str, object]) -> PreparedFeatures:
        """Validate and vectorize features without mutating schema state."""
        plain_values = self._well_formed_values(features)
        if plain_values is not None:
            return self._prepare(plain_values)
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

        names = self._names or tuple(sorted(features) if self._sort_names else features)
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

    value: int | float
    implicit: bool
    arrival_index: int
    revision: int


class MonotonicClock:
    """Own implicit arrivals and explicit monotonic timestamp state."""

    def __init__(self, *, integer_like: bool = False) -> None:
        self._integer_like = integer_like
        self._arrival_index = 0
        self._max_time: int | float = float("-inf")
        self._revision = 0

    @property
    def arrival_index(self) -> int:
        """Return the number of committed implicit arrivals."""
        return self._arrival_index

    @property
    def max_time(self) -> int | float:
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
        current_time: int | float
        if implicit:
            current_time = arrival_index
        elif isinstance(value, int | np.integer):
            current_time = int(value)
        else:
            current_time = coerce_finite_number(value, label="Timestamp value")
            if self._integer_like:
                rounded = int(round(current_time))
                if abs(current_time - float(rounded)) > _INTEGER_TOLERANCE:
                    raise ValueError("Timestamp value must be integer-like")
                current_time = rounded

        if current_time < self._max_time:
            raise ValueError(
                f"Non-monotonic timestamp: received {current_time}, "
                f"current {self._max_time}"
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
        plain_values = self.schema._well_formed_values(sample, self.time_key)
        if plain_values is not None:
            # The keys are exactly the committed features and the time key, so
            # only the timestamp can still be rejected below, with the same
            # error as on the complete path.
            if self.time_key is None:
                timestamp = self.clock.preview(implicit=True)
            else:
                timestamp = self.clock.preview(sample[self.time_key], implicit=False)
            return PreparedNumericEvent(timestamp, self.schema._prepare(plain_values))

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
