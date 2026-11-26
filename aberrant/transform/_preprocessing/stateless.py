"""Stateless feature maps: a signed logarithm and calendar cycles."""

from __future__ import annotations

import calendar
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, tzinfo
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aberrant._utils.validation import coerce_event_number, coerce_feature_values
from aberrant.base.transformer import BaseTransformer
from aberrant.exceptions import ValidationError

Cycle = Literal["day", "week", "year"]
_CYCLES = ("day", "week", "year")


class _Stateless(BaseTransformer):
    @contextmanager
    def learning_transaction(self, x: dict[str, float]) -> Iterator[None]:
        """A stateless transformer needs no restoration."""
        yield

    def learn_one(self, x: dict[str, float]) -> None:
        """Check the event; nothing is learned."""
        self.transform_one(x)


class SignedLog(_Stateless):
    """Compress heavy tails: ``sign(x) * log(1 + |x|)``.

    Counts, sizes and latencies often span orders of magnitude, which lets a
    few large values dominate a distance or a split. The signed logarithm
    keeps the order of the values and their sign, is defined everywhere and
    maps 0 to 0.

    Args:
        apply_to: Features to transform; ``None`` uses every feature.
    """

    def __init__(self, *, apply_to: Sequence[str] | None = None) -> None:
        if isinstance(apply_to, str | bytes):
            raise ValueError("apply_to must be a sequence of names, not a string")
        names = None if apply_to is None else list(apply_to)
        if names is not None and (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError("apply_to must be distinct, non-empty feature names")
        self.apply_to = names

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """The event with each selected value on the signed log scale."""
        values = coerce_feature_values(x)
        selected = values.keys() if self.apply_to is None else set(self.apply_to)
        missing = selected - values.keys()
        if missing:
            raise ValidationError(
                f"Missing features named in apply_to: {sorted(missing)}"
            )
        return {
            name: math.copysign(math.log1p(abs(value)), value)
            if name in selected
            else value
            for name, value in values.items()
        }


class CalendarFeatures(_Stateless):
    """Replace a timestamp by its position in the day, week and year.

    The ``timestamp`` feature, in seconds since the Unix epoch, becomes
    ``{timestamp}_{cycle}_sin`` and ``{timestamp}_{cycle}_cos`` for each
    cycle, appended after the other features. Sine and cosine put the end of
    a cycle next to its start, so 23:59 is close to 00:00. ``"day"`` follows
    the local time of day, ``"week"`` the weekday from Monday and ``"year"``
    the day of the year, all in ``timezone``, so a pattern that follows the
    local clock keeps its place across daylight saving time. A detector
    then learns what is usual at each time instead of seeing a timestamp
    that never repeats. Other features pass through unchanged.

    Args:
        timestamp: The timestamp feature.
        cycles: The cycles to add, each at most once.
        timezone: An IANA time zone name, such as ``"Europe/Berlin"``, or a
            ``tzinfo``. Names other than ``"UTC"`` need the system's time
            zone database or the ``tzdata`` package.
    """

    def __init__(
        self,
        *,
        timestamp: str = "t",
        cycles: Sequence[Cycle] = ("day", "week"),
        timezone: str | tzinfo = "UTC",
    ) -> None:
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError("timestamp must be a non-empty feature name")
        if isinstance(cycles, str | bytes) or not isinstance(cycles, Sequence):
            raise ValueError("cycles must be a sequence of cycles, not a string")
        if (
            not cycles
            or any(cycle not in _CYCLES for cycle in cycles)
            or len(set(cycles)) != len(cycles)
        ):
            raise ValueError(f"cycles must be distinct ones of {', '.join(_CYCLES)}")
        self.timestamp = timestamp
        self.cycles = list(cycles)
        self.timezone = timezone
        self._zone = _zone(timezone)

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """The other features, then the timestamp's cycle positions."""
        if self.timestamp not in x:
            raise ValidationError(f"Missing timestamp feature '{self.timestamp}'")
        seconds = coerce_event_number(
            x[self.timestamp], label=f"Feature '{self.timestamp}'"
        )
        try:
            moment = datetime.fromtimestamp(seconds, self._zone)
        except (OverflowError, OSError, ValueError) as error:
            raise ValidationError(
                f"Feature '{self.timestamp}' is not a representable time"
            ) from error
        result = {name: value for name, value in x.items() if name != self.timestamp}
        for cycle in self.cycles:
            angle = 2 * math.pi * _phase(moment, cycle)
            for part, value in (("sin", math.sin(angle)), ("cos", math.cos(angle))):
                name = f"{self.timestamp}_{cycle}_{part}"
                if name in result:
                    raise ValidationError(f"Features already exist: ['{name}']")
                result[name] = value
        return result


def _zone(value: str | tzinfo) -> tzinfo:
    if isinstance(value, tzinfo):
        return value
    if not isinstance(value, str) or not value:
        raise ValueError("timezone must be a time zone name or a tzinfo")
    if value == "UTC":
        return UTC
    try:
        return ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(
            f"Unknown time zone '{value}'; install tzdata if the system has none"
        ) from error


def _phase(moment: datetime, cycle: str) -> float:
    """How far into the cycle the moment is, from 0 up to 1."""
    day = (
        moment.hour * 3600
        + moment.minute * 60
        + moment.second
        + moment.microsecond / 1e6
    ) / 86400
    if cycle == "day":
        return day
    if cycle == "week":
        return (moment.weekday() + day) / 7
    days = 366 if calendar.isleap(moment.year) else 365
    return (moment.timetuple().tm_yday - 1 + day) / days
