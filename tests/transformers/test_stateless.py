"""SignedLog and CalendarFeatures: stateless maps of one event."""

import math
from datetime import UTC, datetime, timedelta, timezone

import pytest

from aberrant.exceptions import ValidationError
from aberrant.transform import CalendarFeatures, SignedLog


def test_signed_log_keeps_sign_and_order() -> None:
    transform = SignedLog(apply_to=["bytes"])
    row = transform.transform_one({"bytes": -(math.e - 1), "rate": 5.0})
    assert row == {"bytes": pytest.approx(-1.0), "rate": 5.0}
    assert SignedLog().transform_one({"a": 0.0, "b": 9.0}) == {
        "a": 0.0,
        "b": pytest.approx(math.log(10.0)),
    }


def test_signed_log_rejects_what_it_cannot_transform() -> None:
    with pytest.raises(ValidationError, match="apply_to"):
        SignedLog(apply_to=["bytes"]).transform_one({"rate": 1.0})
    with pytest.raises(ValidationError):
        SignedLog().learn_one({"a": math.nan})


def _at(*args: int, tz=UTC) -> float:
    return datetime(*args, tzinfo=tz).timestamp()


def test_calendar_replaces_the_timestamp_by_cycle_positions() -> None:
    transform = CalendarFeatures(cycles=["day", "week", "year"])
    row = transform.transform_one({"load": 2.0, "t": _at(2024, 1, 1, 6)})
    assert list(row) == [
        "load",
        "t_day_sin",
        "t_day_cos",
        "t_week_sin",
        "t_week_cos",
        "t_year_sin",
        "t_year_cos",
    ]
    assert row["load"] == 2.0
    assert (row["t_day_sin"], row["t_day_cos"]) == pytest.approx((1.0, 0.0))
    angle = 2 * math.pi * 0.25 / 7
    assert row["t_week_sin"] == pytest.approx(math.sin(angle))
    angle = 2 * math.pi * 0.25 / 366
    assert row["t_year_cos"] == pytest.approx(math.cos(angle))


def test_the_end_of_a_day_is_next_to_its_start() -> None:
    transform = CalendarFeatures(cycles=["day"])
    late = transform.transform_one({"t": _at(2024, 3, 4, 23, 59)})
    early = transform.transform_one({"t": _at(2024, 3, 5, 0, 1)})
    distance = math.dist(late.values(), early.values())
    assert distance < 0.01


def test_positions_follow_the_given_time_zone() -> None:
    plus_two = timezone(timedelta(hours=2))
    transform = CalendarFeatures(cycles=["day"], timezone=plus_two)
    row = transform.transform_one({"t": _at(2024, 6, 1, 4)})
    assert (row["t_day_sin"], row["t_day_cos"]) == pytest.approx((1.0, 0.0))


def test_calendar_rejects_a_missing_or_unusable_timestamp() -> None:
    transform = CalendarFeatures(timestamp="ts", cycles=["day"])
    for event in ({"load": 1.0}, {"ts": math.inf}, {"ts": 1e300}):
        with pytest.raises(ValidationError):
            transform.learn_one(event)
    with pytest.raises(ValidationError, match="already exist"):
        transform.transform_one({"ts": 0.0, "ts_day_sin": 1.0})


@pytest.mark.parametrize(
    "params",
    [
        {"timestamp": ""},
        {"cycles": "day"},
        {"cycles": []},
        {"cycles": ["hour"]},
        {"cycles": ["day", "day"]},
        {"timezone": "Not/AZone"},
        {"timezone": 2},
    ],
)
def test_calendar_rejects_invalid_configuration(params) -> None:
    with pytest.raises(ValueError):
        CalendarFeatures(**params)


@pytest.mark.parametrize("apply_to", ["a", [], ["a", "a"]])
def test_signed_log_rejects_invalid_configuration(apply_to) -> None:
    with pytest.raises(ValueError):
        SignedLog(apply_to=apply_to)
