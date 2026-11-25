"""Select and Rename: stateless feature operations."""

from typing import Any

import pytest

from aberrant.exceptions import ValidationError
from aberrant.transform import Rename, Select


def test_select_keeps_the_configured_order_and_drops_the_rest() -> None:
    select = Select(features=["b", "a"])

    assert list(select.transform_one({"a": 1, "b": 2.0, "c": 3.0})) == ["b", "a"]
    assert select.transform_one({"a": 1, "b": 2.0}) == {"b": 2.0, "a": 1.0}
    with pytest.raises(ValidationError, match=r"Missing selected features: \['b'\]"):
        select.learn_one({"a": 1.0})


def test_rename_keeps_order_and_rejects_collisions() -> None:
    rename = Rename(mapping={"a": "x"})

    assert list(rename.transform_one({"b": 2.0, "a": 1.0})) == ["b", "x"]
    with pytest.raises(ValidationError, match=r"Missing renamed features: \['a'\]"):
        rename.transform_one({"b": 2.0})
    with pytest.raises(ValidationError, match="duplicate feature 'x'"):
        rename.learn_one({"x": 0.0, "a": 1.0})


@pytest.mark.parametrize(
    ("component", "kwargs", "message"),
    [
        (Select, {"features": "ab"}, "not a string"),
        (Select, {"features": b"ab"}, "not a string"),
        (Select, {"features": []}, "non-empty feature names"),
        (Select, {"features": ["a", ""]}, "non-empty feature names"),
        (Select, {"features": ["a", 1]}, "non-empty feature names"),
        (Select, {"features": ["a", "a"]}, "duplicates"),
        (Rename, {"mapping": {"a": ""}}, "non-empty feature names"),
        (Rename, {"mapping": {"": "a"}}, "non-empty feature names"),
        (Rename, {"mapping": {1: "a"}}, "non-empty feature names"),
        (Rename, {"mapping": {"a": None}}, "non-empty feature names"),
    ],
)
def test_rejects_invalid_configuration(
    component: type, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        component(**kwargs)
