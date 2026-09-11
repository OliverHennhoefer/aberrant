"""Behavioral tests for the public feature-schema boundary."""

import pytest

from aberrant.base import ValidationError
from aberrant.transform import FeatureSchemaGuard


def test_configured_schema_validates_and_orders_features() -> None:
    guard = FeatureSchemaGuard(features=["b", "a"])
    event = {"a": 1, "b": 2}

    guard.learn_one(event)

    assert guard.feature_names == ("b", "a")
    assert list(guard.transform_one(event)) == ["b", "a"]
    assert guard.transform_one(event) == {"b": 2.0, "a": 1.0}


def test_learned_schema_commits_only_after_valid_input() -> None:
    guard = FeatureSchemaGuard()

    with pytest.raises(ValidationError, match="finite"):
        guard.learn_one({"a": float("nan")})

    assert guard.feature_names is None
    guard.learn_one({"b": 2.0, "a": 1.0})
    assert guard.feature_names == ("a", "b")


def test_schema_mismatch_is_rejected_without_changing_the_schema() -> None:
    guard = FeatureSchemaGuard(features=["a", "b"])

    with pytest.raises(ValidationError, match="Inconsistent feature keys"):
        guard.learn_one({"a": 1.0, "c": 2.0})

    assert guard.feature_names == ("a", "b")


def test_reset_forgets_only_a_learned_schema() -> None:
    learned = FeatureSchemaGuard()
    configured = FeatureSchemaGuard(features=["b", "a"])
    learned.learn_one({"a": 1.0})

    learned.reset()
    configured.reset()

    assert learned.feature_names is None
    assert configured.feature_names == ("b", "a")


def test_string_is_not_accepted_as_a_feature_sequence() -> None:
    with pytest.raises(ValueError, match="not a string"):
        FeatureSchemaGuard(features="abc")
