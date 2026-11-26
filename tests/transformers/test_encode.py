"""Categorical encoders: categories become numbers a detector can learn."""

import math

import pytest

from aberrant.base import Readiness
from aberrant.base.pipeline import Pipeline
from aberrant.exceptions import ValidationError
from aberrant.model.forest import HalfSpaceTrees
from aberrant.transform import FrequencyEncoder, Imputer, OneHotEncoder


def test_one_hot_keeps_the_features_and_flags_undeclared_categories() -> None:
    encoder = OneHotEncoder(categories={"proto": ["tcp", "udp"]})
    assert encoder.transform_one({"size": 3.0, "proto": "udp"}) == {
        "size": 3.0,
        "proto_tcp": 0.0,
        "proto_udp": 1.0,
        "proto_other": 0.0,
    }
    assert encoder.transform_one({"size": 3.0, "proto": "icmp"})["proto_other"] == 1.0


def test_categories_compare_as_text() -> None:
    encoder = OneHotEncoder(categories={"code": [200, "404"]})
    for value in (200, 200.0, "200"):
        assert encoder.transform_one({"code": value})["code_200"] == 1.0
    assert encoder.transform_one({"code": 404})["code_404"] == 1.0


def test_one_hot_passes_a_missing_category_on_as_nan() -> None:
    encoder = OneHotEncoder(categories={"proto": ["tcp"]})
    for event in ({"size": 1.0}, {"size": 1.0, "proto": None}, {"proto": math.nan}):
        row = encoder.transform_one(event)
        assert math.isnan(row["proto_tcp"]) and math.isnan(row["proto_other"])


def test_one_hot_rejects_clashing_names_and_non_categories() -> None:
    encoder = OneHotEncoder(categories={"proto": ["tcp"]})
    with pytest.raises(ValidationError, match="clash"):
        encoder.transform_one({"proto": "tcp", "proto_tcp": 1.0})
    for value in (1.5, math.inf, [1]):
        with pytest.raises(ValidationError, match="category"):
            encoder.learn_one({"proto": value})  # type: ignore[dict-item]


def test_frequency_is_the_share_of_learned_categories() -> None:
    encoder = FrequencyEncoder(features=["proto"])
    assert not encoder.readiness.ready
    for proto in ("tcp", "tcp", "tcp", "udp"):
        encoder.learn_one({"size": 1.0, "proto": proto})
    assert encoder.readiness == Readiness(True, 1, 1)
    assert encoder.transform_one({"size": 2.0, "proto": "tcp"}) == {
        "size": 2.0,
        "proto": 0.75,
    }
    assert encoder.transform_one({"proto": "udp"}) == {"proto": 0.25}
    assert encoder.transform_one({"proto": "icmp"}) == {"proto": 0.0}


def test_frequency_forgets_outside_the_window() -> None:
    encoder = FrequencyEncoder(features=["proto"], window_size=2)
    for proto in ("tcp", "tcp", "udp", "udp"):
        encoder.learn_one({"proto": proto})
    assert encoder.transform_one({"proto": "tcp"}) == {"proto": 0.0}
    assert encoder.transform_one({"proto": "udp"}) == {"proto": 1.0}


def test_frequency_does_not_learn_a_missing_category() -> None:
    encoder = FrequencyEncoder(features=["proto"])
    encoder.learn_one({"proto": "tcp"})
    encoder.learn_one({"proto": None})
    encoder.learn_one({})
    assert encoder.transform_one({"proto": "tcp"}) == {"proto": 1.0}
    assert math.isnan(encoder.transform_one({})["proto"])


def test_an_imputer_fills_what_the_encoder_passes_on() -> None:
    detector = (
        FrequencyEncoder(features=["proto"])
        | Imputer(strategy="constant", value=0.0, indicate=True)
        | HalfSpaceTrees(window_size=10, seed=1)
    )
    events = [{"size": 0.5, "proto": "tcp"}] * 30 + [{"size": 0.5, "proto": None}]
    for event in events:
        assert math.isfinite(detector.score_one(event))
        detector.learn_one(event)


def test_a_failed_downstream_update_restores_the_counts() -> None:
    class Failing(Imputer):
        def learn_one(self, x):
            raise RuntimeError("refused")

    encoder = FrequencyEncoder(features=["proto"], window_size=2)
    encoder.learn_one({"proto": "tcp"})
    pipeline = Pipeline(encoder, Failing())
    with pytest.raises(RuntimeError):
        pipeline.learn_one({"proto": "udp"})
    assert encoder.transform_one({"proto": "tcp"}) == {"proto": 1.0}


@pytest.mark.parametrize(
    "categories",
    [{}, {"": ["a"]}, {"p": "ab"}, {"p": []}, {"p": ["a", "a"]}, {"p": ["other"]}],
)
def test_one_hot_rejects_invalid_categories(categories) -> None:
    with pytest.raises(ValueError):
        OneHotEncoder(categories=categories)


@pytest.mark.parametrize(
    "params",
    [{"features": "p"}, {"features": []}, {"features": ["p"], "window_size": 0}],
)
def test_frequency_rejects_invalid_configuration(params) -> None:
    with pytest.raises(ValueError):
        FrequencyEncoder(**params)
