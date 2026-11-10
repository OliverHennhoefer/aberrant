"""Forecasters in the catalog, and residual detectors built from configuration."""

import json
import math

import pytest

from aberrant.base import ConfigurationError, UnknownComponentError
from aberrant.catalog import (
    FORECASTER_CATALOG,
    ComponentKind,
    DetectorConfig,
    EventKind,
    WarmupUnit,
    catalog_manifest,
    get_forecaster_spec,
    get_model_spec,
)
from aberrant.forecast import (
    HoltWinters,
    ResidualDetector,
    RLSRegressor,
    SeasonalNaive,
)


def _config(forecaster: dict, **params: object) -> DetectorConfig:
    return DetectorConfig.from_mapping(
        {
            "model": {
                "id": "residual_detector",
                "params": {"target": "y", "model": forecaster, **params},
            }
        }
    )


def _cycle(i: int) -> dict[str, float]:
    return {"y": math.sin(i * math.pi / 2) + 0.01 * i}


def _first_forecast(detector: ResidualDetector, events: int = 200) -> int | None:
    """Events learned before the detector first compares with a forecast."""
    for learned in range(events):
        x = _cycle(learned)
        if detector.explain_one(x)[1] is not None:
            return learned
        detector.learn_one(x)
    return None


def test_forecasters_are_cataloged_with_their_constructor_schemas() -> None:
    assert dict(FORECASTER_CATALOG).keys() == {
        "seasonal_naive",
        "holt_winters",
        "rls_regressor",
    }
    classes = {
        "seasonal_naive": SeasonalNaive,
        "holt_winters": HoltWinters,
        "rls_regressor": RLSRegressor,
    }
    for component_id, spec in FORECASTER_CATALOG.items():
        assert spec.kind is ComponentKind.FORECASTER
        assert spec.load() is classes[component_id]
    schema = get_forecaster_spec("holt_winters").parameter_schema()
    assert schema["required"] == ["season_length"]
    assert {"horizon", "alpha", "beta", "gamma"} <= schema["properties"].keys()
    with pytest.raises(UnknownComponentError):
        get_forecaster_spec("timestral")


def test_manifest_lists_the_forecasters() -> None:
    manifest = catalog_manifest()

    assert [item["id"] for item in manifest["forecasters"]] == list(FORECASTER_CATALOG)
    json.dumps(manifest, allow_nan=False)


def test_residual_detector_takes_its_forecaster_as_a_nested_component() -> None:
    spec = get_model_spec("residual_detector")
    assert spec.import_path == "aberrant.forecast.ResidualDetector"
    assert [(item.parameter, item.kind) for item in spec.dependencies] == [
        ("model", ComponentKind.FORECASTER)
    ]

    detector = _config(
        {"id": "holt_winters", "params": {"season_length": 4, "horizon": 3}},
        lead=2,
        score="absolute",
    ).build()

    assert isinstance(detector, ResidualDetector)
    assert isinstance(detector.model, HoltWinters)
    assert detector.model.horizon == 3
    assert (detector.target, detector.lead, detector.score) == ("y", 2, "absolute")


@pytest.mark.parametrize(
    ("forecaster", "lead"),
    [
        ({"id": "seasonal_naive", "params": {"season_length": 4}}, None),
        ({"id": "seasonal_naive", "params": {}}, None),
        ({"id": "holt_winters", "params": {"season_length": 4}}, None),
        ({"id": "holt_winters", "params": {"season_length": 6, "horizon": 5}}, 5),
    ],
)
def test_forecaster_warmup_matches_the_first_forecast(forecaster, lead) -> None:
    params = {} if lead is None else {"lead": lead}
    config = _config(forecaster, **params)
    capabilities = config.capabilities()

    assert capabilities.event_kind is EventKind.UNIVARIATE
    assert (capabilities.feature_count.minimum, capabilities.feature_count.maximum) == (
        1,
        1,
    )
    assert capabilities.warmup.unit is WarmupUnit.EVENTS
    assert capabilities.warmup.minimum == _first_forecast(config.build())


def test_regressor_takes_covariates_and_its_warmup_is_left_to_the_events() -> None:
    config = _config({"id": "rls_regressor", "params": {}})
    capabilities = config.capabilities()

    assert capabilities.event_kind is EventKind.TABULAR
    assert capabilities.feature_count.minimum == 2
    assert capabilities.warmup.minimum is None
    detector = config.build()
    assert isinstance(detector.model, RLSRegressor)
    assert detector.lead == 0


def test_directly_created_detectors_resolve_the_same_capabilities() -> None:
    spec = get_model_spec("residual_detector")

    forecaster = spec.capabilities(
        {"model": HoltWinters(season_length=4), "target": "y"}
    )
    regressor = spec.capabilities({"model": RLSRegressor(), "target": "y"})

    assert forecaster.warmup.minimum == 8
    assert regressor.event_kind is EventKind.TABULAR


@pytest.mark.parametrize(
    ("forecaster", "params", "match"),
    [
        ({"id": "timestral", "params": {}}, {}, "timestral"),
        ({"id": "holt_winters", "params": {}}, {}, "season_length"),
        (
            {"id": "holt_winters", "params": {"season_length": 4, "horizon": 2}},
            {"lead": 3},
            "lead must be at most the horizon",
        ),
        ({"id": "rls_regressor", "params": {}}, {"lead": 1}, "lead 0"),
        (
            {"id": "seasonal_naive", "params": {"season_length": 4}},
            {"score": "squared"},
            "score",
        ),
    ],
)
def test_invalid_residual_configurations_are_rejected(
    forecaster, params, match
) -> None:
    with pytest.raises(ConfigurationError, match=match):
        _config(forecaster, **params).build()
