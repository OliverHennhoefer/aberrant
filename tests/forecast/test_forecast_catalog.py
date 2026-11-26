"""Forecasters in the catalog, and residual detectors built from configuration."""

import json
import math

import pytest

from aberrant.base import ReadinessUnit
from aberrant.catalog import (
    FORECASTER_CATALOG,
    ComponentKind,
    DetectorConfig,
    EventKind,
    catalog_manifest,
    get_forecaster_spec,
    get_model_spec,
    model_capabilities,
)
from aberrant.exceptions import ConfigurationError, UnknownComponentError
from aberrant.forecast import (
    AutoRegressive,
    EnsembleForecaster,
    HoltWinters,
    KalmanForecaster,
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
        "kalman_forecaster",
        "autoregressive",
        "ensemble_forecaster",
        "rls_regressor",
    }
    classes = {
        "seasonal_naive": SeasonalNaive,
        "holt_winters": HoltWinters,
        "kalman_forecaster": KalmanForecaster,
        "autoregressive": AutoRegressive,
        "ensemble_forecaster": EnsembleForecaster,
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
        ({"id": "kalman_forecaster", "params": {}}, None),
        ({"id": "kalman_forecaster", "params": {"slope_ratio": 0.0, "horizon": 4}}, 3),
        ({"id": "kalman_forecaster", "params": {"season_length": 12}}, None),
        ({"id": "autoregressive", "params": {}}, None),
        (
            {
                "id": "ensemble_forecaster",
                "params": {
                    "forecasters": [
                        {"id": "seasonal_naive", "params": {"season_length": 4}},
                        {"id": "holt_winters", "params": {"season_length": 4}},
                    ]
                },
            },
            None,
        ),
        (
            {
                "id": "autoregressive",
                "params": {
                    "horizon": 3,
                    "season_length": 4,
                    "seasonal_order": 1,
                    "difference": 1,
                },
            },
            2,
        ),
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
    assert capabilities.warmup_unit is ReadinessUnit.EVENTS
    assert capabilities.warmup == _first_forecast(config.build())


def test_regressor_takes_covariates_and_its_warmup_is_left_to_the_events() -> None:
    config = _config({"id": "rls_regressor", "params": {}})
    capabilities = config.capabilities()

    assert capabilities.event_kind is EventKind.TABULAR
    assert capabilities.feature_count.minimum == 2
    assert capabilities.warmup is None
    detector = config.build()
    assert isinstance(detector.model, RLSRegressor)
    assert detector.lead == 0


def test_directly_created_detectors_resolve_the_same_capabilities() -> None:
    forecaster = model_capabilities(
        "residual_detector", {"model": HoltWinters(season_length=4), "target": "y"}
    )
    regressor = model_capabilities(
        "residual_detector", {"model": RLSRegressor(), "target": "y"}
    )

    assert forecaster.warmup == 8
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


def test_an_ensemble_lists_its_members_as_nested_components() -> None:
    spec = get_forecaster_spec("ensemble_forecaster")
    assert [(item.parameter, item.kind, item.many) for item in spec.dependencies] == [
        ("forecasters", ComponentKind.FORECASTER, True)
    ]
    schema = spec.parameter_schema()["properties"]["forecasters"]
    assert schema["type"] == "array"
    assert schema["items"]["x-aberrant-component-kind"] == "forecaster"

    members = [
        {"id": "seasonal_naive", "params": {"season_length": 4, "horizon": 2}},
        {"id": "kalman_forecaster", "params": {"horizon": 2}},
    ]
    config = _config(
        {"id": "ensemble_forecaster", "params": {"forecasters": members}}, lead=2
    )
    normalized = config.normalized().as_dict()["model"]["params"]["model"]["params"]
    assert [member["id"] for member in normalized["forecasters"]] == [
        "seasonal_naive",
        "kalman_forecaster",
    ]
    detector = config.build()
    assert isinstance(detector.model, EnsembleForecaster)
    assert [type(member) for member in detector.model.forecasters] == [
        SeasonalNaive,
        KalmanForecaster,
    ]


def test_an_ensemble_member_error_names_its_position() -> None:
    members = [{"id": "seasonal_naive"}, {"id": "timestral"}]
    with pytest.raises(UnknownComponentError) as raised:
        _config(
            {"id": "ensemble_forecaster", "params": {"forecasters": members}}
        ).capabilities()
    assert raised.value.issues[0].loc[-3:] == ("forecasters", 1, "id")
    with pytest.raises(ConfigurationError, match="same horizon"):
        _config(
            {
                "id": "ensemble_forecaster",
                "params": {
                    "forecasters": [
                        {"id": "seasonal_naive"},
                        {"id": "kalman_forecaster", "params": {"horizon": 2}},
                    ]
                },
            }
        ).build()
