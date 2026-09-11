"""Contracts for the built-in component catalog and declarative builder."""

import importlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from aberrant.base import ConfigurationError, ModelProtocol, UnknownComponentError
from aberrant.catalog import (
    CATALOG_FORMAT_VERSION,
    MODEL_CATALOG,
    SIMILARITY_ENGINE_CATALOG,
    TRANSFORMER_CATALOG,
    ComponentConfig,
    DetectorConfig,
    EventKind,
    FeatureSchemaKind,
    ScoreKind,
    StateKind,
    WarmupUnit,
    build_detector,
    catalog_manifest,
    create_model,
    get_model_spec,
)
from aberrant.model.distance import SDOStream

PUBLIC_MODEL_MODULES = (
    "aberrant.model",
    "aberrant.model.deep",
    "aberrant.model.distance",
    "aberrant.model.graph",
    "aberrant.model.iforest",
    "aberrant.model.sketch",
    "aberrant.model.stat",
    "aberrant.model.svm",
    "aberrant.model.timeseries",
)


def test_model_catalog_covers_every_public_builtin() -> None:
    public_paths = {
        f"{module_name}.{name}"
        for module_name in PUBLIC_MODEL_MODULES
        for name in importlib.import_module(module_name).__all__
    }
    public_paths.add("aberrant.model.deep.Autoencoder")

    assert {spec.import_path for spec in MODEL_CATALOG.values()} == public_paths


def test_catalogs_and_entries_are_immutable() -> None:
    spec = get_model_spec("online_isolation_forest")

    with pytest.raises(TypeError):
        MODEL_CATALOG["replacement"] = spec  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        spec.id = "replacement"  # type: ignore[misc]


def test_catalog_manifest_is_json_compatible() -> None:
    manifest = catalog_manifest()

    assert manifest["version"] == CATALOG_FORMAT_VERSION
    assert isinstance(manifest["aberrant_version"], str)
    assert len(manifest["models"]) == len(MODEL_CATALOG)
    assert len(manifest["transformers"]) == len(TRANSFORMER_CATALOG)
    assert len(manifest["similarity_engines"]) == len(SIMILARITY_ENGINE_CATALOG)
    json.dumps(manifest, allow_nan=False)


def test_manifest_defers_optional_runtime_inspection(tmp_path: Path) -> None:
    for module in ("torch", "faiss"):
        (tmp_path / f"{module}.py").write_text(
            "raise AssertionError('Optional runtime was imported')\n"
        )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from aberrant.catalog import catalog_manifest\n"
            "print(json.dumps(catalog_manifest()))\n"
            "assert 'torch' not in sys.modules\n"
            "assert 'faiss' not in sys.modules\n",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    manifest = json.loads(result.stdout)
    models = {entry["id"]: entry for entry in manifest["models"]}
    autoencoder = models["autoencoder"]
    faiss = manifest["similarity_engines"][0]

    assert autoencoder["available"] and faiss["available"]
    assert autoencoder["parameters"] is None
    assert autoencoder["capabilities"] is None
    assert faiss["parameters"] is None
    assert models["online_isolation_forest"]["parameters"] is not None
    assert models["online_isolation_forest"]["capabilities"] is not None


def test_parameter_schema_reports_types_defaults_and_required_values() -> None:
    forest_schema = get_model_spec("online_isolation_forest").parameter_schema()
    damp_schema = get_model_spec("x_lag_damp").parameter_schema()
    lof_schema = get_model_spec("local_outlier_factor").parameter_schema()

    assert forest_schema["properties"]["num_trees"] == {
        "type": "integer",
        "default": 100,
    }
    assert forest_schema["properties"]["tree_type"]["enum"] == [
        "fixed",
        "adaptive",
    ]
    assert damp_schema["required"] == ["subsequence_length"]
    assert lof_schema["properties"]["distance"]["enum"] == [
        "euclidean",
        "manhattan",
    ]
    assert (
        get_model_spec("knn").parameter_schema()["properties"]["similarity_engine"][
            "x-aberrant-component-kind"
        ]
        == "similarity_engine"
    )


def test_capabilities_resolve_from_model_parameters() -> None:
    random_cut = get_model_spec("random_cut_forest").capabilities(
        {
            "sample_size": 64,
            "shingle_size": 4,
            "warmup_samples": 10,
            "normalize_score": True,
        }
    )
    xstream = get_model_spec("x_stream").capabilities()
    signed_statistic = get_model_spec("moving_covariance").capabilities(
        {"window_size": 20, "abs_diff": False}
    )
    edge = get_model_spec("midas").capabilities({"time_key": None})

    assert random_cut.warmup.minimum == 13
    assert random_cut.score_kind is ScoreKind.ZERO_TO_ONE
    assert xstream.warmup.minimum == 256
    assert xstream.state is StateKind.BOUNDED
    assert xstream.feature_schema is FeatureSchemaKind.EVOLVING
    assert signed_statistic.score_kind is ScoreKind.SIGNED
    assert signed_statistic.higher_is_more_anomalous is None
    assert edge.event_kind is EventKind.EDGE
    assert edge.feature_count.minimum == 2
    assert edge.feature_count.maximum is None
    assert get_model_spec("half_space_trees").capabilities().requires_unit_interval


@pytest.mark.parametrize(
    ("init_sample_size", "window_size"),
    [(4, 8), (8, 4), (8, 8)],
)
def test_xstream_warmup_matches_scoring_readiness(
    init_sample_size: int, window_size: int
) -> None:
    params = {
        "k": 3,
        "n_chains": 2,
        "depth": 2,
        "cms_width": 8,
        "init_sample_size": init_sample_size,
        "window_size": window_size,
        "seed": 1,
    }
    warmup = get_model_spec("x_stream").capabilities(params).warmup
    model = create_model("x_stream", params)
    query = {"x": 20.0}

    for i in range(7):
        model.learn_one({"x": float(i)})

    assert warmup.unit is WarmupUnit.EVENTS
    assert not warmup.is_satisfied(7)
    assert model.score_one(query) == 0.0

    model.learn_one({"x": 7.0})

    assert warmup.is_satisfied(8)
    assert model.score_one(query) > 0.0


def test_sdo_warmup_counts_sampled_observers_instead_of_events() -> None:
    params = {"k": 8, "T": 1000000.0, "x_neighbors": 2, "seed": 1}
    warmup = get_model_spec("sdo_stream").capabilities(params).warmup
    model = create_model("sdo_stream", params)
    assert isinstance(model, SDOStream)

    model.learn_one({"x": 0.0})
    model.learn_one({"x": 1.0})

    assert model.n_observers == 1
    assert warmup.unit is WarmupUnit.OBSERVERS
    assert warmup.remaining(model.n_observers) == 1
    assert not warmup.is_satisfied(model.n_observers)
    assert model.score_one({"x": 20.0}) == 0.0


@pytest.mark.parametrize(
    ("k", "x_neighbors", "warm_up_observers", "ready_count"),
    [(8, 2, None, 2), (8, 4, 2, 4), (8, 2, 5, 5), (1, 1, None, 1)],
)
def test_sdo_observer_warmup_matches_scoring_readiness(
    k: int, x_neighbors: int, warm_up_observers: int | None, ready_count: int
) -> None:
    params = {
        "k": k,
        "T": 1.0,
        "x_neighbors": x_neighbors,
        "warm_up_observers": warm_up_observers,
        "seed": 1,
    }
    warmup = get_model_spec("sdo_stream").capabilities(params).warmup
    model = create_model("sdo_stream", params)
    assert isinstance(model, SDOStream)
    query = {"x": 20.0}

    for i in range(ready_count - 1):
        model.learn_one({"x": float(i)})

    assert not warmup.is_satisfied(model.n_observers)
    assert model.score_one(query) == 0.0

    model.learn_one({"x": float(ready_count - 1)})

    assert model.n_observers == ready_count
    assert warmup.is_satisfied(model.n_observers)
    assert model.score_one(query) > 0.0


def test_knn_capabilities_resolve_nested_engine_warmup() -> None:
    capabilities = get_model_spec("knn").capabilities(
        {
            "k": 5,
            "similarity_engine": {
                "id": "faiss",
                "params": {"window_size": 100, "warm_up": 20},
            },
        }
    )

    assert capabilities.warmup.minimum == 20
    assert capabilities.warmup.unit is WarmupUnit.EVENTS
    assert capabilities.feature_schema is FeatureSchemaKind.FIXED
    assert capabilities.score_kind is ScoreKind.NON_NEGATIVE
    assert capabilities.state is StateKind.BOUNDED
    assert capabilities.higher_is_more_anomalous
    assert capabilities.warmup.remaining(7) == 13
    assert capabilities.warmup.is_satisfied(20)

    with pytest.raises(ConfigurationError, match="k <= warm_up <= window_size"):
        get_model_spec("knn").capabilities(
            {
                "k": 30,
                "similarity_engine": {
                    "id": "faiss",
                    "params": {"window_size": 100, "warm_up": 20},
                },
            }
        )


def test_build_detector_constructs_a_validated_pipeline() -> None:
    config = DetectorConfig.from_mapping(
        {
            "transformers": [
                {
                    "id": "feature_schema_guard",
                    "params": {"features": ["a", "b", "c"]},
                },
                {"id": "standard_scaler"},
                {
                    "id": "random_projection",
                    "params": {"n_components": 2, "seed": 17},
                },
            ],
            "model": {
                "id": "online_isolation_forest",
                "params": {"num_trees": 2, "window_size": 8, "seed": 17},
            },
        }
    )

    detector = build_detector(config)
    detector.learn_one({"a": 0.0, "b": 1.0, "c": 2.0})

    assert isinstance(detector, ModelProtocol)
    assert isinstance(detector.score_one({"a": 1.0, "b": 2.0, "c": 3.0}), float)
    assert config.capabilities().feature_count.accepts(2)


def test_detector_config_round_trips_and_has_stable_fingerprint() -> None:
    features = ["a", "b"]
    mapping = {
        "model": {
            "params": {"seed": 2, "window_size": 16},
            "id": "online_isolation_forest",
        },
        "transformers": [
            {"id": "feature_schema_guard", "params": {"features": features}},
            {"id": "standard_scaler"},
        ],
    }
    first = DetectorConfig.from_mapping(mapping)
    second = DetectorConfig.from_mapping(first.as_dict())
    explicit_defaults = DetectorConfig.from_mapping(
        {
            "model": {
                "id": "online_isolation_forest",
                "params": {
                    "num_trees": 100,
                    "max_leaf_samples": 32,
                    "tree_type": "adaptive",
                    "subsample": 1.0,
                    "window_size": 16,
                    "branching_factor": 2,
                    "metric": "axisparallel",
                    "n_jobs": 1,
                    "seed": 2,
                },
            },
            "transformers": [
                {
                    "id": "feature_schema_guard",
                    "params": {"features": ["a", "b"], "sort_features": True},
                },
                {"id": "standard_scaler", "params": {"with_std": True}},
            ],
        }
    )

    assert second == first
    assert second.fingerprint() == first.fingerprint()
    assert explicit_defaults.fingerprint() == first.fingerprint()
    assert first.normalized().model.params["num_trees"] == 100
    features.append("c")
    assert first.transformers[0].params["features"] == ("a", "b")
    with pytest.raises(TypeError):
        first.model.params["seed"] = 3  # type: ignore[index]


@pytest.mark.parametrize(
    "mapping",
    [
        {"model": {"id": "null"}, "unexpected": True},
        {"version": True, "model": {"id": "null"}},
        {"version": 999, "model": {"id": "null"}},
        {"model": {"id": "null", "unexpected": True}},
        {"model": {"id": "null"}, "transformers": ["standard_scaler"]},
    ],
)
def test_detector_config_rejects_unknown_or_malformed_fields(mapping) -> None:
    with pytest.raises(ConfigurationError):
        DetectorConfig.from_mapping(mapping)


def test_builder_rejects_unknown_components_and_invalid_parameters() -> None:
    with pytest.raises(UnknownComponentError):
        create_model("not_a_model")
    with pytest.raises(ConfigurationError, match="num_trees"):
        create_model("online_isolation_forest", {"num_trees": 0})
    with pytest.raises(ConfigurationError, match="expected an integer"):
        create_model("online_isolation_forest", {"num_trees": "many"})


def test_non_declarative_model_is_cataloged_but_rejected_by_detector_config() -> None:
    assert not get_model_spec("autoencoder").declarative

    with pytest.raises(ConfigurationError, match="cannot be built from declarative"):
        DetectorConfig(model=ComponentConfig("autoencoder")).normalized()


def test_builder_restores_integer_mapping_keys_from_json() -> None:
    model = create_model(
        "graph_gated_one_class_svm",
        {"graph": {"0": [1], "1": []}},
    )

    assert model.graph == {0: [1], 1: []}


def test_pipeline_feature_count_is_checked_before_construction() -> None:
    config = DetectorConfig(
        transformers=(
            ComponentConfig(
                "feature_schema_guard",
                {"features": ["a", "b"]},
            ),
        ),
        model=ComponentConfig("moving_average", {"window_size": 10}),
    )

    with pytest.raises(ConfigurationError, match="expects 1 features"):
        config.build()


@pytest.mark.parametrize("projection_id", ["incremental_pca", "random_projection"])
def test_projection_output_cannot_exceed_known_input_width(projection_id: str) -> None:
    config = DetectorConfig(
        transformers=(
            ComponentConfig(
                "feature_schema_guard",
                {"features": ["a", "b"]},
            ),
            ComponentConfig(projection_id, {"n_components": 3}),
        ),
        model=ComponentConfig("null"),
    )

    with pytest.raises(ConfigurationError, match="exceeds input feature count"):
        config.capabilities()
    with pytest.raises(ConfigurationError, match="exceeds input feature count"):
        config.build()


@pytest.mark.parametrize(
    ("model_id", "edge_type_key"),
    [
        ("midas", None),
        ("isconna", None),
        ("ano_edge_l", None),
        ("signed_graph_sketch_detector", None),
        ("signed_graph_sketch_detector", "kind"),
    ],
)
@pytest.mark.parametrize("time_key", [None, "t"])
def test_edge_models_accept_guarded_events_with_metadata(
    model_id: str, edge_type_key: str | None, time_key: str | None
) -> None:
    params: dict[str, object] = {"time_key": time_key, "seed": 1}
    event = {"src": 1.0, "dst": 2.0}
    if time_key is not None:
        event[time_key] = 1.0
    if model_id == "signed_graph_sketch_detector":
        params.update(
            {"sketch_dim": 8, "warm_up_graphs": 1, "edge_type_key": edge_type_key}
        )
        event["graph"] = 1.0
        if edge_type_key is not None:
            event[edge_type_key] = 1.0
    else:
        params.update({"count_min_rows": 2, "count_min_cols": 8})

    model_config = ComponentConfig(model_id, params)
    with_metadata = {**event, "weight": 3.0}
    config = DetectorConfig(
        model=model_config,
        transformers=(
            ComponentConfig("feature_schema_guard", {"features": list(with_metadata)}),
        ),
    )
    feature_count = config.capabilities().feature_count
    assert feature_count.minimum == len(event)
    assert feature_count.maximum is None

    pipeline = config.build()
    model = create_model(model_id, params)
    pipeline.learn_one(with_metadata)
    model.learn_one(event)

    assert pipeline.score_one(with_metadata) == model.score_one(event)

    undersized = DetectorConfig(
        model=model_config,
        transformers=(
            ComponentConfig("feature_schema_guard", {"features": list(event)[:-1]}),
        ),
    )
    with pytest.raises(ConfigurationError, match="expects at least"):
        undersized.build()
