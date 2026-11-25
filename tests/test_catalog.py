"""Contracts for the built-in component catalog and declarative builder."""

import importlib
import json
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from aberrant.base import ModelProtocol
from aberrant.catalog import (
    CATALOG_FORMAT_VERSION,
    MODEL_CATALOG,
    SIMILARITY_ENGINE_CATALOG,
    TRANSFORMER_CATALOG,
    ComponentConfig,
    ComponentKind,
    ComponentSpec,
    DetectorConfig,
    EventKind,
    FeatureCount,
    FeatureSchemaKind,
    ScoreKind,
    StateKind,
    WarmupRequirement,
    WarmupUnit,
    build_detector,
    catalog_manifest,
    create_model,
    get_model_spec,
    get_transformer_spec,
)
from aberrant.catalog import _registry as catalog_registry
from aberrant.exceptions import ConfigurationError, UnknownComponentError
from aberrant.model.distance import SDOStream
from aberrant.model.stat import MovingGeometricAverage, MovingHarmonicAverage
from aberrant.model.timeseries import (
    MultivariateRollingMatrixProfile,
    RollingMatrixProfile,
)
from aberrant.similarity import BaseSimilaritySearchEngine

PUBLIC_MODEL_MODULES = (
    "aberrant.model",
    "aberrant.model.deep",
    "aberrant.model.distance",
    "aberrant.model.graph",
    "aberrant.model.forest",
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
    public_paths.add("aberrant.forecast.ResidualDetector")

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
    damp_schema = get_model_spec("damp").parameter_schema()
    lof_schema = get_model_spec("local_outlier_factor").parameter_schema()

    assert forest_schema["properties"]["n_trees"] == {
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
            "warm_up": 10,
            "normalize_score": True,
        }
    )
    xstream = get_model_spec("xstream").capabilities()
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
    ("absolute_values", "ready_count"), [(None, 2), (False, 2), (True, 3)]
)
def test_geometric_average_warmup_matches_scoring_readiness(
    absolute_values: bool | None, ready_count: int
) -> None:
    params: dict[str, object] = {"window_size": 10}
    if absolute_values is not None:
        params["absoluteValues"] = absolute_values
    warmup = get_model_spec("moving_geometric_average").capabilities(params).warmup
    model = create_model("moving_geometric_average", params)
    assert isinstance(model, MovingGeometricAverage)
    query = {"x": 20.0}

    assert warmup.unit is WarmupUnit.RETAINED_VALUES
    assert warmup.minimum == ready_count
    for ignored_value in (-1.0, 0.0):
        model.learn_one({"x": ignored_value})
    assert len(model.window) == 0
    assert warmup.remaining(len(model.window)) == ready_count
    for index in range(ready_count):
        assert not warmup.is_satisfied(len(model.window))
        assert model.score_one(query) == 0.0
        model.learn_one({"x": float(index + 1)})

    assert warmup.is_satisfied(len(model.window))
    assert model.score_one(query) > 0.0


@pytest.mark.parametrize(
    ("absolute_values", "window_size"), [(False, 1), (True, 1), (True, 2)]
)
def test_geometric_average_undersized_window_never_satisfies_warmup(
    absolute_values: bool, window_size: int
) -> None:
    params = {"window_size": window_size, "absoluteValues": absolute_values}
    warmup = get_model_spec("moving_geometric_average").capabilities(params).warmup
    model = create_model("moving_geometric_average", params)
    assert isinstance(model, MovingGeometricAverage)

    for index in range(10):
        model.learn_one({"x": float(index + 1)})
        assert not warmup.is_satisfied(len(model.window))
        assert model.score_one({"x": 20.0}) == 0.0


def test_harmonic_average_warmup_requires_a_retained_nonzero_value() -> None:
    params = {"window_size": 10}
    warmup = get_model_spec("moving_harmonic_average").capabilities(params).warmup
    model = create_model("moving_harmonic_average", params)
    assert isinstance(model, MovingHarmonicAverage)
    query = {"x": 20.0}

    assert warmup.unit is WarmupUnit.RETAINED_VALUES
    assert warmup.minimum == 1
    for _ in range(3):
        model.learn_one({"x": 0.0})
        assert len(model.window) == 0
        assert warmup.remaining(len(model.window)) == 1
        assert model.score_one(query) == 0.0

    model.learn_one({"x": 1.0})
    assert warmup.is_satisfied(len(model.window))
    assert model.score_one(query) > 0.0
    model.learn_one({"x": 0.0})
    assert warmup.is_satisfied(len(model.window))


@pytest.mark.parametrize(
    ("warm_up", "window_size"),
    [(4, 8), (8, 4), (8, 8)],
)
def test_xstream_warmup_matches_scoring_readiness(
    warm_up: int, window_size: int
) -> None:
    params = {
        "k": 3,
        "n_chains": 2,
        "depth": 2,
        "sketch_width": 8,
        "warm_up": warm_up,
        "window_size": window_size,
        "seed": 1,
    }
    warmup = get_model_spec("xstream").capabilities(params).warmup
    model = create_model("xstream", params)
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
    params = {"max_observers": 8, "time_scale": 1000000.0, "n_neighbors": 2, "seed": 1}
    warmup = get_model_spec("sdostream").capabilities(params).warmup
    model = create_model("sdostream", params)
    assert isinstance(model, SDOStream)

    model.learn_one({"x": 0.0})
    model.learn_one({"x": 1.0})

    assert model.n_observers == 1
    assert warmup.unit is WarmupUnit.OBSERVERS
    assert warmup.remaining(model.n_observers) == 1
    assert not warmup.is_satisfied(model.n_observers)
    assert model.score_one({"x": 20.0}) == 0.0


@pytest.mark.parametrize(
    ("k", "n_neighbors", "warm_up", "ready_count"),
    [(8, 2, None, 2), (8, 4, 2, 4), (8, 2, 5, 5), (1, 1, None, 1)],
)
def test_sdo_observer_warmup_matches_scoring_readiness(
    k: int, n_neighbors: int, warm_up: int | None, ready_count: int
) -> None:
    params = {
        "max_observers": k,
        "time_scale": 1.0,
        "n_neighbors": n_neighbors,
        "warm_up": warm_up,
        "seed": 1,
    }
    warmup = get_model_spec("sdostream").capabilities(params).warmup
    model = create_model("sdostream", params)
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


@pytest.mark.parametrize("zone", [None, 0, 3, 7])
def test_matrix_profile_catalog_capabilities_and_warmup(zone) -> None:
    spec = get_model_spec("rolling_matrix_profile")
    params = {"subsequence_length": 4, "exclusion_zone": zone}
    model = create_model(spec.id, params)
    assert isinstance(model, RollingMatrixProfile)
    capabilities = spec.capabilities(params)
    assert spec.family == "time_series"
    assert capabilities.event_kind is EventKind.UNIVARIATE
    assert capabilities.feature_schema is FeatureSchemaKind.FIXED
    assert capabilities.feature_count.accepts(1)
    assert not capabilities.feature_count.accepts(2)
    assert capabilities.score_kind is ScoreKind.NON_NEGATIVE
    assert capabilities.state is StateKind.BOUNDED
    assert capabilities.higher_is_more_anomalous and capabilities.resettable
    assert capabilities.warmup.unit is WarmupUnit.EVENTS
    expected = 4 + (1 if zone is None else zone)
    assert capabilities.warmup.minimum == expected
    for index in range(expected + 2):
        assert model.is_ready == capabilities.warmup.is_satisfied(index)
        assert (model.match_one({"x": float(index)})[1] is not None) == model.is_ready
        model.learn_one({"x": float(index)})


@pytest.mark.parametrize(
    "model_id,model_class",
    [
        ("rolling_matrix_profile", RollingMatrixProfile),
        ("multivariate_rolling_matrix_profile", MultivariateRollingMatrixProfile),
    ],
)
def test_matrix_profile_parameter_schema_and_manifest(model_id, model_class) -> None:
    spec = get_model_spec(model_id)
    schema = spec.parameter_schema()
    assert schema["required"] == ["subsequence_length"]
    assert schema["properties"]["normalize"] == {"type": "boolean", "default": True}
    assert schema["properties"]["window_size"]["default"] is None
    assert schema["properties"]["exclusion_zone"]["default"] is None
    assert spec.capabilities().warmup.minimum is None
    entry = next(
        entry for entry in catalog_manifest()["models"] if entry["id"] == model_id
    )
    assert entry["available"]
    assert entry["import_path"] == f"aberrant.model.timeseries.{model_class.__name__}"


@pytest.mark.parametrize("zone", [None, 0, 3, 7])
def test_multivariate_matrix_profile_capabilities_and_warmup(zone) -> None:
    spec = get_model_spec("multivariate_rolling_matrix_profile")
    params = {"subsequence_length": 4, "exclusion_zone": zone}
    model = create_model(spec.id, params)
    assert isinstance(model, MultivariateRollingMatrixProfile)
    assert model.window_size == 64
    capabilities = spec.capabilities(params)
    assert spec.family == "time_series"
    assert capabilities.event_kind is EventKind.TABULAR
    assert capabilities.feature_schema is FeatureSchemaKind.FIXED
    assert capabilities.feature_count.minimum == 1
    assert capabilities.feature_count.maximum is None
    assert capabilities.score_kind is ScoreKind.NON_NEGATIVE
    assert capabilities.state is StateKind.BOUNDED
    assert capabilities.higher_is_more_anomalous and capabilities.resettable
    assert capabilities.warmup.unit is WarmupUnit.EVENTS
    expected = 4 + (1 if zone is None else zone)
    assert capabilities.warmup.minimum == expected
    for index in range(expected + 2):
        event = {"a": float(index), "b": float(index % 3)}
        assert model.is_ready == capabilities.warmup.is_satisfied(index)
        assert (model.match_one(event)[1] is not None) == model.is_ready
        model.learn_one(event)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("features", [["a"], ["b", "a"], ["a", "b", "c", "d"]])
def test_multivariate_matrix_profile_configured_pipeline_matches_direct(
    normalize, features
) -> None:
    params = {"subsequence_length": 4, "window_size": 16, "normalize": normalize}
    config = DetectorConfig.from_mapping(
        {
            "transformers": [
                {"id": "feature_schema_guard", "params": {"features": features}}
            ],
            "model": {"id": "multivariate_rolling_matrix_profile", "params": params},
        }
    )
    configured = build_detector(DetectorConfig.from_mapping(config.as_dict()))
    direct = MultivariateRollingMatrixProfile(**params)
    for index in range(70):
        event = {
            name: float((index * (channel + 2)) % 13)
            for channel, name in enumerate(features)
        }
        assert configured.score_one(event) == direct.score_one(event)
        configured.learn_one(event)
        direct.learn_one(event)


@pytest.mark.parametrize("normalize", [True, False])
def test_matrix_profile_declarative_pipeline_matches_direct_model(normalize) -> None:
    params = {"subsequence_length": 4, "window_size": 16, "normalize": normalize}
    config = DetectorConfig.from_mapping(
        {
            "transformers": [
                {"id": "feature_schema_guard", "params": {"features": ["value"]}}
            ],
            "model": {"id": "rolling_matrix_profile", "params": params},
        }
    )
    configured = build_detector(DetectorConfig.from_mapping(config.as_dict()))
    direct = RollingMatrixProfile(**params)
    for index in range(70):
        event = {"value": float((index * 7) % 13)}
        assert configured.score_one(event) == direct.score_one(event)
        configured.learn_one(event)
        direct.learn_one(event)
    with pytest.raises(ConfigurationError):
        build_detector(
            DetectorConfig(
                model=ComponentConfig("rolling_matrix_profile", params),
                transformers=(
                    ComponentConfig("feature_schema_guard", {"features": ["a", "b"]}),
                ),
            )
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
                "params": {"n_trees": 2, "window_size": 8, "seed": 17},
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
                    "n_trees": 100,
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
    assert first.normalized().model.params["n_trees"] == 100
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
    with pytest.raises(ConfigurationError, match="n_trees"):
        create_model("online_isolation_forest", {"n_trees": 0})
    with pytest.raises(ConfigurationError, match="expected an integer"):
        create_model("online_isolation_forest", {"n_trees": "many"})


def test_non_declarative_model_is_cataloged_but_rejected_by_detector_config() -> None:
    assert not get_model_spec("autoencoder").declarative

    with pytest.raises(ConfigurationError, match="cannot be built from declarative"):
        DetectorConfig(model=ComponentConfig("autoencoder")).normalized()


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
        ("anoedge_l", None),
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
        params.update({"sketch_dim": 8, "warm_up": 1, "edge_type_key": edge_type_key})
        event["graph"] = 1.0
        if edge_type_key is not None:
            event[edge_type_key] = 1.0
    elif model_id == "anoedge_l":
        params.update({"sketch_rows": 2, "sketch_cols": 8})
    else:
        params.update({"n_hashes": 2, "sketch_width": 8})

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


@pytest.mark.parametrize(
    ("start_index", "ready_count"), [(None, 38), (30, 36)], ids=["default", "explicit"]
)
def test_damp_warmup_matches_scoring_readiness(start_index, ready_count) -> None:
    params: dict[str, object] = {"subsequence_length": 8}
    if start_index is not None:
        params["start_index"] = start_index
    warmup = get_model_spec("damp").capabilities(params).warmup
    model = create_model("damp", params)

    for index in range(ready_count):
        assert not model.is_ready
        model.learn_one({"x": math.sin(index / 3)})

    assert warmup.minimum == ready_count
    assert model.is_ready
    assert get_model_spec("damp").capabilities().warmup.minimum is None


class _WindowedEngine(BaseSimilaritySearchEngine):
    def __init__(self, warm_up: int, window_size: int | None) -> None:
        self.warm_up = warm_up
        self.window_size = window_size

    def append(self, x: dict[str, float]) -> None:
        pass

    def search(self, x: dict[str, float], n_neighbors: int) -> float:
        return 0.0


def test_knn_warmup_reads_a_runtime_engine_instance() -> None:
    spec = get_model_spec("knn")
    engine = _WindowedEngine(warm_up=5, window_size=10)

    assert spec.capabilities({"k": 3, "similarity_engine": engine}).warmup.minimum == 5
    unbounded = _WindowedEngine(warm_up=5, window_size=None)
    assert spec.capabilities({"k": 5, "similarity_engine": unbounded}).warmup.minimum
    for k, broken in [(6, engine), (1, _WindowedEngine(warm_up=11, window_size=10))]:
        with pytest.raises(ConfigurationError, match="k <= warm_up <= window_size"):
            spec.capabilities({"k": k, "similarity_engine": broken})


@pytest.mark.parametrize(
    ("component_id", "params", "names"),
    [
        ("select", {"features": ["c", "a"]}, ("c", "a")),
        ("rename", {"mapping": {"a": "x"}}, ("x", "b", "c")),
        ("passthrough", {}, ("a", "b", "c")),
        ("incremental_pca", {"n_components": 2}, ("component_0", "component_1")),
    ],
)
def test_transformers_resolve_output_names_without_learning(
    component_id, params, names
) -> None:
    spec = get_transformer_spec(component_id)

    assert spec.output_feature_names(params, ("a", "b", "c")) == names
    assert spec.output_feature_count(params, 3) == len(names)


def test_rename_names_stay_unknown_until_inputs_are_known() -> None:
    assert (
        get_transformer_spec("rename").output_feature_names(
            {"mapping": {"a": "x"}}, None
        )
        is None
    )


@pytest.mark.parametrize(
    ("component_id", "params", "message"),
    [
        ("select", {"features": []}, "non-empty sequence"),
        ("select", {"features": [""]}, "non-empty names"),
        ("select", {"features": ["a", "a"]}, "duplicates"),
        ("select", {"features": ["z"]}, "must exist in the input"),
        ("rename", {"mapping": {"z": "x"}}, "must exist in the input"),
        ("rename", {"mapping": {"a": "b"}}, "duplicate features"),
    ],
)
def test_transformers_reject_names_that_cannot_be_resolved(
    component_id, params, message
) -> None:
    spec = get_transformer_spec(component_id)

    with pytest.raises(ConfigurationError, match=message):
        spec.output_feature_names(params, ("a", "b", "c"))


def test_select_narrows_a_pipeline_for_a_univariate_model() -> None:
    config = DetectorConfig(
        transformers=(
            ComponentConfig("feature_schema_guard", {"features": ["a", "b", "c"]}),
            ComponentConfig("select", {"features": ["b"]}),
        ),
        model=ComponentConfig("moving_average", {"window_size": 4}),
    )
    assert config.capabilities().feature_count.accepts(1)
    pipeline = config.build()
    model = create_model("moving_average", {"window_size": 4})

    for index in range(6):
        event = {"a": 100.0, "b": float(index), "c": -100.0}
        assert pipeline.score_one(event) == model.score_one({"b": float(index)})
        pipeline.learn_one(event)
        model.learn_one({"b": float(index)})


@pytest.mark.parametrize(
    ("nested", "message"),
    [
        ({"params": {}}, "forecaster id"),
        ({"id": "seasonal_naive", "params": [4]}, "params must be a mapping"),
    ],
)
def test_residual_detector_rejects_a_malformed_nested_forecaster(
    nested, message
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        get_model_spec("residual_detector").capabilities(
            {"target": "y", "model": nested}
        )


def test_catalog_index_rejects_duplicate_identifiers() -> None:
    spec = get_model_spec("null")

    with pytest.raises(RuntimeError, match="must be unique"):
        catalog_registry._index((spec, spec))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: FeatureCount(0), "must be positive"),
        (lambda: FeatureCount(3, 2), "not be below minimum"),
        (lambda: WarmupRequirement(-1), "non-negative"),
        (lambda: WarmupRequirement(3).remaining(-1), "non-negative"),
    ],
)
def test_capability_records_reject_impossible_bounds(factory, message) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_indeterminate_warmup_is_neither_satisfied_nor_pending() -> None:
    warmup = WarmupRequirement(None, WarmupUnit.ENGINE_DEFINED)

    assert warmup.remaining(10) is None
    assert warmup.is_satisfied(10) is None
    assert WarmupRequirement(3).is_satisfied(3)


@pytest.mark.parametrize(
    ("import_path", "message"),
    [("NoModule", "Invalid catalog import path"), ("math.pi", "not a class")],
)
def test_spec_loading_rejects_entries_that_are_not_classes(
    import_path, message
) -> None:
    spec = ComponentSpec(
        id="broken",
        display_name="Broken",
        family="test",
        kind=ComponentKind.MODEL,
        import_path=import_path,
    )

    with pytest.raises(RuntimeError, match=message):
        spec.load()
