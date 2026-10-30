"""Compatibility and operation counts for reusable declarative resolution."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import FrozenInstanceError
from typing import Literal

import pytest

from aberrant.base import (
    BaseSimilaritySearchEngine,
    ConfigurationError,
    ConfigurationIssue,
    MissingOptionalDependencyError,
    UnknownComponentError,
)
from aberrant.catalog import (
    MODEL_CATALOG,
    SIMILARITY_ENGINE_CATALOG,
    TRANSFORMER_CATALOG,
    ComponentConfig,
    ComponentKind,
    ComponentSpec,
    DetectorConfig,
    ResolvedDetectorConfig,
    get_model_spec,
    resolve_detector_config,
)
from aberrant.evaluate import PrequentialEvaluator


def example():
    return DetectorConfig(
        ComponentConfig("online_isolation_forest", {"seed": 2, "window_size": 16}),
        (ComponentConfig("standard_scaler"),),
    )


@pytest.mark.parametrize("operation", ["build", "resolve", "evaluator"])
def test_each_component_occurrence_resolves_once(monkeypatch, operation):
    original = ComponentSpec.resolved_parameters
    calls = []

    def record(self, *args, **kwargs):
        calls.append(self.id)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ComponentSpec, "resolved_parameters", record)
    config = example()
    if operation == "evaluator":
        PrequentialEvaluator(config, warmup=2)
    elif operation == "resolve":
        resolved = resolve_detector_config(config.as_dict())
        assert isinstance(resolved, ResolvedDetectorConfig)
        resolved.build()
        resolved.build()
        assert (
            resolved.fingerprint
            == "8f87e74a7056c6b4364d6abcaba1ed0015e1a93d2161874e52290b4e96ad2a75"
        )
    else:
        config.build()
    assert calls == ["standard_scaler", "online_isolation_forest"]


def test_inspection_is_shared_within_resolution_only(monkeypatch):
    original = inspect.signature
    calls = []

    def record(component, *args, **kwargs):
        calls.append(component)
        return original(component, *args, **kwargs)

    monkeypatch.setattr(inspect, "signature", record)
    config = DetectorConfig(
        ComponentConfig("moving_average", {"window_size": 8}),
        (ComponentConfig("standard_scaler"), ComponentConfig("standard_scaler")),
    )
    config.resolve().build()
    assert len(calls) == 2
    config.resolve().build()
    assert len(calls) == 4


def test_repeated_builds_have_independent_learned_state_and_parameters():
    resolved = example().resolve()
    first, second = resolved.build(), resolved.build()
    assert first is not second
    first.learn_one({"x": 2.0})
    first.learn_one({"x": 3.0})
    assert first._prefix[0].counts == {"x": 2}
    assert second._prefix[0].counts == {}
    second.learn_one({"x": 10.0})
    assert second._prefix[0].means == {"x": 10.0}
    assert first._prefix[0].means == {"x": 2.5}
    exported = resolved.config.as_dict()
    exported["model"]["params"]["seed"] = 99
    assert resolved.config.model.params["seed"] == 2


def test_normalization_and_fingerprint_do_not_require_feature_compatibility():
    config = DetectorConfig(
        ComponentConfig("moving_average", {"window_size": 8}),
        (ComponentConfig("feature_schema_guard", {"features": ["a", "b"]}),),
    )
    assert config.normalized().model.params["window_size"] == 8
    assert len(config.fingerprint()) == 64
    with pytest.raises(ConfigurationError, match="expects 1 features"):
        config.resolve()
    with pytest.raises(ConfigurationError):
        config.build()


def test_resolution_does_not_construct_or_validate_constructor_only_invariants():
    config = DetectorConfig(
        ComponentConfig("online_isolation_forest", {"num_trees": 0})
    )
    resolved = config.resolve()
    assert resolved.config == config.normalized()
    assert resolved.fingerprint == config.fingerprint()
    with pytest.raises(ConfigurationError) as error:
        resolved.build()
    assert error.value.issues[0].code == "invalid_value"
    assert error.value.issues[0].loc == ("model", "params")


class RuntimeEngine(BaseSimilaritySearchEngine):
    def append(self, x):
        pass

    def search(self, x, n_neighbors):
        return 0.0


def test_runtime_values_remain_buildable_but_cannot_be_fingerprinted():
    config = DetectorConfig(
        ComponentConfig("knn", {"k": 1, "similarity_engine": RuntimeEngine()})
    )
    resolved = config.resolve()
    assert resolved.build() is not resolved.build()
    assert config.build() is not None
    with pytest.raises(ConfigurationError, match="non-JSON"):
        _ = resolved.fingerprint
    with pytest.raises(ConfigurationError, match="non-JSON"):
        config.fingerprint()


def test_existing_core_schema_contract_is_unchanged():
    # Captured from the reviewed 1.3.0/Python 3.11 baseline before this refactor.
    schemas = {
        f"{spec.kind.value}/{spec.id}": spec.parameter_schema()
        for catalog in (MODEL_CATALOG, TRANSFORMER_CATALOG, SIMILARITY_ENGINE_CATALOG)
        for spec in catalog.values()
        if spec.dependency_module is None
    }
    digest = hashlib.sha256(
        json.dumps(schemas, sort_keys=True, allow_nan=False).encode()
    )
    assert (
        digest.hexdigest()
        == "63a3c1e292b5e22361e0304cd81b6a6d4c65fd8159efbb4cafb4e3348682daee"
    )


@pytest.mark.parametrize(
    ("config", "fingerprint"),
    [
        (example(), "8f87e74a7056c6b4364d6abcaba1ed0015e1a93d2161874e52290b4e96ad2a75"),
        (
            DetectorConfig(
                ComponentConfig(
                    "graph_gated_one_class_svm", {"graph": {"0": [1], "1": []}}
                )
            ),
            "f85771fcf13dbdeb1aa5174699e82163b6416ef82a1badd7f286c31261a36a56",
        ),
        (
            DetectorConfig(
                ComponentConfig("moving_average", {"window_size": 8}),
                (ComponentConfig("min_max_scaler", {"feature_range": [0, 1]}),),
            ),
            "8feb46d480239c33fbc9bb2760277444247cfaabd13b14ff47b268a9078db906",
        ),
    ],
)
def test_canonical_fingerprints_are_unchanged(config, fingerprint):
    assert config.fingerprint() == fingerprint
    assert config.resolve().fingerprint == fingerprint


class AnnotatedParameters:
    def __init__(
        self,
        pair: tuple[int, float],
        values: tuple[int, ...] = (1, 2),
        weights: list[float] | None = None,
        choice: Literal["a", "b"] = "a",
        numbers: dict[int, list[int]] | None = None,
        preferred: float | int = 1,
    ):
        pass


def annotated_spec():
    return ComponentSpec(
        id="annotated",
        display_name="Annotated",
        family="test",
        kind=ComponentKind.MODEL,
        import_path=f"{__name__}.AnnotatedParameters",
    )


def test_parameter_descriptor_preserves_containers_nulls_unions_and_defaults():
    resolved = annotated_spec().resolved_parameters(
        {
            "pair": [1, 2],
            "weights": [1, 2.5],
            "numbers": {"0": [1]},
        },
        require_all=True,
    )
    assert resolved == {
        "pair": (1, 2.0),
        "values": (1, 2),
        "weights": [1.0, 2.5],
        "choice": "a",
        "numbers": {0: [1]},
        "preferred": 1.0,
    }
    assert isinstance(resolved["preferred"], float)
    assert annotated_spec().resolved_parameters({"pair": [1, 2]})["weights"] is None
    schema = annotated_spec().parameter_schema()
    assert schema["required"] == ["pair"]
    assert schema["properties"]["weights"]["default"] is None
    schema["properties"]["values"]["default"].append(3)
    assert annotated_spec().parameter_schema()["properties"]["values"]["default"] == [
        1,
        2,
    ]


def test_independent_missing_unexpected_and_type_errors_are_collected():
    with pytest.raises(ConfigurationError) as error:
        annotated_spec().resolved_parameters(
            {"choice": "no", "values": [False, "no"], "unknown": True}, require_all=True
        )
    assert {(issue.code, issue.loc) for issue in error.value.issues} == {
        ("missing_parameter", ("pair",)),
        ("unexpected_parameter", ("unknown",)),
        ("invalid_value", ("choice",)),
        ("invalid_type", ("values", 0)),
        ("invalid_type", ("values", 1)),
    }
    assert "Invalid parameters" in str(error.value)


def test_detector_parameter_errors_retain_component_indexes():
    with pytest.raises(ConfigurationError) as error:
        DetectorConfig.from_mapping(
            {
                "model": {
                    "id": "online_isolation_forest",
                    "params": {
                        "num_trees": "many",
                        "seed": True,
                    },
                },
                "transformers": [
                    {"id": "standard_scaler", "params": {"with_std": "yes"}}
                ],
            }
        ).resolve()
    assert {issue.loc for issue in error.value.issues} == {
        ("model", "params", "num_trees"),
        ("model", "params", "seed"),
        ("transformers", 0, "params", "with_std"),
    }


def test_unknown_component_keeps_its_subclass_and_location():
    with pytest.raises(UnknownComponentError) as error:
        DetectorConfig(ComponentConfig("no-such-model")).resolve()
    assert error.value.component_id == "no-such-model"
    assert error.value.issues[0].code == "unknown_component"
    assert error.value.issues[0].loc == ("model", "id")


def test_missing_dependency_keeps_its_subclass_and_nested_location(monkeypatch):
    monkeypatch.setattr(
        ComponentSpec, "available", property(lambda self: self.id != "faiss")
    )
    with pytest.raises(MissingOptionalDependencyError) as error:
        DetectorConfig(
            ComponentConfig(
                "knn",
                {
                    "k": 1,
                    "similarity_engine": {"id": "faiss"},
                },
            )
        ).resolve()
    assert error.value.extra == "faiss"
    assert error.value.issues[0].code == "missing_dependency"
    assert error.value.issues[0].loc == ("model", "params", "similarity_engine", "id")


@pytest.mark.parametrize(
    ("mapping", "loc"),
    [
        ({"model": {"id": "null", "params": []}}, ("model", "params")),
        ({"model": {"id": "null"}, "transformers": [False]}, ("transformers", 0)),
        ({"model": {"params": {}}}, ("model", "id")),
        ({"model": {"id": "null"}, "version": 0}, ("version",)),
    ],
)
def test_structural_locations_are_relative_to_submitted_configuration(mapping, loc):
    with pytest.raises(ConfigurationError) as error:
        DetectorConfig.from_mapping(mapping)
    assert error.value.issues[0].loc == loc


def test_issue_records_and_issue_collection_are_immutable():
    issue = ConfigurationIssue("invalid_value", ("model", "params"), "invalid")
    error = ConfigurationError("Invalid configuration", issues=[issue])
    with pytest.raises(FrozenInstanceError):
        issue.code = "other"
    with pytest.raises(AttributeError):
        error.issues = ()
    assert error.issues == (issue,)
    assert (
        get_model_spec("online_isolation_forest").resolved_parameters()["num_trees"]
        == 100
    )
