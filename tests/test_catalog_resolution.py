"""Compatibility and operation counts for reusable declarative resolution."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from typing import Literal, Protocol

import pytest

from aberrant.catalog import (
    ComponentConfig,
    ComponentKind,
    ComponentSpec,
    DetectorConfig,
    ResolvedDetectorConfig,
    get_model_spec,
    resolve_detector_config,
)
from aberrant.evaluate import PrequentialEvaluator
from aberrant.exceptions import (
    ConfigurationError,
    ConfigurationIssue,
    MissingOptionalDependencyError,
    UnknownComponentError,
)
from aberrant.similarity import BaseSimilaritySearchEngine


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
            == "b1a48002ccf7c457563a24d01ffc822e6008745424d37759fdc8383bf1b1ad89"
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
    config = DetectorConfig(ComponentConfig("online_isolation_forest", {"n_trees": 0}))
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


# Components added after the baseline below; their schemas are new, not changed.
@pytest.mark.parametrize(
    ("config", "fingerprint"),
    [
        (example(), "b1a48002ccf7c457563a24d01ffc822e6008745424d37759fdc8383bf1b1ad89"),
        (
            DetectorConfig(
                ComponentConfig("moving_average", {"window_size": 8}),
                (ComponentConfig("min_max_scaler", {"feature_range": [0, 1]}),),
            ),
            "bae4638b4a7d65c3627bddae8e0a9cc77d92dc5511ea6e019499b6ae0a993acc",
        ),
    ],
)
def test_canonical_fingerprints_are_unchanged(config, fingerprint):
    assert config.fingerprint() == fingerprint
    assert config.resolve().fingerprint == fingerprint


class AnnotatedParameters:
    def __init__(
        self,
        *,
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
                        "n_trees": "many",
                        "seed": True,
                    },
                },
                "transformers": [
                    {"id": "standard_scaler", "params": {"with_std": "yes"}}
                ],
            }
        ).resolve()
    assert {issue.loc for issue in error.value.issues} == {
        ("model", "params", "n_trees"),
        ("model", "params", "seed"),
        ("transformers", 0, "params", "with_std"),
    }


def test_unknown_component_keeps_its_subclass_and_location():
    with pytest.raises(UnknownComponentError) as error:
        DetectorConfig(ComponentConfig("no-such-model")).resolve()
    assert error.value.component_id == "no-such-model"
    assert error.value.issues[0].code == "unknown_component"
    assert error.value.issues[0].loc == ("model", "id")


def test_collecting_other_errors_preserves_the_first_exception_subclass():
    config = DetectorConfig(
        ComponentConfig("online_isolation_forest", {"n_trees": "many"}),
        (ComponentConfig("no-such-transformer"),),
    )
    with pytest.raises(UnknownComponentError) as error:
        config.resolve()
    assert error.value.component_id == "no-such-transformer"
    assert {issue.code for issue in error.value.issues} == {
        "unknown_component",
        "invalid_type",
    }


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
        ({"version": 2}, ("model",)),
        ({"model": {"id": "null"}, "version": 1}, ("version",)),
        ({"model": "null"}, ("model",)),
        ({"model": {"id": ""}}, ("model", "id")),
        ({"model": {"id": "null", "params": {1: 2}}}, ("model", "params")),
        ({"model": {"id": "null"}, "transformers": "select"}, ("transformers",)),
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
        get_model_spec("online_isolation_forest").resolved_parameters()["n_trees"]
        == 100
    )


def test_every_malformed_transformer_is_reported_at_its_index():
    with pytest.raises(ConfigurationError) as error:
        DetectorConfig.from_mapping(
            {
                "model": {"id": "null"},
                "transformers": [{"id": "passthrough", "extra": 1}, 3],
            }
        )
    assert [issue.loc for issue in error.value.issues] == [
        ("transformers", 0, "extra"),
        ("transformers", 1),
    ]


def test_parsed_configs_accept_component_objects_in_place_of_mappings():
    model, scaler = ComponentConfig("null"), ComponentConfig("standard_scaler")
    parsed = DetectorConfig.from_mapping({"model": model, "transformers": [scaler]})
    plain = DetectorConfig.from_mapping(
        {"model": {"id": "null"}, "transformers": [{"id": "standard_scaler"}]}
    )
    assert parsed == plain


@pytest.mark.parametrize(
    "make",
    [
        lambda: DetectorConfig(model="null"),
        lambda: DetectorConfig(model=ComponentConfig("null"), transformers=3),
        lambda: DetectorConfig(model=ComponentConfig("null"), transformers=["x"]),
        lambda: ComponentConfig(3),
        lambda: ComponentConfig("null", params=[1]),
    ],
)
def test_configs_built_in_python_are_validated_like_parsed_ones(make):
    with pytest.raises(ConfigurationError) as error:
        make()
    assert error.value.issues[0].code == "invalid_type"


def test_nested_configs_and_sets_are_frozen_and_serialized():
    engine = ComponentConfig("faiss", {"window_size": 8})
    config = ComponentConfig("knn", {"similarity_engine": engine, "tags": {"a"}})
    assert config.params["tags"] == frozenset({"a"})
    assert config.as_dict()["params"]["similarity_engine"] == {
        "id": "faiss",
        "params": {"window_size": 8},
    }


class _Opaque:
    def __repr__(self):
        return "_Opaque()"


_OPAQUE = _Opaque()


class _NotRuntimeCheckable(Protocol):
    def method(self) -> None: ...


class LooselyAnnotatedParameters:
    def __init__(
        self,
        *,
        anything,
        ids: set[int] = frozenset(),
        names: frozenset[str] = frozenset(),
        pair: tuple[int, int] = (0, 0),
        opaque: _Opaque = _OPAQUE,
        protocol: _NotRuntimeCheckable | None = None,
    ):
        pass


class IntegerKeyed:
    def __init__(self, by_id: dict[int, int]):
        pass


class UnresolvableHints:
    def __init__(self, value: Undefined = 1):  # noqa: F821
        pass


def _spec_for(component: type) -> ComponentSpec:
    return ComponentSpec(
        id=component.__name__,
        display_name=component.__name__,
        family="test",
        kind=ComponentKind.MODEL,
        import_path=f"{__name__}.{component.__name__}",
    )


def test_descriptors_handle_unannotated_sets_and_python_types():
    spec = _spec_for(LooselyAnnotatedParameters)
    protocol_value = object()
    resolved = spec.resolved_parameters(
        {
            "anything": {"left": "as is"},
            "ids": [1, 2, 2],
            "names": ["a"],
            "protocol": protocol_value,
        }
    )
    assert resolved["anything"] == {"left": "as is"}
    assert resolved["ids"] == {1, 2}
    assert spec.resolved_parameters({"anything": 1})["ids"] == set()
    assert resolved["names"] == frozenset({"a"})
    assert resolved["protocol"] is protocol_value

    properties = spec.parameter_schema()["properties"]
    assert properties["anything"] == {}
    assert properties["opaque"] == {
        "x-python-type": f"{__name__}._Opaque",
        "x-python-default": "_Opaque()",
    }


@pytest.mark.parametrize(
    ("params", "loc"),
    [
        ({"anything": 1, "pair": [1]}, ("pair",)),
        ({"anything": 1, "opaque": 3}, ("opaque",)),
    ],
)
def test_descriptors_reject_values_of_the_wrong_shape(params, loc):
    with pytest.raises(ConfigurationError) as error:
        _spec_for(LooselyAnnotatedParameters).resolved_parameters(params)
    assert error.value.issues[0].loc == loc


def test_integer_mapping_keys_must_parse_as_integers():
    spec = _spec_for(IntegerKeyed)
    assert spec.resolved_parameters({"by_id": {"7": 1}}) == {"by_id": {7: 1}}
    with pytest.raises(ConfigurationError, match="integer object key") as error:
        spec.resolved_parameters({"by_id": {"x": 1}})
    assert error.value.issues[0].loc == ("by_id", "x")


def test_unresolvable_type_hints_fall_back_to_raw_annotations():
    spec = _spec_for(UnresolvableHints)
    assert spec.resolved_parameters({"value": 2}) == {"value": 2}
