"""Safe declarative construction from built-in catalog identifiers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from aberrant.base import (
    BaseSimilaritySearchEngine,
    ConfigurationError,
    ConfigurationIssue,
    ModelProtocol,
    Pipeline,
    TransformerProtocol,
)

from ._parameters import (
    ConstructorParameters,
    Location,
    _invalid,
    _raise_errors,
    copy_parameter,
)
from ._registry import (
    get_model_spec,
    get_similarity_engine_spec,
    get_transformer_spec,
)
from ._specs import (
    ComponentKind,
    ComponentSpec,
    ModelCapabilities,
    ModelSpec,
    TransformerSpec,
)

CONFIG_FORMAT_VERSION = 1


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _plain_value(value: object) -> object:
    if isinstance(value, ComponentConfig):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ComponentConfig:
    """A catalog component identifier and its constructor parameters."""

    id: str
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise _invalid(
                "invalid_type", ("id",), "Component 'id' must be a non-empty string"
            )
        if not isinstance(self.params, Mapping):
            raise _invalid(
                "invalid_type", ("params",), "Component 'params' must be an object"
            )
        if any(not isinstance(name, str) for name in self.params):
            raise _invalid(
                "invalid_type", ("params",), "Component parameter names must be strings"
            )
        object.__setattr__(
            self,
            "params",
            MappingProxyType(
                {name: _freeze_value(value) for name, value in self.params.items()}
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ComponentConfig:
        """Parse a strict JSON-compatible component object."""
        unexpected = set(value) - {"id", "params"}
        if unexpected:
            names = ", ".join(sorted(str(name) for name in unexpected))
            message = f"Unexpected component fields: {names}"
            raise ConfigurationError(
                message,
                issues=tuple(
                    ConfigurationIssue("invalid_configuration", (str(name),), message)
                    for name in sorted(unexpected, key=str)
                ),
            )
        if "id" not in value:
            raise _invalid(
                "invalid_configuration",
                ("id",),
                "Component configuration requires 'id'",
            )
        params = value.get("params", {})
        if not isinstance(params, Mapping):
            raise _invalid(
                "invalid_type", ("params",), "Component 'params' must be an object"
            )
        return cls(id=cast(str, value["id"]), params=params)

    def as_dict(self) -> dict[str, object]:
        """Return a detached JSON-compatible representation when values permit."""
        return {"id": self.id, "params": _plain_value(self.params)}


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Declarative configuration for transformers followed by one model."""

    model: ComponentConfig
    transformers: tuple[ComponentConfig, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model, ComponentConfig):
            raise _invalid(
                "invalid_type",
                ("model",),
                "Detector 'model' must be a component config",
            )
        try:
            transformers = tuple(self.transformers)
        except TypeError as exc:
            raise _invalid(
                "invalid_type",
                ("transformers",),
                "Detector 'transformers' must contain component configs",
            ) from exc
        if any(not isinstance(item, ComponentConfig) for item in transformers):
            raise _invalid(
                "invalid_type",
                ("transformers",),
                "Detector 'transformers' must contain component configs",
            )
        object.__setattr__(self, "transformers", transformers)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> DetectorConfig:
        """Parse the versioned, strict detector configuration format."""
        unexpected = set(value) - {"version", "model", "transformers"}
        if unexpected:
            names = ", ".join(sorted(str(name) for name in unexpected))
            message = f"Unexpected detector fields: {names}"
            raise ConfigurationError(
                message,
                issues=tuple(
                    ConfigurationIssue("invalid_configuration", (str(name),), message)
                    for name in sorted(unexpected, key=str)
                ),
            )
        version = value.get("version", CONFIG_FORMAT_VERSION)
        if type(version) is not int or version != CONFIG_FORMAT_VERSION:
            raise _invalid(
                "invalid_value",
                ("version",),
                f"Unsupported detector config version: {version!r}; "
                f"expected {CONFIG_FORMAT_VERSION}",
            )
        if "model" not in value:
            raise _invalid(
                "invalid_configuration",
                ("model",),
                "Detector configuration requires 'model'",
            )

        raw_model = value["model"]
        if isinstance(raw_model, ComponentConfig):
            model = raw_model
        elif isinstance(raw_model, Mapping):
            with _at(("model",)):
                model = ComponentConfig.from_mapping(raw_model)
        else:
            raise _invalid(
                "invalid_type", ("model",), "Detector 'model' must be an object"
            )

        raw_transformers = value.get("transformers", ())
        if not isinstance(raw_transformers, Sequence) or isinstance(
            raw_transformers, str | bytes
        ):
            raise _invalid(
                "invalid_type",
                ("transformers",),
                "Detector 'transformers' must be an array",
            )
        transformers: list[ComponentConfig] = []
        errors: list[ConfigurationError] = []
        for index, item in enumerate(raw_transformers):
            if isinstance(item, ComponentConfig):
                transformers.append(item)
            elif isinstance(item, Mapping):
                try:
                    with _at(("transformers", index)):
                        transformers.append(ComponentConfig.from_mapping(item))
                except ConfigurationError as error:
                    errors.append(error)
            else:
                errors.append(
                    _invalid(
                        "invalid_type",
                        ("transformers", index),
                        "Every detector transformer must be an object",
                    )
                )
        _raise_errors(errors)
        return cls(model=model, transformers=tuple(transformers))

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible configuration shape."""
        return {
            "version": CONFIG_FORMAT_VERSION,
            "transformers": [item.as_dict() for item in self.transformers],
            "model": self.model.as_dict(),
        }

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 digest of the canonical configuration."""
        return _fingerprint(self.normalized())

    def resolve(self) -> ResolvedDetectorConfig:
        """Resolve parameters and capabilities once, without learning or construction."""
        return resolve_detector_config(self)

    def normalized(self) -> DetectorConfig:
        """Return a copy with constructor defaults and Python types resolved."""
        return normalize_detector_config(self)

    def capabilities(self) -> ModelCapabilities:
        """Resolve and validate the configured detector's operational metadata."""
        return detector_capabilities(self)

    def build(self) -> ModelProtocol:
        """Construct the configured model or model-ending pipeline."""
        return build_detector(self)


def _as_component_config(
    value: ComponentConfig | Mapping[str, object],
) -> ComponentConfig:
    return (
        value
        if isinstance(value, ComponentConfig)
        else ComponentConfig.from_mapping(value)
    )


def _dependency_spec(kind: ComponentKind, component_id: str) -> ComponentSpec:
    if kind is ComponentKind.SIMILARITY_ENGINE:
        return get_similarity_engine_spec(component_id)
    raise RuntimeError(f"Unsupported nested component kind: {kind.value}")


@contextmanager
def _at(loc: Location) -> Iterator[None]:
    try:
        yield
    except ConfigurationError as error:
        error._prepend_location(loc)
        raise


@dataclass(frozen=True, slots=True)
class _ResolvedComponent:
    spec: ComponentSpec
    config: ComponentConfig
    constructor: ConstructorParameters
    params: Mapping[str, object]
    dependencies: tuple[tuple[str, _ResolvedComponent], ...]
    loc: Location

    def build(self) -> object:
        params = {name: copy_parameter(value) for name, value in self.params.items()}
        for name, dependency in self.dependencies:
            params[name] = dependency.build()
        try:
            return self.constructor.component(**params)
        except (TypeError, ValueError) as error:
            message = f"Invalid parameters for {self.spec.kind.value} '{self.spec.id}': {error}"
            raise _invalid("invalid_value", (*self.loc, "params"), message) from error


@dataclass
class _Resolution:
    # Inspection is shared only within this operation. No global mutable cache.
    constructors: dict[str, ConstructorParameters] = field(default_factory=dict)

    def constructor(self, spec: ComponentSpec) -> ConstructorParameters:
        if spec.import_path not in self.constructors:
            self.constructors[spec.import_path] = ConstructorParameters.inspect(
                spec.load()
            )
        return self.constructors[spec.import_path]

    def component(
        self,
        spec: ComponentSpec,
        config: ComponentConfig,
        loc: Location,
        *,
        declarative: bool = True,
    ) -> _ResolvedComponent:
        if declarative and not spec.declarative:
            raise _invalid(
                "invalid_configuration",
                loc,
                f"{spec.kind.value.title()} '{spec.id}' requires runtime Python objects "
                "and cannot be built from declarative configuration",
            )
        with _at((*loc, "id")):
            constructor = self.constructor(spec)
        params = dict(config.params)
        dependencies: list[tuple[str, _ResolvedComponent]] = []
        errors: list[ConfigurationError] = []
        for dependency in spec.dependencies:
            value = params.get(dependency.parameter)
            if not isinstance(value, ComponentConfig | Mapping):
                continue
            nested_loc = (*loc, "params", dependency.parameter)
            try:
                with _at(nested_loc):
                    nested = _as_component_config(value)
                    with _at(("id",)):
                        nested_spec = _dependency_spec(dependency.kind, nested.id)
                resolved_dependency = self.component(nested_spec, nested, nested_loc)
                dependencies.append((dependency.parameter, resolved_dependency))
                params[dependency.parameter] = resolved_dependency.config.as_dict()
            except ConfigurationError as error:
                errors.append(error)
        resolved: dict[str, object] = {}
        try:
            with _at((*loc, "params")):
                resolved = spec.resolved_parameters(
                    params,
                    require_all=True,
                    _constructor=constructor,
                )
        except ConfigurationError as error:
            errors.append(error)
        _raise_errors(errors)
        return _ResolvedComponent(
            spec,
            ComponentConfig(spec.id, resolved),
            constructor,
            MappingProxyType(resolved),
            tuple(dependencies),
            loc,
        )


def _model_instance(component: _ResolvedComponent) -> ModelProtocol:
    model = component.build()
    if not isinstance(model, ModelProtocol):  # pragma: no cover - catalog invariant
        raise RuntimeError(
            f"Catalog model '{component.spec.id}' has an invalid interface"
        )
    return model


def create_model(
    component_id: str,
    params: Mapping[str, object] | None = None,
) -> ModelProtocol:
    """Construct an allowlisted built-in model."""
    spec = get_model_spec(component_id)
    return _model_instance(
        _Resolution().component(
            spec,
            ComponentConfig(component_id, params or {}),
            (),
            declarative=False,
        )
    )


def create_transformer(
    component_id: str,
    params: Mapping[str, object] | None = None,
) -> TransformerProtocol:
    """Construct an allowlisted built-in transformer."""
    spec = get_transformer_spec(component_id)
    transformer = (
        _Resolution()
        .component(
            spec,
            ComponentConfig(component_id, params or {}),
            (),
            declarative=False,
        )
        .build()
    )
    if not isinstance(transformer, TransformerProtocol):  # pragma: no cover
        raise RuntimeError(
            f"Catalog transformer '{component_id}' has an invalid interface"
        )
    return transformer


def create_similarity_engine(
    component_id: str,
    params: Mapping[str, object] | None = None,
) -> BaseSimilaritySearchEngine:
    """Construct an allowlisted built-in similarity-search engine."""
    spec = get_similarity_engine_spec(component_id)
    engine = (
        _Resolution()
        .component(
            spec,
            ComponentConfig(component_id, params or {}),
            (),
            declarative=False,
        )
        .build()
    )
    if not isinstance(engine, BaseSimilaritySearchEngine):  # pragma: no cover
        raise RuntimeError(
            f"Catalog similarity engine '{component_id}' has an invalid interface"
        )
    return engine


@dataclass(frozen=True, slots=True)
class ResolvedDetectorConfig:
    """Reusable canonical configuration and construction plan, without learned state.

    Resolution inspects each constructor once per operation and resolves each
    component occurrence once. Every build creates new components. Fingerprinting
    remains lazy: a runtime dependency may be buildable but not JSON serializable.
    Resolved plans are process-local metadata, not detector checkpoints.
    """

    config: DetectorConfig
    capabilities: ModelCapabilities
    _model: _ResolvedComponent = field(repr=False, compare=False)
    _transformers: tuple[_ResolvedComponent, ...] = field(repr=False, compare=False)

    @property
    def fingerprint(self) -> str:
        """The unchanged canonical JSON digest; rejects non-JSON values."""
        return _fingerprint(self.config)

    def build(self) -> ModelProtocol:
        """Construct independent components, never reusing a learned detector."""
        model = _model_instance(self._model)
        transformers: list[TransformerProtocol] = []
        for component in self._transformers:
            transformer = component.build()
            if not isinstance(transformer, TransformerProtocol):  # pragma: no cover
                raise RuntimeError(
                    f"Catalog transformer '{component.spec.id}' has an invalid interface"
                )
            transformers.append(transformer)
        if not transformers:
            return model
        prefix = transformers[0]
        for transformer in transformers[1:]:
            prefix = Pipeline(prefix, transformer)
        return Pipeline(prefix, model)


def _fingerprint(config: DetectorConfig) -> str:
    try:
        encoded = json.dumps(
            config.as_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "Detector configuration contains non-JSON values and cannot be fingerprinted"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _resolve_components(
    config: DetectorConfig | Mapping[str, object],
) -> tuple[DetectorConfig, _ResolvedComponent, tuple[_ResolvedComponent, ...]]:
    parsed = (
        config
        if isinstance(config, DetectorConfig)
        else DetectorConfig.from_mapping(config)
    )
    resolution = _Resolution()
    errors: list[ConfigurationError] = []
    transformers: list[_ResolvedComponent] = []
    for index, item in enumerate(parsed.transformers):
        loc = ("transformers", index)
        try:
            with _at((*loc, "id")):
                spec = get_transformer_spec(item.id)
            transformers.append(resolution.component(spec, item, loc))
        except ConfigurationError as error:
            errors.append(error)
    model: _ResolvedComponent | None = None
    try:
        with _at(("model", "id")):
            model_spec = get_model_spec(parsed.model.id)
        model = resolution.component(model_spec, parsed.model, ("model",))
    except ConfigurationError as error:
        errors.append(error)
    _raise_errors(errors)
    assert model is not None
    normalized = DetectorConfig(
        model.config, tuple(item.config for item in transformers)
    )
    return normalized, model, tuple(transformers)


def normalize_detector_config(
    config: DetectorConfig | Mapping[str, object],
) -> DetectorConfig:
    """Validate identifiers and parameter types, then apply constructor defaults.

    Does not construct algorithms or require compatible pipeline feature widths.
    """
    normalized, _, _ = _resolve_components(config)
    return normalized


def _capabilities(
    model: _ResolvedComponent,
    transformers: tuple[_ResolvedComponent, ...],
) -> ModelCapabilities:
    feature_count: int | None = None
    for transformer in transformers:
        assert isinstance(transformer.spec, TransformerSpec)
        with _at((*transformer.loc, "params")):
            feature_count = transformer.spec._resolved_output_feature_count(
                transformer.params,
                feature_count,
            )
    assert isinstance(model.spec, ModelSpec)
    with _at((*model.loc, "params")):
        capabilities = model.spec._resolved_capabilities(model.params)
    if feature_count is not None and not capabilities.feature_count.accepts(
        feature_count
    ):
        maximum = capabilities.feature_count.maximum
        expected = (
            str(capabilities.feature_count.minimum)
            if maximum == capabilities.feature_count.minimum
            else f"at least {capabilities.feature_count.minimum}"
            if maximum is None
            else f"{capabilities.feature_count.minimum} to {maximum}"
        )
        raise _invalid(
            "invalid_configuration",
            model.loc,
            f"Model '{model.spec.id}' expects {expected} features, but its "
            f"transformer pipeline produces {feature_count}",
        )
    return capabilities


def resolve_detector_config(
    config: DetectorConfig | Mapping[str, object],
) -> ResolvedDetectorConfig:
    """Resolve a reusable configuration without constructing any algorithms."""
    normalized, model, transformers = _resolve_components(config)
    capabilities = _capabilities(model, transformers)
    return ResolvedDetectorConfig(normalized, capabilities, model, transformers)


def detector_capabilities(
    config: DetectorConfig | Mapping[str, object],
) -> ModelCapabilities:
    """Resolve capabilities and reject statically incompatible feature widths."""
    return resolve_detector_config(config).capabilities


def build_detector(
    config: DetectorConfig | Mapping[str, object],
) -> ModelProtocol:
    """Validate and construct a model-ending pipeline from declarative config."""
    return resolve_detector_config(config).build()
