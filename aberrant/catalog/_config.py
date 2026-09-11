"""Safe declarative construction from built-in catalog identifiers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from aberrant.base import (
    BaseSimilaritySearchEngine,
    ConfigurationError,
    ModelProtocol,
    Pipeline,
    TransformerProtocol,
)

from ._registry import (
    get_model_spec,
    get_similarity_engine_spec,
    get_transformer_spec,
)
from ._specs import ComponentKind, ComponentSpec, ModelCapabilities

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
            raise ConfigurationError("Component 'id' must be a non-empty string")
        if not isinstance(self.params, Mapping):
            raise ConfigurationError("Component 'params' must be an object")
        if any(not isinstance(name, str) for name in self.params):
            raise ConfigurationError("Component parameter names must be strings")
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
            raise ConfigurationError(f"Unexpected component fields: {names}")
        if "id" not in value:
            raise ConfigurationError("Component configuration requires 'id'")
        params = value.get("params", {})
        if not isinstance(params, Mapping):
            raise ConfigurationError("Component 'params' must be an object")
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
            raise ConfigurationError("Detector 'model' must be a component config")
        try:
            transformers = tuple(self.transformers)
        except TypeError as exc:
            raise ConfigurationError(
                "Detector 'transformers' must contain component configs"
            ) from exc
        if any(not isinstance(item, ComponentConfig) for item in transformers):
            raise ConfigurationError(
                "Detector 'transformers' must contain component configs"
            )
        object.__setattr__(self, "transformers", transformers)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> DetectorConfig:
        """Parse the versioned, strict detector configuration format."""
        unexpected = set(value) - {"version", "model", "transformers"}
        if unexpected:
            names = ", ".join(sorted(str(name) for name in unexpected))
            raise ConfigurationError(f"Unexpected detector fields: {names}")
        version = value.get("version", CONFIG_FORMAT_VERSION)
        if type(version) is not int or version != CONFIG_FORMAT_VERSION:
            raise ConfigurationError(
                f"Unsupported detector config version: {version!r}; "
                f"expected {CONFIG_FORMAT_VERSION}"
            )
        if "model" not in value:
            raise ConfigurationError("Detector configuration requires 'model'")

        raw_model = value["model"]
        if isinstance(raw_model, ComponentConfig):
            model = raw_model
        elif isinstance(raw_model, Mapping):
            model = ComponentConfig.from_mapping(raw_model)
        else:
            raise ConfigurationError("Detector 'model' must be an object")

        raw_transformers = value.get("transformers", ())
        if not isinstance(raw_transformers, Sequence) or isinstance(
            raw_transformers, str | bytes
        ):
            raise ConfigurationError("Detector 'transformers' must be an array")
        transformers: list[ComponentConfig] = []
        for item in raw_transformers:
            if isinstance(item, ComponentConfig):
                transformers.append(item)
            elif isinstance(item, Mapping):
                transformers.append(ComponentConfig.from_mapping(item))
            else:
                raise ConfigurationError("Every detector transformer must be an object")
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
        try:
            encoded = json.dumps(
                self.normalized().as_dict(),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "Detector configuration contains non-JSON values and cannot be "
                "fingerprinted"
            ) from exc
        return hashlib.sha256(encoded).hexdigest()

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


def _normalize_component(
    spec: ComponentSpec,
    config: ComponentConfig,
) -> ComponentConfig:
    if not spec.declarative:
        raise ConfigurationError(
            f"{spec.kind.value.title()} '{spec.id}' requires runtime Python objects "
            "and cannot be built from declarative configuration"
        )
    params = dict(config.params)
    for dependency in spec.dependencies:
        value = params.get(dependency.parameter)
        if not isinstance(value, ComponentConfig | Mapping):
            continue
        dependency_config = _as_component_config(value)
        dependency_spec = _dependency_spec(dependency.kind, dependency_config.id)
        params[dependency.parameter] = _normalize_component(
            dependency_spec,
            dependency_config,
        ).as_dict()
    return ComponentConfig(
        spec.id,
        spec.resolved_parameters(params, require_all=True),
    )


def _construct(spec: ComponentSpec, config: ComponentConfig) -> object:
    params = dict(config.params)
    for dependency in spec.dependencies:
        if dependency.parameter not in params:
            continue
        value = params[dependency.parameter]
        if isinstance(value, ComponentConfig | Mapping):
            dependency_config = _as_component_config(value)
            if dependency.kind is ComponentKind.SIMILARITY_ENGINE:
                params[dependency.parameter] = create_similarity_engine(
                    dependency_config.id,
                    dependency_config.params,
                )
            else:  # pragma: no cover - guarded by catalog definitions
                raise RuntimeError(
                    f"Unsupported nested component kind: {dependency.kind.value}"
                )

    resolved = spec.resolved_parameters(params, require_all=True)
    component_type = spec.load()
    try:
        return component_type(**resolved)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Invalid parameters for {spec.kind.value} '{spec.id}': {exc}"
        ) from exc


def create_model(
    component_id: str,
    params: Mapping[str, object] | None = None,
) -> ModelProtocol:
    """Construct an allowlisted built-in model."""
    spec = get_model_spec(component_id)
    model = _construct(spec, ComponentConfig(component_id, params or {}))
    if not isinstance(model, ModelProtocol):  # pragma: no cover - catalog invariant
        raise RuntimeError(f"Catalog model '{component_id}' has an invalid interface")
    return model


def create_transformer(
    component_id: str,
    params: Mapping[str, object] | None = None,
) -> TransformerProtocol:
    """Construct an allowlisted built-in transformer."""
    spec = get_transformer_spec(component_id)
    transformer = _construct(spec, ComponentConfig(component_id, params or {}))
    if not isinstance(
        transformer, TransformerProtocol
    ):  # pragma: no cover - catalog invariant
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
    engine = _construct(spec, ComponentConfig(component_id, params or {}))
    if not isinstance(
        engine, BaseSimilaritySearchEngine
    ):  # pragma: no cover - catalog invariant
        raise RuntimeError(
            f"Catalog similarity engine '{component_id}' has an invalid interface"
        )
    return engine


def _normalize_detector_config(
    config: DetectorConfig | Mapping[str, object],
) -> DetectorConfig:
    return (
        config
        if isinstance(config, DetectorConfig)
        else DetectorConfig.from_mapping(config)
    )


def normalize_detector_config(
    config: DetectorConfig | Mapping[str, object],
) -> DetectorConfig:
    """Validate identifiers and parameter types, then apply constructor defaults."""
    parsed = _normalize_detector_config(config)
    transformers = tuple(
        _normalize_component(get_transformer_spec(item.id), item)
        for item in parsed.transformers
    )
    model = _normalize_component(get_model_spec(parsed.model.id), parsed.model)
    return DetectorConfig(model=model, transformers=transformers)


def detector_capabilities(
    config: DetectorConfig | Mapping[str, object],
) -> ModelCapabilities:
    """Resolve capabilities and reject statically incompatible feature widths."""
    normalized = normalize_detector_config(config)
    feature_count: int | None = None
    for transformer_config in normalized.transformers:
        spec = get_transformer_spec(transformer_config.id)
        feature_count = spec.output_feature_count(
            transformer_config.params,
            feature_count,
        )

    model_spec = get_model_spec(normalized.model.id)
    capabilities = model_spec.capabilities(normalized.model.params)
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
        raise ConfigurationError(
            f"Model '{model_spec.id}' expects {expected} features, but its "
            f"transformer pipeline produces {feature_count}"
        )
    return capabilities


def build_detector(
    config: DetectorConfig | Mapping[str, object],
) -> ModelProtocol:
    """Validate and construct a model-ending pipeline from declarative config."""
    normalized = normalize_detector_config(config)
    detector_capabilities(normalized)
    model = create_model(normalized.model.id, normalized.model.params)
    transformers = [
        create_transformer(item.id, item.params) for item in normalized.transformers
    ]
    if not transformers:
        return model

    prefix: TransformerProtocol = transformers[0]
    for transformer in transformers[1:]:
        prefix = Pipeline(prefix, transformer)
    return Pipeline(prefix, model)
