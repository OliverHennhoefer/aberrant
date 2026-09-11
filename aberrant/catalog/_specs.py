"""Typed, immutable descriptions of built-in ABERRANT components."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from aberrant.base.exceptions import (
    ConfigurationError,
    MissingOptionalDependencyError,
)


class ComponentKind(str, Enum):
    """Kinds of components exposed by the catalog."""

    MODEL = "model"
    TRANSFORMER = "transformer"
    SIMILARITY_ENGINE = "similarity_engine"


class EventKind(str, Enum):
    """Structural shape expected by a model."""

    TABULAR = "tabular"
    UNIVARIATE = "univariate"
    BIVARIATE = "bivariate"
    EDGE = "edge"
    GRAPH_EDGE = "graph_edge"
    SCORE = "score"
    CUSTOM = "custom"


class FeatureSchemaKind(str, Enum):
    """How a model treats feature names after construction or first learning."""

    FIXED = "fixed"
    EVOLVING = "evolving"
    UNRESTRICTED = "unrestricted"
    ENGINE_DEFINED = "engine_defined"


class ScoreKind(str, Enum):
    """Broad numeric contract of a model's anomaly score."""

    ZERO = "zero"
    BINARY = "binary"
    ZERO_TO_ONE = "zero_to_one"
    NON_NEGATIVE = "non_negative"
    SIGNED = "signed"
    ENGINE_DEFINED = "engine_defined"


class StateKind(str, Enum):
    """How model state grows as a stream is processed."""

    STATELESS = "stateless"
    BOUNDED = "bounded"
    GROWING = "growing"
    EXTERNAL = "external"


class WarmupUnit(str, Enum):
    """Unit used to express a model's readiness requirement."""

    EVENTS = "events"
    BUCKETS = "buckets"
    OBSERVERS = "observers"
    ACTIVE_GRAPHS = "active_graphs"
    ENGINE_DEFINED = "engine_defined"


@dataclass(frozen=True, slots=True)
class FeatureCount:
    """Inclusive feature-count bounds for one model input event."""

    minimum: int = 1
    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.minimum <= 0:
            raise ValueError("minimum feature count must be positive")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("maximum feature count must not be below minimum")

    def accepts(self, count: int) -> bool:
        """Return whether ``count`` falls within these bounds."""
        return count >= self.minimum and (self.maximum is None or count <= self.maximum)

    def as_dict(self) -> dict[str, int | None]:
        """Return a JSON-compatible representation."""
        return {"minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True, slots=True)
class WarmupRequirement:
    """Minimum learned history before a model's score contract is ready."""

    minimum: int | None
    unit: WarmupUnit = WarmupUnit.EVENTS

    def __post_init__(self) -> None:
        if self.minimum is not None and self.minimum < 0:
            raise ValueError("warm-up minimum must be non-negative or None")

    def as_dict(self) -> dict[str, int | str | None]:
        """Return a JSON-compatible representation."""
        return {"minimum": self.minimum, "unit": self.unit.value}

    def remaining(self, observed: int) -> int | None:
        """Return the remaining count in ``unit``, or ``None`` if indeterminate."""
        if observed < 0:
            raise ValueError("observed history must be non-negative")
        if self.minimum is None:
            return None
        return max(0, self.minimum - observed)

    def is_satisfied(self, observed: int) -> bool | None:
        """Return readiness, or ``None`` when the catalog cannot determine it."""
        remaining = self.remaining(observed)
        return None if remaining is None else remaining == 0


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Resolved operational facts for one model configuration."""

    event_kind: EventKind
    feature_count: FeatureCount
    feature_schema: FeatureSchemaKind
    score_kind: ScoreKind
    higher_is_more_anomalous: bool | None
    warmup: WarmupRequirement
    state: StateKind
    resettable: bool
    requires_unit_interval: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "event_kind": self.event_kind.value,
            "feature_count": self.feature_count.as_dict(),
            "feature_schema": self.feature_schema.value,
            "score_kind": self.score_kind.value,
            "higher_is_more_anomalous": self.higher_is_more_anomalous,
            "warmup": self.warmup.as_dict(),
            "state": self.state.value,
            "resettable": self.resettable,
            "requires_unit_interval": self.requires_unit_interval,
        }


@dataclass(frozen=True, slots=True)
class ComponentDependency:
    """A constructor parameter containing another catalog component."""

    parameter: str
    kind: ComponentKind


def _qualified_name(value: object) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    return repr(value)


def _annotation_schema(  # noqa: PLR0911, PLR0912
    annotation: object,
) -> dict[str, object]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation is None or annotation is types.NoneType:
        return {"type": "null"}

    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin in (Union, types.UnionType):
        return {"anyOf": [_annotation_schema(argument) for argument in arguments]}
    if origin is Literal:
        return {"enum": list(arguments)}
    if origin is tuple:
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return {"type": "array", "items": _annotation_schema(arguments[0])}
        return {
            "type": "array",
            "prefixItems": [_annotation_schema(argument) for argument in arguments],
            "minItems": len(arguments),
            "maxItems": len(arguments),
        }
    if origin in (list, set, frozenset, Sequence):
        item_schema = _annotation_schema(arguments[0]) if arguments else {}
        return {"type": "array", "items": item_schema}
    if origin in (dict, Mapping):
        key_type, value_type = arguments if len(arguments) == 2 else (Any, Any)
        schema: dict[str, object] = {
            "type": "object",
            "additionalProperties": _annotation_schema(value_type),
        }
        if key_type not in (str, Any):
            schema["x-python-key-type"] = _qualified_name(key_type)
        return schema

    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    return {"x-python-type": _qualified_name(annotation)}


def _json_default(value: object) -> tuple[bool, object]:
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False, repr(value)
    return True, json.loads(encoded)


def _coerce_mapping_key(value: object, annotation: object) -> object:
    if annotation is int and isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise TypeError(f"expected an integer object key, got {value!r}") from exc
    return _coerce_json_value(value, annotation)


def _coerce_json_value(  # noqa: PLR0911, PLR0912, PLR0915
    value: object,
    annotation: object,
) -> object:
    """Validate JSON primitives and restore annotated Python container types."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return value
    if annotation is None or annotation is types.NoneType:
        if value is not None:
            raise TypeError("expected null")
        return None

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, types.UnionType):
        for argument in arguments:
            try:
                return _coerce_json_value(value, argument)
            except (TypeError, ValueError):
                continue
        raise TypeError("value does not match any accepted type")
    if origin is Literal:
        if value not in arguments:
            choices = ", ".join(repr(choice) for choice in arguments)
            raise ValueError(f"expected one of {choices}")
        return value
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise TypeError("expected an array")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_coerce_json_value(item, arguments[0]) for item in value)
        if len(value) != len(arguments):
            raise ValueError(f"expected an array with {len(arguments)} items")
        return tuple(
            _coerce_json_value(item, argument)
            for item, argument in zip(value, arguments, strict=True)
        )
    if origin in (list, set, frozenset, Sequence):
        if not isinstance(value, list | tuple):
            raise TypeError("expected an array")
        item_type = arguments[0] if arguments else Any
        items = [_coerce_json_value(item, item_type) for item in value]
        if origin is set:
            return set(items)
        if origin is frozenset:
            return frozenset(items)
        return items
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise TypeError("expected an object")
        key_type, item_type = arguments if len(arguments) == 2 else (Any, Any)
        return {
            _coerce_mapping_key(key, key_type): _coerce_json_value(item, item_type)
            for key, item in value.items()
        }

    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError("expected a boolean")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("expected an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("expected a number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError("expected a string")
        return value
    if isinstance(annotation, type):
        try:
            matches = isinstance(value, annotation)
        except TypeError:
            return value
        if not matches:
            raise TypeError(f"expected {_qualified_name(annotation)}")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentSpec:
    """Public metadata and construction entry point for one built-in component."""

    id: str
    display_name: str
    kind: ComponentKind
    family: str
    import_path: str
    optional_extra: str | None = None
    dependency_module: str | None = None
    declarative: bool = True
    dependencies: tuple[ComponentDependency, ...] = ()

    @property
    def available(self) -> bool:
        """Whether optional runtime dependencies for this component are installed."""
        return self.dependency_module is None or (
            importlib.util.find_spec(self.dependency_module) is not None
        )

    def load(self) -> type[object]:
        """Load and return the component class without constructing it."""
        if not self.available:
            assert self.optional_extra is not None
            raise MissingOptionalDependencyError(self.id, self.optional_extra)

        module_name, separator, attribute = self.import_path.rpartition(".")
        if not separator:
            raise RuntimeError(f"Invalid catalog import path: {self.import_path}")
        module = importlib.import_module(module_name)
        component = getattr(module, attribute)
        if not isinstance(component, type):
            raise RuntimeError(f"Catalog entry is not a class: {self.import_path}")
        return component

    def resolved_parameters(
        self,
        params: Mapping[str, object] | None = None,
        *,
        require_all: bool = False,
    ) -> dict[str, object]:
        """Bind parameters to the public constructor and apply its defaults."""
        signature = inspect.signature(self.load())
        values = dict(params or {})
        try:
            bound = (
                signature.bind(**values)
                if require_all
                else signature.bind_partial(**values)
            )
        except TypeError as exc:
            raise ConfigurationError(
                f"Invalid parameters for {self.kind.value} '{self.id}': {exc}"
            ) from exc
        bound.apply_defaults()
        try:
            type_hints = get_type_hints(self.load().__init__)
        except (NameError, TypeError):
            type_hints = {}
        resolved: dict[str, object] = {}
        dependency_parameters = {item.parameter for item in self.dependencies}
        for name, value in bound.arguments.items():
            if name in dependency_parameters and isinstance(value, Mapping):
                resolved[name] = value
                continue
            try:
                resolved[name] = _coerce_json_value(
                    value,
                    type_hints.get(name, inspect.Parameter.empty),
                )
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"Invalid parameter '{name}' for {self.kind.value} "
                    f"'{self.id}': {exc}"
                ) from exc
        return resolved

    def parameter_schema(self) -> dict[str, object]:
        """Inspect constructor parameters, loading the component's runtime."""
        component = self.load()
        signature = inspect.signature(component)
        try:
            type_hints = get_type_hints(component.__init__)
        except (NameError, TypeError):
            type_hints = {}

        dependency_by_parameter = {
            dependency.parameter: dependency for dependency in self.dependencies
        }
        properties: dict[str, object] = {}
        required: list[str] = []
        for name, parameter in signature.parameters.items():
            annotation = type_hints.get(name, parameter.annotation)
            dependency = dependency_by_parameter.get(name)
            if dependency is None:
                property_schema = _annotation_schema(annotation)
            else:
                property_schema = {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "params": {"type": "object", "default": {}},
                    },
                    "required": ["id"],
                    "additionalProperties": False,
                    "x-aberrant-component-kind": dependency.kind.value,
                }
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
            else:
                serializable, default = _json_default(parameter.default)
                property_schema["default" if serializable else "x-python-default"] = (
                    default
                )
            properties[name] = property_schema

        schema: dict[str, object] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return schema

    def as_dict(self) -> dict[str, object]:
        """Return catalog metadata, deferring optional-runtime inspection."""
        result: dict[str, object] = {
            "id": self.id,
            "display_name": self.display_name,
            "kind": self.kind.value,
            "family": self.family,
            "import_path": self.import_path,
            "available": self.available,
            "optional_extra": self.optional_extra,
            "declarative": self.declarative,
            "dependencies": [
                {"parameter": dependency.parameter, "kind": dependency.kind.value}
                for dependency in self.dependencies
            ],
        }
        result["parameters"] = (
            self.parameter_schema() if self.dependency_module is None else None
        )
        return result


CapabilityResolver = Callable[[Mapping[str, object]], ModelCapabilities]
FeatureCountResolver = Callable[[Mapping[str, object], int | None], int | None]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSpec(ComponentSpec):
    """Catalog entry for a model with configuration-resolved capabilities."""

    _capability_resolver: CapabilityResolver = field(repr=False, compare=False)

    def capabilities(
        self, params: Mapping[str, object] | None = None
    ) -> ModelCapabilities:
        """Resolve operational capabilities using constructor defaults."""
        resolved = (
            self.resolved_parameters(params) if self.available else dict(params or {})
        )
        try:
            return self._capability_resolver(resolved)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Cannot resolve capabilities for model '{self.id}': {exc}"
            ) from exc

    def as_dict(self) -> dict[str, object]:
        result = ComponentSpec.as_dict(self)
        try:
            result["capabilities"] = (
                self.capabilities().as_dict()
                if self.dependency_module is None
                else None
            )
        except ConfigurationError:
            result["capabilities"] = None
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformerSpec(ComponentSpec):
    """Catalog entry for a transformer and its output feature shape."""

    _feature_count_resolver: FeatureCountResolver = field(
        repr=False,
        compare=False,
    )

    def output_feature_count(
        self,
        params: Mapping[str, object] | None,
        input_feature_count: int | None,
    ) -> int | None:
        """Resolve the transformer's output width when it can be known."""
        resolved = self.resolved_parameters(params)
        try:
            return self._feature_count_resolver(resolved, input_feature_count)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Cannot resolve feature count for transformer '{self.id}': {exc}"
            ) from exc
