"""Typed, immutable descriptions of built-in ABERRANT components."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from aberrant.base.exceptions import (
    ConfigurationError,
    MissingOptionalDependencyError,
)

from ._parameters import ConstructorParameters


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
    RETAINED_VALUES = "retained_values"
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
        _constructor: ConstructorParameters | None = None,
    ) -> dict[str, object]:
        """Bind parameters to the public constructor and apply its defaults."""
        constructor = _constructor or ConstructorParameters.inspect(self.load())
        return constructor.resolve(
            dict(params or {}),
            require_all=require_all,
            dependencies={item.parameter for item in self.dependencies},
            label=f"{self.kind.value} '{self.id}'",
        )

    def parameter_schema(self) -> dict[str, object]:
        """Inspect constructor parameters, loading the component's runtime.

        Each call returns detached metadata that applications may constrain.
        """
        return ConstructorParameters.inspect(self.load()).schema(
            {
                dependency.parameter: dependency.kind.value
                for dependency in self.dependencies
            }
        )

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
FeatureNamesResolver = Callable[
    [Mapping[str, object], tuple[str, ...] | None], tuple[str, ...] | None
]


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
        return self._resolved_capabilities(resolved)

    def _resolved_capabilities(
        self, resolved: Mapping[str, object]
    ) -> ModelCapabilities:
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
    _feature_names_resolver: FeatureNamesResolver | None = field(
        default=None, repr=False, compare=False
    )

    def output_feature_names(
        self,
        params: Mapping[str, object] | None,
        input_feature_names: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        """Resolve public output names without learning any observations."""
        resolved = self.resolved_parameters(params)
        if self._feature_names_resolver is None:
            return input_feature_names
        try:
            return self._feature_names_resolver(resolved, input_feature_names)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Cannot resolve feature names for transformer '{self.id}': {exc}"
            ) from exc

    def output_feature_count(
        self,
        params: Mapping[str, object] | None,
        input_feature_count: int | None,
    ) -> int | None:
        """Resolve the transformer's output width when it can be known."""
        resolved = self.resolved_parameters(params)
        return self._resolved_output_feature_count(resolved, input_feature_count)

    def _resolved_output_feature_count(
        self,
        resolved: Mapping[str, object],
        input_feature_count: int | None,
    ) -> int | None:
        try:
            return self._feature_count_resolver(resolved, input_feature_count)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Cannot resolve feature count for transformer '{self.id}': {exc}"
            ) from exc
