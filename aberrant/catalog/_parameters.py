"""Constructor parameters read from a component's signature."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import get_type_hints

from aberrant.base.descriptors import (
    ParameterDescriptor,
    ValueDescriptor,
    copy_parameter,
    invalid,
    parameters_schema,
    raise_errors,
)
from aberrant.exceptions import ConfigurationError, ConfigurationIssue


def _dependency_schema(kind: str) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "params": {"type": "object", "default": {}},
        },
        "required": ["id"],
        "additionalProperties": False,
        "x-aberrant-component-kind": kind,
    }


@dataclass(frozen=True, slots=True)
class ConstructorParameters:
    component: type[object]
    signature: inspect.Signature
    parameters: tuple[ParameterDescriptor, ...]

    @classmethod
    def inspect(cls, component: type[object]) -> ConstructorParameters:
        signature = inspect.signature(component)
        try:
            hints = get_type_hints(component.__init__)
        except (NameError, TypeError):
            hints = {}
        return cls(
            component,
            signature,
            tuple(
                ParameterDescriptor(
                    name,
                    ValueDescriptor.from_annotation(
                        hints.get(name, parameter.annotation)
                    ),
                    parameter.default,
                )
                for name, parameter in signature.parameters.items()
            ),
        )

    def resolve(
        self,
        params: Mapping[str, object],
        *,
        require_all: bool,
        dependencies: set[str],
        label: str,
    ) -> dict[str, object]:
        errors: list[ConfigurationError] = []
        values: Mapping[str, object]
        try:
            bound = (
                self.signature.bind if require_all else self.signature.bind_partial
            )(**params)
            bound.apply_defaults()
            values = bound.arguments
        except TypeError as error:
            message = f"Invalid parameters for {label}: {error}"
            issues = self._binding_issues(params, require_all, message)
            errors.append(ConfigurationError(message, issues=issues))
            values = dict(params)
            values.update(
                {
                    item.name: item.default
                    for item in self.parameters
                    if item.name not in values and not item.required
                }
            )
        resolved: dict[str, object] = {}
        for parameter in self.parameters:
            name = parameter.name
            if name not in values:
                continue
            value = values[name]
            if name in dependencies and isinstance(value, Mapping):
                resolved[name] = copy_parameter(value)
                continue
            try:
                resolved[name] = copy_parameter(parameter.value.coerce(value, (name,)))
            except ConfigurationError as error:
                errors.append(
                    ConfigurationError(
                        f"Invalid parameter '{name}' for {label}: {error}",
                        issues=error.issues,
                    )
                )
            except (TypeError, ValueError) as error:
                errors.append(
                    invalid(
                        "invalid_type"
                        if isinstance(error, TypeError)
                        else "invalid_value",
                        (name,),
                        f"Invalid parameter '{name}' for {label}: {error}",
                    )
                )
        raise_errors(errors)
        return resolved

    def _binding_issues(
        self, params: Mapping[str, object], require_all: bool, message: str
    ) -> list[ConfigurationIssue]:
        issues = [
            ConfigurationIssue("missing_parameter", (item.name,), message)
            for item in self.parameters
            if require_all
            and item.required
            and item.name not in params
            and self.signature.parameters[item.name].kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if not any(
            item.kind is inspect.Parameter.VAR_KEYWORD
            for item in self.signature.parameters.values()
        ):
            issues.extend(
                ConfigurationIssue("unexpected_parameter", (name,), message)
                for name in params
                if name not in self.signature.parameters
            )
        return issues or [ConfigurationIssue("invalid_configuration", (), message)]

    def schema(self, dependencies: Mapping[str, str]) -> dict[str, object]:
        return parameters_schema(
            self.parameters,
            {name: _dependency_schema(kind) for name, kind in dependencies.items()},
        )
