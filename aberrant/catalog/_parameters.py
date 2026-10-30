"""Operation-local constructor descriptors shared by coercion and JSON Schema."""

from __future__ import annotations

import inspect
import json
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from aberrant.base.exceptions import ConfigurationError, ConfigurationIssue

Location = tuple[str | int, ...]


def copy_parameter(value: object) -> object:
    """Detach containers without copying caller-owned runtime dependencies."""
    if isinstance(value, Mapping):
        return {key: copy_parameter(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_parameter(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_parameter(item) for item in value)
    if isinstance(value, set | frozenset):
        return type(value)(copy_parameter(item) for item in value)
    return value


def _qualified_name(value: object) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    return f"{module}.{name}" if module and name else repr(value)


def _invalid(code: str, loc: Location, message: str) -> ConfigurationError:
    return ConfigurationError(message, issues=(ConfigurationIssue(code, loc, message),))


def _raise_errors(errors: list[ConfigurationError]) -> None:
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ConfigurationError(
            "; ".join(str(error) for error in errors),
            issues=tuple(issue for error in errors for issue in error.issues),
        )


@dataclass(frozen=True, slots=True)
class ValueDescriptor:
    """Interpret an annotation once; both schema and coercion use this shape."""

    kind: str
    annotation: object
    children: tuple[ValueDescriptor, ...] = ()
    choices: tuple[object, ...] = ()

    @classmethod
    def from_annotation(cls, annotation: object) -> ValueDescriptor:  # noqa: PLR0911
        if annotation is inspect.Parameter.empty or annotation is Any:
            return cls("any", annotation)
        if annotation is None or annotation is types.NoneType:
            return cls("null", annotation)
        origin, arguments = get_origin(annotation), get_args(annotation)
        if origin in (Union, types.UnionType):
            return cls("union", annotation, tuple(map(cls.from_annotation, arguments)))
        if origin is Literal:
            return cls("literal", annotation, choices=arguments)
        if origin is tuple:
            variadic = len(arguments) == 2 and arguments[1] is Ellipsis
            children = arguments[:1] if variadic else arguments
            return cls(
                "tuple_items" if variadic else "tuple",
                annotation,
                tuple(map(cls.from_annotation, children)),
            )
        if origin in (list, set, frozenset, Sequence):
            return cls(
                "array",
                origin,
                (cls.from_annotation(arguments[0] if arguments else Any),),
            )
        if origin in (dict, Mapping):
            arguments = arguments if len(arguments) == 2 else (Any, Any)
            return cls("mapping", origin, tuple(map(cls.from_annotation, arguments)))
        primitives: dict[object, str] = {
            str: "string",
            bool: "boolean",
            int: "integer",
            float: "number",
        }
        return cls(primitives.get(annotation, "python"), annotation)

    def schema(self) -> dict[str, object]:  # noqa: PLR0911
        if self.kind == "any":
            return {}
        if self.kind == "union":
            return {"anyOf": [child.schema() for child in self.children]}
        if self.kind == "literal":
            return {"enum": list(self.choices)}
        if self.kind == "tuple":
            return {
                "type": "array",
                "prefixItems": [child.schema() for child in self.children],
                "minItems": len(self.children),
                "maxItems": len(self.children),
            }
        if self.kind in ("array", "tuple_items"):
            return {"type": "array", "items": self.children[0].schema()}
        if self.kind == "mapping":
            key, value = self.children
            result: dict[str, object] = {
                "type": "object",
                "additionalProperties": value.schema(),
            }
            if key.annotation not in (str, Any):
                result["x-python-key-type"] = _qualified_name(key.annotation)
            return result
        if self.kind == "python":
            return {"x-python-type": _qualified_name(self.annotation)}
        return {"type": self.kind}

    def coerce(self, value: object, loc: Location = ()) -> object:  # noqa: PLR0911, PLR0912
        if self.kind == "any":
            return value
        if self.kind == "union":
            for child in self.children:
                try:
                    return child.coerce(value, loc)
                except ConfigurationError:
                    continue
            raise _invalid(
                "invalid_type", loc, "value does not match any accepted type"
            )
        if self.kind == "literal":
            if value not in self.choices:
                choices = ", ".join(repr(choice) for choice in self.choices)
                raise _invalid("invalid_value", loc, f"expected one of {choices}")
            return value
        if self.kind in ("array", "tuple", "tuple_items"):
            return self._array(value, loc)
        if self.kind == "mapping":
            return self._mapping(value, loc)
        if self.kind == "null":
            if value is not None:
                raise _invalid("invalid_type", loc, "expected null")
            return None
        if self.kind == "boolean" and not isinstance(value, bool):
            raise _invalid("invalid_type", loc, "expected a boolean")
        if self.kind == "integer" and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise _invalid("invalid_type", loc, "expected an integer")
        if self.kind == "number":
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise _invalid("invalid_type", loc, "expected a number")
            return float(value)
        if self.kind == "string" and not isinstance(value, str):
            raise _invalid("invalid_type", loc, "expected a string")
        if self.kind == "python" and isinstance(self.annotation, type):
            try:
                matches = isinstance(value, self.annotation)
            except TypeError:
                return value
            if not matches:
                raise _invalid(
                    "invalid_type", loc, f"expected {_qualified_name(self.annotation)}"
                )
        return value

    def _array(self, value: object, loc: Location) -> object:
        if not isinstance(value, list | tuple):
            raise _invalid("invalid_type", loc, "expected an array")
        if self.kind == "tuple" and len(value) != len(self.children):
            raise _invalid(
                "invalid_value",
                loc,
                f"expected an array with {len(self.children)} items",
            )
        errors: list[ConfigurationError] = []
        items: list[object] = []
        for index, item in enumerate(value):
            child = self.children[index] if self.kind == "tuple" else self.children[0]
            try:
                items.append(child.coerce(item, (*loc, index)))
            except ConfigurationError as error:
                errors.append(error)
        _raise_errors(errors)
        if self.kind in ("tuple", "tuple_items"):
            return tuple(items)
        if self.annotation is set:
            return set(items)
        if self.annotation is frozenset:
            return frozenset(items)
        return items

    def _mapping(self, value: object, loc: Location) -> object:
        if not isinstance(value, Mapping):
            raise _invalid("invalid_type", loc, "expected an object")
        key_descriptor, item_descriptor = self.children
        result: dict[object, object] = {}
        errors: list[ConfigurationError] = []
        for key, item in value.items():
            item_loc = (*loc, key if isinstance(key, str | int) else str(key))
            try:
                resolved_key: object
                if key_descriptor.annotation is int and isinstance(key, str):
                    try:
                        resolved_key = int(key)
                    except ValueError as error:
                        raise _invalid(
                            "invalid_type",
                            item_loc,
                            f"expected an integer object key, got {key!r}",
                        ) from error
                else:
                    resolved_key = key_descriptor.coerce(key, item_loc)
                result[resolved_key] = item_descriptor.coerce(item, item_loc)
            except ConfigurationError as error:
                errors.append(error)
        _raise_errors(errors)
        return result


@dataclass(frozen=True, slots=True)
class ParameterDescriptor:
    name: str
    value: ValueDescriptor
    default: object

    @property
    def required(self) -> bool:
        return self.default is inspect.Parameter.empty

    def schema(self, dependency_kind: str | None) -> dict[str, object]:
        result = (
            self.value.schema()
            if dependency_kind is None
            else {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "params": {"type": "object", "default": {}},
                },
                "required": ["id"],
                "additionalProperties": False,
                "x-aberrant-component-kind": dependency_kind,
            }
        )
        if not self.required:
            try:
                result["default"] = json.loads(
                    json.dumps(self.default, allow_nan=False)
                )
            except (TypeError, ValueError):
                result["x-python-default"] = repr(self.default)
        return result


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
                    _invalid(
                        "invalid_type"
                        if isinstance(error, TypeError)
                        else "invalid_value",
                        (name,),
                        f"Invalid parameter '{name}' for {label}: {error}",
                    )
                )
        _raise_errors(errors)
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
        result: dict[str, object] = {
            "type": "object",
            "properties": {
                item.name: item.schema(dependencies.get(item.name))
                for item in self.parameters
            },
            "additionalProperties": False,
        }
        required = [item.name for item in self.parameters if item.required]
        if required:
            result["required"] = required
        return result
