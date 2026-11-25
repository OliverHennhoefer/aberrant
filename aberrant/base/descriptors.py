"""Parameter descriptions shared by validation and JSON Schema.

The component catalog derives them from constructor annotations; the
calibration methods declare theirs. Both validate and describe parameters
the same way.
"""

from __future__ import annotations

import inspect
import json
import math
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin

from aberrant.exceptions import ConfigurationError, ConfigurationIssue

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


def invalid(code: str, loc: Location, message: str) -> ConfigurationError:
    """One configuration error with a single issue at ``loc``."""
    return ConfigurationError(message, issues=(ConfigurationIssue(code, loc, message),))


def raise_errors(errors: list[ConfigurationError]) -> None:
    """Raise the first error, carrying every error's message and issues."""
    if not errors:
        return
    first = errors[0]
    if len(errors) > 1:
        # Keep the first failure's subclass and attributes, as fail-fast callers
        # previously observed, while exposing every independent diagnostic.
        first.args = ("; ".join(str(error) for error in errors),)
        first._issues = tuple(issue for error in errors for issue in error.issues)
    raise first


@dataclass(frozen=True, slots=True)
class ValueDescriptor:
    """The values a parameter accepts, read once for validation and schema.

    ``kind`` is a JSON Schema type (``"number"``, ``"integer"``, ...) or one of
    ``"any"``, ``"union"``, ``"literal"``, ``"tuple"``, ``"tuple_items"``,
    ``"mapping"`` and ``"python"``. Numeric kinds may carry bounds; an
    exclusive bound rejects the bound itself.
    """

    kind: str
    annotation: object
    children: tuple[ValueDescriptor, ...] = ()
    choices: tuple[object, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False

    @classmethod
    def number(
        cls,
        *,
        integer: bool = False,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_minimum: bool = False,
        exclusive_maximum: bool = False,
    ) -> ValueDescriptor:
        """A bounded number, or a bounded integer with ``integer=True``."""
        return cls(
            "integer" if integer else "number",
            int if integer else float,
            minimum=minimum,
            maximum=maximum,
            exclusive_minimum=exclusive_minimum,
            exclusive_maximum=exclusive_maximum,
        )

    @classmethod
    def from_annotation(cls, annotation: object) -> ValueDescriptor:  # noqa: PLR0911
        """Describe the values a type annotation admits."""
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
        """The JSON Schema of the accepted values."""
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
        result = {"type": self.kind}
        if self.minimum is not None:
            result["exclusiveMinimum" if self.exclusive_minimum else "minimum"] = (
                self.minimum
            )
        if self.maximum is not None:
            result["exclusiveMaximum" if self.exclusive_maximum else "maximum"] = (
                self.maximum
            )
        return result

    def coerce(self, value: object, loc: Location = ()) -> object:  # noqa: PLR0911, PLR0912
        """Validate ``value``, raising a ``ConfigurationError`` at ``loc``."""
        if self.kind == "any":
            return value
        if self.kind == "union":
            for child in self.children:
                try:
                    return child.coerce(value, loc)
                except ConfigurationError:
                    continue
            raise invalid("invalid_type", loc, "value does not match any accepted type")
        if self.kind == "literal":
            if value not in self.choices:
                choices = ", ".join(repr(choice) for choice in self.choices)
                raise invalid("invalid_value", loc, f"expected one of {choices}")
            return value
        if self.kind in ("array", "tuple", "tuple_items"):
            return self._array(value, loc)
        if self.kind == "mapping":
            return self._mapping(value, loc)
        if self.kind == "null":
            if value is not None:
                raise invalid("invalid_type", loc, "expected null")
            return None
        if self.kind == "boolean" and not isinstance(value, bool):
            raise invalid("invalid_type", loc, "expected a boolean")
        if self.kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise invalid("invalid_type", loc, "expected an integer")
            return self._bounded(value, loc)
        if self.kind == "number":
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise invalid("invalid_type", loc, "expected a number")
            if not math.isfinite(value):
                raise invalid("invalid_value", loc, "expected a finite number")
            return self._bounded(float(value), loc)
        if self.kind == "string" and not isinstance(value, str):
            raise invalid("invalid_type", loc, "expected a string")
        if self.kind == "python" and isinstance(self.annotation, type):
            try:
                matches = isinstance(value, self.annotation)
            except TypeError:
                return value
            if not matches:
                raise invalid(
                    "invalid_type", loc, f"expected {_qualified_name(self.annotation)}"
                )
        return value

    def _bounded(self, value: float, loc: Location) -> float:
        low, high = self.minimum, self.maximum
        too_low = low is not None and (
            value <= low if self.exclusive_minimum else value < low
        )
        too_high = high is not None and (
            value >= high if self.exclusive_maximum else value > high
        )
        if too_low or too_high:
            left = "(" if self.exclusive_minimum or low is None else "["
            right = ")" if self.exclusive_maximum or high is None else "]"
            lower = "-inf" if low is None else f"{low:g}"
            upper = "inf" if high is None else f"{high:g}"
            raise invalid(
                "invalid_value",
                loc,
                f"expected a value in {left}{lower}, {upper}{right}",
            )
        return value

    def _array(self, value: object, loc: Location) -> object:
        unordered = self.annotation in (set, frozenset)
        if not isinstance(value, list | tuple) and not (
            unordered and isinstance(value, set | frozenset)
        ):
            raise invalid("invalid_type", loc, "expected an array")
        if self.kind == "tuple" and len(value) != len(self.children):
            raise invalid(
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
        raise_errors(errors)
        if self.kind in ("tuple", "tuple_items"):
            return tuple(items)
        if self.annotation is set:
            return set(items)
        if self.annotation is frozenset:
            return frozenset(items)
        return items

    def _mapping(self, value: object, loc: Location) -> object:
        if not isinstance(value, Mapping):
            raise invalid("invalid_type", loc, "expected an object")
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
                        raise invalid(
                            "invalid_type",
                            item_loc,
                            f"expected an integer object key, got {key!r}",
                        ) from error
                else:
                    resolved_key = key_descriptor.coerce(key, item_loc)
                result[resolved_key] = item_descriptor.coerce(item, item_loc)
            except ConfigurationError as error:
                errors.append(error)
        raise_errors(errors)
        return result


@dataclass(frozen=True, slots=True)
class ParameterDescriptor:
    """One named parameter: the values it accepts, its default and its label.

    ``default`` is ``inspect.Parameter.empty`` for a required parameter.
    ``title`` and ``description`` are what an editor shows for it.
    """

    name: str
    value: ValueDescriptor
    default: object = inspect.Parameter.empty
    title: str | None = None
    description: str = ""

    @property
    def required(self) -> bool:
        """Whether the parameter has no default."""
        return self.default is inspect.Parameter.empty

    def coerce(self, value: object) -> object:
        """Validate a value for this parameter."""
        return self.value.coerce(value, (self.name,))

    def schema(self, value: dict[str, object] | None = None) -> dict[str, object]:
        """The JSON Schema of this parameter, around ``value`` if given."""
        result = self.value.schema() if value is None else value
        if self.title is not None:
            result["title"] = self.title
        if self.description:
            result["description"] = self.description
        if not self.required:
            try:
                result["default"] = json.loads(
                    json.dumps(self.default, allow_nan=False)
                )
            except (TypeError, ValueError):
                result["x-python-default"] = repr(self.default)
        return result


def parameters_schema(
    parameters: Sequence[ParameterDescriptor],
    values: Mapping[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """The JSON Schema of an object holding ``parameters``.

    ``values`` replaces the value schema of the parameters it names.
    """
    values = values or {}
    result: dict[str, object] = {
        "type": "object",
        "properties": {
            item.name: item.schema(values.get(item.name)) for item in parameters
        },
        "additionalProperties": False,
    }
    required = [item.name for item in parameters if item.required]
    if required:
        result["required"] = required
    return result


def resolve_parameters(
    parameters: Sequence[ParameterDescriptor],
    params: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    """Every parameter's value: ``params`` validated, defaults for the rest.

    Raises a ``ConfigurationError`` listing each unknown, missing or invalid
    parameter.
    """
    known = {item.name: item for item in parameters}
    errors = [
        invalid("unexpected_parameter", (name,), f"{label} has no parameter {name!r}")
        for name in params
        if name not in known
    ]
    resolved: dict[str, object] = {}
    for item in parameters:
        if item.name not in params:
            if item.required:
                errors.append(
                    invalid(
                        "missing_parameter",
                        (item.name,),
                        f"{label} needs parameter {item.name!r}",
                    )
                )
            else:
                resolved[item.name] = copy_parameter(item.default)
            continue
        try:
            resolved[item.name] = item.coerce(params[item.name])
        except ConfigurationError as error:
            errors.append(
                ConfigurationError(
                    f"Invalid parameter {item.name!r} for {label}: {error}",
                    issues=error.issues,
                )
            )
    raise_errors(errors)
    return resolved
