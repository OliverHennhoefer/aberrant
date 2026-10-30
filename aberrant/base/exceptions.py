"""Custom exceptions for the aberrant library."""

from collections.abc import Sequence
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class ConfigurationIssue:
    """One configuration failure at a path relative to the submitted value.

    Codes are ``missing_parameter``, ``unexpected_parameter``, ``invalid_type``,
    ``invalid_value``, ``unknown_component``, ``missing_dependency`` and
    ``invalid_configuration``. Locations contain object keys and array indexes.
    """

    code: str
    loc: tuple[str | int, ...]
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "loc", tuple(self.loc))


class AberrantError(Exception):
    """Base exception class for all aberrant-specific errors."""

    pass


class ModelNotFittedError(AberrantError):
    """Raised when a model method is called before the model has been fitted."""

    def __init__(self, message: str = "Model has not been fitted yet.") -> None:
        super().__init__(message)


class TransformationError(AberrantError):
    """Raised when a transformation operation fails."""

    pass


class PipelineError(AberrantError):
    """Raised when a pipeline operation fails."""

    pass


class ValidationError(AberrantError):
    """Raised when input validation fails."""

    pass


class ConfigurationError(AberrantError):
    """Raised when declarative component configuration is invalid."""

    def __init__(
        self, message: str = "", *, issues: Sequence[ConfigurationIssue] = ()
    ) -> None:
        super().__init__(message)
        self._issues = tuple(issues) or (
            ConfigurationIssue("invalid_configuration", (), message),
        )

    @property
    def issues(self) -> tuple[ConfigurationIssue, ...]:
        """Immutable diagnostics, independent of the human-readable message."""
        return self._issues

    def _prepend_location(self, loc: tuple[str | int, ...]) -> None:
        self._issues = tuple(
            replace(issue, loc=loc + issue.loc) for issue in self.issues
        )


class UnknownComponentError(ConfigurationError):
    """Raised when a catalog component identifier is unknown."""

    def __init__(self, component_kind: str, component_id: str) -> None:
        message = f"Unknown {component_kind} component: '{component_id}'"
        super().__init__(
            message, issues=(ConfigurationIssue("unknown_component", (), message),)
        )
        self.component_kind = component_kind
        self.component_id = component_id


class MissingOptionalDependencyError(ConfigurationError):
    """Raised when a catalog component requires an unavailable optional extra."""

    def __init__(self, component_id: str, extra: str) -> None:
        message = (
            f"Component '{component_id}' requires optional dependencies. "
            f'Install them via `pip install "aberrant[{extra}]"`.'
        )
        super().__init__(
            message, issues=(ConfigurationIssue("missing_dependency", (), message),)
        )
        self.component_id = component_id
        self.extra = extra


class UnsupportedFeatureError(AberrantError):
    """Raised when an unsupported feature is encountered."""

    def __init__(self, feature_name: str) -> None:
        super().__init__(f"Feature '{feature_name}' has not been seen during training.")
        self.feature_name = feature_name


class IncompatibleComponentError(PipelineError):
    """Raised when pipeline components are incompatible."""

    def __init__(self, component_name: str, expected_type: str) -> None:
        super().__init__(
            f"Component '{component_name}' is not compatible. Expected: {expected_type}"
        )
        self.component_name = component_name
        self.expected_type = expected_type
