# Exceptions API

Every error aberrant raises on purpose derives from `AberrantError`.
`ValidationError` is also a `ValueError` and marks a rejected event.
`MissingOptionalDependencyError` is also an `ImportError` and names the extra
to install.

::: aberrant.exceptions.AberrantError

::: aberrant.exceptions.ValidationError

::: aberrant.exceptions.PipelineError

::: aberrant.exceptions.IncompatibleComponentError

::: aberrant.exceptions.ConfigurationError

::: aberrant.exceptions.ConfigurationIssue

::: aberrant.exceptions.UnknownComponentError

::: aberrant.exceptions.MissingOptionalDependencyError

::: aberrant.exceptions.IncompatibleStateError
