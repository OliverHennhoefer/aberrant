# Base API

The abstract base classes support inheritance; the runtime-checkable protocols
support structural pipeline components without inheritance.

::: aberrant.base.BaseModel

::: aberrant.base.BaseTransformer

::: aberrant.base.BaseSimilaritySearchEngine

::: aberrant.base.Pipeline

::: aberrant.base.LearnerProtocol

::: aberrant.base.TransformerProtocol

::: aberrant.base.ModelProtocol

::: aberrant.base.FeatureMap

## Exceptions

::: aberrant.base.AberrantError

::: aberrant.base.ModelNotFittedError

::: aberrant.base.TransformationError

::: aberrant.base.PipelineError

::: aberrant.base.ValidationError

::: aberrant.base.UnsupportedFeatureError

::: aberrant.base.IncompatibleComponentError

## Optional PyTorch architecture base

The following object requires `aberrant[dl]` at runtime and is intentionally
absent from `from aberrant.base import *`.

::: aberrant.base.architecture.Architecture
