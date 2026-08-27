"""Base classes for online anomaly detection models and components.

This module provides the fundamental abstract base classes that define the
interfaces for models, transformers, pipelines, and other core components
in the aberrant library.
"""

import importlib
from typing import TYPE_CHECKING

from aberrant.base.exceptions import (
    AberrantError,
    IncompatibleComponentError,
    ModelNotFittedError,
    PipelineError,
    TransformationError,
    UnsupportedFeatureError,
    ValidationError,
)
from aberrant.base.model import BaseModel
from aberrant.base.pipeline import Pipeline
from aberrant.base.protocols import (
    FeatureMap,
    LearnerProtocol,
    ModelProtocol,
    TransformerProtocol,
)
from aberrant.base.similarity import BaseSimilaritySearchEngine
from aberrant.base.transformer import BaseTransformer

if TYPE_CHECKING:
    from aberrant.base.architecture import Architecture as Architecture

__all__ = [
    "BaseModel",
    "BaseSimilaritySearchEngine",
    "BaseTransformer",
    "FeatureMap",
    "IncompatibleComponentError",
    "ModelNotFittedError",
    "AberrantError",
    "Pipeline",
    "PipelineError",
    "LearnerProtocol",
    "ModelProtocol",
    "TransformationError",
    "TransformerProtocol",
    "UnsupportedFeatureError",
    "ValidationError",
]


def __getattr__(name: str) -> object:
    """Load the optional torch architecture base on explicit access."""
    if name == "Architecture":
        try:
            module = importlib.import_module("aberrant.base.architecture")
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise ImportError(
                    "Architecture requires the optional 'dl' dependencies"
                ) from exc
            raise
        architecture = module.Architecture
        globals()[name] = architecture
        return architecture
    raise AttributeError(f"module 'aberrant.base' has no attribute '{name}'")
