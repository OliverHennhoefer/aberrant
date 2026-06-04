"""SVM-based models for streaming anomaly detection."""

import importlib
from typing import Any

__all__ = [
    "GADGETSVM",
    "GraphGatedOneClassSVM",
    "IncrementalOneClassSVMAdaptiveKernel",
]


def __getattr__(name: str) -> Any:
    """Lazy import of SVM model classes."""
    if name == "IncrementalOneClassSVMAdaptiveKernel":
        module = importlib.import_module("aberrant.model.svm.adaptive")
        return module.IncrementalOneClassSVMAdaptiveKernel
    if name in {"GADGETSVM", "GraphGatedOneClassSVM"}:
        module = importlib.import_module("aberrant.model.svm.gadget")
        return getattr(module, name)
    raise AttributeError(f"module 'aberrant.model.svm' has no attribute '{name}'")
