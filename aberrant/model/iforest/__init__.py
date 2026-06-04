"""Isolation forest variants for streaming anomaly detection."""

import importlib
from typing import Any

__all__ = [
    "ASDIsolationForest",
    "HalfSpaceTrees",
    "MondrianIsolationForest",
    "MondrianForest",
    "OnlineIsolationForest",
    "RandomCutForest",
    "StreamRandomHistogramForest",
    "XStream",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ASDIsolationForest": ("aberrant.model.iforest.asd", "ASDIsolationForest"),
    "HalfSpaceTrees": ("aberrant.model.iforest.halfspace", "HalfSpaceTrees"),
    "MondrianIsolationForest": (
        "aberrant.model.iforest.mondrian",
        "MondrianIsolationForest",
    ),
    "MondrianForest": ("aberrant.model.iforest.mondrian", "MondrianForest"),
    "OnlineIsolationForest": ("aberrant.model.iforest.online", "OnlineIsolationForest"),
    "RandomCutForest": ("aberrant.model.iforest.random_cut", "RandomCutForest"),
    "StreamRandomHistogramForest": (
        "aberrant.model.iforest.rand_hist",
        "StreamRandomHistogramForest",
    ),
    "XStream": ("aberrant.model.iforest.xstream", "XStream"),
}


def __getattr__(name: str) -> Any:
    """Lazy import of model classes."""
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'aberrant.model.iforest' has no attribute '{name}'"
        ) from exc

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
