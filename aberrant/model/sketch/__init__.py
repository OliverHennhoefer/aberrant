"""Sketch-based models for streaming anomaly detection."""

import importlib
from typing import Any

__all__ = [
    "LODA",
    "MStream",
    "RSHash",
    "StreamingLODA",
    "StreamingRSHash",
]


def __getattr__(name: str) -> Any:
    """Lazy import of sketch model classes."""
    if name in {"LODA", "StreamingLODA"}:
        module = importlib.import_module("aberrant.model.sketch.loda")
        return getattr(module, name)
    if name == "MStream":
        module = importlib.import_module("aberrant.model.sketch.mstream")
        return module.MStream
    if name in {"RSHash", "StreamingRSHash"}:
        module = importlib.import_module("aberrant.model.sketch.rshash")
        return getattr(module, name)
    raise AttributeError(f"module 'aberrant.model.sketch' has no attribute '{name}'")
