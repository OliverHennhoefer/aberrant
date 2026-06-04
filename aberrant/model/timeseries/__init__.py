"""Time-series anomaly detection models."""

import importlib
from typing import Any

__all__ = ["XLagDAMP"]


def __getattr__(name: str) -> Any:
    """Lazy import of time-series model classes."""
    if name == "XLagDAMP":
        module = importlib.import_module("aberrant.model.timeseries.damp")
        return module.XLagDAMP
    raise AttributeError(
        f"module 'aberrant.model.timeseries' has no attribute '{name}'"
    )
