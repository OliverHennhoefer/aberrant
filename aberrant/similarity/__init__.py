"""Similarity-search engines used by distance-based models.

``FaissSimilaritySearchEngine`` loads on first access, so importing this
package does not import FAISS.
"""

import importlib
from typing import TYPE_CHECKING

from aberrant.similarity._base import BaseSimilaritySearchEngine

if TYPE_CHECKING:
    from aberrant.similarity._faiss import (
        FaissSimilaritySearchEngine as FaissSimilaritySearchEngine,
    )

__all__ = ["BaseSimilaritySearchEngine", "FaissSimilaritySearchEngine"]


def __getattr__(name: str) -> object:
    """Load the FAISS engine on explicit access."""
    if name != "FaissSimilaritySearchEngine":
        raise AttributeError(f"module 'aberrant.similarity' has no attribute '{name}'")
    engine = importlib.import_module(
        "aberrant.similarity._faiss"
    ).FaissSimilaritySearchEngine
    globals()[name] = engine
    return engine
