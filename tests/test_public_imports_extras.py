"""Public API import contract tests for optional extras."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("faiss")

from aberrant.model.deep import Autoencoder, OnlineAutoencoderEnsemble
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine


def test_public_imports_extras_smoke() -> None:
    assert Autoencoder is not None
    assert OnlineAutoencoderEnsemble is not None
    assert FaissSimilaritySearchEngine is not None
