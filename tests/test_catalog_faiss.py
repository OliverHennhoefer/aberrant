"""Catalog construction tests for the optional FAISS integration."""

import pytest

pytest.importorskip("faiss")

from aberrant.catalog import catalog_manifest, create_model


def test_catalog_builds_knn_after_manifest_discovery() -> None:
    catalog_manifest()
    model = create_model(
        "knn",
        {
            "k": 1,
            "similarity_engine": {
                "id": "faiss",
                "params": {"window_size": 8, "warm_up": 2},
            },
        },
    )
    model.learn_one({"x": 0.0})
    model.learn_one({"x": 1.0})

    assert isinstance(model.score_one({"x": 0.5}), float)
