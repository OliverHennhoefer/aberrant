"""Catalog construction tests for the optional FAISS integration."""

import pytest

pytest.importorskip("faiss")

from aberrant.base import ConfigurationError
from aberrant.catalog import (
    ComponentSpec,
    DetectorConfig,
    catalog_manifest,
    create_model,
)


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


def test_nested_resolution_is_reused_and_builds_independent_engines(monkeypatch):
    original = ComponentSpec.resolved_parameters
    calls = []

    def record(self, *args, **kwargs):
        calls.append(self.id)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ComponentSpec, "resolved_parameters", record)
    resolved = DetectorConfig.from_mapping(
        {
            "model": {
                "id": "knn",
                "params": {
                    "k": 2,
                    "similarity_engine": {
                        "id": "faiss",
                        "params": {
                            "window_size": 8,
                            "warm_up": 3,
                        },
                    },
                },
            }
        }
    ).resolve()
    first, second = resolved.build(), resolved.build()
    assert first.engine is not second.engine
    first.learn_one({"x": 0.0})
    assert len(first.engine.window) == 1
    assert len(second.engine.window) == 0
    assert calls == ["faiss", "knn"]
    assert (
        resolved.fingerprint
        == "ad1a47f7f6bafce816264110e4fe9bcf367afe3d384ca8410896c768fe465edc"
    )


def test_nested_component_parameter_errors_keep_complete_paths():
    with pytest.raises(ConfigurationError) as error:
        DetectorConfig.from_mapping(
            {
                "model": {
                    "id": "knn",
                    "params": {
                        "k": "many",
                        "similarity_engine": {
                            "id": "faiss",
                            "params": {
                                "window_size": "many",
                                "warm_up": False,
                            },
                        },
                    },
                }
            }
        ).resolve()
    assert {issue.loc for issue in error.value.issues} == {
        ("model", "params", "k"),
        ("model", "params", "similarity_engine", "params", "window_size"),
        ("model", "params", "similarity_engine", "params", "warm_up"),
    }
