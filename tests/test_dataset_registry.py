"""Immutability contracts for trusted built-in dataset metadata."""

from dataclasses import FrozenInstanceError

import pytest

from aberrant.stream.dataset.registry import (
    DATASET_REGISTRY,
    Dataset,
    get_dataset_info,
    list_available,
)


def test_dataset_info_is_frozen() -> None:
    info = get_dataset_info(Dataset.FRAUD)

    with pytest.raises(FrozenInstanceError):
        info.sha256 = "untrusted"  # type: ignore[misc]


def test_dataset_registry_is_read_only() -> None:
    info = get_dataset_info(Dataset.FRAUD)

    with pytest.raises(TypeError):
        DATASET_REGISTRY[Dataset.FRAUD] = info  # type: ignore[index]


def test_listing_returns_an_independent_mapping_with_immutable_values() -> None:
    available = list_available()
    available.pop(Dataset.FRAUD.value)

    assert Dataset.FRAUD in DATASET_REGISTRY
