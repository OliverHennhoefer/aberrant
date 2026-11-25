"""Immutability contracts for trusted built-in dataset metadata."""

from dataclasses import FrozenInstanceError

import pytest

from aberrant.stream import (
    DATASET_REGISTRY,
    Dataset,
    get_categories,
    get_dataset_info,
    list_available,
    list_by_category,
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


def test_categories_partition_the_registry() -> None:
    categories = get_categories()

    assert categories == sorted(set(categories))
    by_category = [list_by_category(category) for category in categories]
    assert sum(map(len, by_category)) == len(DATASET_REGISTRY)
    for category, datasets in zip(categories, by_category, strict=True):
        assert datasets
        assert all(info.category == category for info in datasets.values())
    assert list_by_category("no-such-category") == {}


def test_unregistered_dataset_is_rejected() -> None:
    with pytest.raises(KeyError, match="not found in registry"):
        get_dataset_info("shuttle")  # type: ignore[arg-type]
