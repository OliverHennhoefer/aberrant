"""Typed dataset registry, cache, download, and streaming API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aberrant.stream.dataset.cache import (
    CacheEntry,
    CacheMetadata,
    DatasetCacheStore,
)
from aberrant.stream.dataset.download import (
    DatasetArtifactValidator,
    DownloadBackend,
    UrlLibDownloadBackend,
)
from aberrant.stream.dataset.loader import (
    DatasetManager,
    get_default_manager,
    set_cache_dir,
)
from aberrant.stream.dataset.registry import (
    DATASET_REGISTRY,
    Dataset,
    DatasetInfo,
    get_categories,
    get_dataset_info,
    list_available,
    list_by_category,
)
from aberrant.stream.dataset.streamers import (
    BatchStreamer,
    DatasetStream,
    NpzStreamer,
    Sample,
)


@dataclass(frozen=True, slots=True)
class CacheInfo:
    """Typed summary of the default dataset cache.

    Attributes:
        directory: Configured cache directory.
        size_bytes: Total bytes occupied by registered cached artifacts.
        datasets: Registry values whose artifacts are present in the cache.
    """

    directory: Path
    size_bytes: int
    datasets: tuple[str, ...]


def load(
    dataset: Dataset,
    auto_download: bool = True,
    *,
    feature_prefix: str = "feature_",
    label_column: str = "y",
    feature_column: str = "X",
    show_progress: bool = False,
) -> NpzStreamer:
    """Return a validated NPZ stream from the default manager."""
    return get_default_manager().load(
        dataset,
        auto_download=auto_download,
        feature_prefix=feature_prefix,
        label_column=label_column,
        feature_column=feature_column,
        show_progress=show_progress,
    )


def download(dataset: Dataset, force: bool = False) -> Path:
    """Download one registered dataset into the default cache."""
    return get_default_manager().download(dataset, force=force)


def list_cached() -> dict[str, CacheEntry]:
    """Return metadata for locally cached registered datasets."""
    return get_default_manager().list_cached()


def clear_cache(dataset: Dataset | None = None) -> None:
    """Remove one dataset or all registered datasets from the default cache."""
    get_default_manager().clear_cache(dataset)


def get_cache_info() -> CacheInfo:
    """Return a typed default-cache summary."""
    manager = get_default_manager()
    cached = manager.list_cached()
    return CacheInfo(
        directory=manager.cache_dir,
        size_bytes=manager.get_cache_size(),
        datasets=tuple(sorted(cached)),
    )


__all__ = [
    "BatchStreamer",
    "CacheEntry",
    "CacheInfo",
    "CacheMetadata",
    "DATASET_REGISTRY",
    "Dataset",
    "DatasetArtifactValidator",
    "DatasetCacheStore",
    "DatasetInfo",
    "DatasetManager",
    "DatasetStream",
    "DownloadBackend",
    "NpzStreamer",
    "Sample",
    "UrlLibDownloadBackend",
    "clear_cache",
    "download",
    "get_cache_info",
    "get_categories",
    "get_dataset_info",
    "get_default_manager",
    "list_available",
    "list_by_category",
    "list_cached",
    "load",
    "set_cache_dir",
]
