"""Streaming data interfaces.

Public streaming dataset APIs are re-exported from ``aberrant.stream.dataset``.
"""

from aberrant.stream.dataset import (
    DATASET_REGISTRY,
    BatchStreamer,
    CacheInfo,
    Dataset,
    DatasetInfo,
    DatasetManager,
    NpzStreamer,
    clear_cache,
    download,
    get_cache_info,
    get_categories,
    get_dataset_info,
    get_default_manager,
    list_available,
    list_by_category,
    list_cached,
    load,
    set_cache_dir,
)

__all__ = [
    "BatchStreamer",
    "CacheInfo",
    "DATASET_REGISTRY",
    "Dataset",
    "DatasetInfo",
    "DatasetManager",
    "NpzStreamer",
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
