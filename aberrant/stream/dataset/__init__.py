"""ABERRANT Dataset Management System.

This module provides a comprehensive dataset management system for anomaly detection
benchmarks, including automatic downloading, caching, and streaming capabilities.

The system is designed to work with datasets hosted on GitHub releases and provides
a clean, type-safe API similar to established ML libraries.

Basic Usage:
    from aberrant.dataset import load, Dataset, list_available

    # Load a dataset (auto-downloads if needed)
    dataset = load(Dataset.FRAUD)

    # Stream through data
    for features, label in dataset.stream():
        # features is a dict of feature_name -> value
        # label is the anomaly label (0=normal, 1=anomaly)
        process_data(features, label)

    # Explore available datasets
    print(list_available())

Advanced Usage:
    from aberrant.dataset import get_dataset_info, DatasetManager, BatchStreamer

    # Get dataset metadata
    info = get_dataset_info(Dataset.FRAUD)
    print(f"Dataset: {info.name}")
    print(f"Samples: {info.n_samples}")
    print(f"Features: {info.n_features}")
    print(f"Anomaly rate: {info.anomaly_rate:.3f}")

    # Custom dataset manager
    manager = DatasetManager(cache_dir="/custom/cache")
    dataset = manager.load(Dataset.FRAUD)

    # Batch streaming
    batch_streamer = BatchStreamer(dataset, batch_size=1000)
    for feature_batch, label_batch in batch_streamer.stream():
        process_batch(feature_batch, label_batch)
"""

from typing import Any

from aberrant import __version__

from .loader import DatasetManager, get_default_manager, set_cache_dir

# Import core classes and functions
from .registry import (
    DATASET_REGISTRY,
    Dataset,
    DatasetInfo,
    get_categories,
    get_dataset_info,
    list_available,
    list_by_category,
)
from .streamers import BatchStreamer, DatasetStreamer, NpzStreamer


def load(
    dataset: Dataset, auto_download: bool = True, **kwargs: Any
) -> DatasetStreamer:
    """Load a dataset for streaming.

    This is the main entry point for loading datasets. It uses the global
    DatasetManager instance and handles automatic downloading if needed.

    Args:
        dataset: Dataset enum value to load
        auto_download: Automatically download if not cached (default: True)
        **kwargs: Additional arguments passed to DatasetStreamer

    Returns:
        DatasetStreamer instance ready for iteration

    Raises:
        FileNotFoundError: If dataset not cached and auto_download is False
        RuntimeError: If download fails
        KeyError: If dataset not found in registry

    Example:
        # Load fraud detection dataset
        dataset = load(Dataset.FRAUD)

        # Stream through data
        for features, label in dataset.stream():
            score = model.score_one(features)
            model.learn_one(features)

            if score > threshold:
                print(f"Anomaly detected: {features}")
    """
    manager = get_default_manager()
    return manager.load(dataset, auto_download=auto_download, **kwargs)


def download(dataset: Dataset, force: bool = False) -> str:
    """Download a dataset to local cache.

    Args:
        dataset: Dataset to download
        force: Force re-download even if cached (default: False)

    Returns:
        Path to cached dataset file

    Example:
        # Pre-download datasets
        download(Dataset.FRAUD)
        download(Dataset.SHUTTLE)
    """
    manager = get_default_manager()
    path = manager.download(dataset, force=force)
    return str(path)


def list_cached() -> dict[str, dict]:
    """List all locally cached datasets.

    Returns:
        Dictionary mapping dataset names to cache metadata

    Example:
        cached = list_cached()
        for name, metadata in cached.items():
            print(f"{name}: {metadata['size'] / 1024 / 1024:.1f} MB")
    """
    manager = get_default_manager()
    return manager.list_cached()


def clear_cache(dataset: Dataset | None = None) -> None:
    """Clear dataset cache.

    Args:
        dataset: Specific dataset to remove, or None to clear all

    Example:
        # Clear specific dataset
        clear_cache(Dataset.FRAUD)

        # Clear all cached datasets
        clear_cache()
    """
    manager = get_default_manager()
    manager.clear_cache(dataset)


def get_cache_info() -> dict[str, Any]:
    """Get information about the dataset cache.

    Returns:
        Dictionary with cache statistics

    Example:
        info = get_cache_info()
        print(f"Cache size: {info['size_mb']:.1f} MB")
        print(f"Cached datasets: {info['count']}")
    """
    manager = get_default_manager()
    size_bytes = manager.get_cache_size()
    cached_datasets = manager.list_cached()

    return {
        "cache_dir": str(manager.cache_dir),
        "size_bytes": size_bytes,
        "size_mb": size_bytes / (1024 * 1024),
        "count": len(cached_datasets),
        "datasets": list(cached_datasets.keys()),
    }


# Export all public symbols
__all__ = [
    "__version__",
    # Main functions
    "load",
    "download",
    "list_available",
    "list_cached",
    "clear_cache",
    "get_cache_info",
    # Registry functions
    "get_dataset_info",
    "list_by_category",
    "get_categories",
    # Classes
    "Dataset",
    "DatasetInfo",
    "DatasetManager",
    "DatasetStreamer",
    "NpzStreamer",
    "BatchStreamer",
    # Advanced functions
    "get_default_manager",
    "set_cache_dir",
    # Registry data
    "DATASET_REGISTRY",
]


# Module-level information
__author__ = "ABERRANT Team"
__description__ = "Dataset management system for anomaly detection benchmarks"
