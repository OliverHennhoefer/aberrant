# Stream Dataset API

The module-level functions operate on a process-local default
`DatasetManager`. Construct a manager directly for explicit cache ownership or
dependency injection.

## Registry

::: aberrant.stream.dataset.Dataset

::: aberrant.stream.dataset.DatasetInfo

::: aberrant.stream.dataset.DATASET_REGISTRY

::: aberrant.stream.dataset.get_dataset_info

::: aberrant.stream.dataset.list_available

::: aberrant.stream.dataset.list_by_category

::: aberrant.stream.dataset.get_categories

## Default-manager functions

::: aberrant.stream.dataset.load

::: aberrant.stream.dataset.download

::: aberrant.stream.dataset.list_cached

::: aberrant.stream.dataset.clear_cache

::: aberrant.stream.dataset.get_cache_info

::: aberrant.stream.dataset.get_default_manager

::: aberrant.stream.dataset.set_cache_dir

## Orchestration and cache values

::: aberrant.stream.dataset.DatasetManager

::: aberrant.stream.dataset.DatasetCacheStore

::: aberrant.stream.dataset.CacheEntry

::: aberrant.stream.dataset.CacheMetadata

::: aberrant.stream.dataset.CacheInfo

## Download and validation

::: aberrant.stream.dataset.DownloadBackend

::: aberrant.stream.dataset.UrlLibDownloadBackend

::: aberrant.stream.dataset.DatasetArtifactValidator

## Streams

::: aberrant.stream.dataset.DatasetStream

::: aberrant.stream.dataset.Sample

::: aberrant.stream.dataset.NpzStreamer

::: aberrant.stream.dataset.BatchStreamer
