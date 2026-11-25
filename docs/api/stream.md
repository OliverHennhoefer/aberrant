# Stream Dataset API

The module-level functions operate on a process-local default
`DatasetManager`. Construct a manager directly for explicit cache ownership or
dependency injection.

## Registry

::: aberrant.stream.Dataset

::: aberrant.stream.DatasetInfo

::: aberrant.stream.DATASET_REGISTRY

::: aberrant.stream.get_dataset_info

::: aberrant.stream.list_available

::: aberrant.stream.list_by_category

::: aberrant.stream.get_categories

## Default-manager functions

::: aberrant.stream.load

::: aberrant.stream

::: aberrant.stream.list_cached

::: aberrant.stream.clear_cache

::: aberrant.stream.get_cache_info

::: aberrant.stream.get_default_manager

::: aberrant.stream.set_cache_dir

## Orchestration and cache values

::: aberrant.stream.DatasetManager

::: aberrant.stream.DatasetCacheStore

::: aberrant.stream.CacheEntry

::: aberrant.stream.CacheMetadata

::: aberrant.stream.CacheInfo

## Download and validation

::: aberrant.stream.DownloadBackend

::: aberrant.stream.UrlLibDownloadBackend

::: aberrant.stream.DatasetArtifactValidator

## Streams

::: aberrant.stream.DatasetStream

::: aberrant.stream.Sample

::: aberrant.stream.NpzStreamer

::: aberrant.stream.BatchStreamer
