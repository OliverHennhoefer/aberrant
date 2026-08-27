# Stream Dataset API

Main entry points:

- `aberrant.stream.dataset.load`
- `aberrant.stream.dataset.download`
- `aberrant.stream.dataset.list_available`
- `aberrant.stream.dataset.get_dataset_info`
- `aberrant.stream.dataset.Dataset`
- `aberrant.stream.dataset.DatasetManager`
- `aberrant.stream.dataset.DatasetCacheStore`
- `aberrant.stream.dataset.UrlLibDownloadBackend`
- `aberrant.stream.dataset.NpzStreamer`
- `aberrant.stream.dataset.BatchStreamer`

`DatasetManager` coordinates the typed cache store, download backend, artifact
validator, and NPZ stream. Cache metadata is exposed through immutable
`CacheMetadata` and `CacheEntry` values. Metadata publication and removal are
serialized across processes and use atomic file replacement.

`aberrant.stream` re-exports dataset APIs for convenience.
