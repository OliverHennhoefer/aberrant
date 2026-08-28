# Streaming Datasets

`aberrant.stream.dataset` provides two related capabilities:

1. a registry, downloader, validator, and local cache for built-in benchmark
   datasets;
2. typed row-wise and batch streaming over NPZ files containing feature and
   label arrays.

Dataset labels are evaluation metadata. Unsupervised anomaly models receive
only the feature dictionary passed to `learn_one`.

## Inspect the registry without downloading

Registry metadata is immutable. `list_available` returns a new mapping whose
values are frozen `DatasetInfo` objects.

```python
from aberrant.stream.dataset import get_categories, list_available

available = list_available()
print(f"datasets={len(available)}")
print(f"categories={get_categories()}")

for dataset_name in sorted(available)[:5]:
    info = available[dataset_name]
    print(
        f"{dataset_name}: samples={info.n_samples}, "
        f"features={info.n_features}, anomaly_rate={info.anomaly_rate:.3f}"
    )
```

Use `get_dataset_info(Dataset.SHUTTLE)` for one entry or
`list_by_category("medical")` to filter the registry.

## Load a registered stream

The first call can access the network. `load` downloads a missing artifact,
checks its trusted SHA-256 digest and NPZ structure, publishes it under the
cache lock using per-file atomic replacement, and returns an `NpzStreamer`.
Later calls validate and reuse the cached artifact.

```python
from itertools import islice

from aberrant.stream.dataset import Dataset, load

dataset = load(Dataset.SHUTTLE)

for features, label in islice(dataset.stream(), 3):
    print(f"label={label!r}, features={features}")
```

`NpzStreamer.stream()` opens and closes the archive around iteration. Feature
columns are exposed as `feature_0`, `feature_1`, and so on unless
`feature_prefix` is changed. Labels preserve the scalar type stored in the NPZ
file and are therefore typed as `object`.

Set `auto_download=False` when a missing or invalid cache entry should raise
`FileNotFoundError` instead of using the network:

```python
from aberrant.stream.dataset import Dataset, load

try:
    cached_only = load(Dataset.SHUTTLE, auto_download=False)
except FileNotFoundError:
    print("SHUTTLE is not present in the validated local cache")
else:
    first_features, first_label = next(cached_only.stream())
    print(first_features, first_label)
```

## Stream a local NPZ file

A local archive needs a two-dimensional `X` array and a `y` array with the same
number of rows. The complete example uses a temporary directory and leaves no
file behind:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aberrant.stream.dataset import NpzStreamer

with TemporaryDirectory() as directory:
    path = Path(directory) / "example.npz"
    np.savez(
        path,
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        y=np.array([0, 1], dtype=np.int64),
    )

    streamer = NpzStreamer(path, feature_prefix="sensor_")
    for features, label in streamer.stream():
        print(features, label)
```

Custom array names are supported through `feature_column=` and
`label_column=`. The registry validator deliberately requires `X` and `y` for
built-in artifacts, even though a standalone `NpzStreamer` can use custom
names.

## Batch any compatible stream

`BatchStreamer` wraps any object with `stream()` and `get_metadata()` methods.
It yields two lists: feature mappings and labels. The final batch can be
smaller than `batch_size`.

```python
from collections.abc import Iterator

from aberrant.stream.dataset import BatchStreamer, DatasetInfo, Sample


class InMemoryStream:
    def stream(self) -> Iterator[Sample]:
        yield {"x": 1.0}, 0
        yield {"x": 2.0}, 0
        yield {"x": 9.0}, 1

    def get_metadata(self) -> DatasetInfo | None:
        return None


batch_streamer = BatchStreamer(InMemoryStream(), batch_size=2)
for feature_batch, label_batch in batch_streamer.stream():
    print(feature_batch, label_batch)
```

## Configure and inspect the cache

The default cache is `~/.aberrant/datasets`. `set_cache_dir(path)` replaces the
process-local default manager for subsequent module-level calls. Construct a
`DatasetManager` directly when an application needs explicit ownership,
dependency injection, or separate caches:

```python
from tempfile import TemporaryDirectory

from aberrant.stream.dataset import DatasetManager

with TemporaryDirectory() as directory:
    manager = DatasetManager(cache_dir=directory, show_progress=False)
    print(manager.cache_dir)
    print(manager.list_cached())
    print(manager.get_cache_size())
```

For the default manager, `get_cache_info()` returns a typed `CacheInfo` summary
and `list_cached()` returns valid `CacheEntry` values. `clear_cache(dataset)`
removes one registered artifact; `clear_cache()` removes all artifacts owned by
that manager, so call it only as an intentional destructive operation.

## Cache integrity and concurrency

- Downloads use bounded retries, a timeout, and exponential backoff.
- A temporary artifact is structurally validated and compared with the
  registry's trusted SHA-256 digest before publication.
- Artifact and metadata files are each published with atomic replacement under
  the cross-process lock. An artifact without matching valid metadata, or
  metadata without a valid artifact, is treated as a cache miss.
- Metadata publication and removal are serialized with a cross-process file
  lock.
- Cache metadata is returned as immutable snapshots, so callers cannot mutate
  process-wide registry or cache truth through a returned mapping.
- Cache clearing targets only the registered artifact paths owned by the
  manager; it does not recursively delete arbitrary cache contents.

Download and iteration progress bars are opt-in through `show_progress=True`.
See [Stream Dataset API](../api/stream.md) for complete constructor and function
signatures.
