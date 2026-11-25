# Benchmark Datasets

`aberrant.datasets` replays labelled anomaly detection benchmarks as event
streams. Labels are there for evaluation: an unsupervised model learns from the
features alone.

## Inspect the registry without downloading

Every `Dataset` describes itself through `info`:

```python
from aberrant.datasets import DATASETS, Dataset

info = Dataset.SHUTTLE.info
print(info.name, info.n_samples, info.n_features, info.anomaly_rate)

medical = [dataset for dataset, info in DATASETS.items() if info.category == "medical"]
print(medical)
```

## Load and replay a dataset

The first `load` downloads the archive from a fixed release. Its SHA-256 is
checked before the file is cached, and the cached file is checked again on
every later `load`, so a corrupt or tampered file is never parsed.

```python
from itertools import islice

from aberrant.datasets import Dataset, load

stream = load(Dataset.SHUTTLE)

for features, label in islice(stream, 3):
    print(label, features)
```

Iterating yields `(features, label)` pairs. Features are named `feature_0`,
`feature_1`, and so on; the label is `1` for an anomaly and `0` otherwise. Each
iteration reads the archive afresh, so the same stream can be replayed.
`stream.arrays()` returns the whole feature matrix and label vector.

Pass `fetch=False` to raise `FileNotFoundError` instead of downloading when no
verified copy is cached:

```python
from aberrant.datasets import Dataset, load

try:
    stream = load(Dataset.SHUTTLE, fetch=False)
except FileNotFoundError as error:
    print(error)
```

## Where the files live

Archives are cached in the directory named by `ABERRANT_DATA_DIR`, or in
`~/.cache/aberrant` when it is unset; `data_dir()` returns it. A download is
written to a temporary file in that directory and moved into place only once
its checksum matches, so interrupted or concurrent downloads never leave a
partial file under a dataset's name. To free the space, delete the directory.

## Replay your own archive

`DatasetStream` replays any `.npz` file with a two-dimensional array `X` and a
label array `y` of the same length:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aberrant.datasets import DatasetStream

with TemporaryDirectory() as directory:
    path = Path(directory) / "example.npz"
    np.savez(path, X=np.array([[1.0, 2.0], [3.0, 4.0]]), y=np.array([0, 1]))

    for features, label in DatasetStream(path):
        print(features, label)
```

See the [Datasets API](../api/datasets.md) for every signature.
