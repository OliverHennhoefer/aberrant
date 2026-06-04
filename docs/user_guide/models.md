# Models

ABERRANT ships multiple online detector families.

## Isolation forest family

Imports:

```python
from aberrant.model.iforest import (
    ASDIsolationForest,
    HalfSpaceTrees,
    MondrianIsolationForest,
    OnlineIsolationForest,
    RandomCutForest,
    StreamRandomHistogramForest,
    XStream,
)
```

Use these for general-purpose unsupervised streaming anomaly detection.

`MondrianIsolationForest` combines Mondrian block extension with Isolation
Forest path-length scoring. It uses `lambda_` as the Mondrian lifetime budget:
- smaller `lambda_` yields coarser partitions
- larger `lambda_` yields finer partitions

For evaluation on a stream, prefer scoring before learning each sample to avoid
training on the point being evaluated.

## Distance family

Imports:

```python
from aberrant.model.distance import (
    CellNeighborhoodDetector,
    KNN,
    LocalOutlierFactor,
    SDOStream,
    StationaryRegionNeighborDetector,
)
```

`KNN` requires a similarity engine (for example FAISS):

```python
from aberrant.model.distance import KNN
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
model = KNN(k=25, similarity_engine=engine)
```

`LocalOutlierFactor` is a bounded sliding-window local-density detector:

```python
from aberrant.model.distance import LocalOutlierFactor

model = LocalOutlierFactor(k=10, window_size=1000, distance="euclidean")
```

- Returns a continuous LOF score, where values above `1` are more outlier-like.
- Uses bounded-memory window state (`window_size`).

`CellNeighborhoodDetector` is a bounded point-scoring detector using
NETS-inspired cell-level net-effect pruning:

```python
from aberrant.model.distance import CellNeighborhoodDetector

model = CellNeighborhoodDetector(
    k=30,
    radius=1.5,
    window_size=2048,
    slide_size=128,
    subspace_dim=3,
    seed=42,
)
```

- Returns bounded scores in `[0, 1]`.
- Supports optional explicit time handling through `time_key`.
- Uses bounded-memory full/subspace cell state over a sliding window.

`SDOStream` is a bounded-memory observer-based online detector:

```python
from aberrant.model.distance import SDOStream

model = SDOStream(k=128, T=256.0, qv=0.3, x_neighbors=8, seed=42)
```

- Returns a continuous non-negative anomaly score (distance-based).
- Supports optional explicit time handling through `time_key`.
- Uses fixed-size observer state (`k`) with exponential fading (`T`).

`StationaryRegionNeighborDetector` is a bounded sliding-window local outlier
detector with stationary-cell cache invalidation:

```python
from aberrant.model.distance import StationaryRegionNeighborDetector

model = StationaryRegionNeighborDetector(
    k=40,
    radius=1.5,
    window_size=2048,
    slide_size=128,
    skip_threshold=0.1,
)
```

- Returns bounded scores in `[0, 1]`.
- Supports optional explicit time handling through `time_key`.
- Uses bounded-memory sliding window state (`window_size`).

## Sketch family

Imports:

```python
from aberrant.model.sketch import MStream, StreamingLODA, StreamingRSHash
```

Use `StreamingLODA`, `MStream`, and `StreamingRSHash` for bounded-memory
sketch-based streaming detection.

`StreamingLODA` is a random-projection histogram detector:

```python
from aberrant.model.sketch import StreamingLODA

model = StreamingLODA(
    n_projections=64,
    n_bins=24,
    sparsity=0.3,
    warm_up_samples=256,
    decay=1.0,
    seed=42,
)
```

- Sketch models support optional explicit time handling via `time_key`.
- If `time_key=None`, it uses arrival order as the implicit time axis.
- Returns a continuous non-negative anomaly score.

## Graph family

Imports:

```python
from aberrant.model.graph import AnoEdgeL, ISCONNA, MIDAS, SignedGraphSketchDetector
```

Use `AnoEdgeL`, `ISCONNA`, `MIDAS`, and `SignedGraphSketchDetector` for dynamic
edge streams where each sample encodes source and destination IDs (plus
optional timestamp).

- `AnoEdgeL`, `ISCONNA`, and `MIDAS` use bounded-memory sketch state.
- Supports optional explicit timestamp handling via `time_key`.
- Returns a continuous non-negative anomaly score.

`AnoEdgeL` scores whether a mapped edge belongs to a locally dense submatrix:

```python
from aberrant.model.graph import AnoEdgeL

model = AnoEdgeL(
    source_key="src",
    destination_key="dst",
    count_min_rows=128,
    count_min_cols=128,
    num_hashes=4,
    num_dense_submatrices=1,
)
```

`SignedGraphSketchDetector` models graph-level structural change with bounded
per-graph signed shingle sketches and online Euclidean cluster summaries:

```python
from aberrant.model.graph import SignedGraphSketchDetector

model = SignedGraphSketchDetector(
    graph_key="graph",
    source_key="src",
    destination_key="dst",
    edge_type_key="etype",
    sketch_dim=256,
    shingle_size=2,
    num_clusters=8,
    max_graphs=256,
)
```

## SVM family

Imports:

```python
from aberrant.model.svm import (
    GraphGatedOneClassSVM,
    IncrementalOneClassSVMAdaptiveKernel,
)
```

Use when margin-based decision boundaries are preferred.

`GADGETSVM` is retained only as a backward-compatible alias for
`GraphGatedOneClassSVM`; it is not the published GADGET algorithm.

## Statistical family

Imports:

```python
from aberrant.model.stat import (
    MovingAverage,
    MovingCorrelationCoefficient,
    MovingCovariance,
    MovingMahalanobisDistance,
)
```

Use for compact, interpretable change detectors.

## Deep family

Imports:

```python
from aberrant.model.deep import Autoencoder, OnlineAutoencoderEnsemble
```

- `Autoencoder` depends on `torch` (`aberrant[dl]`).
- `OnlineAutoencoderEnsemble` is an online ensemble of lightweight autoencoders with explicit
  warm-up phases (`feature_map_grace`, `ad_grace`).

The historical names `NETS`, `STARE`, `StreamSpot`, `KitNET`,
`MondrianForest`, `RSHash`, and `LODA` remain compatibility aliases. They refer
to the variants documented above, not score-compatible reproductions of the
published algorithms.

## Core utility models

```python
from aberrant.model import NullModel, QuantileThreshold, RandomModel, ThresholdModel
```

- `ThresholdModel`: static rule-based boundary detector
- `QuantileThreshold`: adaptive threshold on a streaming score signal
