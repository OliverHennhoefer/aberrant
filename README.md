# ABERRANT

Online anomaly detection for real streams.

ABERRANT is for systems that score and adapt one event at a time.  
No batch retraining loop required. No framework lock-in.

## Why ABERRANT

- Streaming-first anomaly detection with a broad algorithm surface under one API
- Coverage for low-latency scoring, bounded-memory setups, interpretable baselines, and optional deep models
- One interface across detector families: `learn_one`, `score_one`
- Composable online preprocessing with `|` pipelines
- Built-in benchmark dataset streaming with local caching
- Drift detectors and threshold models for production alerting policies

## Core interface

```python
from aberrant.model.iforest import OnlineIsolationForest

model = OnlineIsolationForest(window_size=512, num_trees=50)

for x in live_stream:
    score = model.score_one(x)  # score current point
    model.learn_one(x)          # update state incrementally
```

Each `x` is `dict[str, float]`.

## Algorithms included

| Family | Algorithms |
| --- | --- |
| Isolation Forest | `ASDIsolationForest`, `HalfSpaceTrees`, `MondrianIsolationForest`, `OnlineIsolationForest`, `RandomCutForest`, `StreamRandomHistogramForest`, `XStream` |
| Distance | `CellNeighborhoodDetector`, `KNN`, `LocalOutlierFactor`, `SDOStream`, `StationaryRegionNeighborDetector` |
| Sketch | `MStream`, `StreamingLODA`, `StreamingRSHash` |
| Graph | `AnoEdgeL`, `ISCONNA`, `MIDAS`, `SignedGraphSketchDetector` |
| SVM | `GraphGatedOneClassSVM`, `IncrementalOneClassSVMAdaptiveKernel` |
| Statistical | `MovingAverage`, `MovingAverageAbsoluteDeviation`, `MovingGeometricAverage`, `MovingHarmonicAverage`, `MovingInterquartileRange`, `MovingKurtosis`, `MovingMedian`, `MovingQuantile`, `MovingSkewness`, `MovingVariance`, `MovingCorrelationCoefficient`, `MovingCovariance`, `MovingMahalanobisDistance` |
| Deep (`aberrant[dl]`) | `Autoencoder`, `OnlineAutoencoderEnsemble` |
| Utility | `ThresholdModel`, `QuantileThreshold`, `NullModel`, `RandomModel` |
| Drift Detection | `ADWIN`, `KSWIN`, `PageHinkley` |

## Pipelines

```python
from aberrant.model.distance import KNN
from aberrant.transform.preprocessing import StandardScaler
from aberrant.transform.projection import IncrementalPCA
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
detector = StandardScaler() | IncrementalPCA(n_components=3, n0=100) | KNN(
    k=45,
    similarity_engine=engine,
)
```

`FaissSimilaritySearchEngine` requires the optional `aberrant[faiss]` extra.

## Streaming datasets

```python
from aberrant.stream.dataset import Dataset, load

dataset = load(Dataset.SHUTTLE)
for x, y in dataset.stream():
    ...
```

`load(...)` auto-downloads and caches benchmark datasets locally.

## Install

```bash
pip install aberrant
```

Optional extras:

- `aberrant[eval]`: evaluation metrics (`scikit-learn`)
- `aberrant[dl]`: deep models (`torch`)
- `aberrant[faiss]`: FAISS similarity engine (`faiss-cpu`)
- `aberrant[dev]`, `aberrant[docs]`, `aberrant[benchmark]`, `aberrant[all]`

## Public modules

- `aberrant.model.iforest`
- `aberrant.model.distance`
- `aberrant.model.sketch`
- `aberrant.model.graph`
- `aberrant.model.svm`
- `aberrant.model.stat`
- `aberrant.model.deep` (optional extra)
- `aberrant.model`
- `aberrant.drift`
- `aberrant.transform.preprocessing`
- `aberrant.transform.projection`
- `aberrant.stream.dataset`
- `aberrant.utils.similar`

## Scoring notes

- Isolation-forest variants return bounded scores in `[0, 1]`
- `ThresholdModel` returns binary decisions (`0.0` or `1.0`)
- Other models return continuous family-specific scores

## Development

```bash
uv sync --extra dev --extra docs
uv run python -m ruff check .
uv run python -m pytest -q
```

## License

MIT (`LICENSE`)
