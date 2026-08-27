<h1 align="center">
  <a href="https://oliverhennhoefer.github.io/aberrant/">
    <img src="https://raw.githubusercontent.com/OliverHennhoefer/aberrant/main/docs/img/aberrant.png" alt="ABERRANT" width="900">
  </a>
</h1>

<p align="center">
  <strong>Score events. Learn continuously. Adapt to the stream.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/aberrant/"><img src="https://img.shields.io/pypi/v/aberrant.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/aberrant/"><img src="https://img.shields.io/pypi/pyversions/aberrant.svg" alt="Supported Python versions"></a>
  <a href="https://github.com/OliverHennhoefer/aberrant/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/OliverHennhoefer/aberrant/ci.yml?branch=main&label=CI" alt="CI status"></a>
  <a href="https://oliverhennhoefer.github.io/aberrant/"><img src="https://img.shields.io/github/actions/workflow/status/OliverHennhoefer/aberrant/docs.yml?branch=main&label=Documentation" alt="Documentation status"></a>
  <a href="https://github.com/OliverHennhoefer/aberrant/blob/main/LICENSE"><img src="https://img.shields.io/github/license/OliverHennhoefer/aberrant" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://oliverhennhoefer.github.io/aberrant/">Documentation</a> ·
  <a href="https://oliverhennhoefer.github.io/aberrant/quickstart/">Quickstart</a> ·
  <a href="https://oliverhennhoefer.github.io/aberrant/user_guide/models/">Model guide</a> ·
  <a href="https://oliverhennhoefer.github.io/aberrant/api/">API reference</a> ·
  <a href="https://github.com/OliverHennhoefer/aberrant/blob/main/CHANGELOG.md">Changelog</a>
</p>

`aberrant` is a typed Python library for unsupervised anomaly detection on data
that arrives one event at a time. Its models share a compact online interface:
`score_one(x)` evaluates the current event and `learn_one(x)` updates the model.
This lets an application adapt continuously without coordinating an external
batch-retraining loop.

Most models consume a `dict[str, float]`, while graph and time-aware models
document their required keys explicitly. Detector state, warm-up behavior,
memory policy, and score scale remain model-specific rather than being hidden
behind a batch-estimator abstraction.

> [!NOTE]
> ABERRANT is pre-1.0 and under active development. Public APIs may change as
> the model contracts and implementations mature.

## Why ABERRANT?

- **Use one streaming contract** across isolation forests, distance methods,
  sketches, graph detectors, online statistics, SVMs, time-series methods, and
  reconstruction models.
- **Choose the right state strategy** from sliding windows, bounded sketches,
  fading summaries, and model-specific incremental updates.
- **Compose online preprocessing** with detectors using `|` pipelines.
- **Separate detection from policy** with drift detectors and static or adaptive
  score thresholds.
- **Run repeatable experiments** with registry-backed benchmark streams and a
  validated local dataset cache.
- **Extend without framework coupling** through typed, structural transformer
  and model protocols. The distribution includes `py.typed` metadata.

## Installation

ABERRANT supports Python 3.10, 3.11, and 3.12.

```bash
pip install aberrant
```

<details>
<summary><strong>Optional extras</strong></summary>

| Extra | Adds |
| --- | --- |
| `eval` | scikit-learn metrics for model evaluation |
| `dl` | the PyTorch-backed `Autoencoder` |
| `faiss` | the FAISS similarity-search engine used by models such as `KNN` |
| `benchmark` | River and pytest-benchmark |
| `docs` | the documentation build toolchain |
| `dev` | linting, typing, testing, and development dependencies |
| `all` | all optional and development dependencies |

For example:

```bash
pip install "aberrant[eval,faiss]"
```

</details>

## Quick start

The following core-only example learns a scaled isolation forest from a
synthetic stream. The first 64 events warm up the pipeline; every later event is
scored before it is learned.

```python
import numpy as np

from aberrant.model.iforest import OnlineIsolationForest
from aberrant.transform.preprocessing import StandardScaler

rng = np.random.default_rng(42)
stream = np.vstack(
    [
        rng.normal(size=(400, 2)),
        rng.normal(loc=5.0, size=(20, 2)),
    ]
)

detector = StandardScaler() | OnlineIsolationForest(
    num_trees=25,
    window_size=256,
    seed=42,
)

scores = []
for step, values in enumerate(stream):
    event = {"x": float(values[0]), "y": float(values[1])}

    if step >= 64:
        scores.append((step, detector.score_one(event)))

    detector.learn_one(event)

for step, score in sorted(scores, key=lambda item: item[1], reverse=True)[:5]:
    print(f"event={step}, anomaly_score={score:.3f}")
```

Higher scores are more anomalous under the common model contract, but their
numeric range and calibration differ by detector. Compare or threshold scores
only according to the selected model's documented semantics.

## Choose a starting point

| Goal | Start with |
| --- | --- |
| General multivariate detection | [`OnlineIsolationForest` or another isolation-forest variant](https://oliverhennhoefer.github.io/aberrant/user_guide/models/#isolation-forest-family) |
| Local-neighborhood or density anomalies | [`LocalOutlierFactor`, `KNN`, `SDOStream`, or a cell-based detector](https://oliverhennhoefer.github.io/aberrant/user_guide/models/#distance-family) |
| Compact projection or frequency sketches | [`StreamingLODA`, `MStream`, or `StreamingRSHash`](https://oliverhennhoefer.github.io/aberrant/user_guide/models/#sketch-family) |
| Anomalous edges and graph evolution | [`AnoEdgeL`, `ISCONNA`, `MIDAS`, or `SignedGraphSketchDetector`](https://oliverhennhoefer.github.io/aberrant/user_guide/models/#graph-family) |
| Discords in a scalar time series | [`XLagDAMP`](https://oliverhennhoefer.github.io/aberrant/user_guide/models/#time-series-family) |
| Interpretable rolling statistics | [Univariate and multivariate moving statistics](https://oliverhennhoefer.github.io/aberrant/user_guide/models/#statistical-family) |
| Adaptive margin-based detection | [Online SVM models](https://oliverhennhoefer.github.io/aberrant/user_guide/models/#svm-family) |
| Learned reconstruction error | [`OnlineAutoencoderEnsemble` or the optional PyTorch `Autoencoder`](https://oliverhennhoefer.github.io/aberrant/user_guide/models/#deep-family) |
| Detecting distribution drift | [`ADWIN`, `KSWIN`, or `PageHinkley`](https://oliverhennhoefer.github.io/aberrant/api/drift/) |
| Turning a score into an alert signal | [`QuantileThreshold` or `ThresholdModel`](https://oliverhennhoefer.github.io/aberrant/user_guide/models/#core-utility-models) |

See the [model guide](https://oliverhennhoefer.github.io/aberrant/user_guide/models/)
for inputs, score interpretation, warm-up behavior, and memory characteristics.

<details>
<summary><strong>Included public model families</strong></summary>

| Family | Implementations |
| --- | --- |
| Isolation forest | `ASDIsolationForest`, `HalfSpaceTrees`, `MondrianIsolationForest`, `OnlineIsolationForest`, `RandomCutForest`, `StreamRandomHistogramForest`, `XStream` |
| Distance | `CellNeighborhoodDetector`, `KNN`, `LocalOutlierFactor`, `SDOStream`, `StationaryRegionNeighborDetector` |
| Sketch | `MStream`, `StreamingLODA`, `StreamingRSHash` |
| Graph | `AnoEdgeL`, `ISCONNA`, `MIDAS`, `SignedGraphSketchDetector` |
| Time series | `XLagDAMP` |
| SVM | `GraphGatedOneClassSVM`, `IncrementalOneClassSVMAdaptiveKernel` |
| Statistical | `MovingAverage`, `MovingAverageAbsoluteDeviation`, `MovingGeometricAverage`, `MovingHarmonicAverage`, `MovingInterquartileRange`, `MovingKurtosis`, `MovingMedian`, `MovingQuantile`, `MovingSkewness`, `MovingVariance`, `MovingCorrelationCoefficient`, `MovingCovariance`, `MovingMahalanobisDistance` |
| Reconstruction | `OnlineAutoencoderEnsemble`, optional `Autoencoder` |
| Score policy | `QuantileThreshold`, `ThresholdModel` |
| Baselines | `NullModel`, `RandomModel` |
| Drift detection | `ADWIN`, `KSWIN`, `PageHinkley` |

</details>

## Pipelines and custom components

Transformers compose left to right, with at most one terminal model:

```python
from aberrant.model.iforest import OnlineIsolationForest
from aberrant.transform import IncrementalPCA, StandardScaler

detector = (
    StandardScaler()
    | IncrementalPCA(n_components=3, n0=100)
    | OnlineIsolationForest(window_size=512, seed=42)
)
```

Any custom object satisfying `TransformerProtocol` or `ModelProtocol` can join
a pipeline; subclassing an ABERRANT base class is optional. Read the
[pipeline guide](https://oliverhennhoefer.github.io/aberrant/user_guide/pipelines/)
for lifecycle and composition rules.

## Streaming datasets

The dataset API downloads, validates, and caches registered benchmark data,
then exposes it as feature dictionaries and evaluation labels:

```python
from aberrant.stream.dataset import Dataset, load

dataset = load(Dataset.SHUTTLE)

for event, label in dataset.stream():
    score = detector.score_one(event)
    detector.learn_one(event)
```

Labels are provided for evaluation; unsupervised detectors learn only from the
event mapping. Cache location and download behavior are configurable through
the [streaming guide](https://oliverhennhoefer.github.io/aberrant/user_guide/streaming/).

## Update and scoring semantics

> [!IMPORTANT]
> For an honest prequential evaluation, call `score_one(event)` before
> `learn_one(event)`. Scoring does not update pipeline state. During
> `Pipeline.learn_one`, each transformer first learns the event and then passes
> its post-update transform to the next stage. Score scales and warm-up behavior
> are model-specific, so a single numeric threshold is not portable across
> detector families.

The [evaluation guide](https://oliverhennhoefer.github.io/aberrant/user_guide/evaluation/)
covers warm-up separation, label leakage, and useful metrics for imbalanced
anomaly streams.

## Project

Read the [documentation](https://oliverhennhoefer.github.io/aberrant/), browse
the [examples](https://github.com/OliverHennhoefer/aberrant/tree/main/examples),
or report a problem in the [issue tracker](https://github.com/OliverHennhoefer/aberrant/issues).
Contributions are welcome; start with the
[contributing guide](https://github.com/OliverHennhoefer/aberrant/blob/main/CONTRIBUTING.md).

ABERRANT is distributed under the
[MIT License](https://github.com/OliverHennhoefer/aberrant/blob/main/LICENSE).
