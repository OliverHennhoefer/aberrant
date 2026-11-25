# Examples

The scripts are complete programs. Run them from the repository root; CI runs
every script and notebook on each change, so they stay in step with the API.

=== "uv"

    ```bash
    uv sync
    uv run python examples/forest.py
    ```

=== "pip"

    ```bash
    python -m pip install -e ".[eval]"
    python examples/forest.py
    ```

The family scripts evaluate prequentially with `PrequentialEvaluator`: every
detector scores an event before it learns it and learns every event, normal or
not. Labels only feed the metrics, so the numbers are what an unsupervised
detector achieves on the stream. Metrics need the `eval` extra, and the
Shuttle-based scripts download the dataset on first use.

## Notebooks

These notebooks install aberrant themselves and run in Google Colab without a local setup.

| Notebook | What it shows | Run |
| --- | --- | --- |
| [`01_quickstart.ipynb`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/notebooks/01_quickstart.ipynb) | Quickstart: stream the Shuttle benchmark through a scaled Half-Space Trees detector and evaluate it prequentially | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OliverHennhoefer/aberrant/blob/main/examples/notebooks/01_quickstart.ipynb) |
| [`02_graph_streams.ipynb`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/notebooks/02_graph_streams.ipynb) | Graph streams: catch a burst of repeated edges with MIDAS | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OliverHennhoefer/aberrant/blob/main/examples/notebooks/02_graph_streams.ipynb) |
| [`03_scores_to_alerts.ipynb`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/notebooks/03_scores_to_alerts.ipynb) | From scores to alerts: an adaptive quantile threshold plus Page-Hinkley drift detection | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OliverHennhoefer/aberrant/blob/main/examples/notebooks/03_scores_to_alerts.ipynb) |

## One script per model family

| Script | Detectors | Data |
| --- | --- | --- |
| [`forest.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/forest.py) | HalfSpaceTrees, OnlineIsolationForest, ASDIsolationForest, XStream | First 20,000 Shuttle events |
| [`distance.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/distance.py) | KNN, LocalOutlierFactor, SDOStream | First 20,000 Shuttle events |
| [`sketch.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/sketch.py) | LODA, RSHash, MStream | First 20,000 Shuttle events |
| [`stat.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/stat.py) | OnlineECOD, OnlineCOPOD, MovingMahalanobisDistance, MovingZScore, MovingMAD | Shuttle, and a synthetic sensor with spikes |
| [`svm.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/svm.py) | AdaptiveOneClassSVM | First 20,000 Shuttle events |
| [`deep.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/deep.py) | OnlineAutoencoderEnsemble, Autoencoder (needs the `dl` extra) | First 20,000 Shuttle events |
| [`graph.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/graph.py) | MIDAS, ISCONNA, AnoEdgeL, StreamSpot | Synthetic edge streams with injected anomalous edges |
| [`timeseries.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/timeseries.py) | DAMP, RollingMatrixProfile, MultivariateRollingMatrixProfile | A periodic series with an injected discord |
| [`forecast.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/forecast.py) | ResidualDetector over HoltWinters | A seasonal request count with one anomalous value |

## Composition and evaluation

| Script | What it shows |
| --- | --- |
| [`pipeline.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/pipeline.py) | A scaled KNN with and without an incremental PCA in front of it |
| [`graph_pipeline.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/graph_pipeline.py) | Branched preprocessing with a raw feature bypass in a `GraphPipeline` |
| [`prequential_evaluation.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/prequential_evaluation.py) | Detectors built from catalog configurations, compared with metrics and timings |
| [`rolling_robust_scaler.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/rolling_robust_scaler.py) | A recent-window scaler against a frozen one on a level shift |
