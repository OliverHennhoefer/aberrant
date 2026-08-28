# Examples

The repository's scripts are complete programs rather than fragments. Run them
from the repository root so relative project configuration and the local
package checkout are available.

=== "uv"

    ```bash
    uv sync --extra eval
    uv run python examples/models/online_iforest.py
    ```

=== "pip"

    ```bash
    python -m pip install -e ".[eval]"
    python examples/models/online_iforest.py
    ```

Most model scripts compute average precision and ROC AUC and therefore require
the `eval` extra. Dataset-backed scripts can download and cache a registered NPZ
artifact on first use.

## General numeric streams

| Family | Example | Additional requirement |
| --- | --- | --- |
| Isolation forest | [`online_iforest.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/online_iforest.py) | Registered SHUTTLE dataset |
| Isolation forest | [`mondrian_isolation_forest.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/mondrian_isolation_forest.py) | Registered SHUTTLE dataset |
| Isolation forest | [`asd_iforest.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/asd_iforest.py) | Registered SHUTTLE dataset |
| Isolation forest | [`streamRHF.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/streamRHF.py) | Registered SHUTTLE dataset |
| Isolation forest | [`xstream.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/xstream.py) | Registered SHUTTLE dataset |
| Random cut forest | [`random_cut_forest.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/random_cut_forest.py) | Registered SHUTTLE dataset |
| Half-space trees | [`half_space_trees.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/half_space_trees.py) | Registered SHUTTLE dataset |
| Local density | [`local_outlier_factor.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/local_outlier_factor.py) | Registered SHUTTLE dataset |
| Cell neighborhood | [`cell_neighborhood.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/cell_neighborhood.py) | Registered SHUTTLE dataset |
| Observer distance | [`sdostream.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/sdostream.py) | Registered SHUTTLE dataset |
| Stationary region | [`stationary_region_neighbor.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/stationary_region_neighbor.py) | Registered SHUTTLE dataset |
| Multi-aspect sketch | [`mstream.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/mstream.py) | Registered SHUTTLE dataset |
| Projection histogram | [`streaming_loda.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/streaming_loda.py) | Registered SHUTTLE dataset |
| Randomized hashing | [`streaming_rshash.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/streaming_rshash.py) | Registered SHUTTLE dataset |

## Graph and time-series streams

| Model | Example | Data source |
| --- | --- | --- |
| AnoEdge-L | [`anoedge.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/anoedge.py) | Maps two SHUTTLE features to integer-like edge identifiers for demonstration |
| ISCONNA | [`isconna.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/isconna.py) | Seeded synthetic edge stream |
| MIDAS-R | [`midas.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/midas.py) | Seeded synthetic edge stream |
| Signed graph sketch | [`signed_graph_sketch.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/signed_graph_sketch.py) | Seeded synthetic multi-graph stream |
| X-Lag DAMP | [`xlag_damp.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/xlag_damp.py) | Seeded synthetic periodic series with an injected discord; no `eval` extra needed |

The AnoEdge example is an API demonstration, not a claim that tabular SHUTTLE
rows are a scientifically meaningful graph benchmark.

## Experimental, reconstruction, and pipeline examples

| Example | Purpose | Additional requirement |
| --- | --- | --- |
| [`adaptive_svm.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/adaptive_svm.py) | Budgeted adaptive-kernel SVM heuristic | Registered SHUTTLE dataset |
| [`graph_gated_svm.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/graph_gated_svm.py) | Graph-gated linear SVM heuristic | Registered FRAUD dataset |
| [`online_autoencoder_ensemble.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/online_autoencoder_ensemble.py) | NumPy online autoencoder ensemble | Registered SHUTTLE dataset |
| [`autoencoder.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/autoencoder.py) | User-supplied PyTorch architecture and optimizer | `aberrant[dl,eval]` and registered SHUTTLE dataset |
| [`knn.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/knn.py) | FAISS-backed KNN distance | `aberrant[faiss,eval]` and registered SHUTTLE dataset |
| [`pipeline.py`](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/pipeline.py) | Scaler/KNN versus scaler/PCA/KNN | `aberrant[faiss,eval]` and registered SHUTTLE dataset |

Read each script's warm-up and learning policy before comparing its metrics.
Several demonstrations intentionally warm up only on labeled-normal rows; that
is a curated-normal protocol and is not equivalent to fully unsupervised
test-then-train learning.
