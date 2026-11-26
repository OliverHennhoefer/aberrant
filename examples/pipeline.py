"""Compare a scaled KNN with and without an incremental PCA in front of it.

Pipelines compose with ``|``: each transformer learns from the events it sees
and passes on its representation. Both pipelines score each of the first
20,000 Shuttle events before they learn it; labels only feed the metrics. Run
with ``uv run --extra eval python examples/pipeline.py``.
"""

from itertools import islice

from aberrant.datasets import Dataset, load
from aberrant.evaluate import PrequentialEvaluator
from aberrant.model.distance import KNN
from aberrant.transform import IncrementalPCA, StandardScaler


def main() -> None:
    events = list(islice(load(Dataset.SHUTTLE), 20_000))
    pipelines = {
        "StandardScaler | KNN": StandardScaler() | KNN(k=10, window_size=500),
        "StandardScaler | IncrementalPCA | KNN": StandardScaler()
        | IncrementalPCA(n_components=3, warm_up=100)
        | KNN(k=10, window_size=500),
    }
    for name, pipeline in pipelines.items():
        evaluator = PrequentialEvaluator(
            pipeline, warm_up=500, metrics=True, metric_window_size=None
        )
        result = evaluator.evaluate(events)
        print(
            f"{name:<38} average precision {result.average_precision:.3f}  "
            f"ROC AUC {result.roc_auc:.3f}"
        )


if __name__ == "__main__":
    main()
