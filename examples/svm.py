"""The adaptive one-class SVM on the Shuttle benchmark.

It scores an event before it learns it, and learns every event; labels only
feed the metrics. Run with ``uv run --extra eval python examples/svm.py``.
"""

from itertools import islice

from aberrant.datasets import Dataset, load
from aberrant.evaluate import PrequentialEvaluator
from aberrant.model.svm import AdaptiveOneClassSVM
from aberrant.transform import StandardScaler


def main() -> None:
    events = list(islice(load(Dataset.SHUTTLE), 20_000))
    detector = StandardScaler() | AdaptiveOneClassSVM(seed=1)
    evaluator = PrequentialEvaluator(
        detector, warmup=500, metrics=True, metric_window_size=None
    )
    result = evaluator.evaluate(events)
    print(
        f"AdaptiveOneClassSVM average precision {result.average_precision:.3f}  "
        f"ROC AUC {result.roc_auc:.3f}"
    )


if __name__ == "__main__":
    main()
