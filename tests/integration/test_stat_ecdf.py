"""Integration tests for the online ECOD and COPOD detectors."""

import pytest
from sklearn.metrics import average_precision_score

from aberrant.model.stat import OnlineCOPOD, OnlineECOD
from aberrant.stream.dataset import Dataset, load
from tests.integration._settings import MAX_TEST_STANDARD, WARMUP_SAMPLES


def _shuttle_pr_auc(model) -> float:
    labels, scores = [], []
    warmed = 0
    for features, label in load(Dataset.SHUTTLE).stream():
        if warmed < WARMUP_SAMPLES:
            if label == 0:
                model.learn_one(features)
                warmed += 1
            continue
        if len(scores) >= MAX_TEST_STANDARD:
            break
        scores.append(model.score_one(features))
        model.learn_one(features)
        labels.append(label)
    return float(average_precision_score(labels, scores))


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Measured at 0.866, 0.925, 0.911, 0.869 and 0.922; the ranges leave
        # room for platform rounding but catch a broken reference or tail.
        (OnlineECOD(), (0.80, 0.93)),
        (OnlineECOD(memory="window"), (0.86, 0.98)),
        (OnlineECOD(memory="hybrid", tail="pareto"), (0.85, 0.97)),
        (OnlineECOD(half_life=2_000.0), (0.80, 0.93)),
        (OnlineCOPOD(aggregation="pyod"), (0.86, 0.98)),
    ],
    ids=["sketch", "window", "hybrid-pareto", "decay", "copod-pyod"],
)
def test_shuttle_pr_auc_stays_in_expected_range(model, expected) -> None:
    lower, upper = expected
    assert lower <= _shuttle_pr_auc(model) <= upper
