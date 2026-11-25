"""Slow quality checks that the nightly Benchmark workflow runs.

The pull request lane checks every detector with one seed. These tests check
the same floors over more seeds, check the detectors on a harder, shuffled
dataset and compare detectors with river where river implements the same
algorithm. Run them with ``pytest tests/integration -m benchmark``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from aberrant.drift import ADWIN, PageHinkley
from aberrant.model.forest import HalfSpaceTrees
from aberrant.stream import Dataset
from aberrant.transform import MinMaxScaler
from tests.integration._quality import Case, assert_detects, prequential
from tests.integration.test_detection_quality import CASES, N_TEST, WARM_UP

pytestmark = pytest.mark.benchmark

EXTRA_SEEDS = (0, 1, 2, 3)

COVER_N_TEST = 10_000
COVER_SEED = 42
# Shuffled COVER holds about 98 anomalies in 10,000 rows, a 1% base rate.
# Measured on main with model and shuffle seeds 42, 0 and 1: each ROC-AUC
# floor sits 0.05 below the lowest of the three, each PR-AUC floor at 70% of
# it. The best multivariate detector reached 0.985, so a ROC-AUC above the
# alarm means the labels leaked into the scores. The univariate cases are left
# out: elevation alone separates COVER's anomalies (ROC-AUC 0.999), and the
# torch autoencoder is sized for SHUTTLE.
COVER_LEAKAGE_ROC_AUC = 0.999
COVER_FLOORS: dict[str, tuple[float, float]] = {
    "asd-isolation-forest": (0.81, 0.03),
    "half-space-trees": (0.66, 0.02),
    "mondrian-isolation-forest": (0.8, 0.024),
    "online-isolation-forest": (0.69, 0.014),
    "random-histogram-forest": (0.73, 0.015),
    "random-cut-forest": (0.79, 0.037),
    "xstream": (0.81, 0.025),
    "cell-neighborhood": (0.81, 0.028),
    "sdostream": (0.61, 0.009),
    "knn-faiss": (0.85, 0.039),
    "loda": (0.82, 0.024),
    "rshash": (0.7, 0.015),
    "online-autoencoder-ensemble": (0.66, 0.012),
    "adaptive-kernel-svm": (0.84, 0.056),
    "ecod-sketch": (0.86, 0.067),
    "ecod-window": (0.84, 0.055),
    "ecod-hybrid-pareto": (0.85, 0.059),
    "ecod-decay": (0.85, 0.062),
    "copod-pyod": (0.84, 0.046),
    "moving-covariance": (0.71, 0.094),
    "moving-mahalanobis": (0.85, 0.038),
}


def _skip_without(case: Case) -> None:
    for module in case.requires:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{case.name} needs {module}")


@pytest.mark.parametrize("seed", EXTRA_SEEDS)
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_detector_keeps_its_quality_for_other_seeds(case: Case, seed: int) -> None:
    _skip_without(case)
    result = prequential(
        case, Dataset.SHUTTLE, warm_up=WARM_UP, n_test=N_TEST, seed=seed
    )
    assert_detects(
        result.labels,
        result.scores,
        roc_auc_floor=case.roc_auc_floor,
        pr_auc_floor=case.pr_auc_floor,
    )


COVER_CASES = [case for case in CASES if case.name in COVER_FLOORS]


@pytest.mark.parametrize("case", COVER_CASES, ids=[case.name for case in COVER_CASES])
def test_detector_ranks_shuffled_cover_without_leakage(case: Case) -> None:
    _skip_without(case)
    roc_auc_floor, pr_auc_floor = COVER_FLOORS[case.name]
    result = prequential(
        case,
        Dataset.COVER,
        warm_up=WARM_UP,
        n_test=COVER_N_TEST,
        seed=COVER_SEED,
        shuffle_seed=COVER_SEED,
    )
    assert_detects(
        result.labels,
        result.scores,
        roc_auc_floor=roc_auc_floor,
        pr_auc_floor=pr_auc_floor,
    )
    assert result.roc_auc < COVER_LEAKAGE_ROC_AUC, "labels leak into the scores"


class _RiverHalfSpaceTrees:
    def __init__(self, seed: int) -> None:
        from river import anomaly, compose, preprocessing  # noqa: PLC0415

        self._model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            anomaly.HalfSpaceTrees(n_trees=25, height=15, window_size=250, seed=seed),
        )

    def learn_one(self, x: dict[str, float]) -> None:
        self._model.learn_one(x)

    def score_one(self, x: dict[str, float]) -> float:
        return float(self._model.score_one(x))


def test_half_space_trees_ranks_shuttle_as_well_as_river() -> None:
    pytest.importorskip("river")
    ours = Case(
        "aberrant",
        lambda s: (
            MinMaxScaler()
            | HalfSpaceTrees(n_trees=25, height=15, window_size=250, seed=s)
        ),
        0.51,
        0.0,
    )
    river = Case("river", _RiverHalfSpaceTrees, 0.51, 0.0)
    seeds = (42, *EXTRA_SEEDS)
    runs = {
        case.name: [
            prequential(
                case, Dataset.SHUTTLE, warm_up=WARM_UP, n_test=N_TEST, seed=seed
            )
            for seed in seeds
        ]
        for case in (ours, river)
    }
    roc_auc = {name: np.mean([r.roc_auc for r in rs]) for name, rs in runs.items()}
    pr_auc = {name: np.mean([r.pr_auc for r in rs]) for name, rs in runs.items()}
    assert roc_auc["aberrant"] >= roc_auc["river"] - 0.02, roc_auc
    assert pr_auc["aberrant"] >= pr_auc["river"] - 0.05, pr_auc


def _drift_streams() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "mean-shift": np.r_[
            rng.normal(0, 1, 1000), rng.normal(1.5, 1, 1000), rng.normal(-1, 1, 1000)
        ],
        "binary-rate": np.r_[
            rng.binomial(1, 0.2, 2000), rng.binomial(1, 0.6, 2000)
        ].astype(float),
        "variance": np.r_[rng.normal(0, 1, 1500), rng.normal(0, 4, 1500)],
    }


def _detections(detector: object, stream: np.ndarray) -> list[int]:
    found = []
    for index, value in enumerate(stream):
        detector.update(float(value))  # type: ignore[attr-defined]
        if detector.drift_detected:  # type: ignore[attr-defined]
            found.append(index)
    return found


@pytest.mark.parametrize("stream", sorted(_drift_streams()))
@pytest.mark.parametrize("name", ["ADWIN", "PageHinkley"])
def test_drift_detector_flags_the_same_points_as_river(name: str, stream: str) -> None:
    river_drift = pytest.importorskip("river.drift")
    ours = {"ADWIN": ADWIN, "PageHinkley": PageHinkley}[name]()
    theirs = getattr(river_drift, name)()
    values = _drift_streams()[stream]
    expected = _detections(theirs, values)
    assert expected, "the reference finds no drift, so the stream proves nothing"
    assert _detections(ours, values) == expected
