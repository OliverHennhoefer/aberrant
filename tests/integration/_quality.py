"""Shared prequential harness for integration quality tests.

Every detector learns a warm-up of normal rows, then scores each test row
before learning it. Quality is checked with floors, never with caps: a
detector that gets better must keep passing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.stream import Dataset, load

Features = dict[str, float]
Adapter = Callable[[int, Features], Features]


def unchanged(_: int, features: Features) -> Features:
    return features


def with_arrival_time(index: int, features: Features) -> Features:
    sample = dict(features)
    sample["t"] = float(index)
    return sample


def first_features(count: int) -> Adapter:
    def adapt(_: int, features: Features) -> Features:
        return dict(list(features.items())[:count])

    return adapt


@dataclass(frozen=True)
class Case:
    """One detector configuration and the quality it must keep."""

    name: str
    build: Callable[[int], object]
    roc_auc_floor: float
    pr_auc_floor: float
    adapt: Adapter = unchanged
    requires: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Result:
    labels: np.ndarray
    scores: np.ndarray

    @property
    def roc_auc(self) -> float:
        return float(roc_auc_score(self.labels, self.scores))

    @property
    def pr_auc(self) -> float:
        return float(average_precision_score(self.labels, self.scores))


def _rows(dataset: Dataset, shuffle_seed: int | None) -> Iterator[tuple[Features, int]]:
    rows = load(dataset).stream()
    if shuffle_seed is None:
        yield from rows
        return
    materialized = list(rows)
    order = np.random.default_rng(shuffle_seed).permutation(len(materialized))
    for index in order:
        yield materialized[index]


def prequential(
    case: Case,
    dataset: Dataset,
    *,
    warm_up: int,
    n_test: int,
    seed: int,
    shuffle_seed: int | None = None,
) -> Result:
    """Learn ``warm_up`` normal rows, then score and learn ``n_test`` rows."""
    model = case.build(seed)
    labels: list[int] = []
    scores: list[float] = []
    warmed = 0
    for index, (features, label) in enumerate(_rows(dataset, shuffle_seed)):
        sample = case.adapt(index, features)
        if warmed < warm_up:
            if label == 0:
                model.learn_one(sample)  # type: ignore[attr-defined]
                warmed += 1
            continue
        if len(scores) >= n_test:
            break
        scores.append(float(model.score_one(sample)))  # type: ignore[attr-defined]
        model.learn_one(sample)  # type: ignore[attr-defined]
        labels.append(int(label))
    return Result(np.asarray(labels), np.asarray(scores))


def assert_detects(
    labels: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    *,
    roc_auc_floor: float,
    pr_auc_floor: float,
) -> None:
    """Fail unless the scores rank anomalies above both floors."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    assert labels.size > 0, "no test rows were scored"
    assert 0 < labels.sum() < labels.size, "test rows hold only one class"
    assert np.all(np.isfinite(scores)), "scores are not all finite"
    assert np.ptp(scores) > 0.0, "every score is the same"
    assert roc_auc_floor > 0.5, "a detector's floor must beat chance"
    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))
    assert roc_auc >= roc_auc_floor, f"ROC-AUC {roc_auc:.4f} fell below {roc_auc_floor}"
    assert pr_auc >= pr_auc_floor, f"PR-AUC {pr_auc:.4f} fell below {pr_auc_floor}"
