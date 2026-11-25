"""Range-aware ranking metrics: the volume under the range ROC and PR surfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aberrant._utils.validation import coerce_integer_count


@dataclass(frozen=True, slots=True)
class VolumeUnderSurface:
    """VUS-ROC and VUS-PR of one score series against its anomaly ranges."""

    roc: float
    pr: float


def vus(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    window: int,
    thresholds: int = 250,
) -> VolumeUnderSurface:
    """Volume under the range-based ROC and PR surfaces (Paparrizos et al. 2022).

    Point-wise AUC rewards a detector for each anomalous event it ranks high,
    so a long anomaly found late outweighs a short one found on time, and a
    score just before or after a range counts as a false positive. The range
    metrics count each labeled range as one event, weigh its detection by
    how much of it is covered, and give partial credit to scores within a
    soft margin around it. VUS averages them over every margin from 0 to
    ``window``, so the result does not hinge on one margin.

    This follows ``RangeAUC_volume_opt`` of the reference package ``vus``
    0.0.7, the threshold grid included.

    Args:
        labels: Binary labels in time order, ``1`` for an anomalous event.
        scores: One finite score per event, higher is more anomalous.
        window: Largest margin, in events, around each range; about one
            period of the series is the usual choice.
        thresholds: Number of thresholds, at evenly spaced ranks of the
            sorted scores.

    Raises:
        ValueError: The series are empty, of different lengths, not binary
            or not finite, or the labels are all normal or all anomalous.
    """
    label, score = _series(labels, scores)
    window = coerce_integer_count(window, label="window", minimum=0)
    thresholds = coerce_integer_count(thresholds, label="thresholds", minimum=2)
    n = len(score)
    positives = float(label.sum())
    ranges = _ranges(label)

    order_desc = -np.sort(-score)
    grid = order_desc[np.linspace(0, n - 1, thresholds).astype(int)]
    predicted = n - np.searchsorted(np.sort(score), grid, side="left")

    near = _merged(ranges, window, n)
    region = np.concatenate([np.arange(a, b + 1) for a, b in near])
    order = region[np.argsort(-score[region], kind="stable")]
    reached = np.searchsorted(-score[order], -grid, side="right")
    anomalous = label[order] > 0
    found = np.concatenate([[0], np.cumsum(anomalous)])

    rocs, prs = [], []
    for margin in range(window + 1):
        soft = _extended(label, ranges, margin)
        segments = _merged(ranges, margin, n)
        credit = np.concatenate([[0], np.cumsum(np.where(anomalous, 1, soft[order]))])
        peaks = np.array([score[a : b + 1].max() for a, b in segments])

        tp = credit[reached]
        positive_mass = (positives + (found[-1] + tp - found[reached])) / 2
        existence = (peaks[:, None] >= grid).sum(0) / len(segments)
        tpr = np.concatenate([[0], np.minimum(tp / positive_mass, 1) * existence, [1]])
        fpr = np.concatenate([[0], (predicted - tp) / (n - positive_mass), [1]])
        precision = np.concatenate([[1], tp / predicted])

        rocs.append(np.dot(np.diff(fpr), (tpr[1:] + tpr[:-1]) / 2))
        prs.append(np.dot(np.diff(tpr[:-1]), precision[1:]))
    return VolumeUnderSurface(float(np.mean(rocs)), float(np.mean(prs)))


def _series(
    labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    label = np.asarray(labels, dtype=float)
    score = np.asarray(scores, dtype=float)
    if label.ndim != 1 or score.shape != label.shape or not len(label):
        raise ValueError("labels and scores must be equally long, non-empty series")
    if not np.isin(label, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    if not np.isfinite(score).all():
        raise ValueError("scores must be finite")
    if not 0 < label.sum() < len(label):
        raise ValueError("labels must hold both normal and anomalous events")
    return label, score


def _ranges(label: np.ndarray) -> list[tuple[int, int]]:
    """Each run of anomalous events as ``(first, last)``."""
    edges = np.diff(np.concatenate([[0], label, [0]]))
    starts = np.flatnonzero(edges > 0)
    ends = np.flatnonzero(edges < 0) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _merged(
    ranges: list[tuple[int, int]], margin: int, n: int
) -> list[tuple[int, int]]:
    """The ranges widened by ``margin // 2`` on each side, overlaps merged."""
    half = margin // 2
    merged = []
    start = max(ranges[0][0] - half, 0)
    for (_, end), (following, _) in zip(ranges, ranges[1:], strict=False):
        if end + half < following - half:
            merged.append((start, end + half))
            start = following - half
    merged.append((start, min(ranges[-1][1] + half, n - 1)))
    return merged


def _extended(
    label: np.ndarray, ranges: list[tuple[int, int]], margin: int
) -> np.ndarray:
    """Labels with credit ``sqrt(1 - d / margin)`` at distance ``d`` outside a range."""
    soft = label.copy()
    n = len(soft)
    distance = np.arange(1, margin // 2 + 1)
    if not len(distance):
        return soft
    weight = np.sqrt(1 - distance / margin)
    for start, end in ranges:
        after, before = end + distance, start - distance
        soft += np.bincount(after[after < n], weight[after < n], n)
        soft += np.bincount(before[before >= 0], weight[before >= 0], n)
    capped: np.ndarray = np.minimum(soft, 1)
    return capped
