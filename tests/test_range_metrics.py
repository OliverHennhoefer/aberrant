"""VUS: range-aware ROC and PR volumes against a plain transcription."""

import numpy as np
import pytest

from aberrant.evaluate import VolumeUnderSurface, vus


def _ranges(label):
    ranges, start = [], None
    for i, value in enumerate(label):
        if value and start is None:
            start = i
        if not value and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(label) - 1))
    return ranges


def _merged(ranges, margin, n):
    half, merged = margin // 2, []
    start = max(ranges[0][0] - half, 0)
    for i in range(len(ranges) - 1):
        if ranges[i][1] + half < ranges[i + 1][0] - half:
            merged.append((start, ranges[i][1] + half))
            start = ranges[i + 1][0] - half
    merged.append((start, min(ranges[-1][1] + half, n - 1)))
    return merged


def _reference(label, score, window, thresholds=250):
    """One threshold and one margin at a time, as the reference defines them."""
    label, score = np.asarray(label, float), np.asarray(score, float)
    n, positives, ranges = len(label), label.sum(), _ranges(label)
    grid = -np.sort(-score)[np.linspace(0, n - 1, thresholds).astype(int)]
    rocs, prs = [], []
    for margin in range(window + 1):
        soft = label.copy()
        for start, end in ranges:
            for d in range(1, margin // 2 + 1):
                weight = np.sqrt(1 - d / margin)
                if end + d < n:
                    soft[end + d] += weight
                if start - d >= 0:
                    soft[start - d] += weight
        soft = np.minimum(soft, 1)
        segments = _merged(ranges, margin, n)
        tpr, fpr, precision = [0.0], [0.0], [1.0]
        for threshold in grid:
            predicted = score >= threshold
            tp = np.sum(np.where(label > 0, 1, soft) * predicted)
            mass = (positives + positives + tp - np.sum(label * predicted)) / 2
            found = sum(score[a : b + 1].max() >= threshold for a, b in segments)
            tpr.append(min(tp / mass, 1) * found / len(segments))
            fpr.append((predicted.sum() - tp) / (n - mass))
            precision.append(tp / predicted.sum())
        tpr.append(1.0)
        fpr.append(1.0)
        roc = sum(
            (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2
            for i in range(len(fpr) - 1)
        )
        pr = sum((tpr[i + 1] - tpr[i]) * precision[i + 1] for i in range(len(tpr) - 2))
        rocs.append(roc)
        prs.append(pr)
    return np.mean(rocs), np.mean(prs)


def _case(n, k, seed, ties=False, edges=False):
    rng = np.random.default_rng(seed)
    label = np.zeros(n, dtype=int)
    for _ in range(k):
        start = int(rng.integers(0, n - 30))
        label[start : start + int(rng.integers(1, 30))] = 1
    if edges:
        label[:4] = 1
        label[-3:] = 1
    score = rng.normal(0, 1, n) + 1.5 * label
    score += 0.7 * np.convolve(label, np.ones(9) / 9, mode="same")
    return label, np.round(score, 1) if ties else score


@pytest.mark.parametrize(
    ("n", "k", "window", "ties", "edges"),
    [
        (200, 1, 0, False, False),
        (200, 2, 1, False, False),
        (500, 3, 7, False, False),
        (600, 4, 20, True, False),
        (400, 2, 30, False, True),
        (800, 6, 41, True, True),
    ],
)
def test_vus_matches_one_threshold_at_a_time(n, k, window, ties, edges):
    label, score = _case(n, k, seed=n + window, ties=ties, edges=edges)
    expected = _reference(label, score, window)

    result = vus(label, score, window=window)

    assert (result.roc, result.pr) == pytest.approx(expected, abs=1e-12)


def test_a_perfect_score_has_full_volume():
    label, _ = _case(1000, 4, seed=5)

    result = vus(label, label.astype(float), window=20)

    assert result.roc == pytest.approx(1.0, abs=0.01)
    assert result.pr == pytest.approx(1.0, abs=0.01)


def test_a_margin_credits_a_detection_just_before_its_range():
    label = np.zeros(1000, dtype=int)
    label[[200, 500, 800]] = 1
    early = np.roll(label, -2).astype(float)

    tight = vus(label, early, window=0)
    loose = vus(label, early, window=10)

    assert loose.pr > tight.pr + 0.3
    assert isinstance(loose, VolumeUnderSurface)


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([], [], "non-empty"),
        ([0, 1], [0.1], "equally long"),
        ([0, 2], [0.1, 0.2], "0 or 1"),
        ([0, 1], [0.1, float("nan")], "finite"),
        ([0, 0], [0.1, 0.2], "both normal and anomalous"),
        ([1, 1], [0.1, 0.2], "both normal and anomalous"),
    ],
)
def test_invalid_series_are_rejected(labels, scores, message):
    with pytest.raises(ValueError, match=message):
        vus(labels, scores, window=2)
