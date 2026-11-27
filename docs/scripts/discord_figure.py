"""Render the README discord figure from a real aberrant run.

The series is ``013_UCR_Anomaly_DISTORTEDECG3`` from the UCR Time Series
Anomaly Archive (Wu and Keogh, 2021): 30,000 noisy ECG samples with baseline
wander and a single labelled anomalous heartbeat at samples 16,000-16,100.
Every value of that beat lies inside the range of the normal beats, so a
detector that looks at one value at a time has nothing to find.

``DAMP`` scores each event before learning it, in one pass, from the first
sample, without the archive's training split and with bounded memory. Its
highest score lands on the anomalous beat.

The figure draws the heartbeats around that beat large enough to read, between
two dotted lines at the lowest and highest normal value of the series. Below
them it draws the DAMP score of the whole series and marks the stretch shown
above.

Requires matplotlib, which is not an aberrant dependency::

    pip install aberrant matplotlib
    python docs/scripts/discord_figure.py

Downloads TimeEval's CSV copy of the series (about 540 kB, checksum verified)
and writes ``docs/img/discord-light.svg`` and ``docs/img/discord-dark.svg``.
"""

import csv
import hashlib
import io
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from figure_style import THEMES, envelope, heading, save, use_theme
from matplotlib.patches import Polygon

from aberrant.model.timeseries import DAMP

SERIES = "013_UCR_Anomaly_DISTORTEDECG3"
DATA_URL = (
    "https://raw.githubusercontent.com/mbsa-tud/tsad_platform/"
    "4b47f217d176bd8942cfbb2f17d8cbc849b87fe4/datasets/univariate/"
    f"semi-supervised/KDD-TSAD/{SERIES}/test/{SERIES}.test.csv"
)
DATA_SHA256 = "3811edc0a3dbc6b2aa4fdfb314f56fde32bb352a79cd55acfadc4be0e8485e44"
SUBSEQUENCE_LENGTH = 167  # about one heartbeat
R_PEAK_HEIGHT = 100.0  # normal R peaks rise far above this; the anomaly does not
EXCERPT_BEATS = 5  # heartbeats drawn on each side of the anomalous one
WARMUP = 4 * SUBSEQUENCE_LENGTH  # DAMP's default start_index
SCORE_CEILING = 1.12  # top of the score axis; the highest score is drawn at 1


def load_series() -> tuple[np.ndarray, tuple[int, int]]:
    """Download the series, verify it and return values and the label span."""
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DATA_SHA256:
        raise RuntimeError(f"unexpected checksum {digest} for {DATA_URL}")
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    values = np.array([float(row["value"]) for row in rows])
    labelled = np.flatnonzero([int(row["is_anomaly"]) for row in rows])
    return values, (int(labelled[0]), int(labelled[-1]) + 1)


def score_series(values: np.ndarray) -> np.ndarray:
    """Score every event with DAMP before learning it."""
    damp = DAMP(subsequence_length=SUBSEQUENCE_LENGTH)
    scores = np.zeros(len(values))
    for index, value in enumerate(values):
        event = {"ecg": float(value)}
        scores[index] = damp.score_one(event)
        damp.learn_one(event)
    return scores


def find_beats(values: np.ndarray, span: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Return one peak index per heartbeat and the position of the anomaly."""
    half = 60
    peaks = [
        index
        for index in range(len(values))
        if values[index] > R_PEAK_HEIGHT
        and values[index] == values[max(0, index - half) : index + half].max()
    ]
    start, stop = span
    anomaly_peak = start + int(np.argmax(values[start:stop]))
    peaks = sorted([*peaks, anomaly_peak])
    return np.array(peaks), peaks.index(anomaly_peak)


def excerpt_window(peaks: np.ndarray, anomaly: int) -> tuple[int, int]:
    """Return the samples that span the heartbeats around the anomalous one."""
    half_beat = int(np.median(np.diff(peaks))) // 2
    return (
        int(peaks[anomaly - EXCERPT_BEATS]) - half_beat,
        int(peaks[anomaly + EXCERPT_BEATS]) + half_beat,
    )


def draw_excerpt(
    ax: plt.Axes,
    colors: dict[str, str],
    values: np.ndarray,
    span: tuple[int, int],
    window: tuple[int, int],
) -> None:
    """Draw the heartbeats around the labelled one inside the normal range."""
    left, right = window
    start, stop = span
    normal = np.concatenate([values[:start], values[stop:]])
    for bound in (normal.min(), normal.max()):
        ax.axhline(
            bound, color=colors["muted"], linewidth=0.7, linestyle=(0, (1, 4)), zorder=0
        )
    ax.plot(
        np.arange(left, right), values[left:right], color=colors["line"], linewidth=1.0
    )
    ax.plot(
        np.arange(start, stop + 1),
        values[start : stop + 1],
        color=colors["accent"],
        linewidth=2.2,
        solid_capstyle="round",
    )
    ax.set_xlim(left, right)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_scores(
    ax: plt.Axes,
    colors: dict[str, str],
    scores: np.ndarray,
    window: tuple[int, int],
) -> None:
    """Draw the DAMP score of the whole series and mark its highest score."""
    warm = scores[WARMUP:]
    xs, ys = envelope(warm / warm.max())
    ax.plot(xs + WARMUP, ys, color=colors["score"], linewidth=0.8)
    ax.axvspan(*window, color=colors["zoom"], linewidth=0)
    top = int(np.argmax(scores))
    ax.scatter([top], [1.0], s=38, color=colors["accent"], zorder=3, linewidth=0)
    ax.annotate(
        "highest score",
        (top, 1.0),
        xytext=(9, 0),
        textcoords="offset points",
        va="center",
        fontsize=9.5,
    )
    ax.set_xlim(0, len(scores))
    ax.set_ylim(0, SCORE_CEILING)
    ax.set_yticks([])
    ax.set_xticks([0, 10_000, 20_000, 30_000], ["0", "10,000", "20,000", "30,000"])


def draw_zoom(
    fig: plt.Figure,
    colors: dict[str, str],
    excerpt_ax: plt.Axes,
    score_ax: plt.Axes,
    window: tuple[int, int],
) -> None:
    """Join the excerpt to the stretch of the score axis that it shows."""
    left, right = window
    to_figure = fig.transFigure.inverted()
    corners = [
        excerpt_ax.transAxes.transform((0, 0)),
        excerpt_ax.transAxes.transform((1, 0)),
        score_ax.transData.transform((right, SCORE_CEILING)),
        score_ax.transData.transform((left, SCORE_CEILING)),
    ]
    fig.add_artist(
        Polygon(
            to_figure.transform(corners),
            facecolor=colors["zoom"],
            edgecolor="none",
            transform=fig.transFigure,
        )
    )


def render(
    values: np.ndarray,
    span: tuple[int, int],
    damp_scores: np.ndarray,
    theme: str,
) -> None:
    """Write one themed SVG with a transparent background."""
    colors = use_theme(theme)
    peaks, anomaly = find_beats(values, span)
    window = excerpt_window(peaks, anomaly)
    fig = plt.figure(figsize=(9.6, 4.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.62)
    excerpt_ax = fig.add_subplot(grid[0])
    score_ax = fig.add_subplot(grid[1])

    draw_excerpt(excerpt_ax, colors, values, span, window)
    heading(
        excerpt_ax,
        colors,
        f"ECG: {2 * EXCERPT_BEATS + 1} of {len(peaks)} heartbeats",
        "wrong shape, but every sample inside the normal range",
        "accent",
    )
    draw_scores(score_ax, colors, damp_scores, window)
    heading(
        score_ax,
        colors,
        f"DAMP score for all {len(values):,} samples",
        "one pass, no training split",
        "muted",
    )
    draw_zoom(fig, colors, excerpt_ax, score_ax, window)

    excerpt_ax.spines["bottom"].set_visible(False)
    save(fig, "discord", theme)


def main() -> None:
    values, span = load_series()
    damp_scores = score_series(values)
    for theme in THEMES:
        render(values, span, damp_scores, theme)
    print(f"label {span}, DAMP top {int(np.argmax(damp_scores))}")


if __name__ == "__main__":
    main()
