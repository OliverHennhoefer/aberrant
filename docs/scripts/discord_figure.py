"""Render the README discord figure from a real aberrant run.

The series is ``013_UCR_Anomaly_DISTORTEDECG3`` from the UCR Time Series
Anomaly Archive (Wu and Keogh, 2021): 30,000 noisy ECG samples with baseline
wander and a single labelled anomalous heartbeat at samples 16,000-16,100.
Every value of that beat lies inside the range of the normal beats, so a
detector that looks at one value at a time has nothing to find.

``XLagDAMP`` scores each event before learning it, in one pass, from the first
sample, without the archive's training split and with bounded memory. For
contrast, ``HalfSpaceTrees`` scores the same raw values the same way.

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
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from aberrant.model.iforest import HalfSpaceTrees
from aberrant.model.timeseries import XLagDAMP
from aberrant.transform import MinMaxScaler

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "img"
SERIES = "013_UCR_Anomaly_DISTORTEDECG3"
DATA_URL = (
    "https://raw.githubusercontent.com/mbsa-tud/tsad_platform/"
    "4b47f217d176bd8942cfbb2f17d8cbc849b87fe4/datasets/univariate/"
    f"semi-supervised/KDD-TSAD/{SERIES}/test/{SERIES}.test.csv"
)
DATA_SHA256 = "3811edc0a3dbc6b2aa4fdfb314f56fde32bb352a79cd55acfadc4be0e8485e44"
SUBSEQUENCE_LENGTH = 167  # about one heartbeat
# UCR archive convention: a detection counts when the top score lies within
# 100 samples of the labelled region. DAMP scores the subsequence that ends
# at each event, so the region is widened by one subsequence on the right.
TOLERANCE = 100
ZOOM = (15690, 16430)
WARMUP = 4 * SUBSEQUENCE_LENGTH  # XLagDAMP's default start_index
BUCKETS = 1600  # min/max envelope per bucket keeps the SVGs small
THEMES = {
    "light": {
        "text": "#52514e",
        "muted": "#8a8984",
        "series": "#52514e",
        "score": "#2a78d6",
        "baseline": "#8a8984",
        "alert": "#eb6834",
        "band": "#eb683426",
    },
    "dark": {
        "text": "#c3c2b7",
        "muted": "#8f8e86",
        "series": "#c3c2b7",
        "score": "#3987e5",
        "baseline": "#8f8e86",
        "alert": "#d95926",
        "band": "#d9592633",
    },
}


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


def score_series(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score every event before learning it with DAMP and with HalfSpaceTrees."""
    damp = XLagDAMP(subsequence_length=SUBSEQUENCE_LENGTH)
    point = MinMaxScaler() | HalfSpaceTrees(seed=0)
    damp_scores = np.zeros(len(values))
    point_scores = np.zeros(len(values))
    for index, value in enumerate(values):
        event = {"ecg": float(value)}
        damp_scores[index] = damp.score_one(event)
        damp.learn_one(event)
        point_scores[index] = point.score_one(event)
        point.learn_one(event)
    return damp_scores, point_scores


def is_hit(top: int, span: tuple[int, int]) -> bool:
    start, stop = span
    return start - TOLERANCE <= top < stop + SUBSEQUENCE_LENGTH + TOLERANCE


def envelope(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a min/max envelope that draws like the full-resolution line."""
    edges = np.linspace(0, len(values), BUCKETS + 1).astype(int)
    xs, ys = [], []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        chunk = values[left:right]
        low, high = int(np.argmin(chunk)), int(np.argmax(chunk))
        for offset in sorted((low, high)):
            xs.append(left + offset)
            ys.append(chunk[offset])
    return np.array(xs), np.array(ys)


def render(
    values: np.ndarray,
    span: tuple[int, int],
    damp_scores: np.ndarray,
    point_scores: np.ndarray,
    theme: str,
) -> None:
    """Write one themed SVG with a transparent background."""
    colors = THEMES[theme]
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "aberrant",  # stable element ids across runs
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": colors["text"],
            "axes.labelcolor": colors["text"],
            "xtick.color": colors["muted"],
            "ytick.color": colors["muted"],
            "axes.edgecolor": colors["muted"],
        }
    )
    fig = plt.figure(figsize=(12, 5.4))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[3, 1.15],
        height_ratios=[1.2, 1, 1],
        hspace=0.35,
        wspace=0.12,
    )
    stream_ax = fig.add_subplot(grid[0, 0])
    damp_ax = fig.add_subplot(grid[1, 0], sharex=stream_ax)
    point_ax = fig.add_subplot(grid[2, 0], sharex=stream_ax)
    zoom_ax = fig.add_subplot(grid[:, 1])
    start, stop = span

    stream_ax.plot(*envelope(values), color=colors["series"], linewidth=0.5)
    stream_ax.set_title(
        "30,000 noisy ECG samples, one labelled anomalous beat",
        loc="left",
        fontsize=10,
    )
    for name, ax, scores, color in (
        ("XLagDAMP", damp_ax, damp_scores, colors["score"]),
        ("HalfSpaceTrees on raw values", point_ax, point_scores, colors["baseline"]),
    ):
        warm = scores[WARMUP:]
        scaled = (warm - warm.min()) / (warm.max() - warm.min())
        xs, ys = envelope(scaled)
        ax.plot(xs + WARMUP, ys, color=color, linewidth=0.7)
        top = int(np.argmax(scores))
        verdict = "finds the beat" if is_hit(top, span) else "misses it"
        ax.scatter([top], [1.0], s=30, color=colors["alert"], zorder=3)
        ax.set_title(
            f"{name}: top score at event {top:,}, {verdict}",
            loc="left",
            fontsize=10,
        )
        ax.set_ylim(0, 1.15)
    for ax in (stream_ax, damp_ax, point_ax):
        ax.axvspan(start - 150, stop + 150, color=colors["band"], linewidth=0)
        ax.set_yticks([])
        ax.set_xlim(0, len(values))
    for ax in (stream_ax, damp_ax):
        ax.tick_params(labelbottom=False)
    point_ax.set_xlabel("event")

    left, right = ZOOM
    window = np.arange(left, right)
    low, high = values.min(), values.max()
    for level in (low, high):
        zoom_ax.axhline(level, color=colors["muted"], linewidth=0.8, linestyle="--")
    zoom_ax.set_ylim(low - 0.06 * (high - low), high + 0.2 * (high - low))
    zoom_ax.plot(window, values[left:right], color=colors["series"], linewidth=0.9)
    zoom_ax.plot(
        np.arange(start, stop),
        values[start:stop],
        color=colors["alert"],
        linewidth=1.4,
    )
    zoom_ax.set_title("the anomalous beat stays in range", loc="left", fontsize=10)
    zoom_ax.annotate(
        "dashed: lowest and highest value\nin the whole series",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        color=colors["muted"],
        fontsize=9,
        va="top",
    )
    zoom_ax.set_yticks([])
    zoom_ax.set_xlim(left, right)
    zoom_ax.set_xticks([16000, 16200])
    zoom_ax.set_xlabel("event")

    for ax in (stream_ax, damp_ax, point_ax, zoom_ax):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    fig.savefig(
        OUTPUT_DIR / f"discord-{theme}.svg",
        bbox_inches="tight",
        transparent=True,
        metadata={"Date": None},  # keep re-renders byte-identical
    )
    plt.close(fig)


def main() -> None:
    values, span = load_series()
    damp_scores, point_scores = score_series(values)
    for theme in THEMES:
        render(values, span, damp_scores, point_scores, theme)
    print(
        f"label {span}, DAMP top {int(np.argmax(damp_scores))}, "
        f"HalfSpaceTrees top {int(np.argmax(point_scores))}"
    )


if __name__ == "__main__":
    main()
