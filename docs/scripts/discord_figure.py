"""Render the README discord figure from a real aberrant run.

The series is ``013_UCR_Anomaly_DISTORTEDECG3`` from the UCR Time Series
Anomaly Archive (Wu and Keogh, 2021): 30,000 noisy ECG samples with baseline
wander and a single labelled anomalous heartbeat at samples 16,000-16,100.
Every value of that beat lies inside the range of the normal beats, so a
detector that looks at one value at a time has nothing to find.

``XLagDAMP`` scores each event before learning it, in one pass, from the first
sample, without the archive's training split and with bounded memory. Its
highest score lands on the anomalous beat. An overlay of every heartbeat shows
why a detector that scores single values has no outlier to find there.

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
from matplotlib.transforms import blended_transform_factory

from aberrant.model.timeseries import XLagDAMP

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "img"
SERIES = "013_UCR_Anomaly_DISTORTEDECG3"
DATA_URL = (
    "https://raw.githubusercontent.com/mbsa-tud/tsad_platform/"
    "4b47f217d176bd8942cfbb2f17d8cbc849b87fe4/datasets/univariate/"
    f"semi-supervised/KDD-TSAD/{SERIES}/test/{SERIES}.test.csv"
)
DATA_SHA256 = "3811edc0a3dbc6b2aa4fdfb314f56fde32bb352a79cd55acfadc4be0e8485e44"
SUBSEQUENCE_LENGTH = 167  # about one heartbeat
R_PEAK_HEIGHT = 100.0  # normal R peaks rise far above this; the anomaly does not
BEAT_WINDOW = (-75, 115)  # samples around each beat's peak in the overlay
WARMUP = 4 * SUBSEQUENCE_LENGTH  # XLagDAMP's default start_index
BUCKETS = 1600  # min/max envelope per bucket keeps the SVGs small
THEMES = {
    "light": {
        "text": "#3d3c39",
        "muted": "#8a8984",
        "faint": "#c9c8c2",
        "line": "#52514e",
        "beats": "#52514e17",
        "score": "#2a78d6",
        "accent": "#eb6834",
        "band": "#eb68342e",
    },
    "dark": {
        "text": "#d8d7cd",
        "muted": "#8f8e86",
        "faint": "#4a4945",
        "line": "#c3c2b7",
        "beats": "#c3c2b724",
        "score": "#3987e5",
        "accent": "#f07a45",
        "band": "#f07a452b",
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


def score_series(values: np.ndarray) -> np.ndarray:
    """Score every event with XLagDAMP before learning it."""
    damp = XLagDAMP(subsequence_length=SUBSEQUENCE_LENGTH)
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


def heading(
    ax: plt.Axes, colors: dict[str, str], title: str, note: str, strong: bool = False
) -> None:
    """Write a bold panel title on the left and a note on the right."""
    ax.text(0, 1.2, title, transform=ax.transAxes, fontweight="bold", va="bottom")
    ax.text(
        1,
        1.2,
        note,
        transform=ax.transAxes,
        color=colors["text" if strong else "muted"],
        ha="right",
        va="bottom",
        fontsize=9,
    )


def draw_stream(
    ax: plt.Axes,
    colors: dict[str, str],
    values: np.ndarray,
    span: tuple[int, int],
    n_beats: int,
) -> None:
    """Draw the raw stream with the labelled beat marked."""
    start, stop = span
    ax.plot(*envelope(values), color=colors["line"], linewidth=0.45)
    ax.axvspan(start - 120, stop + 120, color=colors["band"], linewidth=0)
    ax.set_xlim(0, len(values))
    ax.set_xticks([])
    ax.set_yticks([])
    heading(
        ax,
        colors,
        "The stream",
        f"{len(values):,} noisy ECG samples, {n_beats} heartbeats, one anomalous",
    )


def draw_scores(
    ax: plt.Axes,
    colors: dict[str, str],
    scores: np.ndarray,
    span: tuple[int, int],
) -> None:
    """Draw the XLagDAMP score stream and mark its highest score."""
    start, stop = span
    warm = scores[WARMUP:]
    scaled = warm / warm.max()
    xs, ys = envelope(scaled)
    ax.plot(xs + WARMUP, ys, color=colors["score"], linewidth=0.7)
    top = int(np.argmax(scores))
    ax.scatter([top], [1.0], s=34, color=colors["accent"], zorder=3, linewidth=0)
    ax.axvspan(start - 120, stop + 120, color=colors["band"], linewidth=0)
    ax.set_xlim(0, len(scores))
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.set_xticks([0, 10_000, 20_000, 30_000], ["0", "10k", "20k", "30k"])
    ax.set_xlabel("sample")
    heading(
        ax,
        colors,
        "XLagDAMP score",
        f"one pass, no training split  ·  peak at sample {top:,}, the anomalous beat",
        strong=True,
    )


def draw_beats(
    ax: plt.Axes,
    colors: dict[str, str],
    values: np.ndarray,
    peaks: np.ndarray,
    anomaly: int,
) -> None:
    """Overlay every normal beat and draw the anomalous beat on top."""
    left, right = BEAT_WINDOW
    offsets = np.arange(left, right)
    for index, peak in enumerate(peaks):
        if index == anomaly or peak + left < 0 or peak + right > len(values):
            continue
        ax.plot(
            offsets,
            values[peak + offsets],
            color=colors["beats"],
            linewidth=0.8,
            rasterized=True,  # 180 overlaid lines would bloat the SVG
        )
    ax.plot(
        offsets,
        values[peaks[anomaly] + offsets],
        color=colors["accent"],
        linewidth=1.8,
    )
    ax.set_xlim(left, right)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        f"orange, over all {len(peaks) - 1} normal beats:\n"
        "same value range, different shape",
        loc="left",
        fontsize=9,
    )


def render(
    values: np.ndarray,
    span: tuple[int, int],
    damp_scores: np.ndarray,
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
            "axes.labelcolor": colors["muted"],
            "xtick.color": colors["muted"],
            "ytick.color": colors["muted"],
            "axes.edgecolor": colors["faint"],
        }
    )
    peaks, anomaly = find_beats(values, span)
    fig = plt.figure(figsize=(12, 4.6))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[2.7, 1],
        height_ratios=[1, 1],
        hspace=0.5,
        wspace=0.08,
    )
    stream_ax = fig.add_subplot(grid[0, 0])
    damp_ax = fig.add_subplot(grid[1, 0])
    beat_ax = fig.add_subplot(grid[:, 1])

    draw_stream(stream_ax, colors, values, span, len(peaks))
    draw_scores(damp_ax, colors, damp_scores, span)
    draw_beats(beat_ax, colors, values, peaks, anomaly)
    beat_ax.text(  # aligned with the stream title
        0,
        1.2,
        "The anomalous beat",
        transform=blended_transform_factory(beat_ax.transAxes, stream_ax.transAxes),
        fontweight="bold",
        va="bottom",
    )

    for ax in (stream_ax, damp_ax, beat_ax):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.patch.set_alpha(0)
    beat_ax.spines["bottom"].set_visible(False)
    fig.patch.set_alpha(0)
    fig.savefig(
        OUTPUT_DIR / f"discord-{theme}.svg",
        bbox_inches="tight",
        transparent=True,
        dpi=200,  # resolution of the rasterized beat overlay
        metadata={"Date": None},  # keep re-renders byte-identical
    )
    plt.close(fig)


def main() -> None:
    values, span = load_series()
    damp_scores = score_series(values)
    for theme in THEMES:
        render(values, span, damp_scores, theme)
    print(f"label {span}, XLagDAMP top {int(np.argmax(damp_scores))}")


if __name__ == "__main__":
    main()
