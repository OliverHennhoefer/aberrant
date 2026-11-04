"""Render the README stream figure from a real aberrant run.

Three synthetic sensors receive one injected anomaly each: a spike, a level
shift and a dropout. A scaled ``LocalOutlierFactor`` scores every event before
learning it, and an online ``QuantileThreshold`` turns the scores into alerts.

Requires matplotlib, which is not an aberrant dependency::

    pip install aberrant matplotlib
    python docs/scripts/stream_figure.py

Writes ``docs/img/stream-light.svg`` and ``docs/img/stream-dark.svg``.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from aberrant.model import QuantileThreshold
from aberrant.model.distance import LocalOutlierFactor
from aberrant.transform import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "img"
N_EVENTS = 1500
ALERT_START = 300  # leave the detector and the threshold time to warm up
# label: (channel, first event, end event, label height in axes fraction)
INJECTED = {
    "spike": (0, 520, 523, 0.88),
    "level shift": (1, 900, 950, 0.88),
    "dropout": (2, 1250, 1280, 0.22),
}
THEMES = {
    "light": {
        "text": "#52514e",
        "muted": "#8a8984",
        "sensor": "#52514e",
        "score": "#2a78d6",
        "alert": "#eb6834",
        "band": "#00000012",
    },
    "dark": {
        "text": "#c3c2b7",
        "muted": "#8f8e86",
        "sensor": "#c3c2b7",
        "score": "#3987e5",
        "alert": "#d95926",
        "band": "#ffffff14",
    },
}


def make_stream() -> np.ndarray:
    """Return three seasonal sensor channels with one injected anomaly each."""
    rng = np.random.default_rng(7)
    t = np.arange(N_EVENTS)
    temperature = 20 + 3 * np.sin(2 * np.pi * t / 300) + rng.normal(0, 0.25, N_EVENTS)
    pressure = (
        101 + 0.8 * np.sin(2 * np.pi * t / 450 + 1) + rng.normal(0, 0.12, N_EVENTS)
    )
    vibration = 1 + 0.15 * np.sin(2 * np.pi * t / 120) + rng.normal(0, 0.05, N_EVENTS)
    temperature[520:523] += np.array([9.0, 7.0, 4.0])
    pressure[900:950] += 2.2
    vibration[1250:1280] = 0.0
    return np.column_stack([temperature, pressure, vibration])


def score_stream(stream: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Score each event before learning it and collect threshold alerts."""
    detector = StandardScaler() | LocalOutlierFactor(k=10, window_size=300)
    threshold = QuantileThreshold(quantile=0.998, window_size=1000)
    scores = np.zeros(len(stream))
    alerts = []
    for index, row in enumerate(stream):
        event = {"temperature": row[0], "pressure": row[1], "vibration": row[2]}
        if index > 0:  # the scaler needs one event before it can transform
            scores[index] = detector.score_one(event)
            if (
                index >= ALERT_START
                and threshold.score_one({"score": scores[index]}) >= 1.0
            ):
                alerts.append(index)
            threshold.learn_one({"score": scores[index]})
        detector.learn_one(event)
    return scores, alerts


def render(
    stream: np.ndarray, scores: np.ndarray, alerts: list[int], theme: str
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
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1.4], "hspace": 0.15},
    )
    names = ["temperature", "pressure", "vibration"]
    for channel, ax in enumerate(axes[:3]):
        ax.plot(stream[:, channel], color=colors["sensor"], linewidth=0.9)
        ax.set_ylabel(names[channel], rotation=0, ha="right", va="center")
        ax.set_yticks([])
    for label, (channel, start, stop, height) in INJECTED.items():
        ax = axes[channel]
        ax.axvspan(start - 8, stop + 8, color=colors["band"], linewidth=0)
        ax.annotate(
            label,
            xy=(stop + 14, height),
            xycoords=("data", "axes fraction"),
            color=colors["muted"],
            fontsize=9,
            va="top",
        )
    score_ax = axes[3]
    score_ax.plot(scores, color=colors["score"], linewidth=1.0, label="anomaly score")
    score_ax.scatter(
        alerts,
        scores[alerts],
        s=36,
        color=colors["alert"],
        zorder=3,
        label="alert",
    )
    for alert in alerts:
        for ax in axes:
            ax.axvline(alert, color=colors["alert"], linewidth=0.8, alpha=0.6, zorder=0)
    score_ax.set_ylabel("anomaly\nscore", rotation=0, ha="right", va="center")
    score_ax.set_xlabel("event")
    score_ax.set_xlim(0, N_EVENTS)
    score_ax.legend(loc="upper left", frameon=False, ncols=2)
    for ax in axes:
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    fig.savefig(
        OUTPUT_DIR / f"stream-{theme}.svg",
        bbox_inches="tight",
        transparent=True,
        metadata={"Date": None},  # keep re-renders byte-identical
    )
    plt.close(fig)


def main() -> None:
    stream = make_stream()
    scores, alerts = score_stream(stream)
    for theme in THEMES:
        render(stream, scores, alerts, theme)
    print(f"alerts at events {alerts}")


if __name__ == "__main__":
    main()
