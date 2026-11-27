"""Render the README taxi figure from a real aberrant run.

The series is ``realKnownCause/nyc_taxi`` from the Numenta Anomaly Benchmark
(Lavin and Ahmad, 2015): the number of New York taxi passengers in each half
hour from July 2014 to January 2015, 10,320 counts. The benchmark labels five
windows in it and names what happened: the New York marathon, Thanksgiving,
Christmas, New Year's Day and a snowstorm.

``RollingMatrixProfile`` scores each count before learning it. A score says
how far the latest 24 hours are from the most similar 24 hours in the 16 days
that the detector keeps by default. The detector starts with no history and
never sees a label.

The figure draws the scores from the moment those 16 days are full. The five
highest peaks from then on fall in the five labelled windows, one in each, and
the script fails if they do not.

Requires matplotlib, which is not an aberrant dependency::

    pip install aberrant matplotlib
    python docs/scripts/taxi_figure.py

Downloads the series from the benchmark's repository (about 270 kB, checksum
verified) and writes ``docs/img/taxi-light.svg`` and
``docs/img/taxi-dark.svg``.
"""

import csv
import hashlib
import io
import urllib.request
from datetime import datetime
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from figure_style import THEMES, envelope, heading, save, use_theme

from aberrant.model.timeseries import RollingMatrixProfile

DATA_URL = (
    "https://raw.githubusercontent.com/numenta/NAB/v1.1/"
    "data/realKnownCause/nyc_taxi.csv"
)
DATA_SHA256 = "d8fa6f7f0734bf5c8be12c52a94e20a82664c397d9dec4449156bd453d32856d"
PER_DAY = 48  # one count per half hour
PEAK_GAP = 2 * PER_DAY  # two peaks closer than this belong to one event
SCORE_CEILING = 1.34  # top of the score axis; the highest score is drawn at 1
# The benchmark's windows (labels/combined_windows.json), named after its
# description of the series (data/README.md).
LABELLED = {
    "Marathon": ("2014-10-30 15:30", "2014-11-03 22:30"),
    "Thanksgiving": ("2014-11-25 12:00", "2014-11-29 19:00"),
    "Christmas": ("2014-12-23 11:30", "2014-12-27 18:30"),
    "New Year": ("2014-12-29 21:30", "2015-01-03 04:30"),
    "Snowstorm": ("2015-01-24 20:30", "2015-01-29 03:30"),
}
# Christmas and New Year are a week apart, so one name sits a line higher. The
# last name ends at its peak to stay inside the figure.
NAME_PLACEMENT = {"Christmas": ("center", 21), "Snowstorm": ("right", 7)}


class Scored(NamedTuple):
    """The score of every count and the first count that the figure draws."""

    scores: np.ndarray
    start: int


def load_series() -> tuple[list[datetime], np.ndarray]:
    """Download the series, verify it and return its times and counts."""
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DATA_SHA256:
        raise RuntimeError(f"unexpected checksum {digest} for {DATA_URL}")
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    times = [datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S") for row in rows]
    return times, np.array([float(row["value"]) for row in rows])


def score_series(values: np.ndarray) -> Scored:
    """Score every count before learning it, with the default 16-day history."""
    detector = RollingMatrixProfile(subsequence_length=PER_DAY)
    scores = np.zeros(len(values))
    for index, value in enumerate(values):
        event = {"passengers": float(value)}
        scores[index] = detector.score_one(event)
        detector.learn_one(event)
    return Scored(scores, detector.window_size)


def highest_peaks(scored: Scored, count: int) -> list[int]:
    """Return the highest scores that the figure draws, one per event."""
    peaks: list[int] = []
    for index in np.argsort(scored.scores)[::-1]:
        apart = all(abs(index - peak) > PEAK_GAP for peak in peaks)
        if index >= scored.start and apart:
            peaks.append(int(index))
        if len(peaks) == count:
            break
    return peaks


def name_peaks(times: list[datetime], peaks: list[int]) -> dict[str, int]:
    """Match each peak to the labelled window that holds it, or fail."""
    named = {}
    for name, (start, stop) in LABELLED.items():
        first = datetime.strptime(start, "%Y-%m-%d %H:%M")
        last = datetime.strptime(stop, "%Y-%m-%d %H:%M")
        inside = [peak for peak in peaks if first <= times[peak] <= last]
        if len(inside) != 1:
            raise RuntimeError(f"{len(inside)} of the highest peaks fall on {name}")
        named[name] = inside[0]
    return named


def month_ticks(times: list[datetime]) -> tuple[list[int], list[str]]:
    """Return the first count of each month and a label for it."""
    positions = [
        index
        for index, time in enumerate(times)
        if time.day == 1 and time.hour == 0 and time.minute == 0
    ]
    labels = [
        times[index].strftime("%b %Y" if index in {0, positions[-1]} else "%b")
        for index in positions
    ]
    return positions, labels


def draw_stream(ax: plt.Axes, colors: dict[str, str], values: np.ndarray) -> None:
    """Draw the counts as they arrived."""
    ax.plot(*envelope(values), color=colors["line"], linewidth=0.5)
    ax.set_xlim(0, len(values))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)


def draw_scores(
    ax: plt.Axes, colors: dict[str, str], scored: Scored, named: dict[str, int]
) -> None:
    """Draw the scores and name their highest peaks."""
    scores, start = scored
    scaled = scores / scores[start:].max()
    xs, ys = envelope(scaled[start:])
    ax.plot(xs + start, ys, color=colors["score"], linewidth=0.8)
    for name, peak in named.items():
        anchor, lift = NAME_PLACEMENT.get(name, ("center", 7))
        ax.scatter(
            [peak], [scaled[peak]], s=30, color=colors["accent"], zorder=3, linewidth=0
        )
        ax.annotate(
            name,
            (peak, scaled[peak]),
            xytext=(0, lift),
            textcoords="offset points",
            ha=anchor,
            fontsize=9.5,
        )
    ax.set_xlim(0, len(scores))
    ax.set_ylim(0, SCORE_CEILING)
    ax.set_yticks([])


def render(
    times: list[datetime],
    values: np.ndarray,
    scored: Scored,
    named: dict[str, int],
    theme: str,
) -> None:
    """Write one themed SVG with a transparent background."""
    colors = use_theme(theme)
    fig = plt.figure(figsize=(9.6, 4.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.4)
    stream_ax = fig.add_subplot(grid[0])
    score_ax = fig.add_subplot(grid[1])

    draw_stream(stream_ax, colors, values)
    heading(
        stream_ax,
        colors,
        "New York taxi passengers, every half hour",
        f"{len(values):,} counts",
        "muted",
    )
    draw_scores(score_ax, colors, scored, named)
    heading(
        score_ax,
        colors,
        "RollingMatrixProfile score",
        "each count scored as it arrives, no training set",
        "muted",
    )
    score_ax.set_xticks(*month_ticks(times))
    score_ax.get_xticklabels()[0].set_horizontalalignment("left")
    for peak in named.values():  # where to look in the counts
        stream_ax.axvline(
            peak, color=colors["muted"], linewidth=0.7, linestyle=(0, (1, 4)), zorder=0
        )
    save(fig, "taxi", theme)


def main() -> None:
    times, values = load_series()
    scored = score_series(values)
    named = name_peaks(times, highest_peaks(scored, len(LABELLED)))
    for theme in THEMES:
        render(times, values, scored, named, theme)
    print(f"scores drawn from {times[scored.start]:%Y-%m-%d %H:%M}")
    for name, peak in named.items():
        print(f"{name}: {times[peak]:%Y-%m-%d %H:%M}, score {scored.scores[peak]:.2f}")


if __name__ == "__main__":
    main()
