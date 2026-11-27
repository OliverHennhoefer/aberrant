"""The look that the README figures share: palette, type and small helpers.

``discord_figure.py`` and ``taxi_figure.py`` import it, so both draw a raw
stream in grey, a score in blue and what the score found in orange, in a light
and a dark variant with a transparent background.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "img"
BUCKETS = 1600  # min/max envelope per bucket keeps the SVGs small
THEMES = {
    "light": {
        "text": "#3d3c39",
        "muted": "#8a8984",
        "faint": "#c9c8c2",
        "line": "#52514e",
        "score": "#2a78d6",
        "accent": "#eb6834",
        "zoom": "#8a898422",
    },
    "dark": {
        "text": "#d8d7cd",
        "muted": "#8f8e86",
        "faint": "#4a4945",
        "line": "#c3c2b7",
        "score": "#3987e5",
        "accent": "#f07a45",
        "zoom": "#8f8e8630",
    },
}


def use_theme(theme: str) -> dict[str, str]:
    """Set matplotlib up for one theme and return its colors."""
    colors = THEMES[theme]
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "aberrant",  # stable element ids across runs
            "font.family": "sans-serif",
            "font.size": 11,
            "text.color": colors["text"],
            "axes.labelcolor": colors["muted"],
            "xtick.color": colors["muted"],
            "ytick.color": colors["muted"],
            "axes.edgecolor": colors["faint"],
        }
    )
    return colors


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
    ax: plt.Axes, colors: dict[str, str], title: str, note: str, note_color: str
) -> None:
    """Write a bold panel title on the left and a note on the right."""
    ax.text(0, 1.07, title, transform=ax.transAxes, fontweight="bold", va="bottom")
    ax.text(
        1,
        1.07,
        note,
        transform=ax.transAxes,
        color=colors[note_color],
        ha="right",
        va="bottom",
        fontsize=9.5,
    )


def save(fig: plt.Figure, name: str, theme: str) -> None:
    """Drop the frames and write one themed SVG with a transparent background."""
    for ax in fig.axes:
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    fig.savefig(
        OUTPUT_DIR / f"{name}-{theme}.svg",
        bbox_inches="tight",
        transparent=True,
        metadata={"Date": None},  # keep re-renders byte-identical
    )
    plt.close(fig)
