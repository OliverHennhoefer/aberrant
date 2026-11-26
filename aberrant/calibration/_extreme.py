"""Extreme value calibrators: streaming peaks over threshold."""

from __future__ import annotations

import bisect
import math
from collections import deque
from collections.abc import Hashable

import numpy as np

from aberrant.calibration._window import ScoreHistory, conformal_quantile

# Fewest excesses the generalized Pareto fit is made from.
MIN_PEAKS = 10
# Fewest scores in the window before the first fit.
MIN_SCORES = 5 * MIN_PEAKS
# Grid points of the root search in Grimshaw's equation: a geometric grid
# brackets each root, a linear one inside the bracket narrows it, and a
# secant step between the last two points places it.
_GRID = 96


class SPOT:
    """Streaming peaks over threshold over a window of recent scores.

    The window holds the last ``W`` scores. Their ``level`` quantile is the
    initial threshold ``t``, which is set again once the window has turned
    over; the excesses over ``t`` are fitted with a generalized Pareto
    distribution, whose tail quantile is the threshold a score exceeds with
    probability ``alpha``. The fit is redone only when an excess enters or
    leaves the window. With ``depth`` above 0 this is DSPOT: the method
    calibrates the deviation from the mean of the previous ``depth`` scores,
    so a drifting level does not count as extreme.

    Until the window holds enough scores for a fit, the threshold is the
    window's conformal quantile.
    """

    def __init__(
        self, history: ScoreHistory, alpha: float, level: float, depth: int = 0
    ) -> None:
        self.alpha = alpha
        self.level = level
        self.depth = depth
        self.size = history.window_size
        self.window: deque[float] = deque(maxlen=self.size)
        self._sorted: list[float] = []
        self._recent: deque[float] = deque(maxlen=depth or 1)
        self._initial: float | None = None
        self._since_initial = 0
        self._tail: float | None = None

    def _base(self) -> float | None:
        if not self.depth:
            return 0.0
        if len(self._recent) < self.depth:
            return None
        return math.fsum(self._recent) / self.depth

    def predict(self, key: Hashable) -> float:
        base = self._base()
        if base is None:
            return math.inf
        if self._tail is None:
            return base + conformal_quantile(self._sorted, 1 - self.alpha)
        return base + self._tail

    def update(self, key: Hashable, score: float) -> None:
        base = self._base()
        self._recent.append(score)
        if base is None:
            return
        value = score - base
        changed = False
        if len(self.window) == self.size:
            leaving = self.window[0]
            _remove_sorted(self._sorted, leaving)
            changed = self._initial is None or leaving > self._initial
        self.window.append(value)
        bisect.insort_left(self._sorted, value)
        changed = changed or self._initial is None or value > self._initial
        self._since_initial += 1
        if self._initial is None or self._since_initial >= self.size:
            self._initial = initial_threshold(self._sorted, self.level)
            self._since_initial = 0
            changed = True
        if changed:
            self._tail = self._fit()

    def _fit(self) -> float | None:
        if self._initial is None:
            return None
        start = bisect.bisect_right(self._sorted, self._initial)
        peaks = np.asarray(self._sorted[start:]) - self._initial
        return tail_quantile(self._initial, peaks, len(self._sorted), self.alpha)


def initial_threshold(ordered: list[float], level: float) -> float | None:
    """The ``level`` quantile of ``ordered``, lowered to leave enough excesses.

    ``None`` while there are too few scores for a fit.
    """
    n = len(ordered)
    if n < MIN_SCORES:
        return None
    return ordered[min(int(level * n), n - MIN_PEAKS - 1)]


def tail_quantile(
    initial: float, peaks: np.ndarray, n: int, alpha: float
) -> float | None:
    """The ``1 - alpha`` quantile of ``n`` scores whose excesses are ``peaks``.

    ``None`` when there are too few excesses for a fit, or ``alpha`` is not
    below their share, where the empirical quantile serves instead.
    """
    if len(peaks) < MIN_PEAKS:
        return None
    ratio = alpha * n / len(peaks)
    if ratio >= 1:
        return None
    gamma, sigma = fit_gpd(peaks)
    if gamma == 0:
        return float(initial - sigma * math.log(ratio))
    return float(initial + sigma / gamma * (ratio**-gamma - 1))


def fit_gpd(peaks: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood ``(gamma, sigma)`` of a generalized Pareto sample.

    Grimshaw's reduction to one equation in ``x = gamma / sigma``, as in
    Siffer et al.: every root is a candidate, with the exponential case
    ``gamma = 0``, and the one with the highest likelihood wins.
    """
    smallest, largest, mean = peaks.min(), peaks.max(), peaks.mean()
    candidates = [(0.0, float(mean))]
    # Roots lie in (-1 / largest, 0) and (0, 2 (mean - smallest) / smallest^2);
    # geometric grids resolve both ends of each side.
    near = 1e-6 / mean
    grids = [-np.geomspace(near, (1 - 1e-9) / largest, _GRID)[::-1]]
    if mean > smallest > 0:
        upper = 2 * (mean - smallest) / smallest**2
        if upper > near:
            grids.append(np.geomspace(near, upper, _GRID))
    for grid in grids:
        for x in _roots(peaks, grid):
            gamma = float(np.mean(np.log1p(x * peaks)))
            if gamma != 0:
                candidates.append((gamma, gamma / x))
    return max(candidates, key=lambda c: _log_likelihood(peaks, *c))


def _grimshaw(peaks: np.ndarray, xs: np.ndarray) -> np.ndarray:
    shifted = 1 + np.outer(xs, peaks)
    u = np.mean(1 / shifted, axis=1)
    v = 1 + np.mean(np.log(shifted), axis=1)
    values: np.ndarray = u * v - 1
    return values


def _roots(peaks: np.ndarray, grid: np.ndarray) -> list[float]:
    """The roots of Grimshaw's equation that ``grid`` brackets."""
    roots = []
    values = _grimshaw(peaks, grid)
    for index in np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:])):
        fine = np.linspace(grid[index], grid[index + 1], _GRID)
        signs = _grimshaw(peaks, fine)
        step = int(np.flatnonzero(np.signbit(signs[:-1]) != np.signbit(signs[1:]))[0])
        (a, b), (fa, fb) = fine[step : step + 2], signs[step : step + 2]
        roots.append(float(a - fa * (b - a) / (fb - fa)) if fb != fa else float(a))
    return roots


def _log_likelihood(peaks: np.ndarray, gamma: float, sigma: float) -> float:
    if sigma <= 0:
        return -math.inf
    if gamma == 0:
        return float(-len(peaks) * math.log(sigma) - peaks.sum() / sigma)
    shifted = 1 + gamma / sigma * peaks
    if np.any(shifted <= 0):
        return -math.inf
    return float(
        -len(peaks) * math.log(sigma) - (1 + 1 / gamma) * np.log(shifted).sum()
    )


def _remove_sorted(values: list[float], value: float) -> None:
    index = bisect.bisect_left(values, value)
    if index < len(values) and values[index] == value:
        del values[index]
