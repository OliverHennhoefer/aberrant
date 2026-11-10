"""The value a forecaster issues: point, spread and quantiles per lead."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite_tuple(values: object, label: str) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{label} must be a tuple")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{label} must hold numbers")
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
    return tuple(float(value) for value in values)


def _check_leads(leads: object) -> None:
    if not isinstance(leads, tuple) or not leads:
        raise ValueError("leads must be a non-empty tuple")
    previous = -1
    for lead in leads:
        if isinstance(lead, bool) or not isinstance(lead, int) or lead <= previous:
            raise ValueError("leads must be strictly increasing integers >= 0")
        previous = lead


def _quantile_rows(
    quantiles: object, levels: tuple[float, ...], n_leads: int
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(quantiles, tuple):
        raise ValueError("quantiles must be a tuple of rows")
    if not levels:
        if quantiles:
            raise ValueError("quantiles need levels")
        return ()
    if len(quantiles) != n_leads:
        raise ValueError("quantiles need one row per lead")
    rows = tuple(_finite_tuple(row, "quantiles") for row in quantiles)
    for row in rows:
        if len(row) != len(levels):
            raise ValueError("each quantile row needs one value per level")
        if any(low > high for low, high in zip(row, row[1:], strict=False)):
            raise ValueError("quantiles must be non-decreasing along levels")
    return rows


@dataclass(frozen=True, slots=True)
class Forecast:
    """Forecasts of one target at one or more leads, issued at one origin.

    Lead ``k`` forecasts the target of the ``origin + k``-th learned event,
    where ``origin`` counts the events learned when the forecast was issued.
    A forecaster issues leads ``1..horizon``; a regressor issues lead ``0``,
    the target of the event whose covariates it was given.

    Args:
        origin: Number of events learned when the forecast was issued.
        leads: Strictly increasing, non-negative leads.
        mean: Point forecast for each lead.
        scale: Positive spread for each lead, in target units, or ``None``.
            Each forecaster documents what its spread estimates.
        levels: Strictly increasing quantile levels in (0, 1), or empty.
        quantiles: One row per lead, one non-decreasing value per level;
            empty when ``levels`` is empty.
    """

    origin: int
    leads: tuple[int, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...] | None = None
    levels: tuple[float, ...] = ()
    quantiles: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.origin, bool) or not isinstance(self.origin, int):
            raise ValueError("origin must be a non-negative integer")
        if self.origin < 0:
            raise ValueError("origin must be a non-negative integer")
        _check_leads(self.leads)
        object.__setattr__(self, "mean", _finite_tuple(self.mean, "mean"))
        if len(self.mean) != len(self.leads):
            raise ValueError("mean needs one value per lead")
        if self.scale is not None:
            scale = _finite_tuple(self.scale, "scale")
            if len(scale) != len(self.leads) or any(value <= 0 for value in scale):
                raise ValueError("scale needs one positive value per lead")
            object.__setattr__(self, "scale", scale)
        levels = _finite_tuple(self.levels, "levels")
        if any(not 0 < level < 1 for level in levels) or any(
            low >= high for low, high in zip(levels, levels[1:], strict=False)
        ):
            raise ValueError("levels must be strictly increasing and in (0, 1)")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(
            self, "quantiles", _quantile_rows(self.quantiles, levels, len(self.leads))
        )

    @property
    def horizon(self) -> int:
        """The largest lead."""
        return self.leads[-1]

    def _index(self, lead: int) -> int:
        try:
            return self.leads.index(lead)
        except ValueError:
            raise KeyError(f"No forecast for lead {lead!r}") from None

    def point(self, lead: int) -> float:
        """The point forecast at ``lead``."""
        return self.mean[self._index(lead)]

    def spread(self, lead: int) -> float | None:
        """The spread at ``lead``, or ``None`` without one."""
        if self.scale is None:
            return None
        return self.scale[self._index(lead)]

    def quantile(self, lead: int, level: float) -> float:
        """The quantile at ``level``, interpolated linearly between grid levels.

        Raises:
            ValueError: Without quantiles, or for a level outside the grid.
        """
        if not self.levels:
            raise ValueError("This forecast has no quantiles")
        row = self.quantiles[self._index(lead)]
        levels = self.levels
        if not levels[0] <= level <= levels[-1]:
            raise ValueError(
                f"level must be within [{levels[0]:g}, {levels[-1]:g}], the grid"
            )
        for position in range(1, len(levels)):
            if level <= levels[position]:
                low, high = levels[position - 1], levels[position]
                weight = (level - low) / (high - low)
                return row[position - 1] + weight * (row[position] - row[position - 1])
        return row[0]
