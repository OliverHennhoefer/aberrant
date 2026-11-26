"""Fill in missing feature values at the input boundary."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Literal

from aberrant._utils.validation import coerce_event_number, coerce_finite_number
from aberrant.base.readiness import Readiness
from aberrant.base.transformer import BaseTransformer
from aberrant.exceptions import ValidationError

Strategy = Literal["last", "mean", "constant"]
_STRATEGIES = ("last", "mean", "constant")


class Imputer(BaseTransformer):
    """Fill in a missing value: an absent feature, ``None`` or NaN.

    Every other component rejects missing values, so an imputer goes first
    in a pipeline that receives them. ``"last"`` carries a feature's last
    observed value forward, ``"mean"`` uses the mean of its observed values
    and ``"constant"`` uses ``value``. Until a feature has been observed,
    ``"last"`` and ``"mean"`` fill in ``value`` too, and the imputer is not
    ready. Infinite and non-numeric values are rejected, not filled.

    The schema is ``features`` or else the keys of the first learned event;
    an event with any other key is rejected. Missing values never update the
    learned values. Output keeps the schema order, followed by
    ``{feature}_missing`` (1.0 or 0.0) for each feature when ``indicate``.

    Args:
        strategy: ``"last"``, ``"mean"`` or ``"constant"``.
        value: Fill for ``"constant"``, and for a feature not yet observed.
        indicate: Add a missing indicator per feature, so a detector can
            learn how often values go missing.
        features: The schema, in output order; ``None`` takes the first
            learned event's keys.
    """

    def __init__(
        self,
        *,
        strategy: Strategy = "last",
        value: float = 0.0,
        indicate: bool = False,
        features: Sequence[str] | None = None,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(_STRATEGIES)}")
        if type(indicate) is not bool:
            raise ValueError("indicate must be a boolean")
        if isinstance(features, str | bytes):
            raise ValueError("features must be a sequence of names, not a string")
        names = None if features is None else tuple(features)
        if names is not None and (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError("features must be distinct, non-empty feature names")
        self.strategy = strategy
        self.value = coerce_finite_number(value, label="value")
        self.indicate = indicate
        self.features = None if names is None else list(names)
        self._names = names
        self._last: dict[str, float] = {}
        self._count: dict[str, int] = {}
        self._mean: dict[str, float] = {}

    @property
    def readiness(self) -> Readiness:
        """Ready once the schema is known and, unless constant, every feature
        has been observed."""
        if self._names is None:
            return Readiness(False, 0, 1)
        if self.strategy == "constant":
            return Readiness(True, 1, 1)
        observed = min(self._count.get(name, 0) for name in self._names)
        return Readiness(observed >= 1, min(observed, 1), 1)

    def _split(
        self, x: Mapping[str, object]
    ) -> tuple[tuple[str, ...], dict[str, float]]:
        names = self._names if self._names is not None else tuple(x)
        extra = [key for key in x if key not in names]
        if extra:
            raise ValidationError(f"Unexpected features: {sorted(extra)}")
        observed: dict[str, float] = {}
        for name in names:
            raw = x.get(name)
            if raw is None:
                continue
            try:
                number = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError) as error:
                raise ValidationError(f"Feature '{name}' must be numeric") from error
            if math.isnan(number):
                continue
            observed[name] = coerce_event_number(number, label=f"Feature '{name}'")
        if self.indicate:
            clashes = [
                f"{name}_missing" for name in names if f"{name}_missing" in names
            ]
            if clashes:
                raise ValidationError(f"Features already exist: {clashes}")
        return names, observed

    def learn_one(self, x: dict[str, float]) -> None:
        """Learn the observed values; missing ones change nothing."""
        names, observed = self._split(x)
        self._names = names
        for name, number in observed.items():
            count = self._count.get(name, 0) + 1
            mean = self._mean.get(name, 0.0)
            self._count[name] = count
            self._mean[name] = mean + number / count - mean / count
            self._last[name] = number

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """The event with every missing value filled in, in schema order."""
        names, observed = self._split(x)
        result = {name: observed.get(name, self._fill(name)) for name in names}
        if self.indicate:
            for name in names:
                result[f"{name}_missing"] = 0.0 if name in observed else 1.0
        return result

    def _fill(self, name: str) -> float:
        if self.strategy == "last":
            return self._last.get(name, self.value)
        if self.strategy == "mean":
            return self._mean.get(name, self.value)
        return self.value

    @contextmanager
    def learning_transaction(self, x: dict[str, float]) -> Iterator[None]:
        """Restore the schema and learned values if a pipeline update fails."""
        saved = (self._names, dict(self._last), dict(self._count), dict(self._mean))
        try:
            yield
        except BaseException:
            self._names, self._last, self._count, self._mean = saved
            raise
