"""One calibrated threshold over a score stream."""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import Any, Self

from aberrant._utils.validation import coerce_event_number, coerce_finite_number
from aberrant.base._parameters import checked_state, versioned_state
from aberrant.base.descriptors import invalid
from aberrant.calibration._registry import CALIBRATORS, CalibratorSpec, OnlineCalibrator
from aberrant.calibration._window import ScoreHistory

_DEFAULT_WINDOW = 100


class ScoreCalibrator:
    """A threshold on a stream of scores, learned online with delayed feedback.

    ``predict(key)`` issues the threshold for a prediction identified by
    ``key``; ``update(key, score)`` reports its score once known, which may be
    several steps later (a forecast's lead). Scores may be negative.

    Args:
        method: Identifier of a method in :data:`CALIBRATORS`.
        alpha: Target miscoverage, so ``1 - alpha`` of scores should fall at
            or below the threshold.
        window_size: Number of recent scores in the calibration window.
        **params: Method parameters; omitted ones take their defaults.

    Raises:
        ConfigurationError: The method is unknown, or a parameter is unknown
            or out of range. Its ``issues`` locate each failure.

    ``get_params()``, ``clone()``, ``reset()`` and ``repr`` work as on every
    component.

    Examples:
        ```python
        from aberrant.calibration import ScoreCalibrator

        calibrator = ScoreCalibrator("aci", alpha=0.1, window_size=100, gamma=0.01)
        for t, score in enumerate([0.2, 0.4, 0.1, 0.3]):
            threshold = calibrator.predict(t)
            calibrator.update(t, score)
        ```
    """

    def __init__(
        self,
        method: str,
        *,
        alpha: float,
        window_size: int = _DEFAULT_WINDOW,
        **params: object,
    ) -> None:
        spec = CALIBRATORS.get(method)
        if spec is None:
            known = ", ".join(sorted(CALIBRATORS))
            raise invalid(
                "unknown_component",
                ("method",),
                f"Unknown calibration method {method!r}; use one of {known}",
            )
        alpha = coerce_finite_number(alpha, label="alpha")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.method_id = method
        self.alpha = alpha
        self.params = spec.resolve(params)
        self.history = ScoreHistory(window_size)
        self.method: OnlineCalibrator = spec.factory(self.history, alpha, self.params)

    def __getstate__(self) -> dict[str, Any]:
        """The calibrator state, stamped with the aberrant version."""
        return versioned_state(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state saved by this aberrant version."""
        self.__dict__.update(checked_state(self, state))

    @property
    def spec(self) -> CalibratorSpec:
        """The method's description and parameters."""
        return CALIBRATORS[self.method_id]

    def predict(self, key: Hashable) -> float:
        """The threshold issued for ``key``.

        ``+inf`` before any score. Otherwise a method that extrapolates
        returns its finite threshold as it is; every other threshold is
        clipped to the largest absolute score seen, as the method compares
        its feedback against that value.
        """
        q = self.method.predict(key)
        if self.history.count == 0:
            return math.inf
        if self.spec.extrapolates and math.isfinite(q):
            return q
        return self.history.cap(q)

    def update(self, key: Hashable, score: float) -> None:
        """Learn the score of the prediction issued for ``key``."""
        score = coerce_event_number(score, label="score")
        self.method.update(key, score)
        self.history.push(score)

    def get_params(self) -> dict[str, Any]:
        """Return the method, ``alpha``, ``window_size`` and every method parameter."""
        return {
            "method": self.method_id,
            "alpha": self.alpha,
            "window_size": self.history.window_size,
            **self.params,
        }

    def clone(self) -> Self:
        """Return a new calibrator with the same parameters and no scores."""
        params = self.get_params()
        return type(self)(params.pop("method"), **params)

    def reset(self) -> None:
        """Forget every score, as if this calibrator were just built."""
        self.__dict__.update(self.clone().__dict__)

    def __repr__(self) -> str:
        """Show the method, ``alpha`` and every parameter not at its default."""
        defaults = {"window_size": _DEFAULT_WINDOW, **self.spec.defaults()}
        shown = "".join(
            f", {name}={value!r}"
            for name, value in self.get_params().items()
            if name in defaults and value != defaults[name]
        )
        return f"ScoreCalibrator({self.method_id!r}, alpha={self.alpha!r}{shown})"
