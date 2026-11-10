"""Anomaly scores from the residuals of a forecaster or regressor."""

from __future__ import annotations

from collections import deque
from typing import Literal

from aberrant.base.forecast import Forecast
from aberrant.base.model import BaseModel
from aberrant.base.protocols import FeatureMap, ForecasterProtocol, RegressorProtocol
from aberrant.forecast.scores import nonconformity
from aberrant.utils.validation import coerce_finite_number, coerce_integer_count

_SCORES = ("absolute", "scaled")


class ResidualDetector(BaseModel):
    """Score each event by how far its target misses a forecast or prediction.

    The detector splits each event into the target ``x[target]`` and, for a
    regressor, the covariates (every other key). It scores before learning:

    - A regressor predicts the target from the event's covariates (lead 0).
    - A forecaster's score uses the forecast issued ``lead`` events earlier,
      so the target is compared with what was expected before it arrived.
      A forecaster learns only the target; other keys are ignored.

    Args:
        model: An online forecaster (``forecast_one()``) or regressor
            (``predict_one(x)``).
        target: Name of the target feature in every event.
        lead: Forecast lead to score, from 1 to the forecaster's horizon;
            defaults to 1. Must be 0, the default, for a regressor.
        score: ``"scaled"`` divides the absolute residual by the forecast's
            spread at that lead; ``"absolute"`` keeps target units.

    Notes:
        Scores are non-negative and higher is more anomalous. They are 0 until
        a forecast for the event exists. The detector learns every event, so
        the model adapts to anomalies too; a longer lead compares with a
        forecast issued before an anomaly began. Calibrate the scores, for
        example with :class:`~aberrant.calibration.ScoreCalibrator`.

    Examples:
        ```python
        from aberrant.forecast import HoltWinters, ResidualDetector

        detector = ResidualDetector(HoltWinters(season_length=24), "load")
        for x in stream:
            score = detector.score_one(x)
            detector.learn_one(x)
        ```
    """

    def __init__(
        self,
        model: ForecasterProtocol | RegressorProtocol,
        target: str,
        *,
        lead: int | None = None,
        score: Literal["absolute", "scaled"] = "scaled",
    ) -> None:
        if not isinstance(target, str) or not target:
            raise ValueError("target must be a non-empty string")
        if score not in _SCORES:
            raise ValueError(f"score must be one of {', '.join(_SCORES)}")
        if isinstance(model, ForecasterProtocol):
            horizon = coerce_integer_count(model.horizon, label="horizon")
            lead = 1 if lead is None else lead
            resolved = coerce_integer_count(lead, label="lead")
            if resolved > horizon:
                raise ValueError(f"lead must be at most the horizon, {horizon}")
            self._forecaster: ForecasterProtocol | None = model
            self._regressor: RegressorProtocol | None = None
        elif isinstance(model, RegressorProtocol):
            resolved = coerce_integer_count(
                0 if lead is None else lead, label="lead", minimum=0
            )
            if resolved != 0:
                raise ValueError("A regressor predicts lead 0 only")
            self._forecaster = None
            self._regressor = model
        else:
            raise TypeError("model must be a forecaster or a regressor")
        self.model = model
        self.target = target
        self.lead = resolved
        self.score = score
        self._issued: deque[Forecast | None] = deque(maxlen=max(resolved, 1))

    def _split(self, x: FeatureMap) -> tuple[FeatureMap, float]:
        if self.target not in x:
            raise ValueError(f"Event is missing the target {self.target!r}")
        y = coerce_finite_number(x[self.target], label=f"Feature '{self.target}'")
        covariates = {key: value for key, value in x.items() if key != self.target}
        return covariates, y

    def _forecast_for(self, covariates: FeatureMap) -> Forecast | None:
        if self._regressor is not None:
            return self._regressor.predict_one(covariates)
        if len(self._issued) < self.lead:
            return None
        return self._issued[0]

    def explain_one(self, x: FeatureMap) -> tuple[float, float | None, float | None]:
        """Return (score, expected target, signed residual) without learning.

        Before a forecast exists return ``(0.0, None, None)``. A positive
        residual means the target exceeds what was expected.
        """
        covariates, y = self._split(x)
        forecast = self._forecast_for(covariates)
        if forecast is None:
            return 0.0, None, None
        expected = forecast.point(self.lead)
        return nonconformity(forecast, self.lead, y, self.score), expected, y - expected

    def score_one(self, x: FeatureMap) -> float:
        """Return the nonconformity of ``x[target]``, or 0 without a forecast."""
        return self.explain_one(x)[0]

    def learn_one(self, x: FeatureMap) -> None:
        """Learn the event, then keep the forecaster's next forecast."""
        covariates, y = self._split(x)
        if self._regressor is not None:
            self._regressor.learn_one(covariates, y)
            return
        assert self._forecaster is not None
        self._forecaster.learn_one(y)
        self._issued.append(self._forecaster.forecast_one())

    def __repr__(self) -> str:
        return (
            f"ResidualDetector({self.model!r}, target={self.target!r}, "
            f"lead={self.lead}, score={self.score!r})"
        )
