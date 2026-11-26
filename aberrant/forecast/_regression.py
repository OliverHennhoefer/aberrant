"""Shared machinery of the online regressors that standardize their inputs."""

from __future__ import annotations

import abc
import math
from typing import Any

import numpy as np

from aberrant._utils.validation import (
    FeatureSchema,
    coerce_event_number,
    coerce_integer_count,
)
from aberrant.base.forecast import Forecast
from aberrant.base.protocols import FeatureMap
from aberrant.base.readiness import Readiness
from aberrant.forecast._arithmetic import blend
from aberrant.forecast.base import BaseRegressor, check_spread_parameters


def seeded_rng(seed: object) -> np.random.Generator:
    """A generator from ``None`` or a non-negative integer seed."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(coerce_integer_count(seed, label="seed", minimum=0))


class ScaledRegressor(BaseRegressor, abc.ABC):
    """An online regressor of standardized covariates and target.

    The first ``warm_up`` events fix each covariate's and the target's mean
    and standard deviation; the model then learns them all, in order. Every
    later event's error is measured against the prediction made before
    learning it, and the faded mean absolute error of those errors is the
    spread. Subclasses learn in standardized units, return new attribute
    values from ``_learned`` and never change arrays in place, so a failed
    update leaves the model unchanged.
    """

    def _init_scaling(
        self, warm_up: object, residual_alpha: object, min_scale: object
    ) -> None:
        self.warm_up = coerce_integer_count(warm_up, label="warm_up")
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )

    def _reset_state(self) -> None:
        """Forget the scaling, the model and the spread."""
        self._schema = FeatureSchema()
        self.n_samples_seen = 0
        self._buffer: list[tuple[np.ndarray, float]] = []
        self._center: np.ndarray | None = None
        self._spread_x = np.ones(0)
        self._target = (0.0, 1.0)
        self._mae: float | None = None
        self._reset_model()

    @abc.abstractmethod
    def _reset_model(self) -> None:
        """Forget what the model learned."""

    @abc.abstractmethod
    def _start(self, features: int) -> None:
        """Set up a model of ``features`` standardized covariates."""

    @abc.abstractmethod
    def _predict(self, z: np.ndarray) -> float:
        """The standardized prediction for standardized covariates ``z``."""

    @abc.abstractmethod
    def _learned(self, z: np.ndarray, target: float) -> dict[str, Any]:
        """New attribute values after learning one standardized event."""

    @property
    def readiness(self) -> Readiness:
        """Ready once a spread has been measured, after ``warm_up + 1`` events."""
        return Readiness(self._mae is not None, self.n_samples_seen, self.warm_up + 1)

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        """Covariate names in model order, once learned."""
        return self._schema.names

    def _scaled(self, values: np.ndarray) -> np.ndarray:
        assert self._center is not None
        scaled: np.ndarray = (values - self._center) / self._spread_x
        return scaled

    def _unscaled(self, prediction: float) -> float:
        center, spread = self._target
        mean = center + spread * prediction
        if not math.isfinite(mean):
            raise OverflowError("Regression arithmetic exceeds the float range")
        return mean

    def predict_one(self, x: FeatureMap) -> Forecast | None:
        """Predict the target of the event with covariates ``x`` at lead 0."""
        prepared = self._schema.preview(x)
        if self._mae is None:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            mean = self._unscaled(self._predict(self._scaled(prepared.values)))
        return Forecast(
            origin=self.n_samples_seen,
            leads=(0,),
            mean=(mean,),
            scale=(max(self.min_scale, self._mae),),
        )

    def learn_one(self, x: FeatureMap, y: float) -> None:
        """Learn the target ``y`` of the event with covariates ``x``."""
        target = coerce_event_number(y, label="y")
        prepared = self._schema.preview(x)
        if self._center is None:
            self._warm(prepared.values, target)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                mean = self._unscaled(self._predict(self._scaled(prepared.values)))
                error = abs(target - mean)
                if not math.isfinite(error):
                    raise OverflowError("Regression arithmetic exceeds the float range")
                update = self._checked(
                    self._learned(self._scaled(prepared.values), self._standard(target))
                )
            weight = self.residual_alpha
            mae = error if self._mae is None else blend(self._mae, error, weight)
            self.__dict__.update(update)
            self._mae = mae
        self._schema.commit(prepared)
        self.n_samples_seen += 1

    def _standard(self, target: float) -> float:
        center, spread = self._target
        return (target - center) / spread

    def _warm(self, values: np.ndarray, target: float) -> None:
        buffer = [*self._buffer, (values, target)]
        if len(buffer) < self.warm_up:
            self._buffer = buffer
            return
        rows = np.array([row for row, _ in buffer])
        targets = np.array([value for _, value in buffer])
        saved = dict(self.__dict__)
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                self._center = rows.mean(axis=0)
                deviation = rows.std(axis=0)
                self._spread_x = np.where(deviation > 0, deviation, 1.0)
                spread = float(targets.std())
                self._target = (float(targets.mean()), spread if spread > 0 else 1.0)
                if not (
                    np.isfinite(self._center).all()
                    and np.isfinite(self._spread_x).all()
                    and all(math.isfinite(v) for v in self._target)
                ):
                    raise OverflowError("Regression arithmetic exceeds the float range")
                self._start(rows.shape[1])
                for row, value in buffer:
                    update = self._learned(self._scaled(row), self._standard(value))
                    self.__dict__.update(self._checked(update))
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(saved)
            raise
        self._buffer = []

    @staticmethod
    def _checked(update: dict[str, Any]) -> dict[str, Any]:
        for value in update.values():
            if isinstance(value, np.ndarray) and value.dtype.kind == "f":
                finite = bool(np.isfinite(value).all())
            elif isinstance(value, float):
                finite = math.isfinite(value)
            else:
                continue
            if not finite:
                raise OverflowError("Regression arithmetic exceeds the float range")
        return update
