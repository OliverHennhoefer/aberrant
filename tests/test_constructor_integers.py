"""Every count a constructor takes must be a true integer."""

import importlib
import inspect

import pytest

from aberrant.catalog import MODEL_CATALOG, TRANSFORMER_CATALOG
from aberrant.drift import ADWIN, CUSUM, KSWIN, PageHinkley
from aberrant.forecast import (
    AutoRegressive,
    HoltWinters,
    KalmanForecaster,
    KNNRegressor,
    PassiveAggressiveRegressor,
    RandomFourierRegressor,
    SeasonalNaive,
    SGDRegressor,
)


def _classes() -> list[type]:
    classes: list[type] = [
        ADWIN,
        CUSUM,
        KSWIN,
        PageHinkley,
        AutoRegressive,
        KNNRegressor,
        PassiveAggressiveRegressor,
        RandomFourierRegressor,
        SGDRegressor,
        HoltWinters,
        KalmanForecaster,
        SeasonalNaive,
    ]
    for catalog in (MODEL_CATALOG, TRANSFORMER_CATALOG):
        for spec in catalog.values():
            if not spec.available:
                continue
            module, name = spec.import_path.rsplit(".", 1)
            classes.append(getattr(importlib.import_module(module), name))
    return classes


# Arguments a class needs before it can reject anything else.
_REQUIRED: dict[str, dict[str, object]] = {
    "HoltWinters": {"season_length": 2},
    "IncrementalPCA": {"n_components": 1},
}

_INTEGER_PARAMETERS = [
    pytest.param(cls, parameter.name, id=f"{cls.__name__}.{parameter.name}")
    for cls in _classes()
    for parameter in inspect.signature(cls.__init__).parameters.values()
    if parameter.name != "seed" and type(parameter.default) is int
]


@pytest.mark.parametrize(("cls", "name"), _INTEGER_PARAMETERS)
@pytest.mark.parametrize("value", [2.5, True])
def test_integer_parameters_reject_fractions_and_booleans(cls, name, value) -> None:
    with pytest.raises(ValueError):
        cls(**{**_REQUIRED.get(cls.__name__, {}), name: value})
