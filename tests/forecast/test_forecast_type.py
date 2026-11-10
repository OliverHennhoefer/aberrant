"""The Forecast value: validation, lookups and quantile interpolation."""

import dataclasses
import math

import pytest

from aberrant.base import Forecast, ForecasterProtocol, RegressorProtocol
from aberrant.forecast import HoltWinters, RLSRegressor, SeasonalNaive


def test_lookups_by_lead() -> None:
    forecast = Forecast(origin=5, leads=(1, 2), mean=(1.0, 2), scale=(0.5, 1.0))
    assert forecast.horizon == 2
    assert forecast.point(2) == 2.0
    assert isinstance(forecast.mean[1], float)
    assert forecast.spread(1) == 0.5
    assert Forecast(origin=0, leads=(0,), mean=(3.0,)).spread(0) is None
    with pytest.raises(KeyError):
        forecast.point(3)


def test_is_immutable() -> None:
    forecast = Forecast(origin=0, leads=(1,), mean=(1.0,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        forecast.origin = 1  # type: ignore[misc]


def test_quantile_interpolates_on_the_grid() -> None:
    forecast = Forecast(
        origin=0,
        leads=(1,),
        mean=(0.0,),
        levels=(0.1, 0.5, 0.9),
        quantiles=((-2.0, 0.0, 4.0),),
    )
    assert forecast.quantile(1, 0.1) == -2.0
    assert forecast.quantile(1, 0.3) == pytest.approx(-1.0)
    assert forecast.quantile(1, 0.7) == pytest.approx(2.0)
    assert forecast.quantile(1, 0.9) == 4.0
    with pytest.raises(ValueError, match="grid"):
        forecast.quantile(1, 0.95)
    with pytest.raises(ValueError, match="no quantiles"):
        Forecast(origin=0, leads=(1,), mean=(0.0,)).quantile(1, 0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"origin": -1},
        {"origin": True},
        {"origin": 1.0},
        {"leads": ()},
        {"leads": (2, 1)},
        {"leads": (1, 1)},
        {"leads": (-1,)},
        {"leads": [1]},
        {"mean": (1.0, 2.0)},
        {"mean": (math.nan,)},
        {"mean": (True,)},
        {"scale": (0.0,)},
        {"scale": (math.inf,)},
        {"scale": (1.0, 1.0)},
        {"levels": (0.5,), "quantiles": ()},
        {"levels": (0.0,), "quantiles": ((1.0,),)},
        {"levels": (0.6, 0.4), "quantiles": ((1.0, 2.0),)},
        {"levels": (0.4, 0.6), "quantiles": ((2.0, 1.0),)},
        {"levels": (0.4, 0.6), "quantiles": ((1.0,),)},
        {"quantiles": ((1.0,),)},
    ],
)
def test_rejects_inconsistent_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {"origin": 0, "leads": (1,), "mean": (1.0,)}
    values.update(kwargs)
    with pytest.raises(ValueError):
        Forecast(**values)  # type: ignore[arg-type]


def test_protocols_tell_forecasters_from_regressors() -> None:
    for forecaster in (SeasonalNaive(), HoltWinters(2)):
        assert isinstance(forecaster, ForecasterProtocol)
        assert not isinstance(forecaster, RegressorProtocol)
    assert isinstance(RLSRegressor(), RegressorProtocol)
    assert not isinstance(RLSRegressor(), ForecasterProtocol)
