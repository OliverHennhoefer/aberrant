# Statistical Models API

Univariate models score the candidate-induced change in one moving statistic.
Unless otherwise stated, `abs_diff=True` returns the magnitude and
`abs_diff=False` preserves direction.

All univariate models share these constructor parameters unless their API
signature adds another one:

| Parameter | Meaning |
| --- | --- |
| `window_size` | Maximum number of learned values retained. It must be positive. |
| `key` | Required feature name. If omitted, the first learned sample locks the model to its sole feature name. Every sample must contain exactly that one feature. |
| `abs_diff` | Return the absolute candidate-induced change when `True`; preserve its sign when `False`. |

`MovingQuantile.quantile` is a probability in the closed interval `[0, 1]` and
uses linear interpolation. `MovingGeometricAverage` accepts only positive
values when scoring and ignores non-positive values while learning. Its
`absoluteValues=False` mode computes the geometric mean of the retained values;
despite the inherited parameter name, `absoluteValues=True` instead computes
the geometric mean of successive ratios.

## Univariate

::: aberrant.model.stat.MovingAverage

::: aberrant.model.stat.MovingHarmonicAverage

::: aberrant.model.stat.MovingGeometricAverage

::: aberrant.model.stat.MovingMedian

::: aberrant.model.stat.MovingQuantile

::: aberrant.model.stat.MovingVariance

::: aberrant.model.stat.MovingInterquartileRange

::: aberrant.model.stat.MovingAverageAbsoluteDeviation

::: aberrant.model.stat.MovingKurtosis

::: aberrant.model.stat.MovingSkewness

## Bivariate and multivariate

::: aberrant.model.stat.MovingCovariance

::: aberrant.model.stat.MovingCorrelationCoefficient

::: aberrant.model.stat.MovingMahalanobisDistance
