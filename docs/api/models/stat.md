# Statistical Models API

The univariate models accept events with exactly one feature. The first learned
event fixes its name. Both return `0.0` until `warm_up` values have been
learned; `warm_up` defaults to `window_size`.

## Univariate

::: aberrant.model.stat.MovingZScore

::: aberrant.model.stat.MovingMAD

## Bivariate and multivariate

::: aberrant.model.stat.MovingCovariance

::: aberrant.model.stat.MovingCorrelationCoefficient

::: aberrant.model.stat.MovingMahalanobisDistance

## Tail probability

`OnlineECOD` and `OnlineCOPOD` share their constructor and memories; see the
[model guide](../../user_guide/models.md#online-ecod-and-copod) for how to
choose between them.

::: aberrant.model.stat.OnlineECOD

::: aberrant.model.stat.OnlineCOPOD
