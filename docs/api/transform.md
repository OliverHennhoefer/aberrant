# Transform API

Transformers expose incremental learning and one-event mapping. Pipeline
learning applies their post-update transform; direct `transform_one` calls do
not learn.

## Preprocessing

::: aberrant.transform.FeatureSchemaGuard

::: aberrant.transform.Imputer

::: aberrant.transform.OneHotEncoder

::: aberrant.transform.FrequencyEncoder

::: aberrant.transform.SignedLog

::: aberrant.transform.CalendarFeatures

::: aberrant.transform.MinMaxScaler

::: aberrant.transform.StandardScaler

::: aberrant.transform.RollingRobustScaler

::: aberrant.transform.Select

::: aberrant.transform.Rename

::: aberrant.transform.Passthrough

## Sequences

::: aberrant.transform.Lags

::: aberrant.transform.Difference

::: aberrant.transform.ExponentialSmoother

::: aberrant.transform.Despike

::: aberrant.transform.RollingStatistic

::: aberrant.transform.Deseasonalize

## Projection

::: aberrant.transform.IncrementalPCA

::: aberrant.transform.RandomProjection
