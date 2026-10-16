# Transform API

Transformers expose incremental learning and one-event mapping. Pipeline
learning applies their post-update transform; direct `transform_one` calls do
not learn.

## Preprocessing

::: aberrant.transform.preprocessing.FeatureSchemaGuard

::: aberrant.transform.preprocessing.MinMaxScaler

::: aberrant.transform.preprocessing.StandardScaler

::: aberrant.transform.preprocessing.RollingRobustScaler

::: aberrant.transform.preprocessing.Select

::: aberrant.transform.preprocessing.Rename

::: aberrant.transform.preprocessing.Passthrough

## Projection

::: aberrant.transform.projection.IncrementalPCA

::: aberrant.transform.projection.RandomProjection
