# Transform API

Transformers expose incremental learning and one-event mapping. Pipeline
learning applies their post-update transform; direct `transform_one` calls do
not learn.

## Preprocessing

::: aberrant.transform.preprocessing.FeatureSchemaGuard

::: aberrant.transform.preprocessing.MinMaxScaler

::: aberrant.transform.preprocessing.StandardScaler

::: aberrant.transform.preprocessing.RollingRobustScaler

## Projection

::: aberrant.transform.projection.IncrementalPCA

::: aberrant.transform.projection.RandomProjection
