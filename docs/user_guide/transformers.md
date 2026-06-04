# Transformers

ABERRANT supports streaming transformers that compose with models using `|`.

## Preprocessing

```python
from aberrant.transform.preprocessing import MinMaxScaler, StandardScaler
```

## Projection

```python
from aberrant.transform.projection import IncrementalPCA, RandomProjection
```

## Pipeline example

```python
from aberrant.model.iforest import OnlineIsolationForest
from aberrant.transform.preprocessing import StandardScaler

pipeline = StandardScaler() | OnlineIsolationForest(window_size=512)

for x in stream_of_feature_dicts:
    score = pipeline.score_one(x)
    pipeline.learn_one(x)
```

## Notes

- Keep feature keys consistent across the stream.
- `HalfSpaceTrees` works best after min-max scaling to `[0, 1]`.
