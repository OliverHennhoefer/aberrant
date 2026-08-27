# Isolation Forest Models

Public objects from `aberrant.model.iforest`:

- `ASDIsolationForest`
- `HalfSpaceTrees`
- `MondrianIsolationForest`
- `OnlineIsolationForest`
- `RandomCutForest`
- `StreamRandomHistogramForest`
- `XStream`

Notes:
- `MondrianIsolationForest(lambda_=...)` uses `lambda_` as the Mondrian lifetime
  budget and Isolation Forest path-length scoring.
- The original supervised Mondrian Forest paper defines no anomaly score.
- `OnlineIsolationForest(tree_type=...)` accepts `"fixed"` or `"adaptive"`.
  `n_jobs=1` is sequential and `n_jobs=-1` uses all available logical CPUs.
