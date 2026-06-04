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
- `MondrianForest` remains a compatibility alias. The original supervised
  Mondrian Forest paper defines no anomaly score.
