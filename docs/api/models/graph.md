# Graph Models

Public objects from `aberrant.model.graph`:

- `AnoEdgeL`
- `ISCONNA`
- `MIDAS`
- `SignedGraphSketchDetector`

`AnoEdgeL` maintains the local dense submatrices from the authors' higher-order
count-min sketch implementation.

`ISCONNA` implements the authors' frequency, consecutive-width, and gap
pattern scores.

`MIDAS` is a bounded-memory microcluster detector for dynamic edge streams.

`SignedGraphSketchDetector` is a bounded-memory structural detector using
signed count sketches and online Euclidean centroids. `StreamSpot` remains a
compatibility alias; the variant does not reproduce the original
Hamming-sketch clustering workflow.

Notes:
- Expects source and destination node identifiers per sample.
- Supports optional explicit timestamp handling via `time_key`.
- Returns continuous non-negative anomaly scores.
