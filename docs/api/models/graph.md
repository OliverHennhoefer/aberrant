# Graph Models

Public objects from `aberrant.model.graph`:

- `AnoEdgeL`
- `ISCONNA`
- `MIDAS`
- `StreamSpot`

`AnoEdgeL` maintains the local dense submatrices from the authors' higher-order
count-min sketch implementation.

`ISCONNA` implements the authors' frequency, consecutive-width, and gap
pattern scores.

`MIDAS` is a bounded-memory microcluster detector for dynamic edge streams.

`StreamSpot` is a bounded-memory structural detector for per-graph edge streams.

Notes:
- Expects source and destination node identifiers per sample.
- Supports optional explicit timestamp handling via `time_key`.
- Returns continuous non-negative anomaly scores.
