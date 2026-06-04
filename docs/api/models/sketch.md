# Sketch Models

Public objects from `aberrant.model.sketch`:

- `MStream`
- `StreamingLODA`
- `StreamingRSHash`

`StreamingLODA` is a bounded-memory random-projection histogram detector.

`MStream` is a bounded-memory streaming detector based on fixed-size sketches.

`StreamingRSHash` is a bounded-memory randomized subspace hashing detector.

`LODA` and `RSHash` remain compatibility aliases for these bounded streaming
adaptations; they do not imply paper-procedure or score parity.

Notes:
- Supports optional explicit timestamp handling via `time_key`.
- Returns continuous non-negative anomaly scores.
