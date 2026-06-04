# Time-Series Models

Public objects from `aberrant.model.timeseries`:

- `XLagDAMP`

`XLagDAMP` implements the original authors' X-Lag Amnesic DAMP backward
processing with `lookahead=0`, their pure-online mode. It scores the
subsequence ending at each scalar event by its nearest preceding subsequence.

Notes:
- Input must contain exactly one consistently named numeric feature.
- `subsequence_length` should be roughly 50% to 90% of a typical period.
- Memory is bounded by `x_lag + subsequence_length - 1` learned values.
- Early-abandoned scores are approximate; the highest peaks identify discord
  candidates.
- Constant subsequences are unsupported by the reference algorithm and rejected.
