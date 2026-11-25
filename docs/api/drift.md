# Drift API

Drift detectors consume one finite scalar with `update`, which returns whether
it signalled drift. The `drift_detected` flag repeats that answer until the
next update.

::: aberrant.drift.BaseDriftDetector

::: aberrant.drift.ADWIN

::: aberrant.drift.KSWIN

::: aberrant.drift.PageHinkley
