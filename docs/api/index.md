# API Reference

This section documents the supported ABERRANT public surface.

## Public modules

- `aberrant.base`
- `aberrant.drift`
- `aberrant.model`
- `aberrant.model.iforest`
- `aberrant.model.distance`
- `aberrant.model.sketch`
- `aberrant.model.graph`
- `aberrant.model.timeseries`
- `aberrant.model.svm`
- `aberrant.model.stat`
- `aberrant.model.deep`
- `aberrant.transform`
- `aberrant.transform.preprocessing`
- `aberrant.transform.projection`
- `aberrant.stream.dataset`

## Scoring conventions

- Binary: `ThresholdModel` and threshold wrappers.
- Bounded `[0, 1]`: isolation-forest variants.
- Continuous/unbounded: sketch, distance, SVM, statistical, and deep models.

## Stability policy

- Objects exported through package `__init__.py` files above are considered public.
- Internal helpers and private names (`_...`) can change between releases.
