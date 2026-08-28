# API Reference

The API reference is generated from the shipped type annotations and
docstrings. It is the authoritative source for constructor signatures, defaults,
public methods, properties, return types, and model-specific caveats. Use the
[user guide](../user_guide/index.md) for lifecycle and selection guidance.

## Public package surface

- [Base interfaces and protocols](base.md)
- [Anomaly models](models/index.md)
- [Transformers](transform.md)
- [Drift detectors](drift.md)
- [Dataset registry, cache, and streams](stream.md)

Objects exported from the documented package `__init__.py` files are public.
Private names beginning with `_` are implementation details. Optional
PyTorch objects are public only through their explicitly documented import
paths and are excluded from wildcard exports when the dependency is absent.

!!! note "Public does not mean frozen"

    ABERRANT is pre-1.0. Public APIs are deliberate and typed, but can still
    change between releases. Consult the changelog when upgrading.

## Shared model shape

Most anomaly models satisfy `ModelProtocol`:

- `learn_one(x) -> None` learns one feature mapping;
- `score_one(x) -> float` scores without learning the candidate event.

That shared method shape does not standardize input fields, warm-up, memory, or
numeric scale. In particular:

- isolation-family scores are not universally bounded because
  `RandomCutForest` defaults to raw CoDisp and
  `StreamRandomHistogramForest` returns a raw log-mass score;
- graph detectors default to raw scores but expose optional normalization;
- moving statistics can return signed changes when `abs_diff=False`;
- `RandomModel` is a generator-backed baseline rather than an anomaly model.

Read the [model score contracts](../user_guide/models.md#score-contracts)
before calibrating or comparing scores.
