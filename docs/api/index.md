# API Reference

The API reference is generated from the shipped type annotations and
docstrings. It is the authoritative source for constructor signatures, defaults,
public methods, properties, return types, and model-specific caveats. Use the
[user guide](../user_guide/index.md) for lifecycle and selection guidance.

## Public package surface

- [Base interfaces and protocols](base.md)
- [Built-in catalog and declarative configuration](catalog.md)
- [Anomaly models](models/index.md)
- [Transformers](transform.md)
- [Drift detectors](drift.md)
- [Dataset registry, cache, and streams](stream.md)

Objects exported from the documented package `__init__.py` files are public.
Private names beginning with `_` are implementation details. Optional
PyTorch objects are public only through their explicitly documented import
paths and are excluded from wildcard exports when the dependency is absent.

## Compatibility policy

Starting with 1.0.0, ABERRANT follows Semantic Versioning for the documented
public API. Within 1.x, existing public import paths, constructor parameters,
methods, properties, return types, and documented behavior remain compatible.
Minor releases may add compatible capabilities; incompatible public API changes
require a new major version.

Private modules, names beginning with `_`, and undocumented internal attributes
are excluded from this guarantee. Experimental model descriptions qualify
algorithm maturity; they do not exempt documented public interfaces from the
compatibility policy.

Bug fixes may correct numeric results, including score values and seeded score
sequences, to match documented semantics. Exact floating-point results or random
sequences across package, dependency, and platform versions are not guaranteed.
Review the changelog, revalidate score thresholds, and run a chronological canary
stream when upgrading.

Model object layout and pickle/joblib checkpoints are not a stable interchange
format. Cross-version serialization compatibility is not promised; see
[persistence and upgrades](../user_guide/best_practices.md#persistence-and-upgrades).

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
