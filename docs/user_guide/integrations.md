# Application Integration

aberrant exposes its built-in components through a typed, immutable catalog.
Applications can discover supported models, generate configuration forms,
validate deployment configuration, and build pipelines without maintaining
class import paths or model-specific warm-up formulas.

## Reuse resolved configuration

Resolve once when initializing a service or evaluator. Resolution applies
constructor defaults, validates types and feature widths, and resolves
capabilities without constructing or learning any algorithms.

```python
from aberrant.catalog import DetectorConfig

resolved = DetectorConfig.from_mapping(
    {
        "transformers": [{"id": "standard_scaler"}],
        "model": {"id": "online_isolation_forest", "params": {"seed": 42}},
    }
).resolve()
canonical = resolved.config.as_dict()
capabilities = resolved.capabilities.as_dict()
fingerprint = resolved.fingerprint
first = resolved.build()
second = resolved.build()  # Independent components and learned state.
```

`resolve_detector_config(mapping_or_config)` is the equivalent module function.
Constructor inspection is shared within one resolution; parameters resolve once
per component occurrence. Builds reuse the construction plan, never a learned
model. Persist canonical configuration and learned detector state using the
application's checkpoint policy; keep resolved plans as process-local metadata.
Caller-supplied runtime dependencies retain their own ownership and lifetime.

`normalized()` and `fingerprint()` keep their narrower validation boundary:
they do not construct algorithms or require compatible pipeline feature counts.
Fingerprints use the existing canonical JSON representation and reject non-JSON
values, including Python runtime dependencies, even when those are buildable.
Algorithm constructor invariants are checked when building.

Configuration exceptions retain their messages and subclasses and expose an
immutable tuple of `ConfigurationIssue` records through `error.issues`. A record
contains a stable `code`, a tuple `loc` of object keys and array indexes, and a
`message`. A nested forecaster field can have location
`("model", "params", "model", "params", "season_length")`.
Independent parameter failures are collected. Constructor-only invariants use
the component's `params` location when a field cannot be identified reliably.
Use these locations directly in forms; do not parse human-readable messages.
The stable codes are documented on `aberrant.base.ConfigurationIssue`.

Public schemas and default parameters are detached containers. Applications may
constrain them for resource limits or presentation without modifying the library
catalog or changing executable fingerprints.

## Inspect the catalog

`MODEL_CATALOG`, `TRANSFORMER_CATALOG`, and `FORECASTER_CATALOG` are read-only mappings keyed by stable component identifiers. Each entry exposes
its public import path, family, optional extra, availability in the current
environment, constructor parameter schema, and operational capabilities.
Identifiers and the versioned configuration shape are part of the public API;
new catalog entries may be added compatibly within 2.x.

```python
from aberrant.catalog import get_model_spec, model_capabilities

parameter_schema = get_model_spec("random_cut_forest").parameter_schema()
capabilities = model_capabilities(
    "random_cut_forest",
    {
        "sample_size": 256,
        "shingle_size": 4,
        "warm_up": 128,
    },
)

assert capabilities.warmup == 131
assert capabilities.score_kind.value == "non_negative"
assert capabilities.state.value == "bounded"
```

`catalog_manifest()` returns a detached JSON-compatible description of all
entries. Parameter schemas describe JSON types, defaults, required fields, and
literal choices. Nested component parameters carry an
`x-aberrant-component-kind` schema extension. Constructors remain the authority
for numeric ranges and cross-parameter invariants. Applications should add their
own resource ceilings for values such as tree counts and window sizes.

Bulk discovery does not import optional runtimes. Optional components retain
availability metadata, but their `parameters` and model `capabilities` are
`None` in catalog snapshots. Inspect a selected component explicitly with
`spec.parameter_schema()` or `model_capabilities(component_id, params)` when
needed; these calls may load its optional runtime.

Catalog warm-up is the `required` count of a freshly built model's
`readiness`, in `warmup_unit`, so the catalog and the model cannot disagree. It
is `None` when the parameters are incomplete or the count depends on the data.
It is the minimum history required by the algorithm's scoring
contract. It is not a recommendation for threshold calibration. A production
application can require a longer calibration period without replacing the
model-specific metadata.

Always supply counts in `warmup.unit` to `remaining()` and `is_satisfied()`.
For SDOStream the unit is `observers`: use `model.n_observers`, since observer
insertion is sampled and learned event counts cannot determine readiness.
Its minimum accounts for both `warm_up` and `n_neighbors`.

`feature_schema` distinguishes fixed-key models from feature-evolving and
unrestricted ones. `requires_unit_interval` identifies hard input-domain
requirements such as Half-Space Trees; it does not imply that applications must
use a particular transformer when their source data already satisfies them.

## Build from configuration

The versioned detector format contains an ordered transformer list and one
terminal model. Only catalog identifiers are accepted; configuration cannot
select an arbitrary Python import path.

```python
from aberrant.catalog import DetectorConfig

config = DetectorConfig.from_mapping(
    {
        "version": 2,
        "transformers": [
            {
                "id": "feature_schema_guard",
                "params": {"features": ["temperature", "pressure", "flow"]},
            },
            {"id": "standard_scaler"},
            {
                "id": "random_projection",
                "params": {"n_components": 2, "seed": 17},
            },
        ],
        "model": {
            "id": "online_isolation_forest",
            "params": {"n_trees": 25, "window_size": 512, "seed": 17},
        },
    }
)

detector = config.build()
minimum_model_warmup = config.capabilities().warmup
config_fingerprint = config.fingerprint()
```

Parsing rejects unknown fields and unsupported format versions. Construction
rejects unknown component IDs, missing or extra constructor arguments, invalid
JSON types, model constructor validation failures, and statically detectable
feature-width mismatches. These failures derive from `ConfigurationError`, so
configuration faults can be separated from stream-processing failures.
Entries with `declarative: false` require runtime Python objects and remain
available for direct construction, but are intentionally rejected by
`DetectorConfig`.

`config.normalized()` validates parameter types and materializes constructor
defaults. The fingerprint is a SHA-256 digest of that canonical configuration,
so omitting a default and specifying it explicitly produce the same value. It
is useful checkpoint metadata, but it is not a serialization format for learned
model state and does not replace the package version in a checkpoint envelope.

## Enforce an event schema

Place `FeatureSchemaGuard` first when an application knows its input fields. It
rejects missing, additional, non-string, non-numeric, and non-finite features
with `ValidationError` before downstream transformers can update their state.

```python
from aberrant.model.forest import OnlineIsolationForest
from aberrant.transform import FeatureSchemaGuard, StandardScaler

detector = (
    FeatureSchemaGuard(features=["temperature", "pressure"])
    | StandardScaler()
    | OnlineIsolationForest(window_size=512, seed=7)
)
```

With no explicit `features`, the first successfully learned event establishes a
sorted schema. The guard does not remove message metadata or choose which fields
belong to a model; that remains an application boundary decision.

## Handle rejected events

Every model, transformer, drift detector and forecaster raises
`aberrant.exceptions.ValidationError` for an event it rejects: a missing, extra,
non-string, non-numeric or non-finite field, or an out-of-order timestamp. It
subclasses `ValueError`. Catch it to route bad events to a dead-letter queue
while letting configuration errors and faults in your own code stop the
service.

```python
from aberrant.exceptions import ValidationError
from aberrant.model.forest import HalfSpaceTrees

detector = HalfSpaceTrees(seed=7)
dead_letters = []
for event in [{"x": 0.2}, {"x": float("nan")}, {"x": 0.4}]:
    try:
        score = detector.score_one(event)
        detector.learn_one(event)
    except ValidationError as error:
        dead_letters.append((event, str(error)))

assert len(dead_letters) == 1
```

## Configure KNN and FAISS

`KNN` keeps its own window of recent events and computes exact distances with
NumPy. For wide or long windows, `backend="faiss"` searches a FAISS index
instead:

```python
from aberrant.catalog import create_model

knn = create_model("knn", {"k": 5, "window_size": 1000})
```

The FAISS backend requires `aberrant[faiss]`. Building `KNN` with
`backend="faiss"` without that extra raises `MissingOptionalDependencyError`.

Some macOS PyTorch and `faiss-cpu` wheel combinations bundle separate OpenMP
runtimes and abort when both are activated in one process. aberrant does not set
the unsafe `KMP_DUPLICATE_LIB_OK` workaround. Use a compatible native package
set or run FAISS-backed and PyTorch-backed detectors in separate processes.

## Configure a forecast-based detector

`residual_detector` scores a target against a forecast. Its `model` is a nested
forecaster from `FORECASTER_CATALOG`: `seasonal_naive`, `holt_winters`,
`kalman_forecaster`, `autoregressive`, or `rls_regressor`.

```python
from aberrant.catalog import DetectorConfig

config = DetectorConfig.from_mapping(
    {
        "model": {
            "id": "residual_detector",
            "params": {
                "target": "load",
                "model": {
                    "id": "holt_winters",
                    "params": {"season_length": 24, "horizon": 6},
                },
                "lead": 6,
            },
        }
    }
)
config.capabilities().warmup  # 53: two seasons, then the lead
detector = config.build()
```

A forecaster takes one feature, the target, and its warm-up is two seasons plus
`lead - 1` events. A regressor takes the target and at least one covariate; its
warm-up depends on how many covariates the events carry, so the catalog reports
no minimum. See [forecast-based detection](forecasting.md) for the models.

## Ownership boundary

The catalog owns facts about aberrant components and their construction. The
application still owns message decoding, tenant or sensor routing, calibration
policy, resource budgets, model-instance concurrency, checkpoint I/O and
signing, alert delivery, circuit breaking, and health endpoints.
