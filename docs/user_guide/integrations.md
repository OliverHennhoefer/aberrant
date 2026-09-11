# Application Integration

ABERRANT exposes its built-in components through a typed, immutable catalog.
Applications can discover supported models, generate configuration forms,
validate deployment configuration, and build pipelines without maintaining
class import paths or model-specific warm-up formulas.

## Inspect the catalog

`MODEL_CATALOG`, `TRANSFORMER_CATALOG`, and `SIMILARITY_ENGINE_CATALOG` are
read-only mappings keyed by stable component identifiers. Each entry exposes
its public import path, family, optional extra, availability in the current
environment, constructor parameter schema, and operational capabilities.
Identifiers and the versioned configuration shape are part of the public API;
new catalog entries may be added compatibly within 1.x.

```python
from aberrant.catalog import get_model_spec

spec = get_model_spec("random_cut_forest")
parameter_schema = spec.parameter_schema()
capabilities = spec.capabilities(
    {
        "sample_size": 256,
        "shingle_size": 4,
        "warmup_samples": 128,
        "normalize_score": True,
    }
)

assert capabilities.warmup.minimum == 131
assert capabilities.score_kind.value == "zero_to_one"
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
`spec.parameter_schema()` or `spec.capabilities(params)` when needed; these
calls may load its optional runtime.

Catalog warm-up is the minimum history required by the algorithm's scoring
contract. It is not a recommendation for threshold calibration. A production
application can require a longer calibration period without replacing the
model-specific metadata.

Always supply counts in `warmup.unit` to `remaining()` and `is_satisfied()`.
For SDOStream the unit is `observers`: use `model.n_observers`, since observer
insertion is sampled and learned event counts cannot determine readiness.
Its minimum accounts for both `warm_up_observers` and `x_neighbors`.

`feature_schema` distinguishes fixed-key models from feature-evolving or
engine-defined behavior. `requires_unit_interval` identifies hard input-domain
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
        "version": 1,
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
            "params": {"num_trees": 25, "window_size": 512, "seed": 17},
        },
    }
)

detector = config.build()
minimum_model_warmup = config.capabilities().warmup.minimum
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
from aberrant.model.iforest import OnlineIsolationForest
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

## Configure KNN and FAISS

FAISS has a stable public import at `aberrant.similarity`. Declarative KNN
configuration accepts a nested similarity-engine component:

```python
from aberrant.catalog import create_model

knn = create_model(
    "knn",
    {
        "k": 5,
        "similarity_engine": {
            "id": "faiss",
            "params": {"window_size": 1000, "warm_up": 20},
        },
    },
)
```

This requires `aberrant[faiss]`. The catalog reports the engine as unavailable
when that extra is not installed and raises `MissingOptionalDependencyError`
when construction is attempted.

Some macOS PyTorch and `faiss-cpu` wheel combinations bundle separate OpenMP
runtimes and abort when both are activated in one process. ABERRANT does not set
the unsafe `KMP_DUPLICATE_LIB_OK` workaround. Use a compatible native package
set or run FAISS-backed and PyTorch-backed detectors in separate processes.

## Ownership boundary

The catalog owns facts about ABERRANT components and their construction. The
application still owns message decoding, tenant or sensor routing, calibration
policy, resource budgets, model-instance concurrency, checkpoint I/O and
signing, alert delivery, circuit breaking, and health endpoints.
