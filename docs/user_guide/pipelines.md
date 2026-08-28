# Pipelines

A pipeline is an ordered sequence of one or more transformers followed by zero
or one terminal anomaly model. `|` composes ABERRANT transformers left to right.

## Valid shapes

| Shape | Available operation |
| --- | --- |
| `transformer` | `learn_one`, `transform_one` |
| `transformer | transformer | ...` | `learn_one`, `transform_one` |
| `transformer | ... | model` | `learn_one`, `score_one` |

A model is terminal. Appending a transformer or another model after it raises
`IncompatibleComponentError`. Calling `score_one` on a transformer-ending
pipeline or `transform_one` on a model-ending pipeline raises `PipelineError`.

## Learning uses post-update transforms

`Pipeline.learn_one(event)` processes each transformer stage as follows:

1. Call the transformer's `learn_one` with its current input.
2. Call `transform_one` on the same input using the updated transformer state.
3. Pass that output to the next stage.
4. After the final transform, call the terminal model's `learn_one`, if present.

This is a deliberate post-update contract. In contrast,
`Pipeline.score_one(event)` only transforms and scores; it does not call any
component's `learn_one`. It therefore evaluates against transformer and model
state learned from earlier events.

!!! important "Prequential order"

    When an evaluated event must not influence its own representation or
    reference model, call `pipeline.score_one(event)` before
    `pipeline.learn_one(event)`.

## A model-ending pipeline

The first event is used only to establish scaler and model state. Every later
event is scored before it is learned.

```python
from aberrant.model.iforest import OnlineIsolationForest
from aberrant.transform.preprocessing import StandardScaler
from aberrant.transform.projection import RandomProjection

pipeline = (
    StandardScaler()
    | RandomProjection(n_components=2, seed=17)
    | OnlineIsolationForest(
        num_trees=10,
        window_size=32,
        seed=17,
    )
)

events = [
    {"a": 0.0, "b": 0.1, "c": -0.1},
    {"a": 0.2, "b": 0.0, "c": 0.1},
    {"a": -0.1, "b": 0.2, "c": 0.0},
    {"a": 5.0, "b": 5.0, "c": 5.0},
]

for index, event in enumerate(events):
    if index > 0:
        print(f"event={index}, score={pipeline.score_one(event):.3f}")
    pipeline.learn_one(event)
```

## A transformer-ending pipeline

The same composition mechanism can expose transformed features without a
model:

```python
from aberrant.transform.preprocessing import StandardScaler
from aberrant.transform.projection import RandomProjection

transformer = StandardScaler() | RandomProjection(
    n_components=2,
    keys=["x", "y", "z"],
    seed=23,
)

for event in [
    {"x": 1.0, "y": 0.0, "z": 0.0},
    {"x": 0.0, "y": 1.0, "z": 0.0},
    {"x": 0.0, "y": 0.0, "z": 1.0},
]:
    transformer.learn_one(event)

print(transformer.transform_one({"x": 1.0, "y": 1.0, "z": 1.0}))
```

## Custom structural components

Pipeline compatibility is structural. A custom component does not have to
inherit `BaseTransformer` or `BaseModel`; it must satisfy exactly one of these
runtime-checkable protocols:

- `TransformerProtocol`: callable `learn_one` and `transform_one` methods;
- `ModelProtocol`: callable `learn_one` and `score_one` methods.

An object implementing both shapes is ambiguous and rejected. Plain structural
objects do not inherit the `|` operator, so construct `Pipeline` explicitly:

```python
from aberrant.base import Pipeline


class SelectCoordinates:
    def learn_one(self, x: dict[str, float]) -> None:
        return None

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        return {"x": float(x["x"]), "y": float(x["y"])}


class L1Magnitude:
    def learn_one(self, x: dict[str, float]) -> None:
        return None

    def score_one(self, x: dict[str, float]) -> float:
        return abs(x["x"]) + abs(x["y"])


detector = Pipeline(SelectCoordinates(), L1Magnitude())
event = {"x": -2.0, "y": 3.0, "ignored": 99.0}

assert detector.score_one(event) == 5.0
detector.learn_one(event)
print(detector)
```

Subclass `BaseTransformer` when a custom transformer should inherit `|`.
Subclassing remains optional for explicitly constructed pipelines.

## Keep score policy outside the pipeline

`ThresholdModel` and `QuantileThreshold` are models, not transformers. They
cannot be appended after another model. Score the detector, pass the resulting
scalar in a new mapping to the threshold model, and update each component in
the intended order. This keeps model learning separate from alert-policy
changes and makes threshold contamination decisions explicit.

See [Base API](../api/base.md) for the protocols, overloads, and pipeline
exceptions.
