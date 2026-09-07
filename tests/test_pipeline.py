"""Behavioral contracts for typed online pipelines."""

import pickle

import pytest

from aberrant.base import (
    BaseModel,
    BaseTransformer,
    IncompatibleComponentError,
    ModelPipeline,
    ModelProtocol,
    Pipeline,
    TransformerPipeline,
    TransformerProtocol,
)


class _StatefulTransformer(BaseTransformer):
    def __init__(self) -> None:
        self.state = 0.0
        self.learned: list[float] = []
        self.transformed: list[tuple[float, float]] = []

    def learn_one(self, x: dict[str, float]) -> None:
        self.learned.append(x["x"])
        self.state += 1.0

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        self.transformed.append((x["x"], self.state))
        return {"x": x["x"] + self.state}


class _RecordingModel(BaseModel):
    def __init__(self) -> None:
        self.learned: list[float] = []

    def learn_one(self, x: dict[str, float]) -> None:
        self.learned.append(x["x"])

    def score_one(self, x: dict[str, float]) -> float:
        return x["x"]


def test_protocols_accept_structural_components() -> None:
    assert isinstance(_StatefulTransformer(), TransformerProtocol)
    assert isinstance(_RecordingModel(), ModelProtocol)


def test_parenthesized_pipeline_composition_is_flattened() -> None:
    first = _StatefulTransformer()
    second = _StatefulTransformer()
    third = _StatefulTransformer()
    pipeline = first | (second | third)

    pipeline.learn_one({"x": 1.0})

    assert pipeline.transform_one({"x": 1.0}) == {"x": 4.0}
    assert str(pipeline) == (
        "_StatefulTransformer | _StatefulTransformer | _StatefulTransformer"
    )


def test_learning_uses_each_transformers_post_update_state() -> None:
    first = _StatefulTransformer()
    second = _StatefulTransformer()
    model = _RecordingModel()
    pipeline = first | second | model

    pipeline.learn_one({"x": 1.0})

    assert first.learned == [1.0]
    assert first.transformed == [(1.0, 1.0)]
    assert second.learned == [2.0]
    assert second.transformed == [(2.0, 1.0)]
    assert model.learned == [3.0]


def test_scoring_does_not_update_transformers() -> None:
    first = _StatefulTransformer()
    second = _StatefulTransformer()
    model = _RecordingModel()
    pipeline = first | second | model
    pipeline.learn_one({"x": 1.0})

    assert pipeline.score_one({"x": 4.0}) == 6.0
    assert first.learned == [1.0]
    assert second.learned == [2.0]


def test_model_ending_pipeline_cannot_be_extended() -> None:
    pipeline = _StatefulTransformer() | _RecordingModel()

    with pytest.raises(IncompatibleComponentError):
        pipeline | _StatefulTransformer()  # type: ignore[misc]


def test_pipeline_rejects_model_as_first_component() -> None:
    with pytest.raises(IncompatibleComponentError):
        Pipeline(_RecordingModel(), _StatefulTransformer())  # type: ignore[arg-type]


def test_pipeline_protocols_match_available_capabilities() -> None:
    transformer_pipeline = _StatefulTransformer() | _StatefulTransformer()
    model_pipeline = _StatefulTransformer() | _RecordingModel()

    assert isinstance(transformer_pipeline, Pipeline)
    assert isinstance(transformer_pipeline, TransformerProtocol)
    assert not isinstance(transformer_pipeline, ModelProtocol)
    assert not hasattr(transformer_pipeline, "score_one")
    assert isinstance(model_pipeline, Pipeline)
    assert isinstance(model_pipeline, ModelProtocol)
    assert not isinstance(model_pipeline, TransformerProtocol)
    assert not hasattr(model_pipeline, "transform_one")


def test_parenthesized_model_pipeline_preserves_learning_and_pickling() -> None:
    first, second, model = (
        _StatefulTransformer(),
        _StatefulTransformer(),
        _RecordingModel(),
    )
    pipeline = Pipeline(first, Pipeline(second, model))
    pipeline.learn_one({"x": 1.0})
    assert model.learned == [3.0]
    assert pipeline.score_one({"x": 2.0}) == 4.0
    restored = pickle.loads(pickle.dumps(pipeline))
    assert restored.score_one({"x": 2.0}) == 4.0
    assert isinstance(restored, ModelProtocol)
    assert not isinstance(restored, TransformerProtocol)


def test_transformer_terminal_is_not_transformed_during_learning() -> None:
    first, second = _StatefulTransformer(), _StatefulTransformer()
    pipeline = Pipeline(first, second)
    pipeline.learn_one({"x": 1.0})
    assert second.learned == [2.0]
    assert second.transformed == []


def test_pipeline_rejects_ambiguous_structural_components() -> None:
    class Ambiguous(_StatefulTransformer):
        def score_one(self, x: dict[str, float]) -> float:
            return x["x"]

    with pytest.raises(IncompatibleComponentError):
        Pipeline(_StatefulTransformer(), Ambiguous())
    with pytest.raises(IncompatibleComponentError):
        Pipeline(Ambiguous(), _RecordingModel())


@pytest.mark.parametrize(
    ("pipeline_type", "terminal_type"),
    [
        (ModelPipeline, _StatefulTransformer),
        (TransformerPipeline, _RecordingModel),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_concrete_pipeline_rejects_incompatible_terminal(
    pipeline_type, terminal_type, nested
) -> None:
    terminal = terminal_type()
    if nested:
        terminal = Pipeline(_StatefulTransformer(), terminal)
    with pytest.raises(IncompatibleComponentError, match=pipeline_type.__name__):
        pipeline_type(_StatefulTransformer(), terminal)


@pytest.mark.parametrize(
    ("pipeline_type", "terminal_type", "method"),
    [
        (ModelPipeline, _RecordingModel, "score_one"),
        (TransformerPipeline, _StatefulTransformer, "transform_one"),
    ],
)
def test_concrete_pipeline_preserves_type_behavior_and_pickling(
    pipeline_type, terminal_type, method
) -> None:
    pipeline = pipeline_type(_StatefulTransformer(), terminal_type())
    assert type(pipeline) is pipeline_type
    pipeline.learn_one({"x": 1.0})
    result = getattr(pipeline, method)({"x": 2.0})
    assert result == (3.0 if method == "score_one" else {"x": 4.0})
    restored = pickle.loads(pickle.dumps(pipeline))
    assert type(restored) is pipeline_type
    assert getattr(restored, method)({"x": 2.0}) == result


@pytest.mark.parametrize("base_type", [ModelPipeline, TransformerPipeline])
def test_concrete_pipeline_subclasses_preserve_their_type(base_type) -> None:
    class CustomPipeline(base_type):
        pass

    terminal = (
        _RecordingModel() if base_type is ModelPipeline else _StatefulTransformer()
    )
    pipeline = CustomPipeline(_StatefulTransformer(), terminal)
    assert type(pipeline) is CustomPipeline
