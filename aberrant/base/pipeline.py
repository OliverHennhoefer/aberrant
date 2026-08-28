"""Pipeline for chaining transformers and models together."""

from typing import Generic, TypeVar, overload

from .exceptions import IncompatibleComponentError, PipelineError
from .protocols import FeatureMap, ModelProtocol, TransformerProtocol

_Terminal = TypeVar("_Terminal", TransformerProtocol, ModelProtocol)


class Pipeline(Generic[_Terminal]):
    """Chain one or more transformers and an optional terminal model.

    ``learn_one`` uses post-update transformations: each transformer first
    learns from its current input and then transforms that input for the next
    stage. ``score_one`` and ``transform_one`` call only the transformers'
    ``transform_one`` methods; the pipeline does not invoke learning methods in
    either path.

    The first component must be a transformer. The second component may be
    another transformer or a terminal model. A pipeline ending in a model
    cannot be extended further.

    Args:
        first: A transformer or a transformer-ending pipeline.
        second: A transformer or terminal anomaly model.

    Examples:
        ```python
        from aberrant.base import Pipeline
        from aberrant.model import RandomModel
        from aberrant.transform.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        model = RandomModel()
        pipeline = Pipeline(scaler, model)
        pipeline.learn_one({"feature": 1.0})
        score = pipeline.score_one({"feature": 2.0})
        ```
    """

    def __init__(self, first: TransformerProtocol, second: _Terminal) -> None:
        self._validate_first(first)
        terminal_kind = self._component_kind(second)

        self.first: TransformerProtocol = first
        self.second: _Terminal = second

        self._transformers: tuple[TransformerProtocol, ...]
        if isinstance(first, Pipeline):
            self._transformers = first._transformers
        else:
            self._transformers = (first,)

        self._model: ModelProtocol | None = None
        if isinstance(second, Pipeline):
            self._transformers += second._transformers
            self._model = second._model
        elif terminal_kind == "transformer":
            assert isinstance(second, TransformerProtocol)
            self._transformers += (second,)
        else:
            assert isinstance(second, ModelProtocol)
            self._model = second

    @property
    def ends_in_transformer(self) -> bool:
        """Whether this pipeline can transform output and accept another stage."""
        return self._model is None

    def learn_one(self, x: FeatureMap) -> None:
        """Learn from one sample using each transformer's updated state."""
        current = x
        final_transformer_index = len(self._transformers) - 1

        for index, transformer in enumerate(self._transformers):
            transformer.learn_one(current)
            needs_output = index < final_transformer_index or self._model is not None
            if needs_output:
                current = self._checked_transform(transformer, current)

        if self._model is not None:
            self._model.learn_one(current)

    def transform_one(
        self: "Pipeline[TransformerProtocol]", x: FeatureMap
    ) -> FeatureMap:
        """Transform one sample through a transformer-ending pipeline."""
        if self._model is not None:
            raise PipelineError(
                "transform_one is only available on a transformer-ending pipeline."
            )

        current = x
        for transformer in self._transformers:
            current = self._checked_transform(transformer, current)
        return current

    def score_one(self: "Pipeline[ModelProtocol]", x: FeatureMap) -> float:
        """Score one sample without invoking a pipeline learning method."""
        if self._model is None:
            raise PipelineError("score_one requires a model-ending pipeline.")

        current = x
        for transformer in self._transformers:
            current = self._checked_transform(transformer, current)

        score = self._model.score_one(current)
        if not isinstance(score, int | float):
            raise PipelineError(
                "The final component must return a numeric score from score_one."
            )
        return float(score)

    @overload
    def __or__(
        self: "Pipeline[TransformerProtocol]", other: TransformerProtocol
    ) -> "Pipeline[TransformerProtocol]": ...

    @overload
    def __or__(
        self: "Pipeline[TransformerProtocol]", other: ModelProtocol
    ) -> "Pipeline[ModelProtocol]": ...

    def __or__(
        self: "Pipeline[TransformerProtocol]",
        other: TransformerProtocol | ModelProtocol,
    ) -> "Pipeline[TransformerProtocol] | Pipeline[ModelProtocol]":
        """Append a transformer or terminal model to this pipeline."""
        if not self.ends_in_transformer:
            raise IncompatibleComponentError(
                self.second.__class__.__name__,
                "a transformer-ending pipeline before another component",
            )
        if self._component_kind(other) == "transformer":
            assert isinstance(other, TransformerProtocol)
            return Pipeline(self, other)
        assert isinstance(other, ModelProtocol)
        return Pipeline(self, other)

    def __repr__(self) -> str:
        """Return a string representation of the pipeline."""
        return f"Pipeline({self.first!r} | {self.second!r})"

    def __str__(self) -> str:
        """Return a human-readable string representation of the pipeline."""
        names = [component.__class__.__name__ for component in self._transformers]
        if self._model is not None:
            names.append(self._model.__class__.__name__)
        return " | ".join(names)

    @staticmethod
    def _checked_transform(
        transformer: TransformerProtocol, x: FeatureMap
    ) -> FeatureMap:
        transformed = transformer.transform_one(x)
        if not isinstance(transformed, dict):
            raise PipelineError(
                "Each transformer must return a dict[str, float] from transform_one."
            )
        return transformed

    @staticmethod
    def _validate_first(first: TransformerProtocol) -> None:
        if isinstance(first, Pipeline):
            if not first.ends_in_transformer:
                raise IncompatibleComponentError(
                    first.__class__.__name__, "a transformer-ending pipeline"
                )
            return

        if not isinstance(first, TransformerProtocol):
            raise IncompatibleComponentError(
                first.__class__.__name__,
                "component with callable 'learn_one' and 'transform_one' methods",
            )

    @staticmethod
    def _component_kind(
        component: TransformerProtocol | ModelProtocol,
    ) -> str:
        if isinstance(component, Pipeline):
            return "transformer" if component.ends_in_transformer else "model"

        is_transformer = isinstance(component, TransformerProtocol)
        is_model = isinstance(component, ModelProtocol)
        if is_transformer and not is_model:
            return "transformer"
        if is_model and not is_transformer:
            return "model"
        if is_transformer and is_model:
            raise IncompatibleComponentError(
                component.__class__.__name__,
                "an unambiguous transformer or model component",
            )
        raise IncompatibleComponentError(
            component.__class__.__name__,
            "component with callable 'learn_one' plus 'transform_one' or 'score_one'",
        )
