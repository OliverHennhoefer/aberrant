"""Composition with distinct transformer and model capabilities."""

from __future__ import annotations

from typing import NoReturn, overload

from .exceptions import IncompatibleComponentError, PipelineError
from .protocols import FeatureMap, ModelProtocol, TransformerProtocol


class Pipeline:
    """Chain transformers with a transformer or model terminal.

    Construction returns a :class:`TransformerPipeline` or :class:`ModelPipeline`
    exposing only the terminal's capability. Learning uses post-update
    transformations; transformation and scoring never invoke learning methods.
    Nested pipelines are flattened into one prefix and terminal.
    """

    @overload
    def __new__(
        cls, first: TransformerProtocol, second: TransformerProtocol
    ) -> TransformerPipeline: ...

    @overload
    def __new__(
        cls, first: TransformerProtocol, second: ModelProtocol
    ) -> ModelPipeline: ...

    # Mypy rejects a union in __new__ even when both variants subclass Pipeline.
    def __new__(  # type: ignore[misc]
        cls, first: TransformerProtocol, second: TransformerProtocol | ModelProtocol
    ) -> TransformerPipeline | ModelPipeline:
        if not isinstance(first, TransformerProtocol) or isinstance(
            first, ModelProtocol
        ):
            raise IncompatibleComponentError(
                first.__class__.__name__, "an unambiguous transformer component"
            )
        is_transformer = isinstance(second, TransformerProtocol)
        is_model = isinstance(second, ModelProtocol)
        if is_transformer == is_model:
            raise IncompatibleComponentError(
                second.__class__.__name__,
                "an unambiguous transformer or model component",
            )
        pipeline: TransformerPipeline | ModelPipeline
        if cls is Pipeline:
            pipeline = (
                object.__new__(TransformerPipeline)
                if is_transformer
                else object.__new__(ModelPipeline)
            )
        elif (issubclass(cls, TransformerPipeline) and is_transformer) or (
            issubclass(cls, ModelPipeline) and is_model
        ):
            pipeline = object.__new__(cls)
        else:
            raise IncompatibleComponentError(
                second.__class__.__name__,
                f"a terminal compatible with {cls.__name__}",
            )
        pipeline._initialize(first, second)
        return pipeline

    def _initialize(
        self, first: TransformerProtocol, second: TransformerProtocol | ModelProtocol
    ) -> None:
        prefix = first.stages if isinstance(first, TransformerPipeline) else (first,)
        self._prefix: tuple[TransformerProtocol, ...]
        self._terminal: TransformerProtocol | ModelProtocol
        if isinstance(second, Pipeline):
            self._prefix = prefix + second._prefix
            self._terminal = second._terminal
        else:
            self._prefix = prefix
            self._terminal = second

    @property
    def first(self) -> TransformerProtocol:
        """Return the transformer prefix as a composable component."""
        first = self._prefix[0]
        for stage in self._prefix[1:]:
            first = Pipeline(first, stage)
        return first

    @property
    def second(self) -> TransformerProtocol | ModelProtocol:
        """Return the terminal component."""
        return self._terminal

    @property
    def ends_in_transformer(self) -> bool:
        """Whether this pipeline can transform output and accept another stage."""
        return isinstance(self, TransformerPipeline)

    def learn_one(self, x: FeatureMap) -> None:
        """Learn from one sample using each prefix transformer's updated state."""
        current = x
        for transformer in self._prefix:
            transformer.learn_one(current)
            current = self._checked_transform(transformer, current)
        self._terminal.learn_one(current)

    def _transform_prefix(self, x: FeatureMap) -> FeatureMap:
        for transformer in self._prefix:
            x = self._checked_transform(transformer, x)
        return x

    def __getnewargs__(
        self,
    ) -> tuple[TransformerProtocol, TransformerProtocol | ModelProtocol]:
        return self.first, self.second

    def __repr__(self) -> str:
        components = (*self._prefix, self._terminal)
        return f"Pipeline({' | '.join(repr(component) for component in components)})"

    def __str__(self) -> str:
        return " | ".join(
            component.__class__.__name__
            for component in (*self._prefix, self._terminal)
        )

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


class TransformerPipeline(Pipeline):
    """A composable pipeline exposing transformed features."""

    _terminal: TransformerProtocol

    @property
    def stages(self) -> tuple[TransformerProtocol, ...]:
        """All transformer stages in execution order."""
        return (*self._prefix, self._terminal)

    def transform_one(self, x: FeatureMap) -> FeatureMap:
        return self._checked_transform(self._terminal, self._transform_prefix(x))

    @overload
    def __or__(self, other: TransformerProtocol) -> TransformerPipeline: ...

    @overload
    def __or__(self, other: ModelProtocol) -> ModelPipeline: ...

    def __or__(
        self, other: TransformerProtocol | ModelProtocol
    ) -> TransformerPipeline | ModelPipeline:
        return Pipeline(self, other)


class ModelPipeline(Pipeline):
    """A terminal pipeline exposing anomaly scores."""

    _terminal: ModelProtocol

    def score_one(self, x: FeatureMap) -> float:
        score = self._terminal.score_one(self._transform_prefix(x))
        if not isinstance(score, int | float):
            raise PipelineError(
                "The final component must return a numeric score from score_one."
            )
        return float(score)

    def __or__(self, other: NoReturn) -> NoReturn:
        raise IncompatibleComponentError(
            self._terminal.__class__.__name__,
            "a transformer-ending pipeline before another component",
        )
