"""The learner transaction contract is independent of the component's role."""

from contextlib import contextmanager

import pytest

from aberrant.base import (
    Pipeline,
    TransactionalLearnerProtocol,
    TransactionalTransformerProtocol,
)
from aberrant.compose import (
    GraphExecutionError,
    GraphNode,
    GraphPipeline,
    GraphStateUncertainError,
)


class RecordingModel:
    def __init__(self, failure=None):
        self.inputs = []
        self.failure = failure

    def learn_one(self, x):
        self.inputs.append(dict(x))
        if self.failure is not None:
            raise self.failure

    def score_one(self, x):
        return x["x"]


class ReversibleModel(RecordingModel):
    @contextmanager
    def learning_transaction(self, x):
        previous = self.inputs.copy()
        try:
            yield
        except BaseException:
            self.inputs = previous
            raise


class ReversibleTransformer:
    def __init__(self):
        self.updates = 0

    def learn_one(self, x):
        self.updates += 1

    def transform_one(self, x):
        return {"x": x["x"] + self.updates}

    @contextmanager
    def learning_transaction(self, x):
        previous = self.updates
        try:
            yield
        except BaseException:
            self.updates = previous
            raise


def pipeline(engine, transformer, model):
    if engine == "graph":
        return GraphPipeline(
            [GraphNode("transform", transformer, minimum_updates=0)], model, "transform"
        )
    return Pipeline(transformer, model)


def test_transaction_capability_does_not_require_transformation():
    assert isinstance(ReversibleModel(), TransactionalLearnerProtocol)
    assert not isinstance(ReversibleModel(), TransactionalTransformerProtocol)
    assert isinstance(ReversibleTransformer(), TransactionalLearnerProtocol)
    assert isinstance(ReversibleTransformer(), TransactionalTransformerProtocol)
    assert not isinstance(RecordingModel(), TransactionalLearnerProtocol)


@pytest.mark.parametrize("engine", ["linear", "graph"])
def test_transactional_terminal_commits_post_update_inputs(engine):
    transformer, model = ReversibleTransformer(), ReversibleModel()
    detector = pipeline(engine, transformer, model)
    detector.learn_one({"x": 1.0})
    detector.learn_one({"x": 1.0})
    assert transformer.updates == 2
    assert model.inputs == [{"x": 2.0}, {"x": 3.0}]


@pytest.mark.parametrize("engine", ["linear", "graph"])
@pytest.mark.parametrize("failure", [ValueError("rejected"), KeyboardInterrupt()])
def test_transactional_model_rolls_back_itself_and_upstream_on_failure(engine, failure):
    transformer, model = ReversibleTransformer(), ReversibleModel(failure)
    detector = pipeline(engine, transformer, model)
    error = type(failure)
    if engine == "graph" and isinstance(failure, Exception):
        error = GraphExecutionError
    with pytest.raises(error):
        detector.learn_one({"x": 1.0})
    assert transformer.updates == 0
    assert model.inputs == []
    if engine == "graph":
        assert not detector.failed
        assert detector.model_updates == 0
    model.failure = None
    detector.learn_one({"x": 2.0})
    assert model.inputs == [{"x": 3.0}]


@pytest.mark.parametrize("engine", ["linear", "graph"])
@pytest.mark.parametrize("failure", [ValueError("rejected"), KeyboardInterrupt()])
def test_nontransactional_model_retains_its_existing_failure_semantics(engine, failure):
    transformer, model = ReversibleTransformer(), RecordingModel(failure)
    detector = pipeline(engine, transformer, model)
    error = type(failure)
    if engine == "graph" and isinstance(failure, Exception):
        error = GraphStateUncertainError
    with pytest.raises(error):
        detector.learn_one({"x": 1.0})
    assert transformer.updates == 0
    assert model.inputs == [{"x": 2.0}]
    if engine == "graph":
        assert detector.failed


@pytest.mark.parametrize("engine", ["linear", "graph"])
def test_transactional_model_rollback_failure_propagates(engine):
    class BrokenRollback(ReversibleModel):
        @contextmanager
        def learning_transaction(self, x):
            try:
                yield
            except BaseException:
                raise RuntimeError("rollback failed") from None

    transformer = ReversibleTransformer()
    model = BrokenRollback(ValueError("rejected"))
    detector = pipeline(engine, transformer, model)
    error = GraphStateUncertainError if engine == "graph" else RuntimeError
    with pytest.raises(error, match="rollback failed"):
        detector.learn_one({"x": 1.0})
    assert transformer.updates == 0
    assert model.inputs == [{"x": 2.0}]
    if engine == "graph":
        assert detector.failed
