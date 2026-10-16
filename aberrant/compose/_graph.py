"""Single-owner, staged online graph execution with explicit failure outcomes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter_ns
from typing import Literal, TypeVar, cast

from aberrant.base.protocols import ModelProtocol
from aberrant.base.readiness import Readiness, ReadinessProtocol
from aberrant.compose._learning import LearningPass, LearningTransactions
from aberrant.compose._types import (
    INPUT,
    GraphComponent,
    GraphExecutionError,
    GraphNode,
    GraphNotReadyError,
    GraphStateUncertainError,
    LearningOutcome,
    MergeNode,
    NodeObservation,
    NodeState,
    Observer,
)
from aberrant.compose._validation import checked_features, ordered_nodes, predecessors
from aberrant.utils.validation import coerce_finite_number, coerce_integer_count

T = TypeVar("T")
MODEL = "$model"


class GraphPipeline:
    """A numeric feature DAG ending in exactly one online model.

    Learning uses each transformer's post-update representation, but only when
    all of its inputs are ready. A newly ready output may feed successors in
    the same observation. Scoring never learns and rejects unready graphs.
    One caller owns execution; concurrent use is not supported.

    This is model-ending composition, not a transformer with provisional
    outputs. Existing linear ``Pipeline`` deliberately retains its eager
    learning behaviour. Applications own calibration and alarm policies.
    """

    def __init__(
        self,
        nodes: Sequence[GraphComponent],
        model: ModelProtocol,
        output: str,
        *,
        model_minimum_updates: int = 0,
        observer: Observer | None = None,
    ) -> None:
        self.nodes = ordered_nodes(tuple(nodes), model, output)
        self.model = model
        self.output = output
        self.model_minimum_updates = coerce_integer_count(
            model_minimum_updates, label="model_minimum_updates", minimum=0
        )
        self.observer = observer
        self.observer_error_count = 0
        self.model_updates = 0
        self._updates = {node.id: 0 for node in self.nodes}
        self._schemas: dict[str, tuple[str, ...]] = {}
        self._failure: GraphStateUncertainError | None = None
        self.last_learning_outcome = self._outcome(False, self._initial_states())

    @property
    def readiness(self) -> Readiness:
        """Last complete learning outcome, detached from mutable model state."""
        if self.failed:
            return Readiness(False, reason="Graph state is uncertain")
        return self.last_learning_outcome.readiness

    @property
    def node_states(self) -> tuple[NodeState, ...]:
        return self.last_learning_outcome.nodes

    @property
    def failed(self) -> bool:
        return self._failure is not None

    def __getstate__(self) -> dict[str, object]:
        """Observers are process-local diagnostics, never checkpoint resources."""
        state = self.__dict__.copy()
        state["observer"] = None
        return state

    def learn_one(self, x: dict[str, float]) -> None:
        """Commit one staged pass, or roll back every transactional transformer.

        Transactions remain open through the model's update. A failure after
        touching any nontransactional learner is uncertain and permanently
        disables this graph; no false all-component atomicity is promised.
        """
        self._require_usable()
        execution = LearningPass(
            {INPUT: self._input(x)}, dict(self._schemas), dict(self._updates)
        )
        transactions = LearningTransactions()
        try:
            with transactions:
                self._learn_nodes(execution, transactions)
                self._learn_model(execution, transactions)
                outcome = self._outcome(
                    execution.model_updated, tuple(execution.states)
                )
        except BaseException as error:
            kind = (
                GraphStateUncertainError
                if transactions.uncertain
                else GraphExecutionError
            )
            failure = kind(execution.node_id, execution.phase, str(error))
            if isinstance(failure, GraphStateUncertainError):
                self._failure = failure
            if not isinstance(error, Exception):
                raise
            raise failure from error
        self._updates = execution.updates
        self.model_updates += int(execution.model_updated)
        execution.schemas[INPUT] = tuple(execution.available[INPUT])
        self._schemas = execution.schemas
        self.last_learning_outcome = outcome

    def _learn_nodes(
        self, execution: LearningPass, transactions: LearningTransactions
    ) -> None:
        for node in self.nodes:
            execution.node_id = node.id
            blocked = tuple(
                key for key in predecessors(node) if key not in execution.available
            )
            if blocked:
                execution.states.append(
                    NodeState(
                        node.id,
                        execution.updates[node.id],
                        Readiness(False, reason="Waiting for inputs"),
                        blocked,
                    )
                )
            elif isinstance(node, MergeNode):
                execution.phase = "merge"
                execution.available[node.id] = self._merge(
                    node, execution.available, execution.schemas
                )
                execution.updates[node.id] += 1
                execution.states.append(
                    NodeState(node.id, execution.updates[node.id], Readiness(True))
                )
            else:
                self._learn_transformer(node, execution, transactions)

    def _learn_transformer(
        self,
        node: GraphNode,
        execution: LearningPass,
        transactions: LearningTransactions,
    ) -> None:
        current = dict(execution.available[node.input])
        execution.phase = "learn"
        transactions.enter_learner(node.transformer, current)
        self._observe_call(
            node.id, "learn", current, lambda: node.transformer.learn_one(dict(current))
        )
        execution.updates[node.id] += 1
        ready = self._component_readiness(
            node.transformer, execution.updates[node.id], node.minimum_updates or 0
        )
        if ready.ready:
            execution.phase = "transform"
            execution.available[node.id] = self._transform(
                node, current, execution.schemas
            )
        execution.states.append(NodeState(node.id, execution.updates[node.id], ready))

    def _learn_model(
        self, execution: LearningPass, transactions: LearningTransactions
    ) -> None:
        if self.output not in execution.available:
            return
        execution.node_id, execution.phase = MODEL, "learn"
        current = dict(execution.available[self.output])
        transactions.enter_learner(self.model, current)
        self._observe_call(
            MODEL, "learn", current, lambda: self.model.learn_one(dict(current))
        )
        execution.model_updated = True

    def score_one(self, x: dict[str, float]) -> float:
        """Execute one read-only pass using the previously committed state."""
        self._require_usable()
        if not self.readiness.ready:
            raise GraphNotReadyError(self.readiness.reason or "Graph is not ready")
        available = {INPUT: self._input(x)}
        schemas = dict(self._schemas)
        node_id, phase = INPUT, "transform"
        try:
            for node in self.nodes:
                node_id = node.id
                if isinstance(node, MergeNode):
                    phase = "merge"
                    available[node.id] = self._merge(node, available, schemas)
                else:
                    phase = "transform"
                    available[node.id] = self._transform(
                        node, available[node.input], schemas
                    )
            node_id, phase = MODEL, "score"
            current = available[self.output]
            score = self._observe_call(
                MODEL, "score", current, lambda: self.model.score_one(dict(current))
            )
            return coerce_finite_number(score, label="Graph model score")
        except Exception as error:
            raise GraphExecutionError(node_id, phase, str(error)) from error

    def _require_usable(self) -> None:
        if self._failure is not None:
            raise GraphStateUncertainError(
                self._failure.node_id,
                self._failure.phase,
                "Graph state is uncertain; restore a known checkpoint or rebuild",
            ) from self._failure

    def _input(self, x: dict[str, float]) -> dict[str, float]:
        expected = self._schemas.get(INPUT)
        return checked_features(x, expected or tuple(sorted(x)))

    def _transform(
        self,
        node: GraphNode,
        current: dict[str, float],
        schemas: dict[str, tuple[str, ...]],
    ) -> dict[str, float]:
        output = self._observe_call(
            node.id,
            "transform",
            current,
            lambda: node.transformer.transform_one(dict(current)),
        )
        values = checked_features(output, schemas.get(node.id))
        schemas[node.id] = tuple(values)
        return values

    def _merge(
        self,
        node: MergeNode,
        available: dict[str, dict[str, float]],
        schemas: dict[str, tuple[str, ...]],
    ) -> dict[str, float]:
        started = perf_counter_ns()
        merged: dict[str, float] = {}
        for source in node.inputs:
            values = available[source]
            if set(merged).intersection(values):
                raise ValueError(
                    "Merge produces duplicate features; rename them explicitly"
                )
            merged.update(values)
        result = checked_features(merged, schemas.get(node.id))
        schemas[node.id] = tuple(result)
        self._emit(
            NodeObservation(
                node.id,
                "merge",
                tuple(merged.items()),
                tuple(result.items()),
                perf_counter_ns() - started,
            )
        )
        return result

    def _observe_call(
        self,
        node_id: str,
        phase: Literal["learn", "transform", "score"],
        inputs: dict[str, float],
        operation: Callable[[], T],
    ) -> T:
        if self.observer is None:
            return operation()
        before = tuple(inputs.items())
        started = perf_counter_ns()
        output = operation()
        detached = cast(
            "tuple[tuple[str, float], ...] | float | None",
            tuple(output.items()) if isinstance(output, dict) else output,
        )
        self._emit(
            NodeObservation(
                node_id, phase, before, detached, perf_counter_ns() - started
            )
        )
        return output

    def _emit(self, observation: NodeObservation) -> None:
        if self.observer is not None:
            try:
                self.observer(observation)
            except Exception:
                self.observer_error_count += 1

    @staticmethod
    def _component_readiness(
        component: object, updates: int, minimum: int
    ) -> Readiness:
        declared = (
            component.readiness
            if isinstance(component, ReadinessProtocol)
            else Readiness(True, updates, minimum)
        )
        if not isinstance(declared, Readiness):
            raise TypeError("Component readiness must be a Readiness value")
        if not declared.ready:
            return declared
        if updates < minimum:
            return Readiness(False, updates, minimum, "Learning reference")
        return declared

    def _initial_states(self) -> tuple[NodeState, ...]:
        result: list[NodeState] = []
        ready = {INPUT}
        for node in self.nodes:
            blocked = tuple(key for key in predecessors(node) if key not in ready)
            state = (
                self._component_readiness(
                    node.transformer, 0, node.minimum_updates or 0
                )
                if isinstance(node, GraphNode)
                else Readiness(True)
            )
            if blocked:
                state = Readiness(False, reason="Waiting for inputs")
            if state.ready:
                ready.add(node.id)
            result.append(NodeState(node.id, 0, state, blocked))
        return tuple(result)

    def _outcome(
        self, model_updated: bool, states: tuple[NodeState, ...]
    ) -> LearningOutcome:
        updates = self.model_updates + int(model_updated)
        model_readiness = self._component_readiness(
            self.model, updates, self.model_minimum_updates
        )
        features_ready = self.output == INPUT or any(
            state.node_id == self.output and state.readiness.ready for state in states
        )
        ready = (
            model_readiness
            if features_ready
            else Readiness(
                False, updates, self.model_minimum_updates, "Waiting for feature graph"
            )
        )
        return LearningOutcome(model_updated, updates, states, ready)
