"""Standalone graph execution against real numeric components and references."""

import pickle
from contextlib import contextmanager, suppress
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from aberrant.base import ModelProtocol, Readiness
from aberrant.catalog import create_transformer, get_transformer_spec
from aberrant.compose import (
    INPUT,
    GraphExecutionError,
    GraphNode,
    GraphPipeline,
    GraphStateUncertainError,
    MergeNode,
)
from aberrant.exceptions import ConfigurationError
from aberrant.model.forest import OnlineIsolationForest
from aberrant.model.null import NullModel
from aberrant.model.stat import (
    MovingCovariance,
    MovingZScore,
)
from aberrant.transform import (
    IncrementalPCA,
    MinMaxScaler,
    Passthrough,
    Rename,
    Select,
    StandardScaler,
)


def test_branched_pca_raw_bypass_matches_explicit_numeric_reference():
    scaler = StandardScaler()
    pca = IncrementalPCA(n_components=1, warm_up=3)
    model = MovingCovariance(window_size=20)
    graph = GraphPipeline(
        [
            GraphNode("selected", Select(features=["a", "b"])),
            GraphNode("scaled", scaler, "selected"),
            GraphNode("pca", pca, "scaled"),
            GraphNode("raw", Select(features=["temperature"])),
            MergeNode("features", ("pca", "raw")),
        ],
        model,
        "features",
    )
    assert isinstance(graph, ModelProtocol)
    reference_scaler = StandardScaler()
    reference_pca = IncrementalPCA(n_components=1, warm_up=3)
    reference_model = MovingCovariance(window_size=20)
    for index in range(1, 10):
        sample = {"a": float(index), "b": float(index**2), "temperature": 20.0 + index}
        selected = {key: sample[key] for key in ("a", "b")}
        if graph.readiness.ready:
            projected = reference_pca.transform_one(
                reference_scaler.transform_one(selected)
            )
            expected = reference_model.score_one(
                {**projected, "temperature": sample["temperature"]}
            )
            assert graph.score_one(sample) == pytest.approx(expected)
        graph.learn_one(sample)
        reference_scaler.learn_one(selected)
        transformed = reference_scaler.transform_one(selected)
        reference_pca.learn_one(transformed)
        if reference_pca.readiness.ready:
            reference_model.learn_one(
                {
                    **reference_pca.transform_one(transformed),
                    "temperature": sample["temperature"],
                }
            )
        assert graph.model_updates == max(0, index - 2)
        assert graph.readiness.ready == (index >= 4)
    assert model.window == reference_model.window
    assert scaler.counts == {"a": 9, "b": 9}


def test_shared_node_updates_once_and_score_is_read_only():
    scaler = StandardScaler()
    graph = GraphPipeline(
        [
            GraphNode("shared", scaler),
            GraphNode("a", Select(features=["a"]), "shared"),
            GraphNode("b", Select(features=["b"]), "shared"),
            MergeNode("joined", ("b", "a")),
        ],
        MovingCovariance(window_size=20),
        "joined",
    )
    for index in range(3):
        graph.learn_one({"a": float(index), "b": float(index + 2)})
    assert scaler.counts == {"a": 3, "b": 3}
    before = pickle.dumps(graph)
    graph.score_one({"b": 8.0, "a": 4.0})
    assert pickle.dumps(graph) == before
    assert tuple(graph.model.window) == ("a", "b")


def test_staggered_branches_gate_downstream_initialization():
    left, right = (
        IncrementalPCA(n_components=1, warm_up=2),
        IncrementalPCA(n_components=1, warm_up=4),
    )
    model = MovingCovariance(window_size=20)
    graph = GraphPipeline(
        [
            GraphNode("left", left),
            GraphNode("right", right),
            GraphNode("left_name", Rename(mapping={"component_0": "left"}), "left"),
            GraphNode("right_name", Rename(mapping={"component_0": "right"}), "right"),
            MergeNode("joined", ("left_name", "right_name")),
        ],
        model,
        "joined",
    )
    for index in range(1, 5):
        graph.learn_one({"a": float(index), "b": float(index + 1)})
        assert graph.model_updates == int(index == 4)
        assert graph.last_learning_outcome.model_updated == (index == 4)
        if index < 4:
            assert not model.window
            assert graph.score_one({"a": 5.0, "b": 6.0}) == 0.0
    counts = {state.node_id: state.updates for state in graph.node_states}
    assert counts["left_name"] == 3
    assert counts["right_name"] == 1
    assert not graph.readiness.ready
    graph.learn_one({"a": 5.0, "b": 6.0})
    assert graph.readiness.ready


def test_terminal_rejection_restores_all_transformers_and_disables_graph():
    scaler = StandardScaler()
    graph = GraphPipeline(
        [GraphNode("scaled", scaler)], MovingZScore(window_size=5), "scaled"
    )
    previous = pickle.dumps(scaler)
    with pytest.raises(GraphStateUncertainError, match="exactly one"):
        graph.learn_one({"a": 1.0, "b": 2.0})
    assert pickle.dumps(scaler) == previous
    assert graph.model_updates == 0
    assert graph.failed
    with pytest.raises(GraphStateUncertainError, match="restore"):
        graph.learn_one({"a": 1.0})
    with pytest.raises(GraphStateUncertainError):
        graph.score_one({"a": 1.0})
    restored = pickle.loads(pickle.dumps(graph))
    assert restored.failed


def test_branch_rejection_rolls_back_sibling_and_retains_last_outcome():
    scaler = StandardScaler()
    graph = GraphPipeline(
        [
            GraphNode("a", scaler),
            GraphNode("b", Rename(mapping={"missing": "named"})),
            MergeNode("joined", ("a", "b")),
        ],
        MovingCovariance(window_size=5),
        "joined",
    )
    before = pickle.dumps(graph)
    with pytest.raises(GraphExecutionError) as error:
        graph.learn_one({"x": 1.0})
    assert not isinstance(error.value, GraphStateUncertainError)
    assert pickle.dumps(graph) == before
    assert not graph.failed


def test_mutating_custom_transformer_cannot_corrupt_raw_sibling():
    class MutatesInput:
        @property
        def readiness(self) -> Readiness:
            return Readiness(True)

        def learn_one(self, x):
            x.clear()

        def transform_one(self, x):
            x.clear()
            return {"processed": 3.0}

    model = MovingCovariance(window_size=5)
    graph = GraphPipeline(
        [
            GraphNode("a_mutates", MutatesInput()),
            GraphNode("b_raw", Passthrough()),
            MergeNode("joined", ("a_mutates", "b_raw")),
        ],
        model,
        "joined",
    )
    sample = {"raw": 7.0}
    graph.learn_one(sample)
    assert sample == {"raw": 7.0}
    assert model.window["raw"][-1] == 7.0
    graph.score_one(sample)
    assert sample == {"raw": 7.0}


def test_observations_are_detached_and_do_not_reexecute_components():
    observations = []
    scaler = StandardScaler()
    graph = GraphPipeline(
        [GraphNode("scaled", scaler)],
        NullModel(),
        "scaled",
        observer=observations.append,
    )
    graph.learn_one({"x": 2.0})
    graph.score_one({"x": 3.0})
    assert [(event.node_id, event.phase) for event in observations] == [
        ("scaled", "learn"),
        ("scaled", "transform"),
        ("$model", "learn"),
        ("scaled", "transform"),
        ("$model", "score"),
    ]
    assert scaler.counts["x"] == 1
    assert dict(observations[3].inputs) == {"x": 3.0}
    with pytest.raises(FrozenInstanceError):
        observations[0].duration_ns = 0
    assert all(event.duration_ns >= 0 for event in observations)
    restored = pickle.loads(pickle.dumps(graph))
    assert restored.observer is None
    assert restored.score_one({"x": 3.0}) == graph.score_one({"x": 3.0})


def test_observer_failure_never_changes_learning_or_score():
    def broken(_event):
        raise RuntimeError("diagnostic failure")

    graph = GraphPipeline(
        [GraphNode("scaled", StandardScaler())],
        MovingZScore(window_size=5, warm_up=2),
        "scaled",
        observer=broken,
    )
    clean = GraphPipeline(
        [GraphNode("scaled", StandardScaler())],
        MovingZScore(window_size=5, warm_up=2),
        "scaled",
    )
    for value in (1.0, 2.0, 3.0):
        graph.learn_one({"x": value})
        clean.learn_one({"x": value})
    assert graph.score_one({"x": 9.0}) == clean.score_one({"x": 9.0})
    assert graph.observer_error_count == 11


def test_duplicate_keys_are_rejected_before_terminal_learning():
    graph = GraphPipeline(
        [
            GraphNode("a", Passthrough()),
            GraphNode("b", Passthrough()),
            MergeNode("joined", ("a", "b")),
        ],
        MovingZScore(window_size=5),
        "joined",
    )
    with pytest.raises(GraphExecutionError, match="duplicate"):
        graph.learn_one({"x": 1.0})
    assert graph.model_updates == 0


@pytest.mark.parametrize(
    "nodes,output",
    [
        ([GraphNode("a", Passthrough(), "missing")], "a"),
        ([GraphNode("a", Passthrough(), "b"), GraphNode("b", Passthrough(), "a")], "a"),
        ([GraphNode("a", Passthrough()), GraphNode("b", Passthrough())], "a"),
        ([GraphNode("a", Passthrough()), GraphNode("a", Passthrough())], "a"),
        ([GraphNode(INPUT, Passthrough())], INPUT),
        ([], "missing"),
    ],
)
def test_invalid_graphs_are_rejected(nodes, output):
    with pytest.raises(ConfigurationError):
        GraphPipeline(nodes, MovingZScore(window_size=5), output)


def test_component_instances_cannot_be_owned_by_two_nodes():
    scaler = StandardScaler()
    with pytest.raises(ConfigurationError, match="one node"):
        GraphPipeline(
            [GraphNode("a", scaler), GraphNode("b", scaler, "a")],
            MovingZScore(window_size=5),
            "b",
        )


def test_input_schema_and_nonfinite_values_reject_before_learning():
    graph = GraphPipeline(
        [GraphNode("scaled", StandardScaler())], MovingZScore(window_size=5), "scaled"
    )
    graph.learn_one({"x": 1.0})
    before = pickle.dumps(graph)
    for sample in ({"other": 2.0}, {"x": float("inf")}, {}):
        with pytest.raises(ValueError):
            graph.learn_one(sample)
        assert pickle.dumps(graph) == before


@pytest.mark.parametrize("passthrough", [False, True])
def test_first_input_order_is_frozen_through_learning_scoring_and_checkpoint(
    passthrough,
):
    observations = []
    terminal = OnlineIsolationForest(n_trees=2, max_leaf_samples=4, seed=42)
    reference = OnlineIsolationForest(n_trees=2, max_leaf_samples=4, seed=42)
    nodes = [GraphNode("raw", Passthrough())] if passthrough else []
    graph = GraphPipeline(
        nodes, terminal, "raw" if passthrough else INPUT, observer=observations.append
    )
    for index in range(12):
        sample = {"z": float(index * 3), "a": float(index % 5)}
        graph.learn_one(sample)
        reference.learn_one(sample)
    assert all(
        tuple(name for name, _ in event.inputs) == ("z", "a") for event in observations
    )

    # A differently ordered caller mapping cannot change the established order.
    observations.clear()
    candidate = {"a": 2.0, "z": 40.0}
    assert graph.score_one(candidate) == reference.score_one(candidate)
    assert all(
        tuple(name for name, _ in event.inputs) == ("z", "a") for event in observations
    )
    restored = pickle.loads(pickle.dumps(graph))
    restored.observer = observations.append
    observations.clear()
    restored.learn_one(candidate)
    reference.learn_one(candidate)
    assert all(
        tuple(name for name, _ in event.inputs) == ("z", "a") for event in observations
    )
    assert restored.score_one(candidate) == reference.score_one(candidate)


def test_select_rename_and_clip_are_public_catalog_components():
    selected = create_transformer("select", {"features": ["b", "a"]})
    renamed = create_transformer("rename", {"mapping": {"b": "right"}})
    pipeline = selected | renamed
    pipeline.learn_one({"a": 1.0, "b": 2.0, "c": 3.0})
    assert pipeline.transform_one({"a": 1.0, "b": 2.0, "c": 3.0}) == {
        "right": 2.0,
        "a": 1.0,
    }
    assert get_transformer_spec("select").output_feature_names(
        {"features": ["b", "a"]}, ("a", "b", "c")
    ) == ("b", "a")
    assert get_transformer_spec("incremental_pca").output_feature_names(
        {"n_components": 2}, ("a", "b")
    ) == ("component_0", "component_1")
    scaler, clipped = MinMaxScaler(), MinMaxScaler(clip=True)
    for sample in ({"x": 1.0}, {"x": 3.0}):
        scaler.learn_one(sample)
        clipped.learn_one(sample)
    assert scaler.transform_one({"x": 5.0}) == {"x": 2.0}
    assert clipped.transform_one({"x": 5.0}) == {"x": 1.0}
    assert clipped.transform_one({"x": -1.0}) == {"x": 0.0}


def test_custom_readiness_gates_each_node():
    class LearnsReference:
        def __init__(self):
            self.count = 0

        @property
        def readiness(self):
            return Readiness.after(self.count, 3)

        def learn_one(self, x):
            self.count += 1

        def transform_one(self, x):
            return x

    graph = GraphPipeline(
        [
            GraphNode("custom", LearnsReference()),
        ],
        MovingZScore(window_size=5, warm_up=2),
        "custom",
    )
    for index in range(1, 5):
        graph.learn_one({"x": float(index)})
        assert graph.model_updates == max(0, index - 2)
        assert graph.readiness.ready == (index == 4)


def test_rollback_failure_disables_graph():
    class BrokenRollback(Passthrough):
        @contextmanager
        def learning_transaction(self, x):
            try:
                yield
            except Exception:
                raise RuntimeError("rollback failed") from None

    graph = GraphPipeline(
        [
            GraphNode("a", BrokenRollback()),
            GraphNode("b", Rename(mapping={"missing": "x"}), "a"),
        ],
        MovingZScore(window_size=5),
        "b",
    )
    with pytest.raises(GraphStateUncertainError, match="rollback failed"):
        graph.learn_one({"x": 1.0})
    assert graph.failed
    assert not graph.readiness.ready


def test_cancellation_after_nontransactional_model_update_disables_graph():
    class InterruptedModel:
        @property
        def readiness(self) -> Readiness:
            return Readiness(True)

        def learn_one(self, x):
            raise KeyboardInterrupt()

        def score_one(self, x):
            return 0.0

    graph = GraphPipeline([], InterruptedModel(), INPUT)
    with pytest.raises(KeyboardInterrupt):
        graph.learn_one({"x": 1.0})
    assert graph.failed
    assert not graph.readiness.ready


def test_old_minmax_checkpoint_restores_default_extrapolation():
    scaler = MinMaxScaler()
    scaler.learn_one({"x": 1.0})
    scaler.learn_one({"x": 2.0})
    del scaler.clip
    restored = pickle.loads(pickle.dumps(scaler))
    assert not restored.clip
    assert restored.transform_one({"x": 3.0}) == {"x": 2.0}


def test_online_pca_keeps_a_real_orthogonal_basis_on_degenerate_stream():
    scaler = StandardScaler()
    projection = IncrementalPCA(n_components=2, warm_up=3)
    for index in range(30):
        data = {"a": float(index), "b": float(index**2), "c": float(index % 3)}
        scaler.learn_one(data)
        current = scaler.transform_one(data)
        projection.learn_one(current)
        if projection.readiness.ready:
            output = projection.transform_one(current)
            assert np.isrealobj(projection.values)
            assert np.isrealobj(projection.vectors)
            assert np.isfinite(list(output.values())).all()
            np.testing.assert_allclose(
                projection.vectors.T @ projection.vectors, np.eye(2), atol=1e-10
            )


class _UndeclaredTransformer:
    def learn_one(self, x):
        return None

    def transform_one(self, x):
        return x


@pytest.mark.parametrize(
    ("nodes", "model", "message"),
    [
        ([], StandardScaler(), "unambiguous model"),
        (["a"], MovingZScore(window_size=5), "GraphNode or MergeNode"),
        (
            [GraphNode("a", MovingZScore(window_size=5))],
            MovingZScore(window_size=5),
            "unambiguous transformer",
        ),
        (
            [GraphNode("a", _UndeclaredTransformer())],
            MovingZScore(window_size=5),
            "unambiguous transformer",
        ),
        ([MergeNode("a", ())], MovingZScore(window_size=5), "named inputs"),
        (
            [GraphNode("b", Passthrough()), MergeNode("a", ("b", "b"))],
            MovingZScore(window_size=5),
            "repeat a predecessor",
        ),
    ],
    ids=[
        "transformer-terminal",
        "not-a-node",
        "model-as-transformer",
        "custom-without-readiness",
        "merge-without-inputs",
        "merge-repeats-input",
    ],
)
def test_graphs_with_ambiguous_components_are_rejected(nodes, model, message):
    with pytest.raises(ConfigurationError, match=message):
        GraphPipeline(nodes, model, "a" if nodes else INPUT)


def test_merge_inputs_must_name_nodes_not_one_string():
    with pytest.raises(ValueError, match="sequence of node IDs"):
        MergeNode("joined", "ab")


def test_scoring_failures_name_the_node_and_keep_the_graph_usable():
    class RejectsNegative(Passthrough):
        def transform_one(self, x):
            if x["x"] < 0:
                raise ValueError("negative reading")
            return x

    model = MovingZScore(window_size=5, warm_up=2)
    reference = MovingZScore(window_size=5, warm_up=2)
    graph = GraphPipeline([GraphNode("guard", RejectsNegative())], model, "guard")
    for value in (1.0, 3.0):
        graph.learn_one({"x": value})
        reference.learn_one({"x": value})

    with pytest.raises(GraphExecutionError, match="negative reading") as error:
        graph.score_one({"x": -1.0})
    assert (error.value.node_id, error.value.phase) == ("guard", "transform")
    assert not graph.failed
    assert graph.score_one({"x": 2.0}) == reference.score_one({"x": 2.0})


def test_transformers_must_emit_string_feature_names():
    class NumberedKeys(Passthrough):
        def transform_one(self, x):
            return {1: x["x"]}

    graph = GraphPipeline(
        [GraphNode("numbered", NumberedKeys())],
        MovingZScore(window_size=5),
        "numbered",
    )
    with pytest.raises(GraphExecutionError, match="non-empty strings"):
        graph.learn_one({"x": 1.0})


def test_component_readiness_must_be_a_readiness_value():
    class ClaimsReadiness(_UndeclaredTransformer):
        readiness = "ready"

    with pytest.raises(TypeError, match="Readiness value"):
        GraphPipeline(
            [GraphNode("claims", ClaimsReadiness())],
            MovingZScore(window_size=5),
            "claims",
        )


def test_a_failing_transaction_entry_disables_the_graph():
    class BrokenTransaction(Passthrough):
        @contextmanager
        def learning_transaction(self, x):
            raise RuntimeError("cannot snapshot")
            yield

    graph = GraphPipeline(
        [GraphNode("broken", BrokenTransaction())],
        MovingZScore(window_size=5),
        "broken",
    )
    with pytest.raises(GraphStateUncertainError, match="cannot snapshot"):
        graph.learn_one({"x": 1.0})
    assert graph.failed


def test_a_transaction_that_swallows_failures_disables_the_graph():
    class Swallowing(Passthrough):
        @contextmanager
        def learning_transaction(self, x):
            with suppress(Exception):
                yield

    graph = GraphPipeline(
        [
            GraphNode("swallowing", Swallowing()),
            GraphNode("b", Rename(mapping={"missing": "x"}), "swallowing"),
        ],
        MovingZScore(window_size=5),
        "b",
    )
    with pytest.raises(GraphStateUncertainError, match="cannot suppress"):
        graph.learn_one({"x": 1.0})
    assert graph.failed
