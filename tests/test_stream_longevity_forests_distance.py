"""Streaming retention, eviction and aging regressions for forest/distance models.

These finite stress tests check structural bounds rather than infer an infinite
stream guarantee from a short process RSS measurement. Optional native FAISS
coverage is isolated at test execution so other tests remain collectible.
"""

from __future__ import annotations

import copy
import gc
import math
import pickle
import sys
import tracemalloc
from collections import deque

import numpy as np
import pytest

from aberrant.model.distance import (
    CellNeighborhoodDetector,
    LocalOutlierFactor,
    SDOStream,
    StationaryRegionNeighborDetector,
)
from aberrant.model.iforest import (
    ASDIsolationForest,
    HalfSpaceTrees,
    MondrianIsolationForest,
    OnlineIsolationForest,
    RandomCutForest,
    StreamRandomHistogramForest,
    XStream,
)
from aberrant.model.iforest._online_tree import _OnlineBranch
from aberrant.model.iforest.halfspace import HSTNode
from aberrant.model.iforest.mondrian import (
    MondrianBranch,
    MondrianLeaf,
    MondrianTree,
    _average_path_length,
)


def _nodes(root, children):
    """Traverse without imposing a test-side recursion depth limit."""
    pending = [] if root is None else [root]
    while pending:
        node = pending.pop()
        yield node
        pending.extend(children(node))


def _mondrian_children(node):
    return (
        [node.left_child, node.right_child] if isinstance(node, MondrianBranch) else []
    )


@pytest.mark.parametrize("pattern", ["constant", "drifting", "alternating"])
def test_mondrian_bounded_mode_replaces_history_and_population(pattern):
    window = 16
    model = MondrianIsolationForest(
        n_estimators=2, subspace_size=2, lambda_=8.0, seed=7, window_size=window
    )
    recent = deque(maxlen=window)
    for i in range(1600):
        value = (
            3.0
            if pattern == "constant"
            else float(i if pattern == "drifting" else (-1) ** (i // 20) * i)
        )
        point = {"x": value, "y": value / 2.0}
        recent.append([value, value / 2.0])
        model.learn_one(point)
        assert len(model._window) <= window
        for tree in model.trees:
            assert tree.n_samples <= 2 * window - 1
            nodes = list(_nodes(tree.root, _mondrian_children))
            assert len(nodes) <= 2 * tree.n_samples - 1
            assert tree.root.count == tree.n_samples
            assert (
                sum(n.count for n in nodes if isinstance(n, MondrianLeaf))
                == tree.n_samples
            )
        assert math.isfinite(model.score_one(point))
        if i + 1 >= 2 * window and (i + 1) % window == 0:
            assert model.trees[0].n_samples == window
            np.testing.assert_allclose(model.trees[0].root.min, np.min(recent, axis=0))
            np.testing.assert_allclose(model.trees[0].root.max, np.max(recent, axis=0))
    assert model.n_samples == 1600
    assert model._compute_c_factor() == _average_path_length(model.trees[0].n_samples)


def test_mondrian_lifetime_mode_remains_explicitly_unbounded():
    model = MondrianIsolationForest(n_estimators=1, subspace_size=1, lambda_=8, seed=1)
    for i in range(512):
        model.learn_one({"x": float(i * 10)})
    assert model.window_size is None
    assert len(model._window) == 0
    assert model.trees[0].n_samples == 512
    assert len(list(_nodes(model.trees[0].root, _mondrian_children))) > 500


def test_mondrian_deep_tree_update_does_not_use_python_recursion():
    tree = MondrianTree(np.array([0]), 8.0, np.random.default_rng(1))
    root = MondrianLeaf.from_point(np.array([0.0]), 8.0)
    # A valid right-growing nested partition with the zero query at depth > the
    # interpreter limit. Matching bounds ensure insertion visits every branch.
    depth = sys.getrecursionlimit() + 100
    for value in range(1, depth + 1):
        root = MondrianBranch.join(
            root,
            MondrianLeaf.from_point(np.array([float(value)]), 8.0),
            0,
            value - 0.5,
            1.0 / (value + 1),
        )
    tree.root = root
    tree.n_samples = depth + 1
    tree.learn_one(np.array([0.0]))
    assert tree.root.count == depth + 2
    assert math.isfinite(tree.score_one(np.array([0.0])))


@pytest.mark.parametrize("forest", ["mondrian", "random_cut"])
def test_expanding_finite_stream_does_not_hit_tree_recursion_limit(forest):
    model = (
        MondrianIsolationForest(n_estimators=1, subspace_size=1, lambda_=1e301, seed=3)
        if forest == "mondrian"
        else RandomCutForest(n_trees=1, sample_size=2500, warmup_samples=2, seed=3)
    )
    for exponent in range(-1000, 1001):
        model.learn_one({"x": math.ldexp(1.0, exponent)})
    # This stream creates a path deeper than 1000 in both seeded forests.
    # Revisiting the oldest leaf previously exhausted Python's default stack.
    point = {"x": math.ldexp(1.0, -1000)}
    model.learn_one(point)
    assert math.isfinite(model.score_one(point))


@pytest.mark.parametrize("window", [0, 1, -1])
def test_mondrian_rejects_invalid_window(window):
    with pytest.raises(ValueError, match="window_size"):
        MondrianIsolationForest(window_size=window)


@pytest.mark.parametrize("subsample", [0.2, 1.0])
@pytest.mark.parametrize("tree_type", ["fixed", "adaptive"])
def test_online_forest_long_turnover_has_bounded_membership_and_tree(
    subsample, tree_type
):
    model = OnlineIsolationForest(
        num_trees=2,
        max_leaf_samples=3,
        tree_type=tree_type,
        subsample=subsample,
        window_size=24,
        seed=5,
    )
    for i in range(2000):
        point = {"x": float((i * 17) % 43), "y": float(i // 100)}
        model.learn_one(point)
        assert model.data_size == min(i + 1, 24)
        assert len(model.data_window) == model.data_size
        for tree in model.trees:
            assert len(tree._sample_membership) == model.data_size
            assert 0 <= tree.data_size <= model.data_size
            if tree.root is not None:
                assert tree.root.data_size == tree.data_size
                nodes = list(
                    _nodes(
                        tree.root,
                        lambda n: n.children if isinstance(n, _OnlineBranch) else [],
                    )
                )
                max_depth = math.ceil(
                    tree.get_random_path_length(
                        model.branching_factor,
                        model.max_leaf_samples,
                        2 * model.window_size,
                    )
                )
                assert len(nodes) <= 2 ** (max_depth + 1) - 1
        if i % 100 == 0:
            assert math.isfinite(model.score_one(point))


def test_online_large_batch_does_not_pin_discarded_batch_or_overgrow_tree_updates():
    model = OnlineIsolationForest(
        num_trees=1, max_leaf_samples=4, window_size=12, seed=5
    )
    batch = np.arange(30_000.0).reshape(-1, 3)
    populations = []
    original_learn = model.trees[0].learn

    def tracked_learn(data):
        populations.append(model.trees[0].data_size + len(data))
        return original_learn(data)

    model.trees[0].learn = tracked_learn
    model.learn_batch(batch)
    assert max(populations) <= 2 * model.window_size
    assert len(model.data_window) == model.window_size
    assert all(row.base is None for row in model.data_window)
    np.testing.assert_array_equal(model.data_window, batch[-model.window_size :])
    batch[:] = -1.0
    assert all(np.all(row >= 0.0) for row in model.data_window)


@pytest.mark.parametrize("shingle", [1, 4])
@pytest.mark.parametrize("pattern", ["duplicates", "unique"])
def test_random_cut_forest_evicts_ids_and_preserves_score_preview(shingle, pattern):
    model = RandomCutForest(
        n_trees=2, sample_size=16, shingle_size=shingle, warmup_samples=4, seed=2
    )
    for i in range(1800):
        value = 2.0 if pattern == "duplicates" else float(i)
        point = {"x": value, "y": value / 2.0}
        model.learn_one(point)
        assert len(model._history) <= shingle
        assert len(model._id_window) <= model.sample_size
        for tree in model._trees:
            assert tree.size == len(model._id_window)
            assert set(tree._id_to_leaf) == set(model._id_window)
            assert (
                sum(
                    len(leaf.point_ids)
                    for leaf in {id(n): n for n in tree._id_to_leaf.values()}.values()
                )
                == tree.size
            )
        if i % 100 == 0:
            before = pickle.dumps(model)
            assert math.isfinite(model.score_one(point))
            assert pickle.dumps(model) == before


@pytest.mark.parametrize("pattern", ["constant", "changing"])
def test_replacement_forests_do_not_accumulate_old_trees_or_samples(pattern):
    asd = ASDIsolationForest(
        n_estimators=2, max_samples=12, window_size=16, retrain_interval=7, seed=2
    )
    rhf = StreamRandomHistogramForest(
        n_estimators=2, max_depth=5, window_size=16, seed=2
    )
    for i in range(1200):
        value = 1.0 if pattern == "constant" else float((i * 11) % 17)
        point = {"x": value, "y": float(i % 5)}
        asd.learn_one(point)
        rhf.learn_one(point)
        assert len(asd.window) <= asd.window_size
        assert len(asd.trees) <= asd.n_estimators
        assert asd._samples_since_retrain < max(asd.window_size, asd.retrain_interval)
        if asd._last_fit_window is not None:
            assert asd._last_fit_window.shape == (asd.window_size, 2)
        assert len(rhf._initial_window) < rhf.window_size
        assert len(rhf._current_window) < rhf.window_size
        assert rhf._forest_size <= 2 * rhf.window_size - 1
        for tree in rhf._trees:
            assert tree.root.size == rhf._forest_size
            assert len(tree.root.collect_points()) == rhf._forest_size
            assert len(tree._node_random) <= 2 ** (rhf.max_depth + 1) - 1
        if i % 100 == 0:
            before = pickle.dumps((asd, rhf))
            assert math.isfinite(asd.score_one(point))
            assert math.isfinite(rhf.score_one(point))
            assert pickle.dumps((asd, rhf)) == before


def test_halfspace_masses_reset_for_many_complete_windows():
    model = HalfSpaceTrees(n_trees=2, height=4, window_size=13, seed=4)
    for i in range(2600):
        point = {"x": (i % 11) / 10.0, "y": (i % 7) / 6.0}
        model.learn_one(point)
        for tree in model._trees:
            nodes = list(
                _nodes(
                    tree, lambda n: [n.left, n.right] if isinstance(n, HSTNode) else []
                )
            )
            assert len(nodes) == 2 ** (model.height + 1) - 1
            assert tree.l_mass == (i + 1) % model.window_size
            assert all(0 <= n.l_mass < model.window_size for n in nodes)
            assert all(0 <= n.r_mass <= model.window_size for n in nodes)
        assert 0.0 <= model.score_one(point) <= 1.0


def test_xstream_key_churn_keeps_projection_cache_and_sketches_bounded():
    model = XStream(
        k=4,
        n_chains=2,
        depth=3,
        cms_width=8,
        cms_num_hashes=2,
        window_size=11,
        init_sample_size=7,
        max_feature_cache_size=5,
        seed=3,
    )
    for i in range(1500):
        model.learn_one({f"feature_{i}": float(i % 13)})
        assert len(model._feature_cache) <= 5
        assert len(model._init_buffer) <= model.init_sample_size
        state = model._state
        if state is not None:
            assert len(model._init_buffer) == 0
            assert state.cms_current.shape == (2, 3, 2, 8)
            assert np.all(state.cms_current >= 0)
            assert np.all(state.cms_current <= model.window_size)
            assert np.all(state.cms_reference >= 0)
            assert np.all(state.cms_reference <= model.window_size)
            if i % 100 == 0:
                current = state.cms_current.copy()
                reference = state.cms_reference.copy()
                for j in range(20):
                    assert math.isfinite(model.score_one({f"query_{i}_{j}": 2.0}))
                    assert len(model._feature_cache) <= 5
                np.testing.assert_array_equal(state.cms_current, current)
                np.testing.assert_array_equal(state.cms_reference, reference)


def test_xstream_large_window_counters_do_not_wrap_at_int32_limit():
    limit = np.iinfo(np.int32).max
    model = XStream(
        k=1,
        n_chains=1,
        depth=1,
        cms_width=1,
        cms_num_hashes=1,
        window_size=limit + 2,
        init_sample_size=1,
        seed=2,
    )
    model.learn_one({"x": 1.0})
    state = model._state
    state.cms_current.fill(limit)
    state.samples_in_window = limit
    model.learn_one({"x": 1.0})
    assert state.cms_current.item() == limit + 1
    model.learn_one({"x": 1.0})
    assert state.cms_reference.item() == limit + 2
    assert state.cms_current.item() == 0
    assert math.isfinite(model.score_one({"x": 1.0}))


@pytest.mark.parametrize(
    "detector", [CellNeighborhoodDetector, StationaryRegionNeighborDetector]
)
@pytest.mark.parametrize("dimensions", [1, 2, 20])
def test_radius_neighbors_churn_matches_brute_force_and_bounds_all_indexes(
    detector, dimensions
):
    model = detector(k=3, radius=1.25, window_size=19, slide_size=1)
    recent = deque(maxlen=model.window_size)
    for i in range(1000):
        vector = np.array(
            [float(i // 25 + (i * 7 + d * 3) % 17) for d in range(dimensions)]
        )
        point = {f"x{d}": value for d, value in enumerate(vector)}
        model.learn_one(point)
        recent.append(vector)
        assert len(model._window_entries) <= model.window_size
        assert len(model._cell_members) <= model.window_size
        assert sum(len(members) for members in model._cell_members.values()) == len(
            recent
        )
        assert len(model._neighbor_cache) <= 1
        assert (
            sum(len(offsets) for offsets in model._neighbor_offsets_cache.values())
            <= 20_000
        )
        if i % 40 == 0 and len(recent) > model.k:
            for offset in [0.0, 0.5, 100_000.0]:
                query = vector + offset
                expected_neighbors = sum(
                    np.dot(p - query, p - query) <= model._distance_limit_sq
                    for p in recent
                )
                expected = 1.0 - min(expected_neighbors / model.k, 1.0)
                assert (
                    model.score_one({f"x{d}": value for d, value in enumerate(query)})
                    == expected
                )
            assert len(model._neighbor_cache) <= 1
    live_ids = {entry_id for entry_id, _ in model._window_entries}
    assert live_ids == {
        entry_id for members in model._cell_members.values() for entry_id in members
    }


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev", "minkowski"])
def test_sdostream_long_turnover_uses_fixed_arrays_and_score_does_not_mutate(metric):
    model = SDOStream(k=8, x_neighbors=2, T=20, distance=metric, seed=4)
    for i in range(2000):
        point = {"x": float((i * 13) % 19), "y": float(i // 100)}
        model.learn_one(point)
        assert model.n_observers <= model.k
        assert model._observers.shape == (8, 2)
        assert model._observations.shape == (8,)
        assert model._time_added.shape == model._time_touched.shape == (8,)
        assert np.all(np.isfinite(model._observations))
        if i % 100 == 0:
            before = pickle.dumps(model)
            assert math.isfinite(model.score_one(point))
            assert pickle.dumps(model) == before


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("duplicates", [True, False])
def test_lof_retains_only_current_window_and_matches_fresh_tail_model(
    metric, duplicates
):
    model = LocalOutlierFactor(k=3, window_size=12, distance=metric)
    recent = deque(maxlen=model.window_size)
    for i in range(2000):
        point = {"x": 1.0 if duplicates else float((i * 13) % 29), "y": float(i % 3)}
        model.learn_one(point)
        recent.append(point)
        assert model.n_points == min(i + 1, model.window_size)
    fresh = LocalOutlierFactor(k=3, window_size=12, distance=metric)
    for point in recent:
        fresh.learn_one(point)
    before = pickle.dumps(model)
    for query in [{"x": 1.0, "y": 0.0}, {"x": 100.0, "y": 30.0}]:
        # An isolated query against duplicate-only neighborhoods has an infinite
        # LOF mathematically; it must agree with a fresh window and never be NaN.
        assert not math.isnan(model.score_one(query))
        assert model.score_one(query) == fresh.score_one(query)
    assert pickle.dumps(model) == before


def test_faiss_and_knn_native_indexes_follow_bounded_fifo_after_thousands_of_evictions():
    pytest.importorskip("faiss")
    from aberrant.model.distance.knn import KNN  # noqa: PLC0415
    from aberrant.utils.similar.faiss_engine import (  # noqa: PLC0415
        FaissSimilaritySearchEngine,
    )

    engine = FaissSimilaritySearchEngine(window_size=17, warm_up=3)
    model = KNN(k=3, similarity_engine=engine)
    recent = deque(maxlen=engine.window_size)
    for i in range(2000):
        point = {"x": float(i), "y": float(i % 11)}
        model.learn_one(point)
        recent.append([point["x"], point["y"]])
        assert len(engine._window) == len(recent)
        assert engine.index.ntotal == len(recent)
        if i % 100 == 0 and len(recent) >= 3:
            query = np.array([i + 0.25, 4.0])
            distances = np.linalg.norm(np.asarray(recent) - query, axis=1)
            assert model.score_one({"x": query[0], "y": query[1]}) == pytest.approx(
                np.sort(distances)[:3].mean(), rel=1e-5
            )
            assert engine.index.ntotal == len(recent)
    # The public snapshot cannot retain or mutate the authoritative vectors.
    snapshot = copy.deepcopy(engine.window)
    snapshot[-1]["x"] = -1e9
    assert engine.window[-1]["x"] == 1999.0


_BOUNDED_MODELS = [
    pytest.param(
        lambda: ASDIsolationForest(
            n_estimators=2, max_samples=16, window_size=16, seed=1
        ),
        id="asd",
    ),
    pytest.param(
        lambda: HalfSpaceTrees(n_trees=2, height=4, window_size=16, seed=1),
        id="halfspace",
    ),
    pytest.param(
        lambda: MondrianIsolationForest(
            n_estimators=2, subspace_size=2, window_size=16, seed=1
        ),
        id="mondrian-bounded",
    ),
    pytest.param(
        lambda: OnlineIsolationForest(
            num_trees=2, max_leaf_samples=4, window_size=16, seed=1
        ),
        id="online",
    ),
    pytest.param(
        lambda: RandomCutForest(n_trees=2, sample_size=16, seed=1),
        id="random-cut",
    ),
    pytest.param(
        lambda: StreamRandomHistogramForest(
            n_estimators=2, window_size=16, max_depth=4, seed=1
        ),
        id="random-histogram",
    ),
    pytest.param(
        lambda: XStream(
            k=4,
            n_chains=2,
            depth=3,
            cms_width=8,
            cms_num_hashes=2,
            window_size=16,
            init_sample_size=16,
            max_feature_cache_size=16,
            seed=1,
        ),
        id="xstream",
    ),
    pytest.param(lambda: SDOStream(k=8, x_neighbors=2, seed=1), id="sdostream"),
    pytest.param(
        lambda: CellNeighborhoodDetector(k=2, window_size=16, slide_size=1),
        id="cell-neighborhood",
    ),
    pytest.param(
        lambda: StationaryRegionNeighborDetector(k=2, window_size=16, slide_size=1),
        id="stationary-region",
    ),
    pytest.param(lambda: LocalOutlierFactor(k=2, window_size=16), id="lof"),
]


@pytest.mark.parametrize("factory", _BOUNDED_MODELS)
def test_bounded_models_retained_python_allocations_plateau(factory):
    model = factory()
    tracemalloc.start()
    try:
        for i in range(1500):
            model.learn_one({"x": float(i % 17), "y": float(i % 5)})
        gc.collect()
        initial = tracemalloc.get_traced_memory()[0]
        for i in range(1500, 3000):
            model.learn_one({"x": float(i % 17), "y": float(i % 5)})
        gc.collect()
        final = tracemalloc.get_traced_memory()[0]
        # Leave room for bounded random tree topology and allocator bookkeeping.
        # Structural tests above independently verify each retained container.
        assert final - initial < 64 * 1024
    finally:
        tracemalloc.stop()


@pytest.mark.xfail(
    strict=True,
    reason="FAISS float32 squared L2 overflows for finite representable vectors",
)
def test_faiss_large_finite_distance_matches_representable_euclidean_norm():
    pytest.importorskip("faiss")
    from aberrant.utils.similar.faiss_engine import (  # noqa: PLC0415
        FaissSimilaritySearchEngine,
    )

    engine = FaissSimilaritySearchEngine(window_size=4, warm_up=1)
    engine.append({"x": 0.0})
    assert engine.search({"x": 1e30}, 1) == pytest.approx(1e30, rel=1e-6)


@pytest.mark.xfail(
    strict=True, reason="SDOStream Euclidean squaring overflows before the norm does"
)
def test_sdostream_large_finite_distance_matches_representable_norm():
    model = SDOStream(k=2, x_neighbors=1, qv=0, T=1, seed=1)
    for _ in range(4):
        model.learn_one({"x": 0.0})
    with np.errstate(over="ignore"):
        assert model.score_one({"x": 1e200}) == pytest.approx(1e200)


@pytest.mark.xfail(
    strict=True, reason="LOF Euclidean distance overflow breaks scale invariance"
)
def test_lof_large_finite_coordinates_preserve_scale_invariant_score():
    large = LocalOutlierFactor(k=2, window_size=8)
    unit = LocalOutlierFactor(k=2, window_size=8)
    for i in range(4):
        large.learn_one({"x": float(i) * 1e200})
        unit.learn_one({"x": float(i)})
    with np.errstate(over="ignore"):
        assert large.score_one({"x": 1.5e200}) == pytest.approx(
            unit.score_one({"x": 1.5})
        )
