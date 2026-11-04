"""Unit tests for Random Cut Forest anomaly detection model."""

import pickle
import time
import unittest

import numpy as np

from aberrant.model.iforest.random_cut import (
    RandomCutForest,
    _RandomCutTree,
    _RCFBranch,
    _RCFLeaf,
    _RCFNode,
    _search_right,
    _weighted_choice,
)
from tests import score_sequences


def _tree_shape(node: _RCFNode | None) -> object:
    """Return a comparable description of a tree's nodes."""
    if node is None:
        return None
    if isinstance(node, _RCFBranch):
        return (
            node.cut_dim,
            node.cut_value,
            node.size,
            node.box.tolist(),
            _tree_shape(node.left),
            _tree_shape(node.right),
        )
    return (node.point.tolist(), list(node.point_ids), node.size)


def _forest_shape(model: RandomCutForest) -> list[object]:
    """Return every tree's shape with its generator state."""
    return [
        (_tree_shape(tree.root), tree.rng.bit_generator.state) for tree in model._trees
    ]


def _leaf_points(node: _RCFNode) -> list[np.ndarray]:
    """Return the points stored under ``node``."""
    if isinstance(node, _RCFLeaf):
        return [node.point]
    return _leaf_points(node.left) + _leaf_points(node.right)


class TestRandomCutForest(unittest.TestCase):
    """Test suite for RandomCutForest model."""

    def create_model(self) -> RandomCutForest:
        return RandomCutForest(
            n_trees=8,
            sample_size=32,
            shingle_size=1,
            warmup_samples=16,
            normalize_score=True,
            score_scale=8.0,
            seed=42,
        )

    def test_initialization_defaults(self):
        """Test default initialization."""
        model = RandomCutForest()
        self.assertEqual(model.n_trees, 40)
        self.assertEqual(model.sample_size, 256)
        self.assertEqual(model.shingle_size, 1)
        self.assertEqual(model.warmup_samples, 256)
        self.assertFalse(model.normalize_score)
        self.assertFalse(model._ready)

    def test_invalid_parameters(self):
        """Test parameter validation."""
        with self.assertRaises(ValueError):
            RandomCutForest(n_trees=0)
        with self.assertRaises(ValueError):
            RandomCutForest(sample_size=1)
        with self.assertRaises(ValueError):
            RandomCutForest(shingle_size=0)
        with self.assertRaises(ValueError):
            RandomCutForest(warmup_samples=0)
        with self.assertRaises(ValueError):
            RandomCutForest(score_scale=0.0)

    def test_score_zero_before_warmup(self):
        """score_one should return 0.0 before warm-up is complete."""
        model = self.create_model()
        for i in range(15):
            model.learn_one({"x": float(i), "y": float(i)})
        score = model.score_one({"x": 0.0, "y": 0.0})
        self.assertEqual(score, 0.0)

    def test_empty_and_non_numeric_input(self):
        """Model should reject empty and non-numeric inputs."""
        model = self.create_model()
        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.score_one({})
        with self.assertRaises(ValueError):
            model.learn_one({"x": "bad"})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.score_one({"x": float("inf")})
        with self.assertRaises(ValueError):
            model.score_one({"x": float("nan")})

    def test_feature_key_consistency(self):
        """Model should enforce consistent feature keys."""
        model = self.create_model()
        for i in range(20):
            model.learn_one({"a": float(i), "b": float(i + 1)})

        with self.assertRaises(ValueError):
            model.score_one({"a": 1.0, "c": 2.0})
        with self.assertRaises(ValueError):
            model.learn_one({"a": 1.0, "c": 2.0})

    def test_deterministic_with_seed(self):
        """Same seed and data should produce identical scores."""
        model_1 = self.create_model()
        model_2 = self.create_model()

        data = [{"x": float(i % 7), "y": float(i % 3)} for i in range(64)]
        for point in data:
            model_1.learn_one(point)
            model_2.learn_one(point)

        test_point = {"x": 2.0, "y": 1.0}
        self.assertAlmostEqual(
            model_1.score_one(test_point),
            model_2.score_one(test_point),
            places=12,
        )

    def test_bounded_memory(self):
        """Stored point count should be bounded by sample_size."""
        model = RandomCutForest(
            n_trees=4,
            sample_size=20,
            warmup_samples=10,
            seed=7,
        )
        for i in range(200):
            model.learn_one({"x": float(i), "y": float(i * 2)})
            self.assertLessEqual(len(model._id_window), 20)
            for tree in model._trees:
                self.assertLessEqual(tree.size, 20)

    def test_outlier_scores_higher_than_normal(self):
        """Outlier should score higher than near-cluster point."""
        model = RandomCutForest(
            n_trees=16,
            sample_size=64,
            warmup_samples=32,
            seed=11,
        )
        rng = np.random.default_rng(123)
        for _ in range(200):
            model.learn_one(
                {
                    "x": float(rng.normal(0.0, 0.25)),
                    "y": float(rng.normal(0.0, 0.25)),
                }
            )

        normal_score = model.score_one({"x": 0.05, "y": -0.02})
        outlier_score = model.score_one({"x": 6.0, "y": 6.0})
        self.assertGreater(outlier_score, normal_score)

    def test_tree_score_matches_reference_insert_codisp_remove(self):
        """score_point should equal candidate-inclusive reference CoDisp."""
        points = [
            np.asarray([0.0, 0.0]),
            np.asarray([0.1, 0.0]),
            np.asarray([0.0, 0.1]),
            np.asarray([3.0, 3.0]),
        ]
        preview_tree = _RandomCutTree(np.random.default_rng(42))
        reference_tree = _RandomCutTree(np.random.default_rng(42))
        for point_id, point in enumerate(points):
            preview_tree.insert(point_id, point)
            reference_tree.insert(point_id, point)

        query = np.asarray([8.0, 8.0])
        preview_score = preview_tree.score_point(query)
        reference_tree.insert(99, query)
        reference_score = reference_tree.codisp(99)

        self.assertEqual(preview_score, reference_score)
        self.assertEqual(preview_tree.size, len(points))
        self.assertNotIn(-1, preview_tree._id_to_leaf)

    def test_tree_score_preview_restores_rng_state(self):
        """Scoring must not change the random cuts used by future inserts."""
        scored_tree = _RandomCutTree(np.random.default_rng(7))
        control_tree = _RandomCutTree(np.random.default_rng(7))
        for point_id in range(8):
            point = np.asarray([float(point_id), float(point_id % 3)])
            scored_tree.insert(point_id, point)
            control_tree.insert(point_id, point)

        scored_tree.score_point(np.asarray([20.0, 20.0]))
        new_point = np.asarray([9.0, 4.0])
        scored_tree.insert(9, new_point)
        control_tree.insert(9, new_point)

        self.assertEqual(scored_tree.codisp(9), control_tree.codisp(9))

    def test_shingle_support(self):
        """Model should support shingled streaming state."""
        model = RandomCutForest(
            n_trees=6,
            sample_size=24,
            shingle_size=3,
            warmup_samples=8,
            seed=5,
        )
        for i in range(50):
            model.learn_one({"x": float(i), "y": float(i % 4)})

        score = model.score_one({"x": 51.0, "y": 1.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_raw_score_mode(self):
        """Raw score mode should return a non-negative continuous score."""
        model = RandomCutForest(
            n_trees=8,
            sample_size=32,
            warmup_samples=16,
            normalize_score=False,
            seed=42,
        )
        for i in range(50):
            model.learn_one({"x": float(i % 5), "y": float(i % 3)})

        score = model.score_one({"x": 10.0, "y": -10.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_reset_restores_state(self):
        """reset() should clear learned state."""
        model = self.create_model()
        for i in range(40):
            model.learn_one({"x": float(i), "y": float(i + 1)})

        self.assertTrue(model._ready)
        model.reset()
        self.assertFalse(model._ready)
        self.assertEqual(model.score_one({"x": 1.0, "y": 2.0}), 0.0)

    def test_repr_contains_key_config(self):
        """repr should include key hyperparameters."""
        model = self.create_model()
        output = repr(model)
        self.assertIn("RandomCutForest", output)
        self.assertIn("n_trees=8", output)
        self.assertIn("sample_size=32", output)


class TestRandomCutForestExactness(unittest.TestCase):
    """Planned walks must reproduce the 1.4.0 trees, generators and scores."""

    def test_score_sequences_match_recorded_reference(self):
        for case in score_sequences.load("random_cut_forest"):
            with self.subTest(case=case["name"]):
                scores = score_sequences.replay(RandomCutForest(**case["params"]), case)
                if case["params"].get("normalize_score"):
                    # Raw CoDisp is plain arithmetic, but np.exp can differ in
                    # the last bit between CPUs (SIMD and libm loops). The
                    # recording platform reproduces every bit.
                    np.testing.assert_allclose(
                        scores, case["scores"], rtol=1e-12, atol=0
                    )
                else:
                    self.assertEqual(scores, case["scores"])

    def test_search_right_matches_searchsorted(self):
        rng = np.random.default_rng(5)
        spans = rng.exponential(size=(40, 6))
        spans[rng.random(spans.shape) < 0.3] = 0.0
        spans[1] = 0.0
        spans[2, 4] = np.inf
        spans[3, [0, 5]] = np.inf
        cumulative = np.cumsum(spans, axis=1)
        targets = rng.random(len(spans)) * cumulative[:, -1]
        targets[4] = cumulative[4, 2]
        targets[5] = 0.0
        targets[2] = np.nan
        targets[3] = np.inf

        found = _search_right(cumulative, targets).tolist()

        for row, (values, target) in enumerate(zip(cumulative, targets, strict=True)):
            expected = int(np.searchsorted(values, target, side="right"))
            self.assertEqual(found[row], expected, msg=f"row {row}")

    def test_weighted_choice_matches_generator_choice(self):
        source = np.random.default_rng(6)
        for trial in range(300):
            first = source.normal(size=1 + trial % 9)
            second = first + source.normal(size=first.size) * 10.0 ** (trial % 7 - 3)
            if trial % 5 == 0:
                second[: first.size // 2] = first[: first.size // 2]
            if trial % 11 == 0:
                first, second = first * 1e-310, second * 1e-310
            diffs = np.abs(first - second)
            total = float(diffs.sum())
            if total <= 0.0:
                continue
            seed = int(source.integers(2**32))
            reference = np.random.default_rng(seed)
            emulated = np.random.default_rng(seed)

            expected = int(reference.choice(diffs.size, p=diffs / total))
            chosen = _weighted_choice(emulated, diffs / total)

            self.assertEqual(chosen, expected)
            self.assertEqual(
                emulated.bit_generator.state, reference.bit_generator.state
            )

    def test_learning_a_scored_point_matches_learning_it_directly(self):
        for kind in ("gaussian", "grid"):
            with self.subTest(kind=kind):
                events = score_sequences.make_stream(
                    seed=7, n_events=260, n_features=3, kind=kind
                )
                params = {"n_trees": 6, "sample_size": 24, "warmup_samples": 4}
                scored = RandomCutForest(**params, seed=8)
                learned = RandomCutForest(**params, seed=8)
                scored_elsewhere = RandomCutForest(**params, seed=8)
                for index, event in enumerate(events):
                    scored.score_one(event)
                    scored.learn_one(event)
                    learned.learn_one(event)
                    # Walks planned for another point must not be reused.
                    scored_elsewhere.score_one(events[-1 - index])
                    scored_elsewhere.learn_one(event)

                self.assertEqual(_forest_shape(scored), _forest_shape(learned))
                self.assertEqual(
                    _forest_shape(scored_elsewhere), _forest_shape(learned)
                )

    def test_branch_boxes_bound_their_points(self):
        events = score_sequences.make_stream(seed=11, n_events=400, n_features=3)
        model = RandomCutForest(n_trees=3, sample_size=30, warmup_samples=5, seed=12)
        for event in events:
            model.score_one(event)
            model.learn_one(event)
            for tree in model._trees:
                branches = [tree.root]
                while branches:
                    node = branches.pop()
                    if not isinstance(node, _RCFBranch):
                        continue
                    points = np.array(_leaf_points(node))
                    np.testing.assert_array_equal(node.box[0], points.min(axis=0))
                    np.testing.assert_array_equal(-node.box[1], points.max(axis=0))
                    branches.extend((node.left, node.right))

    def test_pickles_leave_out_the_plan(self):
        events = score_sequences.make_stream(seed=9, n_events=80, n_features=2)
        model = RandomCutForest(n_trees=4, sample_size=20, warmup_samples=5, seed=10)
        for event in events[:60]:
            model.learn_one(event)
        before = pickle.dumps(model)

        score = model.score_one(events[60])

        self.assertEqual(pickle.dumps(model), before)
        restored = pickle.loads(pickle.dumps(model))
        self.assertIsNone(restored._planned)
        self.assertEqual(restored.score_one(events[60]), score)
        model.learn_one(events[60])
        restored.learn_one(events[60])
        self.assertEqual(_forest_shape(restored), _forest_shape(model))

    def test_default_forest_speed_smoke(self):
        events = score_sequences.make_stream(seed=1, n_events=600, n_features=8)
        model = RandomCutForest(seed=1)

        start = time.perf_counter()
        score_sequences.score_then_learn(model, events)

        # Generous bound against large regressions; typical runs take ~2 s.
        self.assertLess(time.perf_counter() - start, 20.0)


if __name__ == "__main__":
    unittest.main()
