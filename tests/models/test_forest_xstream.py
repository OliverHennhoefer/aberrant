"""Unit tests for xStream anomaly detection model."""

import pickle
import unittest

import numpy as np

from aberrant.model.forest.xstream import XStream
from tests import score_sequences


def _reference_buckets(model: XStream, y: np.ndarray) -> np.ndarray:
    """Return the buckets of ``y`` from the per-chain traversal of aberrant 1.4.0.

    The result is shaped ``(n_hashes, depth, n_chains)``.
    """
    state = model._state
    assert state is not None
    width = model.sketch_width
    buckets = np.empty((model.n_hashes, model.depth, model.n_chains), dtype=np.int64)
    for chain in range(model.n_chains):
        z = np.zeros(model.k)
        bins = [0] * model.k
        visited: set[int] = set()
        hash_dots = np.zeros(model.n_hashes, dtype=np.int64)
        for level in range(model.depth):
            dim = int(state.chain_dims[chain, level])
            if dim in visited:
                z[dim] = 2.0 * z[dim] - state.shift[chain, dim] / state.deltamax[dim]
            else:
                visited.add(dim)
                z[dim] = (y[dim] + state.shift[chain, dim]) / state.deltamax[dim]
            bin_new = int(np.floor(z[dim]))
            step = (bin_new - bins[dim]) % width
            bins[dim] = bin_new
            hash_dots = (hash_dots + step * state.hash_coeffs_mod[:, dim]) % width
            buckets[:, level, chain] = (
                hash_dots + state.hash_offsets_mod[chain, level]
            ) % width
    return buckets


class TestXStream(unittest.TestCase):
    """Test suite for XStream model."""

    def create_model(self) -> XStream:
        return XStream(
            k=16,
            n_chains=8,
            depth=6,
            sketch_width=128,
            n_hashes=3,
            window_size=16,
            warm_up=16,
            density=0.25,
            max_feature_cache_size=32,
            seed=42,
        )

    def test_initialization_defaults(self):
        """Test default initialization."""
        model = XStream()
        self.assertEqual(model.k, 100)
        self.assertEqual(model.n_chains, 100)
        self.assertEqual(model.depth, 15)
        self.assertEqual(model.window_size, 256)
        self.assertEqual(model.max_feature_cache_size, 10_000)
        self.assertIsNone(model._state)

    def test_invalid_parameters(self):
        """Test parameter validation."""
        with self.assertRaises(ValueError):
            XStream(k=0)
        with self.assertRaises(ValueError):
            XStream(n_chains=0)
        with self.assertRaises(ValueError):
            XStream(depth=0)
        with self.assertRaises(ValueError):
            XStream(sketch_width=0)
        with self.assertRaises(ValueError):
            XStream(n_hashes=0)
        with self.assertRaises(ValueError):
            XStream(window_size=0)
        with self.assertRaises(ValueError):
            XStream(warm_up=0)
        with self.assertRaises(ValueError):
            XStream(density=0.0)
        with self.assertRaises(ValueError):
            XStream(density=1.5)
        with self.assertRaises(ValueError):
            XStream(max_feature_cache_size=0)

    def test_score_zero_before_ready(self):
        """score_one should return 0.0 before model warm-up is complete."""
        model = self.create_model()
        score = model.score_one({"x": 1.0, "y": 2.0})
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

    def test_large_finite_values_do_not_overflow(self):
        """Large finite values should not trigger integer overflow."""
        model = XStream(
            k=8,
            n_chains=4,
            depth=4,
            sketch_width=64,
            n_hashes=2,
            window_size=8,
            warm_up=8,
            density=0.5,
            seed=7,
        )
        for i in range(16):
            model.learn_one({"x": float(i), "y": 0.0})

        large_point = {"x": 1e30, "y": 0.0}
        score = model.score_one(large_point)
        model.learn_one(large_point)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_deterministic_with_seed(self):
        """Same seed and data should produce identical scores."""
        model1 = self.create_model()
        model2 = self.create_model()

        data = [{"x": float(i % 5), "y": float(i % 3)} for i in range(64)]
        for point in data:
            model1.learn_one(point)
            model2.learn_one(point)

        test_point = {"x": 2.0, "y": 1.0}
        self.assertAlmostEqual(
            model1.score_one(test_point), model2.score_one(test_point), places=12
        )

    def test_feature_evolving_support(self):
        """Model should handle changing feature keys over time."""
        model = self.create_model()
        for i in range(80):
            point = {"base": float(i % 5)}
            if i % 3 == 0:
                point["f_a"] = float(i)
            if i % 5 == 0:
                point["f_b"] = float(i * 0.1)
            model.learn_one(point)

        score = model.score_one({"base": 1.0, "f_new": 8.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_feature_cache_is_bounded_lru(self):
        """Feature cache should evict least recently used mappings."""
        model = XStream(
            k=8,
            n_chains=4,
            depth=4,
            sketch_width=64,
            n_hashes=2,
            window_size=8,
            warm_up=8,
            density=0.5,
            max_feature_cache_size=3,
            seed=7,
        )

        model._feature_projection("a")
        model._feature_projection("b")
        model._feature_projection("c")
        model._feature_projection("a")  # refresh "a"
        model._feature_projection("d")  # should evict "b"

        self.assertEqual(len(model._feature_cache), 3)
        self.assertNotIn("b", model._feature_cache)
        self.assertIn("a", model._feature_cache)
        self.assertIn("c", model._feature_cache)
        self.assertIn("d", model._feature_cache)

    def test_window_rollover_sets_reference(self):
        """After enough samples, reference sketch should become available."""
        model = XStream(
            k=8,
            n_chains=4,
            depth=4,
            sketch_width=64,
            n_hashes=2,
            window_size=8,
            warm_up=8,
            density=0.5,
            seed=7,
        )
        for i in range(8):
            model.learn_one({"x": float(i), "y": float(i + 1)})

        self.assertIsNotNone(model._state)
        assert model._state is not None
        self.assertTrue(model._state.reference_ready)
        self.assertEqual(model._state.samples_in_window, 0)

    def test_outlier_scores_higher_than_normal(self):
        """Outlier should score higher than near-cluster point."""
        model = self.create_model()

        rng = np.random.default_rng(123)
        for _ in range(200):
            normal = {
                "x": float(rng.normal(0.0, 0.3)),
                "y": float(rng.normal(0.0, 0.3)),
            }
            model.learn_one(normal)

        normal_score = model.score_one({"x": 0.05, "y": -0.02})
        outlier_score = model.score_one({"x": 6.0, "y": 6.0})

        self.assertGreater(outlier_score, normal_score)

    def test_reset_restores_state(self):
        """reset() should clear learned state."""
        model = self.create_model()
        for i in range(50):
            model.learn_one({"x": float(i), "y": float(i * 2)})

        self.assertIsNotNone(model._state)
        model.reset()
        self.assertIsNone(model._state)
        self.assertEqual(model.score_one({"x": 1.0, "y": 2.0}), 0.0)

    def test_repr_contains_key_config(self):
        """repr should include key hyperparameters."""
        model = self.create_model()
        output = repr(model)
        self.assertIn("XStream", output)
        self.assertIn("k=16", output)
        self.assertIn("n_chains=8", output)


class TestXStreamExactness(unittest.TestCase):
    """Descending all chains together must reproduce the per-chain traversal."""

    create_model = TestXStream.create_model

    def test_score_sequences_match_recorded_reference(self):
        for case in score_sequences.load("xstream"):
            with self.subTest(case=case["name"]):
                model = XStream(**case["params"])
                events = score_sequences.make_stream(**case["stream"])
                scores = score_sequences.score_then_learn(model, events)
                # The tolerance only absorbs last-bit libm differences between
                # platforms; the recording platform reproduces every bit.
                np.testing.assert_allclose(scores, case["scores"], rtol=1e-12, atol=0)

    def test_sketch_cells_match_the_per_chain_traversal(self):
        configurations = [
            # Three dimensions over twelve levels revisit every dimension.
            {"k": 3, "n_chains": 5, "depth": 12, "sketch_width": 37, "density": 1.0},
            {"k": 16, "n_chains": 7, "depth": 9, "sketch_width": 64},
            {"k": 6, "n_chains": 4, "depth": 6, "sketch_width": 1},
            # Widths whose hash sums need int64, a power of two and not.
            {"k": 4, "n_chains": 3, "depth": 5, "sketch_width": 1 << 15},
            {"k": 4, "n_chains": 3, "depth": 5, "sketch_width": 40_009},
        ]
        rng = np.random.default_rng(5)
        for params in configurations:
            model = XStream(n_hashes=3, window_size=10, warm_up=10, seed=2, **params)
            for row in rng.normal(size=(10, 3)):
                model.learn_one({"a": row[0], "b": row[1], "c": row[2]})
            state = model._state
            assert state is not None
            rows = np.arange(model.n_chains * model.depth * model.n_hashes)
            rows = rows.reshape(model.n_chains, model.depth, -1).transpose(2, 1, 0)
            for scale in (1.0, -3.0, 1e6, 1e30, -1e300):
                y = rng.normal(size=model.k) * scale
                with self.subTest(params=params, scale=scale):
                    cells = model._sketch_cells(state, y)
                    np.testing.assert_array_equal(cells // model.sketch_width, rows)
                    np.testing.assert_array_equal(
                        cells % model.sketch_width, _reference_buckets(model, y)
                    )

    def test_reduced_hash_terms_give_the_same_cells(self):
        model = XStream(k=8, n_chains=6, depth=10, sketch_width=97, seed=3)
        events = score_sequences.make_stream(seed=4, n_events=300, n_features=3)
        for event in events:
            model.learn_one(event)
        state = model._state
        assert state is not None
        self.assertFalse(state.levels.reduce_terms)
        for event in events[:20]:
            y = model._project_one(event)
            state.last_cells = None
            state.levels.reduce_terms = False
            unreduced = model._sketch_cells(state, y)
            state.last_cells = None
            state.levels.reduce_terms = True
            np.testing.assert_array_equal(model._sketch_cells(state, y), unreduced)

    def test_chain_score_table_gives_the_same_scores(self):
        events = score_sequences.make_stream(seed=7, n_events=200, n_features=4)
        tabled = XStream(k=8, n_chains=6, depth=5, window_size=20, warm_up=20, seed=1)
        computed = XStream(k=8, n_chains=6, depth=5, window_size=20, warm_up=20, seed=1)
        tabled_scores = []
        computed_scores = []
        for event in events:
            if computed._state is not None:
                computed._state.levels.chain_scores = None
            tabled_scores.append(tabled.score_one(event))
            computed_scores.append(computed.score_one(event))
            tabled.learn_one(event)
            computed.learn_one(event)
        assert tabled._state is not None
        self.assertIsNotNone(tabled._state.levels.chain_scores)
        self.assertGreater(len(set(tabled_scores)), 10)
        self.assertEqual(tabled_scores, computed_scores)

    def test_learning_after_scoring_another_point_counts_the_learned_point(self):
        events = score_sequences.make_stream(seed=5, n_events=80, n_features=3)
        scored_first = self.create_model()
        learned_only = self.create_model()
        for event in events[:40]:
            scored_first.learn_one(event)
            learned_only.learn_one(event)
        for event, other in zip(events[40:], reversed(events[40:]), strict=True):
            scored_first.score_one(other)
            scored_first.learn_one(event)
            learned_only.learn_one(event)
        assert scored_first._state is not None and learned_only._state is not None
        np.testing.assert_array_equal(
            scored_first._state.cms_current, learned_only._state.cms_current
        )
        np.testing.assert_array_equal(
            scored_first._state.cms_reference, learned_only._state.cms_reference
        )

    def test_overflowing_bins_raise_without_counting(self):
        model = XStream(
            k=2,
            n_chains=3,
            depth=4,
            sketch_width=16,
            n_hashes=2,
            window_size=4,
            warm_up=4,
            density=1.0,
            seed=1,
        )
        for i in range(6):
            model.learn_one({"x": 0.01 * i})
        state = model._state
        assert state is not None
        current = state.cms_current.copy()
        samples_seen = state.samples_seen
        with np.errstate(over="ignore"):
            # As in the per-chain traversal, an infinite bin cannot become a
            # bucket.
            with self.assertRaises(OverflowError):
                model.score_one({"x": 1.7e308})
            with self.assertRaises(OverflowError):
                model.learn_one({"x": 1.7e308})
        np.testing.assert_array_equal(state.cms_current, current)
        self.assertEqual(state.samples_seen, samples_seen)

    def test_counters_are_the_smallest_type_that_holds_a_window(self):
        for window_size, dtype in [
            (127, np.int8),
            (128, np.int16),
            (32_767, np.int16),
            (32_768, np.int32),
        ]:
            with self.subTest(window_size=window_size):
                model = XStream(
                    k=2,
                    n_chains=1,
                    depth=1,
                    sketch_width=2,
                    n_hashes=1,
                    window_size=window_size,
                    warm_up=1,
                    seed=1,
                )
                model.learn_one({"x": 1.0})
                assert model._state is not None
                self.assertEqual(model._state.cms_current.dtype, dtype)
                self.assertEqual(model._state.cms_reference.dtype, dtype)

    def test_pickle_stores_window_points_and_recounts_the_sketches(self):
        rng = np.random.default_rng(5)
        events = [
            {"a": float(a), "b": float(b), "c": float(c)}
            for a, b, c in rng.normal(size=(120, 3))
        ]
        for learned in (5, 16, 40, 85):
            with self.subTest(learned=learned):
                model = self.create_model()
                for event in events[:learned]:
                    model.learn_one(event)
                restored = pickle.loads(pickle.dumps(model))

                if model._state is not None:
                    assert restored._state is not None
                    np.testing.assert_array_equal(
                        restored._state.cms_current, model._state.cms_current
                    )
                    np.testing.assert_array_equal(
                        restored._state.cms_reference, model._state.cms_reference
                    )
                for event in events[learned:]:
                    self.assertEqual(restored.score_one(event), model.score_one(event))
                    restored.learn_one(event)
                    model.learn_one(event)

    def test_windows_larger_than_a_sketch_pickle_the_sketches(self):
        model = XStream(
            k=16,
            n_chains=2,
            depth=3,
            sketch_width=8,
            n_hashes=1,
            window_size=64,
            warm_up=16,
            seed=4,
        )
        rng = np.random.default_rng(7)
        events = [{"a": float(a), "b": float(b)} for a, b in rng.normal(size=(200, 2))]
        for event in events[:150]:
            model.learn_one(event)
        assert model._state is not None
        self.assertIsNone(model._state.current_points)

        restored = pickle.loads(pickle.dumps(model))
        assert restored._state is not None
        np.testing.assert_array_equal(
            restored._state.cms_reference, model._state.cms_reference
        )
        for event in events[150:]:
            self.assertEqual(restored.score_one(event), model.score_one(event))
            restored.learn_one(event)
            model.learn_one(event)

    def test_pickle_is_far_smaller_than_the_sketches(self):
        model = XStream(seed=3)
        rng = np.random.default_rng(6)
        for row in rng.normal(size=(600, 10)):
            model.learn_one({f"f{index}": float(v) for index, v in enumerate(row)})
        assert model._state is not None
        sketches = model._state.cms_current.nbytes + model._state.cms_reference.nbytes
        points = 2 * model.window_size * model.k * 8

        size = len(pickle.dumps(model))
        self.assertLess(size, 2 * points)
        self.assertLess(size, sketches / 20)


if __name__ == "__main__":
    unittest.main()
