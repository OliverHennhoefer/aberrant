"""Unit tests for xStream anomaly detection model."""

import time
import unittest

import numpy as np

from aberrant.model.iforest.xstream import XStream
from tests import score_sequences


def _reference_buckets(model: XStream, y: np.ndarray) -> np.ndarray:
    """Return the buckets of ``y`` from the per-chain traversal of aberrant 1.4.0.

    The result is shaped ``(cms_num_hashes, depth, n_chains)``.
    """
    state = model._state
    assert state is not None
    width = model.cms_width
    buckets = np.empty(
        (model.cms_num_hashes, model.depth, model.n_chains), dtype=np.int64
    )
    for chain in range(model.n_chains):
        z = np.zeros(model.k)
        bins = [0] * model.k
        visited: set[int] = set()
        hash_dots = np.zeros(model.cms_num_hashes, dtype=np.int64)
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
            cms_width=128,
            cms_num_hashes=3,
            window_size=16,
            init_sample_size=16,
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
            XStream(cms_width=0)
        with self.assertRaises(ValueError):
            XStream(cms_num_hashes=0)
        with self.assertRaises(ValueError):
            XStream(window_size=0)
        with self.assertRaises(ValueError):
            XStream(init_sample_size=0)
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
            cms_width=64,
            cms_num_hashes=2,
            window_size=8,
            init_sample_size=8,
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
            cms_width=64,
            cms_num_hashes=2,
            window_size=8,
            init_sample_size=8,
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
            cms_width=64,
            cms_num_hashes=2,
            window_size=8,
            init_sample_size=8,
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
            {"k": 3, "n_chains": 5, "depth": 12, "cms_width": 37, "density": 1.0},
            {"k": 16, "n_chains": 7, "depth": 9, "cms_width": 64},
            {"k": 6, "n_chains": 4, "depth": 6, "cms_width": 1},
        ]
        rng = np.random.default_rng(5)
        for params in configurations:
            model = XStream(
                cms_num_hashes=3, window_size=10, init_sample_size=10, seed=2, **params
            )
            for row in rng.normal(size=(10, 3)):
                model.learn_one({"a": row[0], "b": row[1], "c": row[2]})
            state = model._state
            assert state is not None
            rows = np.arange(model.n_chains * model.depth * model.cms_num_hashes)
            rows = rows.reshape(model.n_chains, model.depth, -1).transpose(2, 1, 0)
            for scale in (1.0, -3.0, 1e6, 1e30, -1e300):
                y = rng.normal(size=model.k) * scale
                with self.subTest(params=params, scale=scale):
                    cells = model._sketch_cells(state, y)
                    np.testing.assert_array_equal(cells // model.cms_width, rows)
                    np.testing.assert_array_equal(
                        cells % model.cms_width, _reference_buckets(model, y)
                    )

    def test_reduced_hash_terms_give_the_same_cells(self):
        model = XStream(k=8, n_chains=6, depth=10, cms_width=97, seed=3)
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
            cms_width=16,
            cms_num_hashes=2,
            window_size=4,
            init_sample_size=4,
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
                    cms_width=2,
                    cms_num_hashes=1,
                    window_size=window_size,
                    init_sample_size=1,
                    seed=1,
                )
                model.learn_one({"x": 1.0})
                assert model._state is not None
                self.assertEqual(model._state.cms_current.dtype, dtype)
                self.assertEqual(model._state.cms_reference.dtype, dtype)

    def test_default_model_processes_two_thousand_events_quickly(self):
        events = score_sequences.make_stream(seed=1, n_events=2000, n_features=8)
        model = XStream(seed=1)

        start = time.perf_counter()
        score_sequences.score_then_learn(model, events)

        # Generous bound against large regressions; typical runs take ~0.5 s.
        self.assertLess(time.perf_counter() - start, 5.0)


if __name__ == "__main__":
    unittest.main()
