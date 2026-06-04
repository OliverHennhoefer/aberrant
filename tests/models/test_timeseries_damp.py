"""Unit tests for pure-online X-Lag Amnesic DAMP."""

import math
import unittest

import numpy as np

from aberrant.model.timeseries import XLagDAMP
from aberrant.model.timeseries.damp import _mass_distance_profile, _next_power_of_two


def _brute_force_distance_profile(
    series: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Compute the z-normalized Euclidean profile directly."""
    windows = np.lib.stride_tricks.sliding_window_view(series, query.size)
    normalized_windows = (windows - windows.mean(axis=1, keepdims=True)) / windows.std(
        axis=1,
        keepdims=True,
    )
    normalized_query = (query - query.mean()) / query.std()
    return np.linalg.norm(normalized_windows - normalized_query, axis=1)


def _reference_x_lag_damp(
    series: np.ndarray,
    subsequence_length: int,
    start_index: int,
    x_lag: int,
) -> tuple[np.ndarray, float]:
    """Direct translation of the authors' backward-only MATLAB loop."""
    left_mp = np.zeros(series.size, dtype=np.float64)
    best_so_far = float("-inf")
    initial_search_length = _next_power_of_two(8 * subsequence_length)

    for matlab_index in range(
        start_index,
        series.size - subsequence_length + 2,
    ):
        query_start = matlab_index - 1
        query = series[query_start : query_start + subsequence_length]
        approximate_distance = float("inf")
        search_length = initial_search_length
        first_iteration = True
        expansion_num = 0

        while approximate_distance >= best_so_far:
            far_start = matlab_index - search_length + 1 + (
                expansion_num * subsequence_length
            )
            if (
                far_start <= matlab_index - x_lag
                or far_start < 1
                or x_lag < search_length
            ):
                segment_start = max(0, matlab_index - x_lag - 1)
                segment = series[segment_start:matlab_index]
                approximate_distance = float(
                    np.min(
                        _mass_distance_profile(
                            segment,
                            query,
                            eps=1e-12,
                        )
                    )
                )
                left_mp[query_start] = approximate_distance
                best_so_far = max(best_so_far, approximate_distance)
                break

            if first_iteration:
                first_iteration = False
                segment = series[
                    matlab_index - search_length : matlab_index
                ]
            else:
                segment_start = (
                    matlab_index
                    - search_length
                    + expansion_num * subsequence_length
                )
                segment_end = (
                    matlab_index
                    - search_length // 2
                    + expansion_num * subsequence_length
                )
                segment = series[segment_start:segment_end]

            approximate_distance = float(
                np.min(_mass_distance_profile(segment, query, eps=1e-12))
            )
            if approximate_distance < best_so_far:
                left_mp[query_start] = approximate_distance
                break

            search_length *= 2
            expansion_num += 1

    return left_mp, best_so_far


class TestMassDistanceProfile(unittest.TestCase):
    """Verify the NumPy MASS_V2 translation against its direct formula."""

    def test_matches_brute_force_z_normalized_distances(self) -> None:
        series = np.array([1.0, 2.0, 4.0, 3.0, 0.0, 1.0, 5.0, 2.0])
        query = np.array([0.0, 1.0, 5.0, 2.0])

        actual = _mass_distance_profile(series, query, eps=1e-12)
        expected = _brute_force_distance_profile(series, query)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_large_numeric_offset_preserves_distance_precision(self) -> None:
        offset = 1e12
        series = offset + np.array([1.0, 2.0, 4.0, 3.0, 0.0, 1.0, 5.0, 2.0])
        query = offset + np.array([0.0, 1.0, 5.0, 2.0])

        actual = _mass_distance_profile(series, query, eps=1e-12)
        expected = _brute_force_distance_profile(series, query)

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_rejects_constant_query_and_candidate_subsequences(self) -> None:
        with self.assertRaises(ValueError):
            _mass_distance_profile(
                np.array([0.0, 1.0, 2.0, 3.0]),
                np.array([1.0, 1.0]),
                eps=1e-12,
            )

        with self.assertRaises(ValueError):
            _mass_distance_profile(
                np.array([0.0, 0.0, 1.0, 2.0]),
                np.array([1.0, 2.0]),
                eps=1e-12,
            )


class TestXLagDAMP(unittest.TestCase):
    """Test the author-grounded pure-online X-Lag DAMP implementation."""

    def create_model(self, **overrides: object) -> XLagDAMP:
        defaults: dict[str, object] = {
            "subsequence_length": 8,
            "x_lag": 64,
            "start_index": 32,
        }
        defaults.update(overrides)
        return XLagDAMP(**defaults)

    def test_initialization_defaults_follow_author_source(self) -> None:
        model = XLagDAMP(subsequence_length=16)

        self.assertEqual(model.x_lag, 16 * 16)
        self.assertEqual(model.start_index, 4 * 16)
        self.assertEqual(model._initial_search_length, 128)
        self.assertEqual(model._history_capacity, model.x_lag + 15)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            XLagDAMP(subsequence_length=1)
        with self.assertRaises(ValueError):
            XLagDAMP(subsequence_length=8, x_lag=7)
        with self.assertRaises(ValueError):
            XLagDAMP(subsequence_length=8, start_index=7)
        with self.assertRaises(ValueError):
            XLagDAMP(subsequence_length=8, eps=0.0)

    def test_input_validation_and_feature_locking(self) -> None:
        model = self.create_model()

        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.learn_one({"x": 1.0, "y": 2.0})
        with self.assertRaises(ValueError):
            model.learn_one({"x": "bad"})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.learn_one({"x": float("nan")})

        model.learn_one({"x": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"y": 2.0})
        with self.assertRaises(ValueError):
            model.score_one({"y": 2.0})

    def test_score_is_zero_before_start_index(self) -> None:
        model = self.create_model()
        for index in range(20):
            value = float(np.sin(index))
            self.assertEqual(model.score_one({"value": value}), 0.0)
            model.learn_one({"value": value})

        self.assertFalse(model.is_ready)
        self.assertEqual(model.n_subsequences_processed, 0)

    def test_first_processed_query_matches_exact_left_discord(self) -> None:
        model = self.create_model(
            subsequence_length=4,
            x_lag=32,
            start_index=8,
        )
        history = np.array([0.2, 1.0, -0.4, 0.7, 1.4, -0.8, 0.1, 1.2, -0.2, 0.9])
        for value in history:
            model.learn_one({"value": float(value)})

        current = 1.7
        combined = np.append(history, current)
        query_start = combined.size - model.subsequence_length
        query = combined[query_start:]
        expected = float(
            np.min(
                _brute_force_distance_profile(
                    combined[: query_start + 1],
                    query,
                )
            )
        )

        self.assertAlmostEqual(model.score_one({"value": current}), expected, places=10)

    def test_score_one_is_non_mutating_and_matches_subsequent_learning(self) -> None:
        model = self.create_model()
        for index in range(50):
            model.learn_one({"value": float(np.sin(2.0 * np.pi * index / 8.0))})

        state_before = (
            tuple(model._history),
            model.n_samples_seen,
            model.n_subsequences_processed,
            model.best_score,
            model.last_score,
        )
        query = {"value": 0.35}
        score = model.score_one(query)

        self.assertEqual(
            (
                tuple(model._history),
                model.n_samples_seen,
                model.n_subsequences_processed,
                model.best_score,
                model.last_score,
            ),
            state_before,
        )

        model.learn_one(query)
        self.assertAlmostEqual(model.last_score, score)

    def test_early_abandoned_score_obeys_damp_bounds(self) -> None:
        model = self.create_model()
        period = np.sin(2.0 * np.pi * np.arange(8) / 8.0)
        stream = np.tile(period, 30)
        stream[120:128] = np.array([-1.0, 1.1, -0.8, 1.3, -0.6, 1.5, -0.4, 1.7])

        for value in stream[:170]:
            model.learn_one({"value": float(value)})

        previous_best = model.best_score
        current = float(stream[170])
        approximate = model.score_one({"value": current})

        combined = np.fromiter(
            (*model._history, current),
            dtype=np.float64,
            count=model.n_history + 1,
        )
        query_start = combined.size - model.subsequence_length
        exact = float(
            np.min(
                _mass_distance_profile(
                    combined[
                        max(0, query_start - model.x_lag) : query_start + 1
                    ],
                    combined[query_start:],
                    eps=model.eps,
                )
            )
        )

        self.assertLess(approximate, previous_best)
        self.assertLessEqual(exact, approximate + 1e-6)

    def test_stream_matches_direct_author_loop_translation(self) -> None:
        subsequence_length = 12
        x_lag = 16 * subsequence_length
        start_index = 4 * subsequence_length
        rng = np.random.default_rng(7)
        period = np.sin(
            2.0 * np.pi * np.arange(subsequence_length) / subsequence_length
        )
        stream = np.tile(period, 40) + rng.normal(
            0.0,
            0.01,
            40 * subsequence_length,
        )
        stream[300:312] = np.linspace(-2.0, 2.0, subsequence_length) + np.sign(
            np.sin(
                6.0
                * np.pi
                * np.arange(subsequence_length)
                / subsequence_length
            )
        )

        reference_scores, reference_best = _reference_x_lag_damp(
            stream,
            subsequence_length,
            start_index,
            x_lag,
        )
        model = XLagDAMP(
            subsequence_length=subsequence_length,
            x_lag=x_lag,
            start_index=start_index,
        )
        online_scores: list[float] = []
        for value in stream:
            online_scores.append(model.score_one({"value": float(value)}))
            model.learn_one({"value": float(value)})

        endpoint_scores = np.zeros_like(reference_scores)
        endpoint_scores[subsequence_length - 1 :] = reference_scores[
            : -(subsequence_length - 1)
        ]
        first_endpoint = start_index + subsequence_length - 2
        np.testing.assert_allclose(
            online_scores[first_endpoint:],
            endpoint_scores[first_endpoint:],
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertAlmostEqual(model.best_score, reference_best)

    def test_discord_scores_higher_than_repeated_pattern(self) -> None:
        model = self.create_model(
            subsequence_length=16,
            x_lag=128,
            start_index=32,
        )
        period = np.sin(2.0 * np.pi * np.arange(16) / 16.0)
        stream = np.tile(period, 20).astype(np.float64)
        stream[220:236] = np.sign(np.sin(6.0 * np.pi * np.arange(16) / 16.0))

        scores: list[float] = []
        for value in stream:
            scores.append(model.score_one({"value": float(value)}))
            model.learn_one({"value": float(value)})

        normal_max = max(scores[80:200])
        discord_max = max(scores[220:252])
        self.assertGreater(discord_max, normal_max + 1.0)

    def test_history_is_bounded_by_x_lag_and_query_length(self) -> None:
        model = self.create_model()
        for index in range(1_000):
            value = math.sin(2.0 * math.pi * index / 8.0) + 0.01 * index
            model.learn_one({"value": value})

        self.assertEqual(model.n_history, model.x_lag + model.subsequence_length - 1)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model()
        for index in range(80):
            model.learn_one({"value": float(np.sin(index))})

        self.assertGreater(model.n_subsequences_processed, 0)
        model.reset()

        self.assertEqual(model.n_samples_seen, 0)
        self.assertEqual(model.n_history, 0)
        self.assertEqual(model.n_subsequences_processed, 0)
        self.assertEqual(model.best_score, 0.0)
        self.assertEqual(model.score_one({"value": 1.0}), 0.0)

    def test_repr_contains_configuration_and_state(self) -> None:
        representation = repr(self.create_model())

        self.assertIn("XLagDAMP", representation)
        self.assertIn("subsequence_length=8", representation)
        self.assertIn("x_lag=64", representation)
        self.assertIn("start_index=32", representation)


if __name__ == "__main__":
    unittest.main()
