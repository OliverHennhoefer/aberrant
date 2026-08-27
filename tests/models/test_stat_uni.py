import unittest
from collections import deque

import numpy as np
from scipy.stats import kurtosis, skew

from aberrant.model.stat import (
    MovingAverage,
    MovingAverageAbsoluteDeviation,
    MovingGeometricAverage,
    MovingHarmonicAverage,
    MovingInterquartileRange,
    MovingKurtosis,
    MovingMedian,
    MovingQuantile,
    MovingSkewness,
    MovingVariance,
)


class TestMovingAverage(unittest.TestCase):
    def test_initialization_with_positive_window_size(self):
        ma = MovingAverage(3)
        self.assertEqual(ma.window.maxlen, 3)

    def test_initialization_with_non_positive_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingAverage(0)

    def test_learn_one_updates_window_correctly(self):
        ma = MovingAverage(3)
        ma.learn_one({"value": 1.0})
        self.assertEqual(list(ma.window), [1.0])
        ma.learn_one({"value": 2.0})
        self.assertEqual(list(ma.window), [1.0, 2.0])

    def test_learn_one_with_multiple_keys_raises_assertion_error(self):
        ma = MovingAverage(3)
        with self.assertRaises(ValueError):
            ma.learn_one({"key1": 1.0, "key2": 2.0})

    def test_score_one_calculates_mean_correctly(self):
        ma = MovingAverage(3)
        ma.learn_one({"value": 1.0})
        ma.learn_one({"value": 2.0})
        self.assertEqual(ma.score_one({"value": 3}), 0.5)

    def test_score_one_with_empty_window(self):
        ma = MovingAverage(3)
        self.assertEqual(ma.score_one({"value": 3}), 0)

    def test_window_shifting(self):
        ma = MovingAverage(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))

    def test_zero_division(self):
        ma = MovingAverage(3)
        for x in [0, 0, 0, 0, 0]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one({"value": 0}), 0)


class TestMovingHarmonicAverage(unittest.TestCase):
    def test_initialization_with_positive_window_size(self):
        mah = MovingHarmonicAverage(3)
        self.assertEqual(len(mah.window), 0)

    def test_initialization_with_zero_window_size_raises_error(self):
        with self.assertRaises(ValueError) as context:
            _ = MovingHarmonicAverage(0)
        self.assertEqual(
            str(context.exception), "Window size must be a positive integer."
        )

    def test_initialization_with_negative_window_size_raises_error(self):
        with self.assertRaises(ValueError) as context:
            _ = MovingHarmonicAverage(-5)
        self.assertEqual(
            str(context.exception), "Window size must be a positive integer."
        )

    def test_learn_one_with_single_value_dict(self):
        mah = MovingHarmonicAverage(3)
        mah.learn_one({"value": 10.0})
        self.assertEqual(len(mah.window), 1)
        if mah.feature_name:
            self.assertIn("value", mah.feature_name)

    def test_learn_one_raises_assertion_error_if_more_than_one_key(self):
        mah = MovingHarmonicAverage(3)
        with self.assertRaises(ValueError) as context:
            mah.learn_one({"a": 10.0, "b": 20.0})
        self.assertEqual(
            str(context.exception), "Input must contain exactly one key-value pair."
        )

    def test_learn_one_raises_assertion_error_if_empty_dict(self):
        mah = MovingHarmonicAverage(3)
        with self.assertRaises(ValueError) as context:
            mah.learn_one({})
        self.assertEqual(
            str(context.exception), "Input must contain exactly one key-value pair."
        )

    def test_score_one_with_no_values_returns_zero(self):
        mah = MovingHarmonicAverage(3)
        self.assertEqual(mah.score_one({"value": 4}), 0.0)

    def test_score_one_calculates_correct_harmonic_average(self):
        mah = MovingHarmonicAverage(3, abs_diff=False)
        mah.learn_one({"value": 1.0})
        mah.learn_one({"value": 2.0})
        mah.learn_one({"value": 4.0})
        expected_score = 4 / (1 + 0.5 + 0.25 + 1) - 3 / (1 + 0.5 + 0.25)
        self.assertAlmostEqual(mah.score_one({"value": 1}), expected_score, places=6)

    def test_window_shifting(self):
        ma = MovingHarmonicAverage(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))

    def test_zero_division(self):
        ma = MovingHarmonicAverage(5)
        for x in [0, 0, 0, 0, 0]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one({"value": 0}), 0)
        self.assertEqual(ma.score_one({"value": 1}), 0)

    def test_undefined_harmonic_mean_raises_value_error(self):
        model = MovingHarmonicAverage(3)
        model.learn_one({"value": 1.0})
        model.learn_one({"value": -1.0})

        with self.assertRaisesRegex(ValueError, "undefined for this window"):
            model.score_one({"value": 2.0})


class TestMovingGeometricAverage(unittest.TestCase):
    def test_initialization_with_positive_window_size(self):
        m = MovingGeometricAverage(window_size=5)
        self.assertEqual(len(m.window), 0)

    def test_initialization_with_zero_or_negative_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            MovingGeometricAverage(window_size=0)
        with self.assertRaises(ValueError):
            MovingGeometricAverage(window_size=-1)

    def test_learn_one_with_valid_input(self):
        m = MovingGeometricAverage(window_size=3)
        m.learn_one({"feature": 2.0})
        self.assertEqual(len(m.window), 1)
        self.assertEqual(m.window[0], 2.0)

    def test_learn_one_raises_assertion_error_for_empty_dict(self):
        m = MovingGeometricAverage(window_size=3)
        with self.assertRaises(ValueError):
            m.learn_one({})

    def test_learn_one_raises_assertion_error_for_multiple_keys(self):
        m = MovingGeometricAverage(window_size=3)
        with self.assertRaises(ValueError):
            m.learn_one({"feature1": 2.0, "feature2": 4.0})

    def test_score_one_with_empty_window(self):
        m = MovingGeometricAverage(window_size=3)
        self.assertEqual(m.score_one({"value": 1}), 0)

    def test_score_one_with_single_value_in_window(self):
        m = MovingGeometricAverage(window_size=3)
        m.learn_one({"feature": 2.0})
        self.assertEqual(m.score_one({"feature": 1}), 0)

    def test_score_one_with_multiple_values_absolute_values(self):
        m = MovingGeometricAverage(window_size=5, abs_diff=False, absoluteValues=True)
        for point in [1, 1.5, 1.65]:
            m.learn_one({"feature": point})
        geo_window = (1.5 * 1.1) ** (1 / 2)
        geo_score = (1.5 * 1.1 * 1.2) ** (1 / 3)
        expected_score = geo_score - geo_window
        self.assertAlmostEqual(m.score_one({"feature": 1.98}), expected_score)

    def test_score_one_with_multiple_values_relative_values(self):
        m = MovingGeometricAverage(window_size=5, abs_diff=False, absoluteValues=False)
        for point in [1.5, 1.1]:
            m.learn_one({"feature": point})
        geo_window = (1.5 * 1.1) ** (1 / 2)
        geo_score = (1.5 * 1.1 * 1.2) ** (1 / 3)
        expected_score = geo_score - geo_window
        self.assertAlmostEqual(m.score_one({"feature": 1.2}), expected_score)

    def test_window_shifting(self):
        ma = MovingGeometricAverage(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))

    def test_zero_division(self):
        ma = MovingGeometricAverage(3)
        for x in [0, 0, 0, 0, 0]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one({"value": 1}), 0)

    def test_score_one_rejects_nonpositive_value(self):
        model = MovingGeometricAverage(3)
        model.learn_one({"value": 1.0})
        model.learn_one({"value": 2.0})

        with self.assertRaisesRegex(ValueError, "requires positive values"):
            model.score_one({"value": -1.0})


class TestMovingMedian(unittest.TestCase):
    def test_repr_reports_configured_window_size_before_warmup(self):
        self.assertIn("window_size=5", repr(MovingMedian(5)))

    def test_initialization_invalid(self):
        # Testing if ValueError is raised for non-positive window size
        with self.assertRaises(ValueError):
            MovingMedian(0)
        with self.assertRaises(ValueError):
            MovingMedian(-5)

    def test_learn_one_valid_input(self):
        # Test learning a single valid input
        mm = MovingMedian(3)
        mm.learn_one({"value": 10})
        self.assertEqual(list(mm.window), [10])

    def test_learn_one_invalid_input_dict_length(self):
        # Test assertion error for dictionary with more than one key-value pair
        mm = MovingMedian(3)
        with self.assertRaises(ValueError):
            mm.learn_one({"a": 1, "b": 2})

    def test_score_one_empty_window(self):
        # Test scoring when the window is empty
        mm = MovingMedian(3)
        self.assertEqual(mm.score_one({"value": 1}), 0)

    def test_score_one_single_element(self):
        # Test scoring with a single element in the window
        mm = MovingMedian(1, abs_diff=False)
        mm.learn_one({"value": 5})
        self.assertEqual(mm.score_one({"value": 10}), 2.5)

    def test_score_one_odd_window_size(self):
        # Test scoring for an odd-sized window
        mm = MovingMedian(3)
        mm.learn_one({"value": 10})
        mm.learn_one({"value": 20})
        mm.learn_one({"value": 30})
        self.assertEqual(mm.score_one({"value": 40}), 5)

    def test_score_one_even_window_size(self):
        # Test scoring for an even-sized window
        mm = MovingMedian(4, abs_diff=False)
        mm.learn_one({"value": 10})
        mm.learn_one({"value": 20})
        mm.learn_one({"value": 30})
        mm.learn_one({"value": 40})
        self.assertEqual(mm.score_one({"value": 10}), -5)

    def test_window_replacement(self):
        # Test that old values are replaced in a full window
        mm = MovingMedian(3)
        mm.learn_one({"value": 10})
        mm.learn_one({"value": 20})
        mm.learn_one({"value": 30})
        self.assertEqual(list(mm.window), [10, 20, 30])

        # Adding another value should remove the oldest (first added) one
        mm.learn_one({"value": 40})
        self.assertEqual(list(mm.window), [20, 30, 40])

    def test_zero_division(self):
        ma = MovingMedian(3)
        for x in [0, 0, 0, 0, 0]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one({"value": 1}), 0)


class TestMovingQuantile(unittest.TestCase):
    def test_initialization(self):
        with self.assertRaises(ValueError):
            MovingQuantile(window_size=0)
        with self.assertRaises(ValueError):
            MovingQuantile(window_size=3, quantile=-0.1)
        with self.assertRaises(ValueError):
            MovingQuantile(window_size=3, quantile=1.1)

        model = MovingQuantile(window_size=3, quantile=0.5)
        self.assertEqual(model.window.maxlen, 3)
        self.assertEqual(model.quantile, 0.5)

    def test_learn_one_valid(self):
        model = MovingQuantile(window_size=3)
        model.learn_one({"value": 1})
        self.assertEqual(len(model.window), 1)

    def test_learn_one_invalid(self):
        model = MovingQuantile(window_size=3)
        with self.assertRaises(ValueError):
            model.learn_one({})

        with self.assertRaises(ValueError):
            model.learn_one({"value1": 1, "value2": 2})

    def test_score_empty_window(self):
        model = MovingQuantile(window_size=3)
        self.assertEqual(model.score_one({"value": 1}), 0)

    def test_score_single_value(self):
        model = MovingQuantile(window_size=1, abs_diff=False)
        model.learn_one({"value": 5})
        self.assertEqual(
            model.score_one({"value": 0}),
            np.quantile([0, 5], 0.5) - np.quantile(model.window, 0.5),
        )

    def test_score_median(self):
        model = MovingQuantile(window_size=3, quantile=0.5)
        model.learn_one({"value": 1})
        model.learn_one({"value": 3})
        model.learn_one({"value": 2})
        self.assertEqual(model.score_one({"value": 1}), 0.5)

    def test_score_quantile_below_50(self):
        model = MovingQuantile(window_size=4, quantile=0.25, abs_diff=False)
        model.learn_one({"value": 1})
        model.learn_one({"value": 3})
        model.learn_one({"value": 5})
        model.learn_one({"value": 2})
        self.assertEqual(
            model.score_one({"value": 1}),
            np.quantile([1, 1, 2, 3, 5], 0.25) - np.quantile(model.window, 0.25),
        )

    def test_score_quantile_above_50(self):
        model = MovingQuantile(window_size=4, quantile=0.75, abs_diff=False)
        model.learn_one({"value": 1})
        model.learn_one({"value": 3})
        model.learn_one({"value": 5})
        model.learn_one({"value": 2})
        self.assertEqual(
            model.score_one({"value": 1}),
            np.quantile([1, 1, 3, 5, 2], 0.75) - np.quantile(model.window, 0.75),
        )

    def test_interpolation(self):
        model = MovingQuantile(window_size=5, quantile=0.6, abs_diff=False)
        for value in [1, 3, 7, 9, 10]:
            model.learn_one({"value": value})
        self.assertAlmostEqual(
            model.score_one({"value": 1}),
            np.quantile([1, 1, 3, 7, 9, 10], 0.6) - np.quantile(model.window, 0.6),
        )

    def test_window_shifting(self):
        ma = MovingQuantile(3, quantile=0.6)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))

    def test_zero_division(self):
        ma = MovingQuantile(3, quantile=0.6)
        for x in [0, 0, 0, 0, 0]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one({"value": 1}), 0)


class TestUnivariateFeatureSchema(unittest.TestCase):
    def test_all_moving_models_lock_to_first_learned_feature(self):
        models = [
            MovingAverage(3),
            MovingHarmonicAverage(3),
            MovingGeometricAverage(3),
            MovingMedian(3),
            MovingQuantile(3),
            MovingVariance(3),
            MovingInterquartileRange(3),
            MovingAverageAbsoluteDeviation(3),
            MovingKurtosis(3),
            MovingSkewness(3),
        ]

        for model in models:
            with self.subTest(model=model.__class__.__name__):
                model.learn_one({"expected": 1.0})
                self.assertEqual(model.feature_name, "expected")
                with self.assertRaises(ValueError):
                    model.learn_one({"unexpected": 2.0})

    def test_invalid_values_are_rejected_before_feature_or_window_commit(self):
        model = MovingAverage(3)

        with self.assertRaisesRegex(ValueError, "must be numeric"):
            model.learn_one({"value": "bad"})  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "must be finite"):
            model.learn_one({"value": float("nan")})

        self.assertIsNone(model.feature_name)
        self.assertEqual(list(model.window), [])


class TestMovingVariance(unittest.TestCase):
    def test_repr_reports_configured_window_size_before_warmup(self):
        self.assertIn("window_size=5", repr(MovingVariance(5)))

    def test_initialization_with_valid_size(self):
        mv = MovingVariance(window_size=5)
        self.assertEqual(len(mv.window), 0)

    def test_initialization_with_invalid_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingVariance(window_size=-1)

    def test_learn_one_valid_input(self):
        mv = MovingVariance(window_size=3)
        mv.learn_one({"value": 10})
        mv.learn_one({"value": 20})
        self.assertEqual(list(mv.window), [10, 20])

    def test_learn_one_invalid_input_raises_assertion_error(self):
        mv = MovingVariance(window_size=3)
        with self.assertRaises(ValueError):
            mv.learn_one({"a": 1, "b": 2})

    def test_score_one_empty_window_returns_zero(self):
        mv = MovingVariance(window_size=3)
        self.assertEqual(mv.score_one({"value": 10}), 0)

    def test_score_one_single_value_variance_is_zero(self):
        mv = MovingVariance(window_size=5)
        mv.learn_one({"value": 10})
        self.assertEqual(mv.score_one({"value": 10}), 0)

    def test_score_one_calculates_correct_variance(self):
        mv = MovingVariance(window_size=5)
        values = [10, 12, 23, 23, 16]
        for value in values:
            mv.learn_one({"value": value})
        self.assertAlmostEqual(
            mv.score_one({"value": 10}),
            np.var([10, 12, 23, 23, 16, 10]) - np.var(values),
        )

    def test_score_one_with_window_rolling(self):
        mv = MovingVariance(window_size=3, abs_diff=False)
        values = [10, 12, 23]
        for value in values:
            mv.learn_one({"value": value})
        # Expect variance of window [10, 12, 23]
        np.var([10, 12, 23])
        self.assertAlmostEqual(
            mv.score_one({"value": 10}), np.var([10, 12, 23, 10]) - np.var(values)
        )
        mv.learn_one({"value": 18})
        # Expect variance of window [12, 23, 18]
        np.var([12, 23, 18])
        self.assertAlmostEqual(
            mv.score_one({"value": 10}), np.var([12, 23, 18, 10]) - np.var([12, 23, 18])
        )

    def test_zero_division(self):
        ma = MovingVariance(3)
        for x in [0, 0, 0, 0, 0]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one({"value": 0}), 0)


class TestMovingInterquartileRange(unittest.TestCase):
    def test_initialization_with_positive_window_size(self):
        ma = MovingInterquartileRange(3)
        self.assertEqual(ma.window.maxlen, 3)

    def test_initialization_with_non_positive_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingInterquartileRange(0)

    def test_learn_one_updates_window_correctly(self):
        ma = MovingInterquartileRange(3)
        ma.learn_one({"value": 1.0})
        self.assertEqual(list(ma.window), [1.0])
        ma.learn_one({"value": 2.0})
        self.assertEqual(list(ma.window), [1.0, 2.0])

    def test_learn_one_with_multiple_keys_raises_assertion_error(self):
        ma = MovingInterquartileRange(3)
        with self.assertRaises(ValueError):
            ma.learn_one({"key1": 1.0, "key2": 2.0})

    def test_score_one_calculates_mean_correctly(self):
        ma = MovingInterquartileRange(10)
        test_values = [1, 4, 2, 6, 7, 3, 9, 33, 34, -5]
        for x in test_values:
            ma.learn_one({"value": x})
        iqr_window = np.quantile(test_values, 0.75) - np.quantile(test_values, 0.25)
        score_window = [1, 4, 2, 6, 7, 3, 9, 33, 34, -5, 1]
        iqr_score = np.quantile(score_window, 0.75) - np.quantile(score_window, 0.25)
        self.assertEqual(ma.score_one({"value": 1.0}), iqr_score - iqr_window)

    def test_score_one_with_empty_window(self):
        ma = MovingInterquartileRange(3)
        self.assertEqual(ma.score_one({"value": 1.0}), 0)

    def test_window_shifting(self):
        ma = MovingInterquartileRange(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))

    def test_zero_division(self):
        ma = MovingInterquartileRange(3)
        for x in [0, 0, 0, 0, 0]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one({"value": 0.0}), 0)


class TestMovingAverageAbsoluteDeviation(unittest.TestCase):
    def test_initialization_with_positive_window_size(self):
        ma = MovingAverageAbsoluteDeviation(3)
        self.assertEqual(ma.window.maxlen, 3)

    def test_initialization_with_non_positive_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingAverageAbsoluteDeviation(0)

    def test_learn_one_updates_window_correctly(self):
        ma = MovingAverageAbsoluteDeviation(3)
        ma.learn_one({"value": 1.0})
        self.assertEqual(list(ma.window), [1.0])
        ma.learn_one({"value": 2.0})
        self.assertEqual(list(ma.window), [1.0, 2.0])

    def test_learn_one_with_multiple_keys_raises_assertion_error(self):
        ma = MovingAverageAbsoluteDeviation(3)
        with self.assertRaises(ValueError):
            ma.learn_one({"key1": 1.0, "key2": 2.0})

    def test_score_one_calculates_mean_correctly(self):
        ma = MovingAverageAbsoluteDeviation(10, abs_diff=False)
        test_values = [1, 4, 2, 6, 7, 3, 9, 33, 34, -5]
        for x in test_values:
            ma.learn_one({"value": x})
        window = sum(abs(x - np.mean(test_values)) for x in test_values) / len(
            test_values
        )
        score_window = sum(
            abs(x - np.mean([1, 4, 2, 6, 7, 3, 9, 33, 34, -5, 1]))
            for x in [1, 4, 2, 6, 7, 3, 9, 33, 34, -5, 1]
        ) / len([1, 4, 2, 6, 7, 3, 9, 33, 34, -5, 1])
        self.assertEqual(ma.score_one({"value": 1}), score_window - window)

    def test_score_one_with_empty_window(self):
        ma = MovingAverageAbsoluteDeviation(3)
        self.assertEqual(ma.score_one({"value": 1}), 0)

    def test_window_shifting(self):
        ma = MovingAverageAbsoluteDeviation(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))

    def test_zero_division(self):
        ma = MovingAverageAbsoluteDeviation(3)
        for x in [0, 0, 0, 0, 0]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one({"value": 0}), 0)


class TestMovingKurtosis(unittest.TestCase):
    def test_initialization_with_positive_window_size(self):
        ma = MovingKurtosis(3)
        self.assertEqual(ma.window.maxlen, 3)

    def test_initialization_with_non_positive_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingKurtosis(0)

    def test_learn_one_updates_window_correctly(self):
        ma = MovingKurtosis(3)
        ma.learn_one({"value": 1.0})
        self.assertEqual(list(ma.window), [1.0])
        ma.learn_one({"value": 2.0})
        self.assertEqual(list(ma.window), [1.0, 2.0])

    def test_learn_one_with_multiple_keys_raises_assertion_error(self):
        ma = MovingKurtosis(3)
        with self.assertRaises(ValueError):
            ma.learn_one({"key1": 1.0, "key2": 2.0})

    def test_score_one_calculates_pearson_correctly(self):
        ma = MovingKurtosis(5, abs_diff=False)
        test_values = [1, 2, 3, 4]
        for x in test_values:
            ma.learn_one({"value": x})

        self.assertAlmostEqual(
            ma.score_one({"value": 1.0}),
            float(
                kurtosis([1, 2, 3, 4, 1], fisher=False)
                - kurtosis(test_values, fisher=False)
            ),
        )

    def test_score_one_with_empty_window(self):
        ma = MovingKurtosis(3)
        self.assertEqual(ma.score_one({"value": 1.0}), 0)

    def test_window_shifting(self):
        ma = MovingKurtosis(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))

    def test_zero_division(self):
        ma = MovingKurtosis(3)
        for x in [0, 0, 0, 0, 0]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one({"value": 1.0}), 0)


class TestMovingSkewness(unittest.TestCase):
    def test_initialization_with_positive_window_size(self):
        ma = MovingSkewness(3)
        self.assertEqual(ma.window.maxlen, 3)

    def test_initialization_with_non_positive_window_size_raises_error(self):
        with self.assertRaises(ValueError):
            _ = MovingSkewness(0)

    def test_learn_one_updates_window_correctly(self):
        ma = MovingSkewness(3)
        ma.learn_one({"value": 1.0})
        self.assertEqual(list(ma.window), [1.0])
        ma.learn_one({"value": 2.0})
        self.assertEqual(list(ma.window), [1.0, 2.0])

    def test_learn_one_with_multiple_keys_raises_assertion_error(self):
        ma = MovingSkewness(3)
        with self.assertRaises(ValueError):
            ma.learn_one({"key1": 1.0, "key2": 2.0})

    def test_score_one_calculates_skewness_correctly(self):
        ma = MovingSkewness(5, abs_diff=False)
        test_values = [1, 2, 3, 4, 43]
        for x in test_values:
            ma.learn_one({"value": x})
        self.assertEqual(
            ma.score_one({"value": 1}), skew([1, 2, 3, 4, 43, 1]) - skew(test_values)
        )

    def test_score_one_with_empty_window(self):
        ma = MovingSkewness(3)
        self.assertEqual(ma.score_one({"value": 1}), 0)

    def test_window_shifting(self):
        ma = MovingSkewness(3)
        for x in [1, 2, 3, 4]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.window, deque([2, 3, 4]))

    def test_zero_division(self):
        ma = MovingSkewness(3)
        for x in [0, 0, 0, 0, 0]:
            ma.learn_one({"value": x})
        self.assertEqual(ma.score_one({"value": 0}), 0)


if __name__ == "__main__":
    unittest.main()
