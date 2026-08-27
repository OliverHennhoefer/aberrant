import unittest

import numpy as np

from aberrant.transform.projection.incremental_pca import IncrementalPCA


class TestIncrementalPCA(unittest.TestCase):
    data = np.array([[1, 2, 2.5, 5, 5], [10, 10.5, 11, 8, 4], [3, 3.5, 7, 10, 9]])
    x = {f"feature_{i}": val for i, val in enumerate([2, 3, 3.5, 11, 5])}
    y = {f"feature_{i}": val for i, val in enumerate([4, 3.4, 9.5, 1, 1])}

    def test_ipca_transform_before_learn(self):
        x = {f"feature_{i}": val for i, val in enumerate([2, 3, 3.5, 11, 5])}
        ipca = IncrementalPCA(2)
        self.assertEqual(
            ipca.transform_one(x), {"component_0": 0.0, "component_1": 0.0}
        )

    def test_invalid_initialization_parameters(self):
        with self.assertRaisesRegex(ValueError, "n_components must be positive"):
            IncrementalPCA(0)
        with self.assertRaisesRegex(ValueError, "n0 must be positive"):
            IncrementalPCA(1, n0=0)
        with self.assertRaisesRegex(ValueError, "tol must be non-negative"):
            IncrementalPCA(1, tol=-1.0)
        with self.assertRaisesRegex(ValueError, "cannot contain duplicates"):
            IncrementalPCA(1, keys=["x", "x"])

    def test_keys_are_owned_and_non_finite_values_do_not_poison_state(self):
        keys = ["x", "y"]
        ipca = IncrementalPCA(1, n0=2, keys=keys)
        keys[0] = "changed"

        with self.assertRaisesRegex(ValueError, "must be finite"):
            ipca.learn_one({"x": float("inf"), "y": 1.0})

        self.assertEqual(ipca.feature_names, ["x", "y"])
        self.assertEqual(ipca.n_samples_seen, 0)
        self.assertEqual(ipca.window, [])

    def test_q_greater_d_init(self):
        with self.assertRaises(ValueError):
            ipca = IncrementalPCA(  # noqa
                7,
                keys=["key_01", "key_02", "key_03", "key_04", "key_05"],  # noqa
            )  # noqa

    def test_q_greater_d_dict(self):
        x = {f"feature_{i}": val for i, val in enumerate([2, 3, 3.5, 11, 5])}
        ipca = IncrementalPCA(7)
        with self.assertRaises(ValueError):
            ipca.learn_one(x)

    def test_learn_one(self):
        # no keys
        x = {f"feature_{i}": val for i, val in enumerate([2, 3, 3.5, 11, 5])}
        ipca1 = IncrementalPCA(3)
        ipca1.learn_one(x)
        self.assertListEqual(list(ipca1.window[0]), [2, 3, 3.5, 11, 5])
        # keys match
        ipca2 = IncrementalPCA(
            3, keys=["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
        )
        ipca2.learn_one(x)
        self.assertListEqual(list(ipca1.window[0]), [2, 3, 3.5, 11, 5])
        # keys do not match
        ipca3 = IncrementalPCA(
            3, keys=["key_01", "key_02", "key_03", "key_04", "key_05"]
        )
        with self.assertRaises(KeyError):
            ipca3.learn_one(x)

    def test_rejects_schema_extras_during_learning_and_warmup_transform(self):
        ipca = IncrementalPCA(1, n0=3, keys=["x", "y"])

        with self.assertRaisesRegex(ValueError, "unexpected feature"):
            ipca.learn_one({"x": 1.0, "y": 2.0, "z": 3.0})

        ipca.learn_one({"x": 1.0, "y": 2.0})
        with self.assertRaises(KeyError):
            ipca.transform_one({"x": 1.0})

        self.assertEqual(ipca.n_samples_seen, 1)

    def test_initialization(self):  # compare to incRpca from R (onlinePCA package)
        data = np.array([[1, 2, 2.5, 5, 5], [10, 10.5, 11, 8, 4], [3, 3.5, 7, 10, 9]])
        data_stream = [{f"feature_{i}": val for i, val in enumerate(dp)} for dp in data]
        ipca = IncrementalPCA(2, n0=3)
        for data_point in data_stream:
            ipca.learn_one(data_point)

        self.assertTrue(np.allclose(ipca.values, [325.52805, 35.94132]))

        # R vectors are stored as columns, so transpose them to match our format (features x components)
        vectors_R = np.array(
            [
                [-0.3792350, -0.4164001, -0.5165665, -0.5218317, -0.3790019],
                [-0.4771219, -0.4290828, -0.1729495, 0.4036324, 0.6288179],
            ]
        ).T
        self.assertTrue(np.allclose(ipca.vectors, vectors_R))

    def test_learn_after_initialization(
        self,
    ):  # compare to incRpca from R (onlinePCA package)
        data = np.array([[1, 2, 2.5, 5, 5], [10, 10.5, 11, 8, 4], [3, 3.5, 7, 10, 9]])
        x = {f"feature_{i}": val for i, val in enumerate([2, 3, 3.5, 11, 5])}
        y = {f"feature_{i}": val for i, val in enumerate([4, 3.4, 9.5, 1, 1])}
        data_stream = [{f"feature_{i}": val for i, val in enumerate(dp)} for dp in data]
        ipca = IncrementalPCA(2, n0=3)
        for data_point in data_stream:  # learning data including n0
            ipca.learn_one(data_point)
        # learn x
        ipca.learn_one(x)
        self.assertTrue(np.allclose(ipca.values, [263.20439, 31.11496]))

        vectors_R = np.array(
            [
                [-0.3380927, -0.3841677, -0.4758005, -0.5986203, -0.3916327],
                [0.5038719, 0.4439442, 0.2912170, -0.4938852, -0.4693579],
            ]
        ).T
        self.assertTrue(np.allclose(ipca.vectors, vectors_R))

        # learn y
        ipca.learn_one(y)
        self.assertTrue(np.allclose(ipca.values, [215.25829, 31.20006]))

        vectors_R = np.array(
            [
                [-0.3528824, -0.3885382, -0.5330556, -0.5541488, -0.3650793],
                [-0.3990229, -0.3074019, -0.4358383, 0.5795662, 0.4695027],
            ]
        ).T
        self.assertTrue(np.allclose(ipca.vectors, vectors_R))

    def test_transform_one(self):
        q = 2  # subspace
        data = np.array([[1, 2, 2.5, 5, 5], [10, 10.5, 11, 8, 4], [3, 3.5, 7, 10, 9]])
        x = {f"feature_{i}": val for i, val in enumerate([2, 3, 3.5, 11, 5])}

        data_stream = [{f"feature_{i}": val for i, val in enumerate(dp)} for dp in data]
        ipca = IncrementalPCA(q, n0=3)
        for data_point in data_stream:  # learning data including n0
            ipca.learn_one(data_point)

        x_transformed = ipca.transform_one(x)
        print(x_transformed)
        self.assertEqual(q, len(x_transformed))

    def test_transform_one_forgetting(self):
        q = 2  # subspace
        data = np.array([[1, 2, 2.5, 5, 5], [10, 10.5, 11, 8, 4], [3, 3.5, 7, 10, 9]])
        x = {f"feature_{i}": val for i, val in enumerate([2, 3, 3.5, 11, 5])}

        data_stream = [{f"feature_{i}": val for i, val in enumerate(dp)} for dp in data]
        ipca = IncrementalPCA(q, n0=3, forgetting_factor=0.1)
        for data_point in data_stream:  # learning data including n0
            ipca.learn_one(data_point)

        x_transformed = ipca.transform_one(x)
        print(x_transformed)
        self.assertEqual(q, len(x_transformed))

    def test_init_with_wrong_f(self):
        with self.assertRaises(ValueError):
            _ipca = IncrementalPCA(2, n0=3, forgetting_factor=-0.5)

    def test_hardcoded_pca_exact_r_match(self):
        """
        Comprehensive test using exact hardcoded data from R reference
        Tests for exact matching with R onlinePCA::incRpca implementation
        """
        # Hardcoded data for reproducible testing - matches R exactly
        hardcoded_data = [
            [5.74, 3.28, 5.25, 6.12, 5.34, 4.33, 7.33, 4.49],
            [6.13, 5.11, 4.42, 5.08, 5.82, 5.08, 6.49, 4.38],
            [3.16, 4.29, 5.49, 4.03, 3.48, 5.48, 2.82, 6.11],
            [5.91, 5.76, 5.45, 6.88, 5.61, 5.03, 5.61, 4.22],
            [5.94, 5.27, 4.16, 5.54, 4.21, 6.13, 2.44, 4.88],
            [2.28, 5.68, 5.82, 5.56, 5.99, 4.41, 5.12, 4.84],
            [7.02, 3.82, 3.32, 5.11, 6.51, 5.17, 4.79, 4.26],
            [3.88, 4.43, 6.55, 4.58, 4.38, 4.22, 5.75, 4.41],
            [2.89, 4.49, 4.59, 4.43, 3.42, 6.11, 4.29, 2.67],
            [4.25, 4.78, 6.78, 4.22, 6.15, 4.43, 5.73, 5.08],
        ]

        # Parameters matching R test
        n = 10  # Total samples
        q = 2  # Components
        n0 = 5  # Initialization samples

        # Convert to stream format
        data_stream = [
            {f"feature_{i}": val for i, val in enumerate(sample)}
            for sample in hardcoded_data
        ]

        # Initialize PCA
        ipca = IncrementalPCA(q, n0=n0)

        # Learn first n0 samples (initialization phase)
        for i in range(n0):
            ipca.learn_one(data_stream[i])

        # Test initialization results against R
        expected_init_values = np.array([258.355009, 6.026835])
        expected_init_vectors = np.array(
            [
                [0.3781516, 0.14054882],
                [0.3300961, -0.26010689],
                [0.3431899, -0.13366177],
                [0.3881459, 0.07745804],
                [0.3441842, 0.20754754],
                [0.3591014, -0.40351392],
                [0.3513482, 0.72289789],
                [0.3297563, -0.40030537],
            ]
        )

        self.assertTrue(
            np.allclose(ipca.values, expected_init_values, rtol=1e-3),
            f"Initialization values mismatch. Expected: {expected_init_values}, Got: {ipca.values}",
        )
        # Handle sign ambiguity for initialization vectors too
        init_vectors_match = True
        for i in range(expected_init_vectors.shape[1]):
            col_expected = expected_init_vectors[:, i]
            col_actual = ipca.vectors[:, i]
            matches_positive = np.allclose(col_actual, col_expected, rtol=1e-3)
            matches_negative = np.allclose(col_actual, -col_expected, rtol=1e-3)
            if not (matches_positive or matches_negative):
                init_vectors_match = False
                break
        self.assertTrue(init_vectors_match, "Initialization vectors mismatch.")

        # Learn remaining samples and test each step
        expected_values_by_step = [
            [245.926516, 4.844848],  # After sample 6
            [238.546340, 4.314877],  # After sample 7
            [230.42418, 3.70013],  # After sample 8
            [218.621707, 3.420312],  # After sample 9
            [218.186016, 3.040879],  # After sample 10
        ]

        for i in range(n0, n):
            ipca.learn_one(data_stream[i])
            step_idx = i - n0
            expected_vals = np.array(expected_values_by_step[step_idx])

            self.assertTrue(
                np.allclose(ipca.values, expected_vals, rtol=1e-3),
                f"Step {i + 1} values mismatch. Expected: {expected_vals}, Got: {ipca.values}",
            )

        # Test final vectors against R
        expected_final_vectors = np.array(
            [
                [0.3414534, -0.34029140],
                [0.3346197, 0.32369498],
                [0.3689964, 0.24539451],
                [0.3700917, -0.04145283],
                [0.3672606, -0.28126258],
                [0.3561266, 0.42696108],
                [0.3637972, -0.59785818],
                [0.3228903, 0.31408234],
            ]
        )

        # Handle sign ambiguity in eigenvectors - they can be negated and still be correct
        # Check if vectors match (allowing for sign flips)
        vectors_match = True
        for i in range(expected_final_vectors.shape[1]):  # For each component
            # Check if column matches (positive or negative)
            col_expected = expected_final_vectors[:, i]
            col_actual = ipca.vectors[:, i]

            matches_positive = np.allclose(col_actual, col_expected, rtol=1e-4)
            matches_negative = np.allclose(col_actual, -col_expected, rtol=1e-4)

            if not (matches_positive or matches_negative):
                vectors_match = False
                break

        self.assertTrue(
            vectors_match,
            f"Final vectors mismatch (accounting for sign ambiguity).\nExpected:\n{expected_final_vectors}\nGot:\n{ipca.vectors}",
        )

        # Test transformation capability
        test_sample = data_stream[0]
        transformed = ipca.transform_one(test_sample)
        self.assertEqual(len(transformed), q)
        self.assertTrue(all(isinstance(v, float) for v in transformed.values()))


if __name__ == "__main__":
    unittest.main()
