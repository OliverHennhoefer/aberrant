"""Rejected samples must not change transformer learning state."""

import pickle

import numpy as np
import pytest

from aberrant.transform.preprocessing import MinMaxScaler, StandardScaler
from aberrant.transform.projection import IncrementalPCA, RandomProjection


@pytest.mark.parametrize("factory", [MinMaxScaler, StandardScaler])
@pytest.mark.parametrize("warm", [False, True])
def test_scaler_rejects_entire_sample_before_mutating(factory, warm):
    model = factory()
    if warm:
        model.learn_one({"a": 1.0, "b": 2.0})
    before = pickle.dumps(model)
    with pytest.raises(ValueError, match="finite"):
        model.learn_one({"a": 10.0, "b": float("nan")})
    assert pickle.dumps(model) == before


@pytest.mark.parametrize("factory", [IncrementalPCA, RandomProjection])
def test_failed_first_projection_sample_does_not_lock_schema(factory):
    model = factory(n_components=2)
    before = pickle.dumps(model)
    with pytest.raises(ValueError):
        model.learn_one({"x": 1.0})
    assert pickle.dumps(model) == before
    model.learn_one({"b": 2.0, "a": 1.0})
    assert model.feature_names == ["b", "a"]
    assert len(model.transform_one({"a": 1.0, "b": 2.0})) == 2


def test_projection_retry_matches_clean_seeded_initialization():
    failed = RandomProjection(2, seed=42)
    clean = RandomProjection(2, seed=42)
    with pytest.raises(ValueError):
        failed.learn_one({"x": 1.0})
    sample = {"b": 2.0, "a": 1.0, "c": 3.0}
    failed.learn_one(sample)
    clean.learn_one(sample)
    np.testing.assert_array_equal(failed.random_matrix, clean.random_matrix)
    assert failed.transform_one(sample) == clean.transform_one(sample)
