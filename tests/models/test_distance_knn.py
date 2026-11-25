"""Tests for the windowed nearest-neighbor detector."""

import pickle
from importlib import util

import numpy as np
import pytest

from aberrant.catalog import create_model
from aberrant.exceptions import ValidationError
from aberrant.model.distance import KNN
from aberrant.model.distance import knn as knn_module

_needs_faiss = pytest.mark.skipif(
    util.find_spec("faiss") is None, reason="needs aberrant[faiss]"
)
_BACKENDS = ["numpy", pytest.param("faiss", marks=_needs_faiss)]


def _brute_force(window: np.ndarray, point: np.ndarray, k: int) -> float:
    distances = np.sort(np.linalg.norm(window - point, axis=1))
    return float(distances[:k].mean())


def _stream(n: int, seed: int = 0) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    return [{"a": float(a), "b": float(b)} for a, b in rng.normal(size=(n, 2))]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 0}, "k must be a positive integer"),
        ({"window_size": 0}, "window_size must be a positive integer"),
        ({"k": 5, "window_size": 4}, "window_size must be at least k"),
        ({"backend": "annoy"}, "backend must be 'numpy' or 'faiss'"),
    ],
)
def test_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        KNN(**kwargs)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_scores_zero_until_the_window_holds_k_events(backend):
    model = KNN(k=3, window_size=10, backend=backend)
    for event in _stream(2):
        model.learn_one(event)
        assert model.score_one({"a": 5.0, "b": 5.0}) == 0.0
    model.learn_one({"a": 0.0, "b": 0.0})
    assert model.score_one({"a": 5.0, "b": 5.0}) > 0.0


@pytest.mark.parametrize("backend", _BACKENDS)
def test_score_is_the_mean_distance_to_the_k_nearest_retained_events(backend):
    events = _stream(60, seed=1)
    model = KNN(k=4, window_size=25, backend=backend)
    rows = np.array([[e["a"], e["b"]] for e in events])
    for index, event in enumerate(events):
        model.learn_one(event)
        window = rows[max(0, index - 24) : index + 1]
        if len(window) < 4:
            continue
        query = np.array([0.3, -0.2])
        expected = _brute_force(window, query, 4)
        assert model.score_one({"a": 0.3, "b": -0.2}) == pytest.approx(
            expected, rel=1e-6 if backend == "faiss" else 1e-12
        )
    assert model.n_samples_seen == 25


def test_scoring_does_not_learn_the_candidate():
    model = KNN(k=1, window_size=4)
    model.learn_one({"a": 0.0, "b": 0.0})
    first = model.score_one({"a": 3.0, "b": 4.0})
    assert model.score_one({"a": 3.0, "b": 4.0}) == first == 5.0
    assert model.n_samples_seen == 1


def test_invalid_events_leave_the_window_unchanged():
    model = KNN(k=1, window_size=4)
    for event in ({}, {"a": "bad"}, {"a": np.nan}, {"a": np.inf}):
        with pytest.raises(ValidationError):
            model.learn_one(event)
        assert model.n_samples_seen == 0
    model.learn_one({"a": 1.0})
    with pytest.raises(ValidationError):
        model.learn_one({"b": 1.0})
    assert model.n_samples_seen == 1


@_needs_faiss
def test_faiss_rejects_values_outside_float32():
    model = KNN(k=1, window_size=2, backend="faiss")
    with pytest.raises(ValidationError, match="float32"):
        model.learn_one({"a": 1e100})
    assert model.n_samples_seen == 0


@_needs_faiss
def test_faiss_and_numpy_agree_and_survive_pickling():
    events = _stream(80, seed=3)
    numpy_model = KNN(k=5, window_size=30)
    faiss_model = KNN(k=5, window_size=30, backend="faiss")
    for event in events[:50]:
        numpy_model.learn_one(event)
        faiss_model.learn_one(event)
    faiss_model.score_one(events[50])
    restored = pickle.loads(pickle.dumps(faiss_model))
    for event in events[50:]:
        expected = numpy_model.score_one(event)
        assert faiss_model.score_one(event) == pytest.approx(expected, rel=1e-5)
        assert restored.score_one(event) == pytest.approx(expected, rel=1e-5)
        for model in (numpy_model, faiss_model, restored):
            model.learn_one(event)


@_needs_faiss
def test_catalog_builds_the_faiss_backend():
    model = create_model("knn", {"k": 1, "window_size": 8, "backend": "faiss"})
    model.learn_one({"x": 0.0})
    assert model.score_one({"x": 2.0}) == pytest.approx(2.0)


def test_faiss_backend_explains_the_missing_extra(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(knn_module, "import_module", missing)
    with pytest.raises(ImportError, match=r"aberrant\[faiss\]"):
        KNN(backend="faiss")
