"""The FAISS similarity engine: mean distances, owned windows and failure recovery."""

import unittest
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from aberrant.similarity import FaissSimilaritySearchEngine
from aberrant.similarity import _faiss as faiss_engine

faiss = pytest.importorskip("faiss")


class TestFaissSimilaritySearchEngine(unittest.TestCase):
    def test_engine_returns_mean_l2_distance(self):
        engine = FaissSimilaritySearchEngine(window_size=4, warm_up=2)
        engine.append({"x": 0.0})
        engine.append({"x": 3.0})

        # IndexFlatL2 reports squared distances [0, 9]; the public engine
        # contract is the mean Euclidean distance [0, 3].
        self.assertAlmostEqual(engine.search({"x": 0.0}, n_neighbors=2), 1.5)

    def test_engine_owns_window_samples(self):
        engine = FaissSimilaritySearchEngine(window_size=2, warm_up=1)
        point = {"x": 1.0}
        engine.append(point)
        point["x"] = 99.0

        self.assertEqual(engine.window[0]["x"], 1.0)

    def test_window_and_key_snapshots_cannot_change_retained_vectors(self):
        engine = FaissSimilaritySearchEngine(window_size=2, warm_up=1)
        engine.append({"x": 1.0})
        engine.window[0]["x"] = 99.0
        engine.keys.append("unexpected")
        engine.append({"x": 2.0})
        engine.append({"x": 3.0})

        self.assertEqual(engine.keys, ["x"])
        self.assertEqual(list(engine.window), [{"x": 2.0}, {"x": 3.0}])
        self.assertEqual(engine.search({"x": 2.0}, n_neighbors=1), 0.0)

    def test_rejects_invalid_vectors_before_schema_or_window_changes(self):
        engine = FaissSimilaritySearchEngine(window_size=2, warm_up=1)
        invalid_points = (
            {},
            {1: 2.0},
            {"x": "bad"},
            {"x": np.nan},
            {"x": np.inf},
            {"x": 1e100},
        )
        for point in invalid_points:
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    engine.append(point)
                self.assertEqual(len(engine.window), 0)
                self.assertIsNone(engine.keys)
                self.assertIsNone(engine.index)

        engine.append({"valid": 1.0})
        with self.assertRaisesRegex(ValueError, "Inconsistent feature keys"):
            engine.append({"other": 2.0})
        self.assertEqual(list(engine.window), [{"valid": 1.0}])
        self.assertEqual(engine.index.ntotal, 1)

    @staticmethod
    def _index_that_fails_after_add(index):
        staged = Mock(wraps=index)

        def fail_after_add(data):
            index.add(data)
            raise RuntimeError("index insertion failed")

        staged.add.side_effect = fail_after_add
        return staged

    def test_incremental_add_failure_recovers_from_retained_vectors(self):
        engine = FaissSimilaritySearchEngine(window_size=3, warm_up=1)
        engine.append({"x": 1.0})
        original_index = engine.index
        revision = engine._schema.preview({"x": 1.0}).revision
        original_add = original_index.add

        def fail_after_add(data):
            original_add(data)
            raise RuntimeError("index insertion failed")

        with (
            patch.object(original_index, "add", side_effect=fail_after_add),
            self.assertRaisesRegex(RuntimeError, "index insertion failed"),
        ):
            engine.append({"x": 2.0})

        self.assertEqual(original_index.ntotal, 2)  # failure followed native mutation
        self.assertIsNone(engine.index)
        self.assertEqual(list(engine.window), [{"x": 1.0}])
        self.assertEqual(engine._schema.preview({"x": 1.0}).revision, revision)
        self.assertEqual(engine.search({"x": 2.0}, n_neighbors=1), 1.0)
        self.assertEqual(engine.index.ntotal, 1)
        engine.append({"x": 2.0})
        self.assertEqual(engine.index.ntotal, 2)
        self.assertEqual(engine.search({"x": 2.0}, n_neighbors=1), 0.0)

    def test_append_rebuilds_invalidated_index_without_preceding_search(self):
        engine = FaissSimilaritySearchEngine(window_size=3, warm_up=1)
        engine.append({"x": 1.0})
        index = engine.index
        original_add = index.add

        def fail_after_add(data):
            original_add(data)
            raise RuntimeError("index insertion failed")

        with (
            patch.object(index, "add", side_effect=fail_after_add),
            self.assertRaisesRegex(RuntimeError, "index insertion failed"),
        ):
            engine.append({"x": 2.0})
        engine.append({"x": 3.0})

        self.assertEqual(list(engine.window), [{"x": 1.0}, {"x": 3.0}])
        self.assertEqual(engine.index.ntotal, 2)
        self.assertEqual(engine.search({"x": 2.0}, n_neighbors=2), 1.0)

    def test_rebuild_failure_preserves_schema_and_fifo_window(self):
        constructor = faiss.IndexFlatL2
        for retained in ((), (1.0, 2.0)):
            with self.subTest(retained=retained):
                engine = FaissSimilaritySearchEngine(window_size=2, warm_up=1)
                for value in retained:
                    engine.append({"x": value})
                original_index = engine.index

                def create_failing_index(dimension):
                    return self._index_that_fails_after_add(constructor(dimension))

                with (
                    patch(
                        "aberrant.similarity._faiss.faiss.IndexFlatL2",
                        side_effect=create_failing_index,
                    ),
                    self.assertRaisesRegex(RuntimeError, "index insertion failed"),
                ):
                    engine.append({"x": 3.0})

                self.assertIs(engine.index, original_index)
                self.assertEqual(
                    list(engine.window), [{"x": value} for value in retained]
                )
                self.assertEqual(engine.keys, ["x"] if retained else None)
                if not retained:
                    engine.append({"new": 4.0})
                    self.assertEqual(engine.keys, ["new"])
                else:
                    engine.append({"x": 3.0})
                    self.assertEqual(list(engine.window), [{"x": 2.0}, {"x": 3.0}])
                    self.assertEqual(engine.search({"x": 2.0}, n_neighbors=1), 0.0)

    def test_search_validates_without_committing_a_schema(self):
        engine = FaissSimilaritySearchEngine(window_size=3, warm_up=2)
        self.assertEqual(engine.search({"query": 1.0}, n_neighbors=1), 0.0)
        self.assertIsNone(engine.keys)
        with self.assertRaises(ValueError):
            engine.search({"query": np.nan}, n_neighbors=1)
        engine.append({"learned": 2.0})
        with self.assertRaisesRegex(ValueError, "Inconsistent feature keys"):
            engine.search({"query": 1.0}, n_neighbors=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_size": 0, "warm_up": 1}, r"window_size \(0\) must be positive"),
        ({"window_size": 2, "warm_up": 0}, r"warm_up \(0\) must be positive"),
        ({"window_size": 2, "warm_up": 3}, r"window_size \(2\) must be >= warm_up"),
    ],
)
def test_rejects_invalid_configuration(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FaissSimilaritySearchEngine(**kwargs)


def test_search_rejects_impossible_neighbour_counts() -> None:
    engine = FaissSimilaritySearchEngine(window_size=3, warm_up=1)
    engine.append({"x": 1.0})

    with pytest.raises(ValueError, match="n_neighbors must be positive"):
        engine.search({"x": 1.0}, n_neighbors=0)
    with pytest.raises(ValueError, match=r"cannot exceed window size \(1\)"):
        engine.search({"x": 1.0}, n_neighbors=2)


def test_explains_how_to_install_faiss_when_it_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(faiss_engine, "faiss", None)

    with pytest.raises(ImportError, match=r"aberrant\[faiss\]"):
        FaissSimilaritySearchEngine(window_size=2, warm_up=1)
