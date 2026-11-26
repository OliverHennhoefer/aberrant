"""Registered datasets: verified download, cache and replay."""

import dataclasses
import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pytest

from aberrant import datasets
from aberrant.datasets import DATASETS, Dataset, DatasetStream, data_dir, load


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _archive(features: np.ndarray, labels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, X=features, y=labels)
    return buffer.getvalue()


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Serve a small shuttle archive and record each requested URL."""
    payload = _archive(np.arange(6.0).reshape(3, 2), np.array([0, 1, 0]))
    info = dataclasses.replace(
        DATASETS[Dataset.SHUTTLE], sha256=hashlib.sha256(payload).hexdigest()
    )
    requests: list[str] = []

    def open_url(url: str) -> io.BytesIO:
        requests.append(url)
        return io.BytesIO(payload)

    monkeypatch.setenv("ABERRANT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(datasets, "DATASETS", {**DATASETS, Dataset.SHUTTLE: info})
    monkeypatch.setattr(datasets, "_open_url", open_url)
    return requests


def test_every_dataset_is_registered_with_a_checksum() -> None:
    assert set(DATASETS) == set(Dataset)
    assert len({info.filename for info in DATASETS.values()}) == len(Dataset)
    for info in DATASETS.values():
        assert len(info.sha256) == 64
        int(info.sha256, 16)
    with pytest.raises(TypeError):
        DATASETS[Dataset.SHUTTLE] = DATASETS[Dataset.FRAUD]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Dataset.SHUTTLE.info.sha256 = "0" * 64  # type: ignore[misc]


def test_the_cache_lives_in_the_configured_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ABERRANT_DATA_DIR", raising=False)
    assert data_dir() == Path.home() / ".cache" / "aberrant"
    monkeypatch.setenv("ABERRANT_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path


def test_a_dataset_downloads_once_and_replays(
    served: list[str], tmp_path: Path
) -> None:
    stream = load("shuttle")
    again = load(Dataset.SHUTTLE)

    assert served == [f"{datasets.RELEASE_URL}/shuttle.npz"]
    assert stream.path == again.path == tmp_path / "shuttle.npz"
    assert stream.info is Dataset.SHUTTLE.info
    assert len(stream) == 3
    assert (
        list(stream)
        == list(stream)
        == [
            ({"feature_0": 0.0, "feature_1": 1.0}, 0),
            ({"feature_0": 2.0, "feature_1": 3.0}, 1),
            ({"feature_0": 4.0, "feature_1": 5.0}, 0),
        ]
    )
    assert [path.name for path in tmp_path.iterdir()] == ["shuttle.npz"]
    assert stream.path.stat().st_mode & 0o444 == 0o444 & ~_umask()


def test_a_download_that_fails_its_checksum_is_never_cached_or_parsed(
    served: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(datasets, "_open_url", lambda url: io.BytesIO(b"not npz"))

    with pytest.raises(ValueError, match="does not match its SHA-256"):
        load(Dataset.SHUTTLE)

    assert list(tmp_path.iterdir()) == []


def test_a_tampered_cache_is_downloaded_again(
    served: list[str], tmp_path: Path
) -> None:
    path = datasets.download(Dataset.SHUTTLE)
    path.write_bytes(_archive(np.zeros((3, 2)), np.ones(3)))

    with pytest.raises(FileNotFoundError, match="not cached"):
        load(Dataset.SHUTTLE, fetch=False)
    assert [label for _, label in load(Dataset.SHUTTLE)] == [0, 1, 0]
    assert len(served) == 2


def test_offline_loading_needs_a_cached_copy(served: list[str]) -> None:
    with pytest.raises(FileNotFoundError, match=r"call download\('shuttle'\)"):
        load(Dataset.SHUTTLE, fetch=False)
    assert served == []


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ({"X": np.zeros((2, 1))}, "must hold arrays X and y"),
        ({"X": np.zeros(2), "y": np.zeros(2)}, "2-D X and a 1-D y"),
        ({"X": np.zeros((2, 1)), "y": np.zeros(3)}, "2 rows but 3 labels"),
    ],
)
def test_a_malformed_archive_is_rejected(
    tmp_path: Path, arrays: dict[str, np.ndarray], message: str
) -> None:
    path = tmp_path / "data.npz"
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match=message):
        list(DatasetStream(path))


def test_synthetic_streams_repeat_for_the_same_arguments() -> None:
    first, second = datasets.synthetic(seed=3), datasets.synthetic(seed=3)
    assert first.anomalies == second.anomalies
    assert np.array_equal(first.arrays()[0], second.arrays()[0])
    assert not np.array_equal(first.arrays()[0], datasets.synthetic(seed=4).arrays()[0])


def test_synthetic_labels_cover_exactly_the_injected_anomalies() -> None:
    stream = datasets.synthetic(n_samples=2000, n_anomalies=10, length=15)
    features, labels = stream.arrays()
    expected = np.zeros(len(stream), dtype=int)
    for anomaly in stream.anomalies:
        expected[anomaly.start : anomaly.stop] = 1
    assert np.array_equal(labels, expected)
    assert not labels[:400].any()
    assert [anomaly.kind for anomaly in stream.anomalies] == [
        "spike",
        "level_shift",
        "variance",
        "flatline",
        "decoupling",
    ] * 2
    rows = list(stream)
    assert len(rows) == 2000
    assert list(rows[0][0]) == ["feature_0", "feature_1", "feature_2"]
    assert [label for _, label in rows] == labels.tolist()


def test_each_kind_changes_its_feature_as_described() -> None:
    clean = datasets.synthetic(n_anomalies=0, seed=5).arrays()[0]
    stream = datasets.synthetic(seed=5)
    features = stream.arrays()[0]
    scale = clean[:1000].std(axis=0)
    for anomaly in stream.anomalies:
        column = int(anomaly.feature.removeprefix("feature_"))
        window = slice(anomaly.start, anomaly.stop)
        change = features[window, column] - clean[window, column]
        others = np.delete(features[window], column, axis=1)
        assert np.array_equal(others, np.delete(clean[window], column, axis=1))
        if anomaly.kind == "spike":
            assert abs(change[0]) == pytest.approx(6 * scale[column])
        elif anomaly.kind == "level_shift":
            assert np.allclose(np.abs(change), 3 * scale[column])
        elif anomaly.kind == "flatline":
            assert np.ptp(features[window, column]) == 0.0
        elif anomaly.kind == "decoupling":
            assert (
                np.corrcoef(features[window, column], clean[window, column])[0, 1] < 0
            )


def test_synthetic_streams_hold_only_the_requested_kinds() -> None:
    stream = datasets.synthetic(n_features=1, anomalies=["flatline"], n_anomalies=3)
    assert {anomaly.kind for anomaly in stream.anomalies} == {"flatline"}
    assert (
        repr(stream) == "SyntheticStream(n_samples=5000, n_features=1, n_anomalies=3)"
    )


@pytest.mark.parametrize(
    "params",
    [
        {"n_samples": 0},
        {"anomalies": []},
        {"anomalies": "spike"},
        {"anomalies": ["spike", "spike"]},
        {"anomalies": ["drift"]},
        {"n_features": 1, "anomalies": ["decoupling"]},
        {"n_samples": 100, "n_anomalies": 10, "length": 20},
        {"clean_prefix": 6000},
        {"noise": -1.0},
        {"season_length": 0},
    ],
)
def test_synthetic_rejects_invalid_arguments(params) -> None:
    with pytest.raises(ValueError):
        datasets.synthetic(**params)
