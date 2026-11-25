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
