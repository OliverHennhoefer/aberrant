"""Archive ownership and shape validation for NPZ sample streams."""

from pathlib import Path

import numpy as np
import pytest

from aberrant.stream.dataset.streamers import NpzStreamer


@pytest.fixture
def opened_archives(
    monkeypatch: pytest.MonkeyPatch,
) -> list[np.lib.npyio.NpzFile]:
    archives: list[np.lib.npyio.NpzFile] = []
    load = np.load

    def open_archive(file_path: Path) -> np.lib.npyio.NpzFile:
        archive = load(file_path)
        archives.append(archive)
        return archive

    monkeypatch.setattr(np, "load", open_archive)
    return archives


def test_interleaved_streams_close_their_own_archives(
    tmp_path: Path, opened_archives: list[np.lib.npyio.NpzFile]
) -> None:
    file_path = tmp_path / "samples.npz"
    np.savez(file_path, X=[[1.0], [2.0]], y=[0, 1])
    streamer = NpzStreamer(file_path)
    first = streamer.stream()
    second = streamer.stream()
    assert next(first) == ({"feature_0": 1.0}, 0)
    assert next(second) == ({"feature_0": 1.0}, 0)
    first.close()
    assert opened_archives[0].zip is None
    assert opened_archives[1].zip is not None
    assert next(second) == ({"feature_0": 2.0}, 1)
    second.close()
    assert opened_archives[1].zip is None


@pytest.mark.parametrize(
    ("arrays", "error", "message"),
    [
        ({"X": [1.0], "y": [0]}, ValueError, "two-dimensional"),
        ({"X": [[1.0]], "y": 0}, ValueError, "sample axis"),
        ({"X": [[1.0], [2.0]], "y": [0]}, ValueError, "different lengths"),
        ({"X": [["bad"]], "y": [0]}, ValueError, "could not convert"),
        ({"X": [[1.0]]}, KeyError, "Label array"),
    ],
)
def test_rejected_stream_closes_its_archive(
    tmp_path: Path,
    arrays: dict[str, object],
    error: type[Exception],
    message: str,
    opened_archives: list[np.lib.npyio.NpzFile],
) -> None:
    file_path = tmp_path / "samples.npz"
    np.savez(file_path, **arrays)
    with pytest.raises(error, match=message):
        list(NpzStreamer(file_path).stream())
    assert len(opened_archives) == 1
    assert opened_archives[0].zip is None


def test_stream_inside_context_preserves_context_archive(tmp_path: Path) -> None:
    file_path = tmp_path / "samples.npz"
    np.savez(file_path, X=[[1.0]], y=[0])
    streamer = NpzStreamer(file_path)
    with streamer:
        assert list(streamer.stream()) == [({"feature_0": 1.0}, 0)]
        assert list(streamer) == [({"feature_0": 1.0}, 0)]
        with pytest.raises(RuntimeError, match="already open"), streamer:
            pass
        assert list(streamer) == [({"feature_0": 1.0}, 0)]

    with pytest.raises(RuntimeError, match="not open"):
        iter(streamer)
