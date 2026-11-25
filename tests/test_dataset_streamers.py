"""Archive ownership and shape validation for NPZ sample streams."""

from pathlib import Path

import numpy as np
import pytest

from aberrant.stream.dataset.registry import Dataset, get_dataset_info
from aberrant.stream.dataset.streamers import BatchStreamer, NpzStreamer


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


@pytest.mark.parametrize("option", ["feature_prefix", "label_column", "feature_column"])
def test_column_names_must_be_non_empty(tmp_path: Path, option: str) -> None:
    with pytest.raises(ValueError, match=option):
        NpzStreamer(tmp_path / "samples.npz", **{option: ""})


def test_missing_file_and_feature_array_are_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        list(NpzStreamer(tmp_path / "absent.npz").stream())

    file_path = tmp_path / "samples.npz"
    np.savez(file_path, features=[[1.0]], y=[0])
    with pytest.raises(KeyError, match="Feature array 'X'"):
        list(NpzStreamer(file_path).stream())
    stream = NpzStreamer(file_path, feature_column="features", feature_prefix="f")
    assert list(stream.stream()) == [({"f0": 1.0}, 0)]


def test_progress_and_metadata_describe_a_registered_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    file_path = tmp_path / "samples.npz"
    np.savez(file_path, X=[[1.0], [2.0]], y=[0, 1])
    info = get_dataset_info(Dataset.SHUTTLE)
    streamer = NpzStreamer(file_path, info, show_progress=True)

    assert len(list(streamer.stream())) == 2
    assert f"Loading {info.name}" in capsys.readouterr().err
    assert streamer.get_metadata() is info
    assert f"dataset={info.name!r}" in repr(streamer)
    assert repr(NpzStreamer(file_path)) == f"NpzStreamer(file_path={str(file_path)!r})"


@pytest.mark.parametrize(("rows", "sizes"), [(5, [2, 2, 1]), (4, [2, 2]), (0, [])])
def test_batches_keep_order_and_yield_the_remainder(
    tmp_path: Path, rows: int, sizes: list[int]
) -> None:
    file_path = tmp_path / "samples.npz"
    np.savez(file_path, X=np.arange(rows, dtype=float).reshape(rows, 1), y=range(rows))
    batches = BatchStreamer(NpzStreamer(file_path), batch_size=2)

    streamed = list(batches.stream())

    assert [len(features) for features, _ in streamed] == sizes
    assert [label for _, labels in streamed for label in labels] == list(range(rows))
    assert batches.get_metadata() is None
    assert repr(batches).endswith("batch_size=2)")


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        BatchStreamer(object(), batch_size=0)  # type: ignore[arg-type]
