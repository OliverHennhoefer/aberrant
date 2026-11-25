"""Dataset cache, download, validation, and orchestration tests."""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import os
from dataclasses import replace
from http.client import RemoteDisconnected
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pytest

import aberrant.stream._dataset as dataset_api
from aberrant.stream import (
    CacheEntry,
    Dataset,
    DatasetCacheStore,
    DatasetManager,
    UrlLibDownloadBackend,
    get_dataset_info,
)
from aberrant.stream._dataset import loader as loader_module


def _publish_metadata_in_process(
    cache_dir: str,
    dataset: Dataset,
    ready: Queue,
    start: Event,
) -> None:
    store = DatasetCacheStore(Path(cache_dir))
    payload = dataset.value.encode()
    process_id = multiprocessing.current_process().pid
    temporary_path = Path(cache_dir) / f".{dataset.value}.{process_id}.tmp"
    temporary_path.write_bytes(payload)
    ready.put(dataset.value)
    start.wait(timeout=10.0)
    store.publish(
        dataset_name=dataset.value,
        temporary_path=temporary_path,
        destination=store.path(get_dataset_info(dataset).filename),
        entry=CacheEntry(
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
            release_tag="test-release",
        ),
    )


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        if size < 0:
            size = len(self._payload) - self._offset
        start = self._offset
        end = min(len(self._payload), start + size)
        self._offset = end
        return self._payload[start:end]


class _BytesDownloadBackend:
    def __init__(self, payload: bytes, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[tuple[str, Path]] = []

    def download(self, url: str, destination: Path) -> None:
        self.calls.append((url, destination))
        if self.error is not None:
            raise self.error
        destination.write_bytes(self.payload)


def _valid_npz_payload() -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        y=np.array([0, 1], dtype=int),
    )
    return buffer.getvalue()


def _cache_entry(payload: bytes, *, release_tag: str = "test-release") -> CacheEntry:
    return CacheEntry(
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        release_tag=release_tag,
    )


def test_url_backend_retries_and_succeeds(tmp_path: Path) -> None:
    destination = tmp_path / "shuttle.tmp"
    backend = UrlLibDownloadBackend(retries=2, backoff_seconds=0.0)

    with patch(
        "urllib.request.urlopen",
        side_effect=[
            URLError("temporary network issue"),
            _FakeResponse(b"payload"),
        ],
    ) as mock_urlopen:
        backend.download("https://example.com/shuttle.npz", destination)

    assert mock_urlopen.call_count == 2
    assert destination.read_bytes() == b"payload"


def test_url_backend_raises_after_retry_exhaustion(tmp_path: Path) -> None:
    destination = tmp_path / "shuttle.tmp"
    backend = UrlLibDownloadBackend(retries=3, backoff_seconds=0.0)

    with (
        patch(
            "urllib.request.urlopen",
            side_effect=RemoteDisconnected("no response"),
        ) as mock_urlopen,
        pytest.raises(RuntimeError, match="after 3 attempts"),
    ):
        backend.download("https://example.com/shuttle.npz", destination)

    assert mock_urlopen.call_count == 3


def test_manager_rejects_checksum_mismatch_and_cleans_temporary_file(
    tmp_path: Path,
) -> None:
    manager = DatasetManager(
        cache_dir=tmp_path,
        download_backend=_BytesDownloadBackend(_valid_npz_payload()),
    )

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        manager.download(Dataset.SHUTTLE, force=True)

    assert not manager.cache.path("shuttle").exists()
    assert not list(tmp_path.glob(".shuttle.npz.*.tmp"))


def test_manager_download_publishes_typed_metadata(tmp_path: Path) -> None:
    payload = _valid_npz_payload()
    payload_hash = hashlib.sha256(payload).hexdigest()
    trusted_info = replace(get_dataset_info(Dataset.SHUTTLE), sha256=payload_hash)
    backend = _BytesDownloadBackend(payload)
    manager = DatasetManager(
        cache_dir=tmp_path,
        release_tag="test-release",
        download_backend=backend,
    )

    with patch(
        "aberrant.stream._dataset.loader.get_dataset_info",
        return_value=trusted_info,
    ):
        cached_path = manager.download(Dataset.SHUTTLE, force=True)

    entry = manager.cache.read().datasets[Dataset.SHUTTLE.value]
    assert cached_path.read_bytes() == payload
    assert entry == _cache_entry(payload)
    assert len(backend.calls) == 1


def test_manager_reuses_validated_cached_artifact_without_downloading(
    tmp_path: Path,
) -> None:
    payload = _valid_npz_payload()
    trusted_info = replace(
        get_dataset_info(Dataset.SHUTTLE),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    backend = _BytesDownloadBackend(b"must not be written")
    manager = DatasetManager(
        cache_dir=tmp_path,
        release_tag="test-release",
        download_backend=backend,
    )
    destination = manager.cache.path(trusted_info.filename)
    temporary_path = tmp_path / ".seed.tmp"
    temporary_path.write_bytes(payload)
    manager.cache.publish(
        dataset_name=Dataset.SHUTTLE.value,
        temporary_path=temporary_path,
        destination=destination,
        entry=_cache_entry(payload),
    )

    with patch(
        "aberrant.stream._dataset.loader.get_dataset_info",
        return_value=trusted_info,
    ):
        result = manager.download(Dataset.SHUTTLE)

    assert result == destination
    assert backend.calls == []


def test_load_rejects_unvalidated_cached_file_without_auto_download(
    tmp_path: Path,
) -> None:
    manager = DatasetManager(cache_dir=tmp_path)
    manager.cache.path("shuttle").write_bytes(b"not-an-npz")

    with pytest.raises(FileNotFoundError, match="validated cache"):
        manager.load(Dataset.SHUTTLE, auto_download=False)


def test_load_wraps_download_backend_failure(tmp_path: Path) -> None:
    backend = _BytesDownloadBackend(b"", error=RuntimeError("offline"))
    manager = DatasetManager(cache_dir=tmp_path, download_backend=backend)

    with pytest.raises(RuntimeError, match="Failed to download dataset shuttle"):
        manager.load(Dataset.SHUTTLE)

    assert len(backend.calls) == 1


def test_old_metadata_version_is_not_migrated(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "datasets": {
                    "shuttle": {
                        "downloaded": True,
                        "hash": "legacy",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = DatasetCacheStore(tmp_path).read()

    assert snapshot.version == "2"
    assert dict(snapshot.datasets) == {}


def test_metadata_snapshot_is_immutable(tmp_path: Path) -> None:
    snapshot = DatasetCacheStore(tmp_path).read()

    with pytest.raises(TypeError):
        snapshot.datasets["shuttle"] = _cache_entry(b"payload")  # type: ignore[index]


@pytest.mark.skipif(os.name != "posix", reason="Unix file-lock backend regression")
@pytest.mark.parametrize("lock_entry", ["rename", "hardlink"])
def test_cache_lock_preserves_unrelated_file_contents(
    tmp_path: Path, lock_entry: str
) -> None:
    store = DatasetCacheStore(tmp_path / "cache")
    unrelated = tmp_path / "unrelated.txt"
    payload = b"unrelated file contents must survive cache locking"
    unrelated.write_bytes(payload)
    lock_path = store.cache_dir / ".aberrant-cache.lock"
    if lock_entry == "rename":
        unrelated.replace(lock_path)
    else:
        lock_path.hardlink_to(unrelated)

    store.clear({})

    assert lock_path.read_bytes() == payload
    if lock_entry == "hardlink":
        assert unrelated.read_bytes() == payload
    assert dict(store.read().datasets) == {}


def test_metadata_publication_is_atomic_and_leaves_no_temporary_file(
    tmp_path: Path,
) -> None:
    store = DatasetCacheStore(tmp_path)
    payload = b"payload"
    temporary_path = tmp_path / ".artifact.tmp"
    temporary_path.write_bytes(payload)
    destination = store.path("shuttle")

    store.publish(
        dataset_name="shuttle",
        temporary_path=temporary_path,
        destination=destination,
        entry=_cache_entry(payload),
    )

    assert destination.read_bytes() == payload
    assert store.metadata_file.exists()
    assert not list(tmp_path.glob(".metadata.*.tmp"))
    assert store.read().datasets["shuttle"] == _cache_entry(payload)


@pytest.mark.parametrize("has_previous_artifact", [False, True])
def test_failed_metadata_publication_restores_previous_cache(
    tmp_path: Path, has_previous_artifact: bool
) -> None:
    store = DatasetCacheStore(tmp_path)
    destination = store.path("shuttle")
    payload = _valid_npz_payload()
    temporary_path = tmp_path / ".artifact.tmp"
    if has_previous_artifact:
        temporary_path.write_bytes(payload)
        store.publish(
            dataset_name="shuttle",
            temporary_path=temporary_path,
            destination=destination,
            entry=_cache_entry(payload),
        )
    previous_metadata = (
        store.metadata_file.read_bytes() if store.metadata_file.exists() else None
    )
    temporary_path.write_bytes(b"replacement")
    replace_file = os.replace

    def fail_metadata_replace(source: Path, target: Path) -> None:
        if target == store.metadata_file:
            raise OSError("metadata publication failed")
        replace_file(source, target)

    with (
        patch(
            "aberrant.stream._dataset.cache.os.replace",
            side_effect=fail_metadata_replace,
        ),
        pytest.raises(OSError, match="metadata publication failed"),
    ):
        store.publish(
            dataset_name="shuttle",
            temporary_path=temporary_path,
            destination=destination,
            entry=_cache_entry(b"replacement"),
        )

    assert not list(tmp_path.glob(".metadata.*.tmp"))
    assert not list(tmp_path.glob("*.bak"))
    if not has_previous_artifact:
        assert not destination.exists()
        assert not store.metadata_file.exists()
        assert dict(store.read().datasets) == {}
        return
    assert destination.read_bytes() == payload
    assert store.metadata_file.read_bytes() == previous_metadata
    trusted_info = replace(
        get_dataset_info(Dataset.SHUTTLE),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    manager = DatasetManager(cache_dir=tmp_path, release_tag="test-release")
    with patch(
        "aberrant.stream._dataset.loader.get_dataset_info", return_value=trusted_info
    ):
        streamer = manager.load(Dataset.SHUTTLE, auto_download=False)
    assert list(streamer.stream()) == [
        ({"feature_0": 1.0, "feature_1": 2.0}, 0),
        ({"feature_0": 3.0, "feature_1": 4.0}, 1),
    ]


def test_failed_artifact_restoration_retains_backup(tmp_path: Path) -> None:
    store = DatasetCacheStore(tmp_path)
    destination = store.path("shuttle")
    payload = b"previous"
    temporary_path = tmp_path / ".artifact.tmp"
    temporary_path.write_bytes(payload)
    store.publish(
        dataset_name="shuttle",
        temporary_path=temporary_path,
        destination=destination,
        entry=_cache_entry(payload),
    )
    temporary_path.write_bytes(b"replacement")
    replace_file = os.replace

    def fail_metadata_and_restoration(source: Path, target: Path) -> None:
        if target == store.metadata_file or source.suffix == ".bak":
            raise OSError("cannot replace")
        replace_file(source, target)

    with (
        patch(
            "aberrant.stream._dataset.cache.os.replace",
            side_effect=fail_metadata_and_restoration,
        ),
        pytest.raises(OSError, match="cannot replace"),
    ):
        store.publish(
            dataset_name="shuttle",
            temporary_path=temporary_path,
            destination=destination,
            entry=_cache_entry(b"replacement"),
        )

    backups = list(tmp_path.glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == payload
    assert store.read().datasets["shuttle"] == _cache_entry(payload)


def test_backup_cleanup_failure_keeps_successful_publication(tmp_path: Path) -> None:
    store = DatasetCacheStore(tmp_path)
    destination = store.path("shuttle")
    temporary_path = tmp_path / ".artifact.tmp"
    unlink_file = Path.unlink

    def fail_backup_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path.suffix == ".bak":
            raise PermissionError("backup is in use")
        unlink_file(path, missing_ok=missing_ok)

    for payload in (b"previous", b"replacement"):
        temporary_path.write_bytes(payload)
        with patch.object(
            Path, "unlink", autospec=True, side_effect=fail_backup_unlink
        ):
            store.publish(
                dataset_name="shuttle",
                temporary_path=temporary_path,
                destination=destination,
                entry=_cache_entry(payload),
            )

    assert destination.read_bytes() == b"replacement"
    assert store.read().datasets["shuttle"] == _cache_entry(b"replacement")
    backups = list(tmp_path.glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"previous"


def test_metadata_updates_are_merged_across_processes(tmp_path: Path) -> None:
    resolved_cache = str(tmp_path.resolve())
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    processes = [
        context.Process(
            target=_publish_metadata_in_process,
            args=(resolved_cache, dataset, ready, start),
        )
        for dataset in (Dataset.FRAUD, Dataset.SHUTTLE)
    ]

    for process in processes:
        process.start()
    for _ in processes:
        ready.get(timeout=10.0)
    start.set()
    for process in processes:
        process.join(timeout=10.0)
        assert process.exitcode == 0

    cached = DatasetManager(cache_dir=resolved_cache).list_cached()
    assert set(cached) == {Dataset.FRAUD.value, Dataset.SHUTTLE.value}
    assert all(isinstance(entry, CacheEntry) for entry in cached.values())


def test_clear_cache_preserves_unrelated_files_in_shared_directory(
    tmp_path: Path,
) -> None:
    unrelated = tmp_path / "application-data.txt"
    unrelated.write_text("keep", encoding="utf-8")
    manager = DatasetManager(cache_dir=tmp_path)
    payload = b"cached"
    temporary_path = tmp_path / ".shuttle.tmp"
    temporary_path.write_bytes(payload)
    destination = manager.cache.path("shuttle")
    manager.cache.publish(
        dataset_name=Dataset.SHUTTLE.value,
        temporary_path=temporary_path,
        destination=destination,
        entry=_cache_entry(payload),
    )

    manager.clear_cache()

    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert not destination.exists()
    assert manager.cache.metadata_file.exists()
    assert dict(manager.cache.read().datasets) == {}


def test_cache_size_counts_only_registered_artifacts(tmp_path: Path) -> None:
    manager = DatasetManager(cache_dir=tmp_path)
    manager.cache.path("shuttle").write_bytes(b"owned")
    (tmp_path / "unrelated.npz").write_bytes(b"not-owned")

    assert manager.get_cache_size() == len(b"owned")


_VALID_ENTRY = {"sha256": "a" * 64, "size": 3, "release_tag": "v1"}


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        {"version": "2", "datasets": []},
        {"version": "2", "datasets": {"": _VALID_ENTRY}},
        {"version": "2", "datasets": {"shuttle": "entry"}},
        {"version": "2", "datasets": {"shuttle": {**_VALID_ENTRY, "sha256": 1}}},
        {"version": "2", "datasets": {"shuttle": {**_VALID_ENTRY, "size": "3"}}},
        {"version": "2", "datasets": {"shuttle": {**_VALID_ENTRY, "release_tag": 1}}},
        {"version": "2", "datasets": {"shuttle": {**_VALID_ENTRY, "sha256": "z" * 64}}},
        {"version": "2", "datasets": {"shuttle": {**_VALID_ENTRY, "size": -1}}},
        {"version": "2", "datasets": {"shuttle": {**_VALID_ENTRY, "size": True}}},
        {"version": "2", "datasets": {"shuttle": {**_VALID_ENTRY, "release_tag": ""}}},
    ],
    ids=[
        "root-array",
        "datasets-array",
        "empty-name",
        "entry-string",
        "sha-number",
        "size-string",
        "tag-number",
        "sha-not-hex",
        "size-negative",
        "size-bool",
        "tag-empty",
    ],
)
def test_corrupt_metadata_reads_as_an_empty_cache(tmp_path: Path, metadata) -> None:
    store = DatasetCacheStore(tmp_path)
    store.metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    assert dict(store.read().datasets) == {}


def test_valid_metadata_entries_are_read_back(tmp_path: Path) -> None:
    store = DatasetCacheStore(tmp_path)
    store.metadata_file.write_text(
        json.dumps({"version": "2", "datasets": {"shuttle": _VALID_ENTRY}}),
        encoding="utf-8",
    )

    assert store.read().datasets["shuttle"] == CacheEntry(**_VALID_ENTRY)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: DatasetCacheStore(Path("."), lock_timeout=0.0), "lock_timeout"),
        (lambda: UrlLibDownloadBackend(retries=0), "retries"),
        (lambda: UrlLibDownloadBackend(timeout=0.0), "timeout"),
        (lambda: UrlLibDownloadBackend(backoff_seconds=-1.0), "backoff_seconds"),
    ],
)
def test_cache_and_download_settings_must_be_positive(make, message) -> None:
    with pytest.raises(ValueError, match=message):
        make()


def _published_manager(
    tmp_path: Path, payload: bytes, *, release_tag: str = "test-release"
) -> tuple[DatasetManager, _BytesDownloadBackend]:
    """A manager whose cache holds ``payload`` as a validated SHUTTLE artifact."""
    backend = _BytesDownloadBackend(payload)
    manager = DatasetManager(
        cache_dir=tmp_path, release_tag=release_tag, download_backend=backend
    )
    temporary_path = tmp_path / ".seed.tmp"
    temporary_path.write_bytes(payload)
    manager.cache.publish(
        dataset_name=Dataset.SHUTTLE.value,
        temporary_path=temporary_path,
        destination=manager.cache.path("shuttle"),
        entry=_cache_entry(payload),
    )
    return manager, backend


def _trusting(payload: bytes):
    return patch(
        "aberrant.stream._dataset.loader.get_dataset_info",
        return_value=replace(
            get_dataset_info(Dataset.SHUTTLE),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
    )


def test_cached_artifact_from_another_release_is_downloaded_again(
    tmp_path: Path,
) -> None:
    payload = _valid_npz_payload()
    manager, backend = _published_manager(tmp_path, payload, release_tag="newer")

    with _trusting(payload):
        manager.download(Dataset.SHUTTLE)

    assert len(backend.calls) == 1
    assert manager.cache.read().datasets["shuttle"].release_tag == "newer"


@pytest.mark.parametrize(
    "tamper",
    [
        lambda path: path.write_bytes(path.read_bytes() + b"tail"),
        lambda path: path.write_bytes(b"x" * path.stat().st_size),
    ],
    ids=["resized", "same-size-garbage"],
)
def test_tampered_cached_artifact_is_not_trusted(tmp_path: Path, tamper) -> None:
    payload = _valid_npz_payload()
    manager, _ = _published_manager(tmp_path, payload)
    tamper(manager.cache.path("shuttle"))

    with _trusting(payload), pytest.raises(FileNotFoundError, match="validated"):
        manager.load(Dataset.SHUTTLE, auto_download=False)


def test_archive_without_labels_is_rejected(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, X=np.ones((2, 2)))
    manager = DatasetManager(
        cache_dir=tmp_path, download_backend=_BytesDownloadBackend(buffer.getvalue())
    )

    with pytest.raises(RuntimeError, match="Invalid NPZ dataset artifact"):
        manager.download(Dataset.SHUTTLE)
    assert not manager.cache.path("shuttle").exists()


def test_failed_backup_copy_leaves_the_cache_untouched(tmp_path: Path) -> None:
    payload = _valid_npz_payload()
    manager, _ = _published_manager(tmp_path, payload)
    replacement = tmp_path / ".replacement.tmp"
    replacement.write_bytes(b"replacement")

    with (
        patch(
            "aberrant.stream._dataset.cache.copyfile", side_effect=OSError("disk full")
        ),
        pytest.raises(OSError, match="disk full"),
    ):
        manager.cache.publish(
            dataset_name="shuttle",
            temporary_path=replacement,
            destination=manager.cache.path("shuttle"),
            entry=_cache_entry(b"replacement"),
        )

    assert manager.cache.path("shuttle").read_bytes() == payload
    assert manager.cache.read().datasets["shuttle"] == _cache_entry(payload)
    assert not list(tmp_path.glob("*.bak"))


def test_clearing_one_dataset_keeps_the_others(tmp_path: Path) -> None:
    payload = _valid_npz_payload()
    manager, _ = _published_manager(tmp_path, payload)
    fraud = tmp_path / ".fraud.tmp"
    fraud.write_bytes(payload)
    manager.cache.publish(
        dataset_name=Dataset.FRAUD.value,
        temporary_path=fraud,
        destination=manager.cache.path(get_dataset_info(Dataset.FRAUD).filename),
        entry=_cache_entry(payload),
    )

    manager.clear_cache(Dataset.SHUTTLE)

    assert set(manager.list_cached()) == {Dataset.FRAUD.value}
    assert not manager.cache.path("shuttle").exists()
    assert "cached_datasets=1" in repr(manager)


def test_url_backend_reports_download_progress(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "shuttle.tmp"
    backend = UrlLibDownloadBackend(show_progress=True)

    with patch("urllib.request.urlopen", return_value=_FakeResponse(b"x" * 20_000)):
        backend.download("https://example.com/shuttle.npz", destination)

    assert destination.stat().st_size == 20_000
    assert "Downloading shuttle.tmp" in capsys.readouterr().err


@pytest.fixture
def default_cache(tmp_path: Path):
    previous = loader_module._DefaultManager.instance
    dataset_api.set_cache_dir(tmp_path)
    yield tmp_path
    loader_module._DefaultManager.instance = previous


def test_module_functions_use_the_default_manager(default_cache: Path) -> None:
    payload = _valid_npz_payload()
    manager = dataset_api.get_default_manager()
    assert manager.cache_dir == default_cache
    manager.download_backend = _BytesDownloadBackend(payload)

    with _trusting(payload):
        path = dataset_api.download(Dataset.SHUTTLE)
        samples = list(dataset_api.load(Dataset.SHUTTLE).stream())

    assert path.read_bytes() == payload
    assert samples[1] == ({"feature_0": 3.0, "feature_1": 4.0}, 1)
    assert set(dataset_api.list_cached()) == {Dataset.SHUTTLE.value}
    info = dataset_api.get_cache_info()
    assert info.directory == default_cache
    assert info.size_bytes == len(payload)
    assert info.datasets == (Dataset.SHUTTLE.value,)

    dataset_api.clear_cache()
    assert dataset_api.get_cache_info().datasets == ()
