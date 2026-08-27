"""Dataset cache, download, validation, and orchestration tests."""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
from dataclasses import replace
from http.client import RemoteDisconnected
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pytest

from aberrant.stream.dataset import Dataset
from aberrant.stream.dataset.cache import CacheEntry, DatasetCacheStore
from aberrant.stream.dataset.download import UrlLibDownloadBackend
from aberrant.stream.dataset.loader import DatasetManager
from aberrant.stream.dataset.registry import get_dataset_info
from aberrant.stream.dataset.streamers import BatchStreamer


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


def test_url_backend_retries_and_succeeds() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        destination = Path(temp_dir) / "shuttle.tmp"
        backend = UrlLibDownloadBackend(retries=2, backoff_seconds=0.0)

        with patch(
            "aberrant.stream.dataset.download.urllib.request.urlopen",
            side_effect=[URLError("temporary network issue"), _FakeResponse(b"payload")],
        ) as mock_urlopen:
            backend.download("https://example.com/shuttle.npz", destination)

        assert mock_urlopen.call_count == 2
        assert destination.read_bytes() == b"payload"


def test_url_backend_raises_after_retry_exhaustion() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        destination = Path(temp_dir) / "shuttle.tmp"
        backend = UrlLibDownloadBackend(retries=3, backoff_seconds=0.0)

        with (
            patch(
                "aberrant.stream.dataset.download.urllib.request.urlopen",
                side_effect=RemoteDisconnected("no response"),
            ) as mock_urlopen,
            pytest.raises(RuntimeError, match="after 3 attempts"),
        ):
            backend.download("https://example.com/shuttle.npz", destination)

        assert mock_urlopen.call_count == 3


def test_manager_rejects_checksum_mismatch_and_cleans_temporary_file() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        manager = DatasetManager(
            cache_dir=temp_dir,
            download_backend=_BytesDownloadBackend(_valid_npz_payload()),
        )

        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            manager.download(Dataset.SHUTTLE, force=True)

        assert not manager.cache.path("shuttle").exists()
        assert not list(Path(temp_dir).glob(".shuttle.npz.*.tmp"))


def test_manager_download_publishes_typed_metadata() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        payload = _valid_npz_payload()
        payload_hash = hashlib.sha256(payload).hexdigest()
        trusted_info = replace(get_dataset_info(Dataset.SHUTTLE), sha256=payload_hash)
        backend = _BytesDownloadBackend(payload)
        manager = DatasetManager(
            cache_dir=temp_dir,
            release_tag="test-release",
            download_backend=backend,
        )

        with patch(
            "aberrant.stream.dataset.loader.get_dataset_info",
            return_value=trusted_info,
        ):
            cached_path = manager.download(Dataset.SHUTTLE, force=True)

        entry = manager.cache.read().datasets[Dataset.SHUTTLE.value]
        assert cached_path.read_bytes() == payload
        assert entry == _cache_entry(payload)
        assert len(backend.calls) == 1


def test_manager_reuses_validated_cached_artifact_without_downloading() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        payload = _valid_npz_payload()
        trusted_info = replace(
            get_dataset_info(Dataset.SHUTTLE),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        backend = _BytesDownloadBackend(b"must not be written")
        manager = DatasetManager(
            cache_dir=temp_dir,
            release_tag="test-release",
            download_backend=backend,
        )
        destination = manager.cache.path(trusted_info.filename)
        temporary_path = Path(temp_dir) / ".seed.tmp"
        temporary_path.write_bytes(payload)
        manager.cache.publish(
            dataset_name=Dataset.SHUTTLE.value,
            temporary_path=temporary_path,
            destination=destination,
            entry=_cache_entry(payload),
        )

        with patch(
            "aberrant.stream.dataset.loader.get_dataset_info",
            return_value=trusted_info,
        ):
            result = manager.download(Dataset.SHUTTLE)

        assert result == destination
        assert backend.calls == []


def test_load_rejects_unvalidated_cached_file_without_auto_download() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        manager = DatasetManager(cache_dir=temp_dir)
        manager.cache.path("shuttle").write_bytes(b"not-an-npz")

        with pytest.raises(FileNotFoundError, match="validated cache"):
            manager.load(Dataset.SHUTTLE, auto_download=False)


def test_load_wraps_download_backend_failure() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        backend = _BytesDownloadBackend(b"", error=RuntimeError("offline"))
        manager = DatasetManager(cache_dir=temp_dir, download_backend=backend)

        with pytest.raises(RuntimeError, match="Failed to download dataset shuttle"):
            manager.load(Dataset.SHUTTLE)

        assert len(backend.calls) == 1


def test_old_metadata_version_is_not_migrated() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        cache_dir = Path(temp_dir)
        (cache_dir / "metadata.json").write_text(
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

        snapshot = DatasetCacheStore(cache_dir).read()

        assert snapshot.version == "2"
        assert dict(snapshot.datasets) == {}


def test_metadata_snapshot_is_immutable() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        snapshot = DatasetCacheStore(Path(temp_dir)).read()

        with pytest.raises(TypeError):
            snapshot.datasets["shuttle"] = _cache_entry(b"payload")  # type: ignore[index]


def test_metadata_publication_is_atomic_and_leaves_no_temporary_file() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        cache_dir = Path(temp_dir)
        store = DatasetCacheStore(cache_dir)
        payload = b"payload"
        temporary_path = cache_dir / ".artifact.tmp"
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
        assert not list(cache_dir.glob(".metadata.*.tmp"))
        assert store.read().datasets["shuttle"] == _cache_entry(payload)


def test_metadata_updates_are_merged_across_processes() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        resolved_cache = str(Path(temp_dir).resolve())
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


def test_clear_cache_preserves_unrelated_files_in_shared_directory() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        cache_dir = Path(temp_dir)
        unrelated = cache_dir / "application-data.txt"
        unrelated.write_text("keep", encoding="utf-8")
        manager = DatasetManager(cache_dir=cache_dir)
        payload = b"cached"
        temporary_path = cache_dir / ".shuttle.tmp"
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


def test_cache_size_counts_only_registered_artifacts() -> None:
    with TemporaryDirectory(dir=".") as temp_dir:
        cache_dir = Path(temp_dir)
        manager = DatasetManager(cache_dir=cache_dir)
        manager.cache.path("shuttle").write_bytes(b"owned")
        (cache_dir / "unrelated.npz").write_bytes(b"not-owned")

        assert manager.get_cache_size() == len(b"owned")


def test_batch_streamer_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        BatchStreamer(object(), batch_size=0)  # type: ignore[arg-type]
