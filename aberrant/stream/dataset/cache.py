"""Typed, cross-process-safe dataset cache storage."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType

from filelock import FileLock

_METADATA_VERSION = "2"


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Immutable metadata for one cached artifact.

    Attributes:
        sha256: Verified 64-character hexadecimal SHA-256 digest.
        size: Artifact size in bytes.
        release_tag: Dataset release tag from which the artifact was obtained.
    """

    sha256: str
    size: int
    release_tag: str

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in self.sha256
        ):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        if isinstance(self.size, bool) or self.size < 0:
            raise ValueError("size must be a non-negative integer")
        if not self.release_tag:
            raise ValueError("release_tag must be non-empty")


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    """Immutable cache metadata snapshot.

    Attributes:
        version: On-disk metadata schema version.
        datasets: Read-only mapping from registry value to cache entry.
    """

    version: str
    datasets: Mapping[str, CacheEntry]


def _empty_metadata() -> CacheMetadata:
    return CacheMetadata(
        version=_METADATA_VERSION,
        datasets=MappingProxyType({}),
    )


class DatasetCacheStore:
    """Own cache paths, metadata transactions, and artifact publication.

    Args:
        cache_dir: Directory in which artifacts, metadata, and the process lock
            are stored. It is created, including parents, when necessary.
        lock_timeout: Positive seconds to wait for the cross-process metadata
            lock before the underlying file lock raises a timeout.
    """

    def __init__(self, cache_dir: Path, *, lock_timeout: float = 60.0) -> None:
        if lock_timeout <= 0.0:
            raise ValueError("lock_timeout must be positive")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self._lock = FileLock(
            self.cache_dir / ".aberrant-cache.lock",
            timeout=lock_timeout,
        )

    def path(self, filename: str) -> Path:
        """Return the cache path for an NPZ artifact name."""
        return self.cache_dir / f"{filename}.npz"

    @staticmethod
    def _parse_entry(name: object, value: object) -> tuple[str, CacheEntry]:
        if not isinstance(name, str) or not name:
            raise ValueError("Dataset metadata keys must be non-empty strings")
        if not isinstance(value, dict):
            raise ValueError(f"Metadata entry '{name}' must be an object")

        sha256 = value.get("sha256")
        size = value.get("size")
        release_tag = value.get("release_tag")
        if not isinstance(sha256, str):
            raise ValueError(f"Metadata entry '{name}' has an invalid SHA256")
        if not isinstance(size, int):
            raise ValueError(f"Metadata entry '{name}' has an invalid size")
        if not isinstance(release_tag, str):
            raise ValueError(f"Metadata entry '{name}' has an invalid release tag")
        try:
            entry = CacheEntry(
                sha256=sha256,
                size=size,
                release_tag=release_tag,
            )
        except ValueError as exc:
            raise ValueError(f"Metadata entry '{name}' is invalid: {exc}") from exc
        return name, entry

    def read(self) -> CacheMetadata:
        """Read and validate an immutable metadata snapshot."""
        if not self.metadata_file.exists():
            return _empty_metadata()
        try:
            with self.metadata_file.open(encoding="utf-8") as file:
                raw: object = json.load(file)
            if not isinstance(raw, dict):
                raise ValueError("Cache metadata root must be an object")
            if raw.get("version") != _METADATA_VERSION:
                raise ValueError("Unsupported cache metadata version")
            raw_datasets = raw.get("datasets")
            if not isinstance(raw_datasets, dict):
                raise ValueError("Cache metadata datasets must be an object")
            entries = dict(
                self._parse_entry(name, value)
                for name, value in raw_datasets.items()
            )
            return CacheMetadata(
                version=_METADATA_VERSION,
                datasets=MappingProxyType(entries),
            )
        except (OSError, json.JSONDecodeError, ValueError):
            return _empty_metadata()

    def _write_unlocked(self, entries: Mapping[str, CacheEntry]) -> None:
        temp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.cache_dir,
                prefix=".metadata.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temp_path = Path(file.name)
                json.dump(
                    {
                        "version": _METADATA_VERSION,
                        "datasets": {
                            name: asdict(entry) for name, entry in entries.items()
                        },
                    },
                    file,
                    indent=2,
                    sort_keys=True,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, self.metadata_file)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def publish(
        self,
        *,
        dataset_name: str,
        temporary_path: Path,
        destination: Path,
        entry: CacheEntry,
    ) -> None:
        """Replace an artifact, then merge metadata under one process lock."""
        with self._lock:
            entries = dict(self.read().datasets)
            os.replace(temporary_path, destination)
            entries[dataset_name] = entry
            self._write_unlocked(entries)

    def remove(self, dataset_name: str, artifact_path: Path) -> None:
        """Remove one artifact and its metadata under one process lock."""
        with self._lock:
            entries = dict(self.read().datasets)
            artifact_path.unlink(missing_ok=True)
            entries.pop(dataset_name, None)
            self._write_unlocked(entries)

    def clear(self, artifacts: Mapping[str, Path]) -> None:
        """Remove the supplied owned artifacts and reset metadata."""
        with self._lock:
            for artifact_path in artifacts.values():
                artifact_path.unlink(missing_ok=True)
            self._write_unlocked({})

    @staticmethod
    def size(artifacts: Mapping[str, Path]) -> int:
        """Return total bytes occupied by the supplied owned artifacts."""
        return sum(
            path.stat().st_size
            for path in artifacts.values()
            if path.is_file()
        )
