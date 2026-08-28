"""Dataset orchestration over typed cache and download components."""

from __future__ import annotations

import hmac
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from aberrant.stream.dataset.cache import CacheEntry, DatasetCacheStore
from aberrant.stream.dataset.download import (
    DatasetArtifactValidator,
    DownloadBackend,
    UrlLibDownloadBackend,
)
from aberrant.stream.dataset.registry import (
    DATASET_REGISTRY,
    Dataset,
    get_dataset_info,
)
from aberrant.stream.dataset.streamers import NpzStreamer


class DatasetManager:
    """Coordinate registered dataset downloads, validation, caching, and streams.

    Args:
        cache_dir: Artifact-cache directory. ``None`` uses
            ``~/.aberrant/datasets``.
        github_repo: GitHub ``owner/repository`` containing the dataset release.
        release_tag: Release tag used in artifact URLs and cache metadata.
        download_retries: Transfer attempts for the default download backend.
        download_timeout: Per-request timeout in seconds for the default
            download backend.
        retry_backoff_seconds: Base seconds for the default backend's
            exponential retry backoff.
        cache_lock_timeout: Seconds to wait for cross-process cache metadata
            transactions.
        show_progress: Show transfer progress in the default download backend.
            Row-iteration progress is configured separately in ``load``.
        logger: Logger for retries and manager diagnostics. ``None`` uses the
            module logger.
        download_backend: Injected transfer implementation. When supplied, the
            default backend settings above do not configure it.
        validator: Injected NPZ and SHA-256 validator. ``None`` uses
            ``DatasetArtifactValidator``.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        github_repo: str = "OliverHennhoefer/nonconform",
        release_tag: str = "v0.9.17-datasets",
        download_retries: int = 3,
        download_timeout: float = 30.0,
        retry_backoff_seconds: float = 1.0,
        cache_lock_timeout: float = 60.0,
        show_progress: bool = False,
        logger: logging.Logger | None = None,
        download_backend: DownloadBackend | None = None,
        validator: DatasetArtifactValidator | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.github_repo = github_repo
        self.release_tag = release_tag
        resolved_cache = (
            Path.home() / ".aberrant" / "datasets"
            if cache_dir is None
            else Path(cache_dir)
        )
        self.cache = DatasetCacheStore(
            resolved_cache,
            lock_timeout=cache_lock_timeout,
        )
        self.download_backend = download_backend or UrlLibDownloadBackend(
            retries=download_retries,
            timeout=download_timeout,
            backoff_seconds=retry_backoff_seconds,
            show_progress=show_progress,
            logger=self.logger,
        )
        self.validator = validator or DatasetArtifactValidator()

    @property
    def cache_dir(self) -> Path:
        """Return the configured cache directory."""
        return self.cache.cache_dir

    def _url(self, dataset: Dataset) -> str:
        info = get_dataset_info(dataset)
        return (
            f"https://github.com/{self.github_repo}/releases/download/"
            f"{self.release_tag}/{info.filename}.npz"
        )

    def _path(self, dataset: Dataset) -> Path:
        return self.cache.path(get_dataset_info(dataset).filename)

    def _validate_cached(self, dataset: Dataset) -> bool:
        path = self._path(dataset)
        entry = self.cache.read().datasets.get(dataset.value)
        if entry is None or not path.exists():
            return False
        if entry.release_tag != self.release_tag or entry.size != path.stat().st_size:
            return False
        try:
            actual_hash = self.validator.validate(path, get_dataset_info(dataset))
        except (OSError, ValueError):
            return False
        return hmac.compare_digest(actual_hash, entry.sha256)

    def download(self, dataset: Dataset, force: bool = False) -> Path:
        """Download, validate, and publish one registered dataset safely."""
        if dataset not in DATASET_REGISTRY:
            raise KeyError(f"Dataset {dataset} not found in registry")
        destination = self._path(dataset)
        if not force and self._validate_cached(dataset):
            return destination

        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=self.cache_dir,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            self.download_backend.download(self._url(dataset), temporary_path)
            digest = self.validator.validate(
                temporary_path,
                get_dataset_info(dataset),
            )
            entry = CacheEntry(
                sha256=digest,
                size=temporary_path.stat().st_size,
                release_tag=self.release_tag,
            )
            self.cache.publish(
                dataset_name=dataset.value,
                temporary_path=temporary_path,
                destination=destination,
                entry=entry,
            )
            temporary_path = None
            return destination
        except Exception as exc:
            raise RuntimeError(f"Failed to download dataset {dataset.value}: {exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load(
        self,
        dataset: Dataset,
        auto_download: bool = True,
        *,
        feature_prefix: str = "feature_",
        label_column: str = "y",
        feature_column: str = "X",
        show_progress: bool = False,
    ) -> NpzStreamer:
        """Return a typed NPZ stream for one validated cached dataset."""
        if dataset not in DATASET_REGISTRY:
            raise KeyError(f"Dataset {dataset} not found in registry")
        if not self._validate_cached(dataset):
            if not auto_download:
                raise FileNotFoundError(
                    f"Dataset {dataset.value} is not present in the validated cache"
                )
            self.download(dataset)

        return NpzStreamer(
            self._path(dataset),
            get_dataset_info(dataset),
            feature_prefix=feature_prefix,
            label_column=label_column,
            feature_column=feature_column,
            show_progress=show_progress,
        )

    def list_cached(self) -> dict[str, CacheEntry]:
        """Return valid metadata entries for artifacts that exist on disk."""
        snapshot = self.cache.read()
        return {
            dataset.value: snapshot.datasets[dataset.value]
            for dataset in DATASET_REGISTRY
            if dataset.value in snapshot.datasets and self._path(dataset).exists()
        }

    def _artifact_paths(self) -> dict[str, Path]:
        """Return the exact registered artifact paths owned by this manager."""
        return {
            dataset.value: self._path(dataset)
            for dataset in DATASET_REGISTRY
        }

    def clear_cache(self, dataset: Dataset | None = None) -> None:
        """Remove one registered artifact or all registered artifacts."""
        if dataset is not None:
            self.cache.remove(dataset.value, self._path(dataset))
            return
        self.cache.clear(self._artifact_paths())

    def get_cache_size(self) -> int:
        """Return total cached NPZ bytes."""
        return self.cache.size(self._artifact_paths())

    def __repr__(self) -> str:
        return (
            f"DatasetManager(cache_dir={str(self.cache_dir)!r}, "
            f"cached_datasets={len(self.list_cached())}, "
            f"cache_size={self.get_cache_size() / (1024 * 1024):.1f}MB)"
        )


class _DefaultManager:
    instance: DatasetManager | None = None


def get_default_manager() -> DatasetManager:
    """Return the process-local default manager."""
    if _DefaultManager.instance is None:
        _DefaultManager.instance = DatasetManager()
    return _DefaultManager.instance


def set_cache_dir(cache_dir: str | Path) -> None:
    """Replace the default manager with one using ``cache_dir``."""
    _DefaultManager.instance = DatasetManager(cache_dir=cache_dir)
