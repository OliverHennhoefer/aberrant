"""Dataset transfer and artifact validation backends."""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import urllib.request
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import Protocol
from urllib.error import URLError

import numpy as np
from tqdm import tqdm

from aberrant.stream.dataset.registry import DatasetInfo


class DownloadBackend(Protocol):
    """Transfer a URL into a caller-owned destination path."""

    def download(self, url: str, destination: Path) -> None:
        """Download one artifact or raise after exhausting retries."""
        ...


class UrlLibDownloadBackend:
    """Retrying urllib transfer backend with optional progress reporting.

    Args:
        retries: Positive maximum number of transfer attempts.
        timeout: Positive per-request timeout in seconds.
        backoff_seconds: Non-negative base delay for exponential retry backoff.
            Delay before retry ``n`` is ``backoff_seconds * 2**(n - 1)``.
        show_progress: Display a byte progress bar when downloading.
        logger: Logger for retry warnings. ``None`` uses the module logger.
    """

    def __init__(
        self,
        *,
        retries: int = 3,
        timeout: float = 30.0,
        backoff_seconds: float = 1.0,
        show_progress: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        if retries <= 0:
            raise ValueError("retries must be positive")
        if timeout <= 0.0:
            raise ValueError("timeout must be positive")
        if backoff_seconds < 0.0:
            raise ValueError("backoff_seconds must be non-negative")
        self.retries = retries
        self.timeout = timeout
        self.backoff_seconds = backoff_seconds
        self.show_progress = show_progress
        self.logger = logger or logging.getLogger(__name__)

    def download(self, url: str, destination: Path) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with (
                    urllib.request.urlopen(url, timeout=self.timeout) as response,
                    destination.open("wb") as file,
                ):
                    progress: tqdm[object] | None = None
                    try:
                        if self.show_progress:
                            progress = tqdm(
                                total=int(response.headers.get("Content-Length", 0)),
                                unit="B",
                                unit_scale=True,
                                desc=f"Downloading {destination.name}",
                            )
                        while chunk := response.read(8192):
                            file.write(chunk)
                            if progress is not None:
                                progress.update(len(chunk))
                    finally:
                        if progress is not None:
                            progress.close()
                return
            except (
                URLError,
                TimeoutError,
                OSError,
                IncompleteRead,
                RemoteDisconnected,
            ) as exc:
                last_error = exc
                if attempt < self.retries:
                    wait_seconds = self.backoff_seconds * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Download attempt %s/%s failed: %s; retrying in %.1fs",
                        attempt,
                        self.retries,
                        exc,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)

        raise RuntimeError(
            f"Failed to download artifact after {self.retries} attempts: {last_error}"
        ) from last_error


class DatasetArtifactValidator:
    """Validate NPZ structure and trusted SHA256 metadata."""

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def validate(self, path: Path, info: DatasetInfo) -> str:
        """Return the verified digest or raise for an invalid artifact."""
        try:
            with np.load(path) as archive:
                if "X" not in archive or "y" not in archive:
                    raise ValueError("Dataset archive must contain X and y arrays")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid NPZ dataset artifact: {path}") from exc

        actual = self.sha256(path)
        if not hmac.compare_digest(actual, info.sha256):
            raise ValueError(
                f"Checksum mismatch for {path.name}: expected {info.sha256}, got {actual}"
            )
        return actual
