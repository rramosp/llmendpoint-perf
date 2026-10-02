"""Unified storage abstraction for local filesystem and Google Cloud Storage (gs://)."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
from typing import Any


def resolve_base_path(explicit_base_path: str | None = None) -> str:
    """Resolve the storage base path from an explicit value or environment variables."""
    if explicit_base_path:
        return explicit_base_path.rstrip("/")
    env_val = os.environ.get("LLMENDPOINTPERF_BASEPATH") or os.environ.get(
        "LLMENDPOINTPERF_GCS_BASEPATH"
    )
    if not env_val:
        raise RuntimeError(
            "Storage base path is not configured. Please set the environment variable "
            "'LLMENDPOINTPERF_BASEPATH' (e.g., 'gs://my-bucket/llmperf' or '/path/to/local/dir') "
            "or pass '--base-path'."
        )
    return env_val.rstrip("/")


def is_gcs_uri(uri: str) -> bool:
    """Return True if the URI starts with gs://."""
    return uri.startswith("gs://")


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split a gs://bucket/prefix URI into (bucket_name, object_prefix)."""
    if not is_gcs_uri(uri):
        raise ValueError(f"Not a valid GCS URI: {uri}")
    without_scheme = uri[len("gs://") :]
    parts = without_scheme.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1].strip("/") if len(parts) > 1 else ""
    if not bucket_name:
        raise ValueError(f"GCS URI is missing bucket name: {uri}")
    return bucket_name, prefix


class AppendStream:
    """Thread-safe line-oriented append stream that persists to local disk or GCS."""

    def __init__(self, storage: TaskStorage, rel_path: str) -> None:
        self._storage = storage
        self._rel_path = rel_path.lstrip("/")
        self._lock = threading.Lock()
        self._closed = False

        if self._storage.is_gcs:
            self._tmp_file = tempfile.NamedTemporaryFile(
                mode="w+", encoding="utf-8", suffix=".tmp", delete=False
            )
            self._file_handle = self._tmp_file
        else:
            full_path = self._storage.local_root / self._rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(full_path, "w", encoding="utf-8")

    def write_line(self, line: str) -> None:
        """Thread-safely write a single line (adding trailing newline if missing) and flush."""
        with self._lock:
            if self._closed:
                raise RuntimeError(f"Cannot write to closed stream: {self._rel_path}")
            if not line.endswith("\n"):
                line = line + "\n"
            self._file_handle.write(line)
            self._file_handle.flush()

    def flush_to_remote(self) -> None:
        """If backed by GCS, upload the current snapshot of the file to the bucket."""
        with self._lock:
            self._file_handle.flush()
            if self._storage.is_gcs:
                tmp_path = Path(self._file_handle.name)
                data = tmp_path.read_text(encoding="utf-8")
                self._storage.write_text(self._rel_path, data)

    def close(self) -> None:
        """Flush and close the stream, uploading the final artifact if on GCS."""
        with self._lock:
            if self._closed:
                return
            self._file_handle.flush()
            self._file_handle.close()
            self._closed = True
            if self._storage.is_gcs:
                tmp_path = Path(self._file_handle.name)
                try:
                    data = tmp_path.read_text(encoding="utf-8")
                    self._storage.write_text(self._rel_path, data)
                finally:
                    tmp_path.unlink(missing_ok=True)

    def __enter__(self) -> AppendStream:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class TaskStorage:
    """Storage interface scoped to `$LLMENDPOINTPERF_BASEPATH/<task-name>`."""

    def __init__(
        self,
        task_name: str,
        base_path: str | None = None,
        gcs_client: Any | None = None,
    ) -> None:
        if not task_name or "/" in task_name.strip("/"):
            raise ValueError(
                f"Invalid task_name '{task_name}'. Task name must be a non-empty identifier."
            )
        self.task_name = task_name.strip()
        self.base_path = resolve_base_path(base_path)
        self.is_gcs = is_gcs_uri(self.base_path)
        self._gcs_client = gcs_client

        if self.is_gcs:
            bucket_name, base_prefix = parse_gcs_uri(self.base_path)
            self.bucket_name = bucket_name
            self.task_prefix = (
                f"{base_prefix}/{self.task_name}" if base_prefix else self.task_name
            )
            self.local_root = Path("/nonexistent")
        else:
            self.bucket_name = ""
            self.task_prefix = ""
            self.local_root = Path(self.base_path).expanduser().resolve() / self.task_name

    @property
    def task_uri(self) -> str:
        """Full URI or path to the task directory."""
        if self.is_gcs:
            return f"gs://{self.bucket_name}/{self.task_prefix}"
        return str(self.local_root)

    def run_rel_dir(self, run_id: str) -> str:
        """Relative path for a specific run directory (`runs/YYYYMMDD-HHMMSS`)."""
        return f"runs/{run_id}"

    def run_uri(self, run_id: str) -> str:
        """Full URI or path to a specific run directory."""
        return f"{self.task_uri}/{self.run_rel_dir(run_id)}"

    def _get_bucket(self) -> Any:
        if self._gcs_client is None:
            from google.cloud import storage  # type: ignore

            self._gcs_client = storage.Client()
        return self._gcs_client.bucket(self.bucket_name)

    def _blob_name(self, rel_path: str) -> str:
        clean = rel_path.lstrip("/")
        return f"{self.task_prefix}/{clean}" if clean else self.task_prefix

    def exists(self, rel_path: str) -> bool:
        """Check whether a relative artifact path exists under the task."""
        clean = rel_path.lstrip("/")
        if self.is_gcs:
            bucket = self._get_bucket()
            blob = bucket.blob(self._blob_name(clean))
            return bool(blob.exists())
        return (self.local_root / clean).exists()

    def write_text(self, rel_path: str, content: str) -> None:
        """Write UTF-8 text content to `rel_path` inside the task directory."""
        clean = rel_path.lstrip("/")
        if self.is_gcs:
            bucket = self._get_bucket()
            blob = bucket.blob(self._blob_name(clean))
            blob.upload_from_string(content, content_type="text/plain; charset=utf-8")
        else:
            target = self.local_root / clean
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def read_text(self, rel_path: str) -> str:
        """Read UTF-8 text content from `rel_path` inside the task directory."""
        clean = rel_path.lstrip("/")
        if self.is_gcs:
            bucket = self._get_bucket()
            blob = bucket.blob(self._blob_name(clean))
            if not blob.exists():
                raise FileNotFoundError(
                    f"Artifact not found in GCS: gs://{self.bucket_name}/{self._blob_name(clean)}"
                )
            return str(blob.download_as_text(encoding="utf-8"))
        target = self.local_root / clean
        if not target.exists():
            raise FileNotFoundError(f"Artifact not found: {target}")
        return target.read_text(encoding="utf-8")

    def write_bytes(self, rel_path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Write raw bytes to `rel_path` inside the task directory."""
        clean = rel_path.lstrip("/")
        if self.is_gcs:
            bucket = self._get_bucket()
            blob = bucket.blob(self._blob_name(clean))
            blob.upload_from_string(data, content_type=content_type)
        else:
            target = self.local_root / clean
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def read_bytes(self, rel_path: str) -> bytes:
        """Read raw bytes from `rel_path` inside the task directory."""
        clean = rel_path.lstrip("/")
        if self.is_gcs:
            bucket = self._get_bucket()
            blob = bucket.blob(self._blob_name(clean))
            if not blob.exists():
                raise FileNotFoundError(
                    f"Artifact not found in GCS: gs://{self.bucket_name}/{self._blob_name(clean)}"
                )
            return bytes(blob.download_as_bytes())
        target = self.local_root / clean
        if not target.exists():
            raise FileNotFoundError(f"Artifact not found: {target}")
        return target.read_bytes()

    def open_append_stream(self, rel_path: str) -> AppendStream:
        """Open a line-buffered AppendStream for streaming logs or JSONL records."""
        return AppendStream(self, rel_path)

    def list_runs(self) -> list[str]:
        """Return a chronologically sorted list of run IDs (`YYYYMMDD-HHMMSS`) for this task."""
        if self.is_gcs:
            bucket = self._get_bucket()
            prefix = f"{self.task_prefix}/runs/"
            blobs = bucket.list_blobs(prefix=prefix)
            run_ids: set[str] = set()
            for blob in blobs:
                suffix = blob.name[len(prefix) :]
                parts = suffix.split("/", 1)
                if parts and parts[0]:
                    run_ids.add(parts[0])
            return sorted(run_ids)

        runs_dir = self.local_root / "runs"
        if not runs_dir.exists():
            return []
        return sorted(
            d.name for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )


def list_external_images(source_uri: str, gcs_client: Any | None = None) -> list[bytes]:
    """Load image bytes from a local directory or GCS prefix for multimodal dataset sampling."""
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    if is_gcs_uri(source_uri):
        bucket_name, prefix = parse_gcs_uri(source_uri)
        if gcs_client is None:
            from google.cloud import storage  # type: ignore

            gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        images: list[bytes] = []
        for blob in blobs:
            if blob.name.lower().endswith(valid_exts):
                images.append(bytes(blob.download_as_bytes()))
        return images

    source_path = Path(source_uri).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Image source path does not exist: {source_path}")
    if source_path.is_file():
        return [source_path.read_bytes()]
    files = sorted(
        p for p in source_path.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts
    )
    return [p.read_bytes() for p in files]


def list_tasks(base_path: str | None = None, gcs_client: Any | None = None) -> list[str]:
    """Return a sorted list of evaluation task names under `$LLMENDPOINTPERF_BASEPATH`."""
    resolved = resolve_base_path(base_path)
    if is_gcs_uri(resolved):
        bucket_name, base_prefix = parse_gcs_uri(resolved)
        if gcs_client is None:
            from google.cloud import storage  # type: ignore

            gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        prefix = f"{base_prefix}/" if base_prefix else ""
        blobs = bucket.list_blobs(prefix=prefix)
        tasks: set[str] = set()
        for blob in blobs:
            suffix = blob.name[len(prefix) :]
            parts = suffix.split("/", 1)
            if len(parts) > 1 and parts[0] and not parts[0].startswith("."):
                tasks.add(parts[0])
        return sorted(tasks)

    root = Path(resolved).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )

