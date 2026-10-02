"""Unit tests for local filesystem and mocked GCS storage backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llmendpoint_perf.storage import TaskStorage, parse_gcs_uri


class FakeBlob:
    """In-memory mock of a google.cloud.storage.Blob."""

    def __init__(self, name: str, store: dict[str, bytes]) -> None:
        self.name = name
        self._store = store

    def exists(self) -> bool:
        return self.name in self._store

    def upload_from_string(self, data: str | bytes, content_type: str | None = None) -> None:
        del content_type
        if isinstance(data, str):
            self._store[self.name] = data.encode("utf-8")
        else:
            self._store[self.name] = bytes(data)

    def download_as_text(self, encoding: str = "utf-8") -> str:
        return self._store[self.name].decode(encoding)

    def download_as_bytes(self) -> bytes:
        return self._store[self.name]


class FakeBucket:
    """In-memory mock of a google.cloud.storage.Bucket."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.store: dict[str, bytes] = {}

    def blob(self, blob_name: str) -> FakeBlob:
        return FakeBlob(blob_name, self.store)

    def list_blobs(self, prefix: str = "") -> list[FakeBlob]:
        return [
            FakeBlob(k, self.store)
            for k in sorted(self.store.keys())
            if k.startswith(prefix)
        ]


class FakeGCSClient:
    """In-memory mock of a google.cloud.storage.Client."""

    def __init__(self) -> None:
        self.buckets: dict[str, FakeBucket] = {}

    def bucket(self, name: str) -> FakeBucket:
        if name not in self.buckets:
            self.buckets[name] = FakeBucket(name)
        return self.buckets[name]


def test_parse_gcs_uri() -> None:
    bucket, prefix = parse_gcs_uri("gs://my-bucket/path/to/tasks/")
    assert bucket == "my-bucket"
    assert prefix == "path/to/tasks"

    bucket2, prefix2 = parse_gcs_uri("gs://only-bucket")
    assert bucket2 == "only-bucket"
    assert prefix2 == ""


def test_local_task_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMENDPOINTPERF_BASEPATH", str(tmp_path))
    storage = TaskStorage("demo-task")
    assert not storage.exists("config.yaml")

    storage.write_text("config.yaml", "hello: world\n")
    assert storage.exists("config.yaml")
    assert storage.read_text("config.yaml") == "hello: world\n"

    with storage.open_append_stream("runs/20261001-120000/calls.jsonl") as stream:
        stream.write_line('{"idx": 1}')
        stream.write_line('{"idx": 2}')

    with storage.open_append_stream("runs/20261001-130000/calls.jsonl") as stream:
        stream.write_line('{"idx": 3}')

    assert storage.list_runs() == ["20261001-120000", "20261001-130000"]
    calls_text = storage.read_text("runs/20261001-120000/calls.jsonl")
    assert calls_text.splitlines() == ['{"idx": 1}', '{"idx": 2}']


def test_gcs_task_storage() -> None:
    fake_gcs: Any = FakeGCSClient()
    storage = TaskStorage(
        task_name="gcs-task",
        base_path="gs://perf-bucket/benchmarks",
        gcs_client=fake_gcs,
    )
    assert storage.is_gcs is True
    assert storage.task_uri == "gs://perf-bucket/benchmarks/gcs-task"

    storage.write_text("config.yaml", "key: val\n")
    assert storage.exists("config.yaml")
    assert storage.read_text("config.yaml") == "key: val\n"

    with storage.open_append_stream("runs/20261001-150000/log.txt") as log_stream:
        log_stream.write_line("line 1")
        log_stream.write_line("line 2")

    assert storage.list_runs() == ["20261001-150000"]
    assert storage.read_text("runs/20261001-150000/log.txt") == "line 1\nline 2\n"
