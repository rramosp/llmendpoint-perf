"""Dual logger writing identical formatted messages to stdout and a task storage log file."""

from __future__ import annotations

from datetime import datetime, timezone
import sys
import threading
from typing import Any

from llmendpoint_perf.storage import AppendStream, TaskStorage


class DualLogger:
    """Thread-safe logger that writes identical lines to stdout and a storage AppendStream."""

    def __init__(
        self,
        storage: TaskStorage,
        log_rel_path: str,
        include_timestamps: bool = True,
    ) -> None:
        self._storage = storage
        self._log_rel_path = log_rel_path
        self._include_timestamps = include_timestamps
        self._stream: AppendStream = storage.open_append_stream(log_rel_path)
        self._lock = threading.Lock()

    def info(self, message: str) -> None:
        """Log an informational message to both stdout and the log file."""
        self._emit("INFO", message)

    def warning(self, message: str) -> None:
        """Log a warning message to both stdout and the log file."""
        self._emit("WARN", message)

    def error(self, message: str) -> None:
        """Log an error message to both stdout and the log file."""
        self._emit("ERROR", message)

    def raw(self, block: str) -> None:
        """Write a pre-formatted text block (e.g., summary table) verbatim to stdout and log file."""
        with self._lock:
            for line in block.splitlines():
                sys.stdout.write(line + "\n")
                self._stream.write_line(line)
            sys.stdout.flush()

    def _emit(self, level: str, message: str) -> None:
        with self._lock:
            if self._include_timestamps:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                formatted = f"[{ts}] [{level}] {message}"
            else:
                formatted = f"[{level}] {message}"
            sys.stdout.write(formatted + "\n")
            sys.stdout.flush()
            self._stream.write_line(formatted)

    def flush(self) -> None:
        """Flush buffered logs to remote storage if applicable."""
        self._stream.flush_to_remote()

    def close(self) -> None:
        """Close the underlying log stream."""
        self._stream.close()

    def __enter__(self) -> DualLogger:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
