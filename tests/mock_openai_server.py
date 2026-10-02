"""Lightweight local mock OpenAI-compatible HTTP server supporting SSE streaming and non-streaming."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any


class MockOpenAIHandler(BaseHTTPRequestHandler):
    """HTTP handler simulating `/v1/chat/completions` in both streaming and non-streaming modes."""

    ttft_delay_secs: float = 0.01
    tpot_delay_secs: float = 0.002
    fail_every_n: int = 0
    request_counter: int = 0
    counter_lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:  # pylint: disable=redefined-builtin
        """Silence standard HTTP server stderr logging during tests."""
        return

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Handle POST /v1/chat/completions."""
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')
            return

        content_len = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_len)
        payload = json.loads(raw_body.decode("utf-8"))

        with self.counter_lock:
            MockOpenAIHandler.request_counter += 1
            req_num = MockOpenAIHandler.request_counter

        if self.fail_every_n > 0 and (req_num % self.fail_every_n == 0):
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "Rate limit exceeded"}}')
            return

        model = payload.get("model", "mock-model")
        stream = bool(payload.get("stream", False))
        messages = payload.get("messages", [])

        # Extract a snippet from user prompt to make responses deterministic yet unique
        user_snippet = f"item-{req_num}"
        if messages:
            last_content = messages[-1].get("content", "")
            if isinstance(last_content, str):
                user_snippet = last_content[:40].replace("\n", " ")
            elif isinstance(last_content, list):
                for part in last_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_snippet = str(part.get("text", ""))[:40].replace("\n", " ")

        if not stream:
            response_obj = {
                "id": f"chatcmpl-mock-{req_num}",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Synthetic question #{req_num}: How does product feature X compare to Y ({user_snippet})?",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 45,
                    "completion_tokens": 16,
                    "total_tokens": 61,
                },
            }
            body = json.dumps(response_obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Streaming SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        if self.ttft_delay_secs > 0:
            time.sleep(self.ttft_delay_secs)

        tokens = ["Here ", "is ", "a ", "detailed ", "answer ", "for ", f"#{req_num}."]
        for idx, tok in enumerate(tokens):
            if idx > 0 and self.tpot_delay_secs > 0:
                time.sleep(self.tpot_delay_secs)
            chunk = {
                "id": f"chatcmpl-stream-{req_num}",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": tok},
                        "finish_reason": None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

        # Final finish_reason chunk + usage chunk
        final_chunk = {
            "id": f"chatcmpl-stream-{req_num}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 64,
                "completion_tokens": len(tokens),
                "total_tokens": 64 + len(tokens),
                "completion_tokens_details": {"reasoning_tokens": 2},
                "prompt_tokens_details": {"cached_tokens": 16},
            },
        }
        self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class MockOpenAIServer:
    """Context manager running a background ThreadingHTTPServer for OpenAI endpoint tests."""

    def __init__(
        self,
        ttft_delay_secs: float = 0.01,
        tpot_delay_secs: float = 0.002,
        fail_every_n: int = 0,
    ) -> None:
        MockOpenAIHandler.ttft_delay_secs = ttft_delay_secs
        MockOpenAIHandler.tpot_delay_secs = tpot_delay_secs
        MockOpenAIHandler.fail_every_n = fail_every_n
        MockOpenAIHandler.request_counter = 0

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
        host, port = self._server.server_address
        self.base_url = f"http://{host}:{port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> MockOpenAIServer:
        """Start the background HTTP server thread."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Shut down the background HTTP server thread."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=3.0)

    def __enter__(self) -> MockOpenAIServer:
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
