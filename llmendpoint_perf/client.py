"""OpenAI-compatible HTTP and SSE streaming client for dataset generation and benchmarking."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import time
from typing import Any
import uuid

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llmendpoint_perf.config import PricingConfig
from llmendpoint_perf.metrics import CallRecord


def build_chat_completions_url(endpoint: str) -> str:
    """Normalize an OpenAI-compatible base endpoint into a `/chat/completions` URL."""
    clean = endpoint.strip().rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return f"{clean}/chat/completions"


def estimate_tokens_from_text(text: str) -> int:
    """Fallback heuristic estimator for token counts when server usage is omitted."""
    if not text:
        return 0
    words = len(text.split())
    char_est = max(1, len(text) // 4)
    return max(int(round(words * 1.3)), char_est)


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate input prompt tokens across text and multimodal content blocks."""
    total = 0
    for msg in messages:
        total += 4  # Message framing tokens
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens_from_text(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    total += estimate_tokens_from_text(str(part.get("text", "")))
                elif part_type == "image_url":
                    total += 258  # Standard vision tile token estimate
    return max(1, total)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class OpenAICompatibleClient:
    """Thread-safe HTTP client for interacting with OpenAI-compatible LLM endpoints."""

    def __init__(
        self,
        endpoint: str,
        api_key: str = "EMPTY",
        timeout_secs: float = 120.0,
        max_connections: int = 200,
    ) -> None:
        self.url = build_chat_completions_url(endpoint)
        self.api_key = api_key or "EMPTY"
        self.timeout_secs = timeout_secs
        limits = httpx.Limits(
            max_connections=max(max_connections, 50),
            max_keepalive_connections=max(max_connections // 2, 25),
        )
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_secs),
            limits=limits,
            follow_redirects=True,
        )
        self._supports_stream_options = True

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
        }
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type((httpx.RequestError, RuntimeError)),
    )
    def generate_text(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> str:
        """Execute a chat completion call to generate text (used for synthetic dataset creation)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = self._client.post(self.url, json=payload, headers=self._headers())
        if resp.status_code >= 500 or resp.status_code == 429:
            raise RuntimeError(
                f"Transient endpoint error HTTP {resp.status_code}: {resp.text[:300]}"
            )
        if resp.status_code != 200:
            raise ValueError(
                f"Endpoint returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            # Handle endpoints that always stream SSE even when stream=False
            pieces: list[str] = []
            for line in resp.text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]" or not data_str:
                    continue
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        text_piece = delta.get("content") or ""
                        pieces.append(text_piece)
                except json.JSONDecodeError:
                    continue
            return "".join(pieces).strip()

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"Response contained no choices: {data}")
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        return str(content).strip()

    def stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        thread_id: int,
        prompt_index: int,
        generation_params: dict[str, Any] | None = None,
        pricing: PricingConfig | None = None,
    ) -> CallRecord:
        """Execute a streaming chat completion request and measure TTFT, TPOT, and E2E latency."""
        request_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if self._supports_stream_options:
            payload["stream_options"] = {"include_usage": True}
        if generation_params:
            for k, v in generation_params.items():
                if k not in ("model", "messages", "stream"):
                    payload[k] = v

        timestamp_start = _iso_utc_now()
        t_start = time.perf_counter()
        t_first_token: float | None = None
        t_last_token: float | None = None
        timestamp_first_token: str | None = None

        status_code: int | None = None
        error_msg: str | None = None
        content_pieces: list[str] = []
        chunk_token_events = 0

        usage_prompt_tokens: int | None = None
        usage_completion_tokens: int | None = None
        usage_reasoning_tokens: int = 0
        usage_cached_tokens: int = 0
        raw_metadata: dict[str, Any] = {"model": model}

        try:
            with self._client.stream(
                "POST", self.url, json=payload, headers=self._headers()
            ) as response:
                status_code = response.status_code

                # If server rejects stream_options with HTTP 400, retry immediately without it
                if status_code == 400 and "stream_options" in payload:
                    body_bytes = response.read()
                    body_text = body_bytes.decode("utf-8", errors="replace")
                    if "stream_options" in body_text:
                        self._supports_stream_options = False
                        payload.pop("stream_options", None)
                        return self._execute_stream_without_options(
                            request_id=request_id,
                            payload=payload,
                            messages=messages,
                            model=model,
                            thread_id=thread_id,
                            prompt_index=prompt_index,
                            pricing=pricing,
                        )
                    error_msg = f"HTTP 400: {body_text[:500]}"
                elif status_code != 200:
                    body_bytes = response.read()
                    body_text = body_bytes.decode("utf-8", errors="replace")
                    error_msg = f"HTTP {status_code}: {body_text[:500]}"
                else:
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type and "text/event-stream" not in content_type:
                        # Non-streaming JSON fallback
                        body_bytes = response.read()
                        now_perf = time.perf_counter()
                        t_first_token = now_perf
                        t_last_token = now_perf
                        timestamp_first_token = _iso_utc_now()
                        data = json.loads(body_bytes.decode("utf-8", errors="replace"))
                        self._extract_non_streaming_data(
                            data,
                            content_pieces,
                            raw_metadata,
                        )
                        usage = data.get("usage") or {}
                        usage_prompt_tokens = usage.get("prompt_tokens")
                        usage_completion_tokens = usage.get("completion_tokens")
                        comp_details = usage.get("completion_tokens_details") or {}
                        usage_reasoning_tokens = int(comp_details.get("reasoning_tokens") or 0)
                        prompt_details = usage.get("prompt_tokens_details") or {}
                        usage_cached_tokens = int(prompt_details.get("cached_tokens") or 0)
                    else:
                        for raw_line in response.iter_lines():
                            line = raw_line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[len("data:") :].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            if "id" in chunk and "id" not in raw_metadata:
                                raw_metadata["id"] = chunk["id"]
                            if "model" in chunk:
                                raw_metadata["model"] = chunk["model"]

                            choices = chunk.get("choices") or []
                            if choices:
                                choice0 = choices[0]
                                if choice0.get("finish_reason"):
                                    raw_metadata["finish_reason"] = choice0["finish_reason"]
                                delta = choice0.get("delta") or {}
                                token_text = delta.get("content") or ""
                                reasoning_text = (
                                    delta.get("reasoning_content")
                                    or delta.get("reasoning")
                                    or ""
                                )
                                if token_text or reasoning_text:
                                    now_perf = time.perf_counter()
                                    if t_first_token is None:
                                        t_first_token = now_perf
                                        timestamp_first_token = _iso_utc_now()
                                    t_last_token = now_perf
                                    chunk_token_events += 1
                                    if token_text:
                                        content_pieces.append(str(token_text))

                            usage = chunk.get("usage")
                            if isinstance(usage, dict):
                                if usage.get("prompt_tokens") is not None:
                                    usage_prompt_tokens = int(usage["prompt_tokens"])
                                if usage.get("completion_tokens") is not None:
                                    usage_completion_tokens = int(usage["completion_tokens"])
                                comp_details = usage.get("completion_tokens_details") or {}
                                if comp_details.get("reasoning_tokens") is not None:
                                    usage_reasoning_tokens = int(comp_details["reasoning_tokens"])
                                prompt_details = usage.get("prompt_tokens_details") or {}
                                if prompt_details.get("cached_tokens") is not None:
                                    usage_cached_tokens = int(prompt_details["cached_tokens"])

        except Exception as exc:  # pylint: disable=broad-except
            error_msg = f"{type(exc).__name__}: {exc}"

        t_end = time.perf_counter()
        timestamp_end = _iso_utc_now()

        return self._finalize_call_record(
            request_id=request_id,
            thread_id=thread_id,
            prompt_index=prompt_index,
            messages=messages,
            timestamp_start=timestamp_start,
            timestamp_first_token=timestamp_first_token,
            timestamp_end=timestamp_end,
            t_start=t_start,
            t_first_token=t_first_token,
            t_last_token=t_last_token,
            t_end=t_end,
            status_code=status_code,
            error_msg=error_msg,
            content_pieces=content_pieces,
            chunk_token_events=chunk_token_events,
            usage_prompt_tokens=usage_prompt_tokens,
            usage_completion_tokens=usage_completion_tokens,
            usage_reasoning_tokens=usage_reasoning_tokens,
            usage_cached_tokens=usage_cached_tokens,
            raw_metadata=raw_metadata,
            pricing=pricing,
        )

    def _execute_stream_without_options(
        self,
        request_id: str,
        payload: dict[str, Any],
        messages: list[dict[str, Any]],
        model: str,
        thread_id: int,
        prompt_index: int,
        pricing: PricingConfig | None,
    ) -> CallRecord:
        del request_id  # Will generate a fresh measurement on retry
        return self.stream_chat_completion(
            messages=messages,
            model=model,
            thread_id=thread_id,
            prompt_index=prompt_index,
            generation_params={
                k: v
                for k, v in payload.items()
                if k not in ("model", "messages", "stream", "stream_options")
            },
            pricing=pricing,
        )

    @staticmethod
    def _extract_non_streaming_data(
        data: dict[str, Any],
        content_pieces: list[str],
        raw_metadata: dict[str, Any],
    ) -> None:
        if "id" in data:
            raw_metadata["id"] = data["id"]
        if "model" in data:
            raw_metadata["model"] = data["model"]
        choices = data.get("choices") or []
        if choices:
            choice0 = choices[0]
            if choice0.get("finish_reason"):
                raw_metadata["finish_reason"] = choice0["finish_reason"]
            msg = choice0.get("message") or {}
            if msg.get("content"):
                content_pieces.append(str(msg["content"]))

    @staticmethod
    def _finalize_call_record(
        request_id: str,
        thread_id: int,
        prompt_index: int,
        messages: list[dict[str, Any]],
        timestamp_start: str,
        timestamp_first_token: str | None,
        timestamp_end: str,
        t_start: float,
        t_first_token: float | None,
        t_last_token: float | None,
        t_end: float,
        status_code: int | None,
        error_msg: str | None,
        content_pieces: list[str],
        chunk_token_events: int,
        usage_prompt_tokens: int | None,
        usage_completion_tokens: int | None,
        usage_reasoning_tokens: int,
        usage_cached_tokens: int,
        raw_metadata: dict[str, Any],
        pricing: PricingConfig | None,
    ) -> CallRecord:
        e2e_latency_ms = max((t_end - t_start) * 1000.0, 0.0)
        full_response = "".join(content_pieces)

        if status_code != 200 or error_msg is not None:
            return CallRecord(
                request_id=request_id,
                thread_id=thread_id,
                prompt_index=prompt_index,
                timestamp_start=timestamp_start,
                timestamp_first_token=timestamp_first_token,
                timestamp_end=timestamp_end,
                status_code=status_code,
                error=error_msg,
                ttft_ms=None,
                tpot_ms=None,
                e2e_latency_ms=round(e2e_latency_ms, 3),
                response_content=full_response,
                raw_response_metadata=raw_metadata,
            )

        input_tokens = (
            usage_prompt_tokens
            if (usage_prompt_tokens is not None and usage_prompt_tokens > 0)
            else estimate_prompt_tokens(messages)
        )
        if usage_completion_tokens is not None and usage_completion_tokens > 0:
            output_tokens = usage_completion_tokens
        elif chunk_token_events > 0:
            output_tokens = max(chunk_token_events, estimate_tokens_from_text(full_response))
        else:
            output_tokens = estimate_tokens_from_text(full_response)

        if t_first_token is not None:
            ttft_ms = max((t_first_token - t_start) * 1000.0, 0.0)
            decode_window_secs = max((t_last_token or t_end) - t_first_token, 0.0)
        else:
            ttft_ms = e2e_latency_ms
            decode_window_secs = 0.0

        if output_tokens > 1 and decode_window_secs > 0:
            tpot_ms = (decode_window_secs * 1000.0) / (output_tokens - 1)
            output_tps = output_tokens / decode_window_secs
        elif output_tokens > 0 and e2e_latency_ms > 0:
            tpot_ms = 0.0
            output_tps = output_tokens / (e2e_latency_ms / 1000.0)
        else:
            tpot_ms = 0.0
            output_tps = 0.0

        cost_usd = (
            pricing.compute_cost(input_tokens, output_tokens, usage_cached_tokens)
            if pricing is not None
            else 0.0
        )

        return CallRecord(
            request_id=request_id,
            thread_id=thread_id,
            prompt_index=prompt_index,
            timestamp_start=timestamp_start,
            timestamp_first_token=timestamp_first_token,
            timestamp_end=timestamp_end,
            status_code=status_code,
            error=None,
            ttft_ms=round(ttft_ms, 3),
            tpot_ms=round(tpot_ms, 3),
            e2e_latency_ms=round(e2e_latency_ms, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=usage_reasoning_tokens,
            cached_input_tokens=usage_cached_tokens,
            output_tokens_per_sec=round(output_tps, 3),
            cost_usd=cost_usd,
            response_content=full_response,
            raw_response_metadata=raw_metadata,
        )

    def close(self) -> None:
        """Close the underlying HTTP client connection pool."""
        self._client.close()

    def __enter__(self) -> OpenAICompatibleClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
