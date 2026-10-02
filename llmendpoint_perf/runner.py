"""Multi-threaded benchmark execution engine for llmendpoint-perf."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import random
import threading
import time
from typing import Any

import numpy as np

from llmendpoint_perf.client import OpenAICompatibleClient
from llmendpoint_perf.config import TaskConfig
from llmendpoint_perf.inspector import format_run_summary_report
from llmendpoint_perf.logging_utils import DualLogger
from llmendpoint_perf.metrics import CallRecord, RunResults, aggregate_run_results
from llmendpoint_perf.storage import TaskStorage


def load_prompts(storage: TaskStorage) -> list[dict[str, Any]]:
    """Load and validate evaluation prompts from `prompts.jsonl`."""
    if not storage.exists("prompts.jsonl"):
        raise FileNotFoundError(
            f"Prompts file not found at {storage.task_uri}/prompts.jsonl. "
            f"Run 'llmendpoint-perf generate_dataset {storage.task_name}' first."
        )
    raw = storage.read_text("prompts.jsonl")
    prompts: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_no} of prompts.jsonl: {exc}"
            ) from exc
        if not isinstance(item, dict) or "messages" not in item:
            raise ValueError(
                f"Line {line_no} of prompts.jsonl must be a JSON object with a 'messages' key."
            )
        prompts.append(item)

    if not prompts:
        raise ValueError(f"prompts.jsonl at {storage.task_uri}/prompts.jsonl is empty.")
    return prompts


def generate_run_id(storage: TaskStorage) -> str:
    """Generate a unique UTC timestamp run ID (`YYYYMMDD-HHMMSS`)."""
    existing = set(storage.list_runs())
    while True:
        candidate = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        if candidate not in existing:
            return candidate
        time.sleep(0.25)


class RequestPacer:
    """Thread-safe pacer enforcing `wait_time_between_requests_ms` between request dispatches."""

    def __init__(self, wait_time_ms: float) -> None:
        self._wait_secs = max(wait_time_ms / 1000.0, 0.0)
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def acquire_slot(self, deadline: float | None = None) -> bool:
        """Wait until the next dispatch slot is available; return False if past deadline."""
        if self._wait_secs <= 0.0:
            return deadline is None or time.perf_counter() < deadline

        with self._lock:
            now = time.perf_counter()
            target_time = max(now, self._next_allowed_time)
            if deadline is not None and target_time >= deadline:
                return False
            self._next_allowed_time = target_time + self._wait_secs
            sleep_duration = target_time - now

        if sleep_duration > 0:
            time.sleep(sleep_duration)
        return deadline is None or time.perf_counter() < deadline


def run_evaluation_task(
    storage: TaskStorage,
    config: TaskConfig | None = None,
    config_overrides: list[str] | tuple[str, ...] = (),
    run_id: str | None = None,
) -> RunResults:
    """Execute a benchmark run for a task and store all run artifacts under `runs/<run_id>/`."""
    if config is None:
        if not storage.exists("config.yaml"):
            raise FileNotFoundError(
                f"Task configuration not found at {storage.task_uri}/config.yaml. "
                f"Run 'llmendpoint-perf init {storage.task_name}' first."
            )
        config = TaskConfig.from_yaml(storage.read_text("config.yaml"))

    if config_overrides:
        config = config.apply_overrides(config_overrides)

    prompts = load_prompts(storage)
    eval_cfg = config.evaluation

    if run_id is None:
        run_id = generate_run_id(storage)

    run_dir = storage.run_rel_dir(run_id)
    storage.write_text(f"{run_dir}/config.yaml", config.to_yaml())

    with DualLogger(storage, f"{run_dir}/log.txt") as logger:
        logger.info(
            f"Starting evaluation run '{run_id}' for task '{storage.task_name}'"
        )
        logger.info(
            f"Artifacts directory: {storage.run_uri(run_id)}"
        )
        logger.info(
            f"Target model='{eval_cfg.model}' at endpoint='{eval_cfg.model_endpoint}' | "
            f"threads={eval_cfg.num_threads} | wait_ms={eval_cfg.wait_time_between_requests_ms} | "
            f"run_time_secs={eval_cfg.run_time_secs} | max_requests={eval_cfg.max_requests} | "
            f"prompts={len(prompts)} ({eval_cfg.sampling_strategy})"
        )

        api_key = eval_cfg.resolve_api_key()
        with OpenAICompatibleClient(
            endpoint=eval_cfg.model_endpoint,
            api_key=api_key,
            timeout_secs=eval_cfg.request_timeout_secs,
            max_connections=max(eval_cfg.num_threads * 4, 50),
        ) as client:
            # Optional Warmup Phase
            if eval_cfg.warmup_requests > 0:
                logger.info(
                    f"Executing {eval_cfg.warmup_requests} warmup requests (excluded from metrics)..."
                )
                warmup_workers = min(eval_cfg.num_threads, eval_cfg.warmup_requests)
                with ThreadPoolExecutor(max_workers=warmup_workers) as warm_pool:
                    warm_futs = [
                        warm_pool.submit(
                            client.stream_chat_completion,
                            messages=prompts[i % len(prompts)]["messages"],
                            model=eval_cfg.model,
                            thread_id=i % warmup_workers,
                            prompt_index=i % len(prompts),
                            generation_params=eval_cfg.generation_params,
                            pricing=eval_cfg.pricing,
                        )
                        for i in range(eval_cfg.warmup_requests)
                    ]
                    for wf in as_completed(warm_futs):
                        _ = wf.result()
                logger.info("Warmup phase complete.")

            # Main Benchmark Phase
            records: list[CallRecord] = []
            records_lock = threading.Lock()
            counter_lock = threading.Lock()
            dispatched_count = 0
            active_requests = 0
            rng = random.Random(42)

            pacer = RequestPacer(eval_cfg.wait_time_between_requests_ms)
            timestamp_start = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            t_bench_start = time.perf_counter()
            deadline = t_bench_start + eval_cfg.run_time_secs
            last_progress_log = t_bench_start
            progress_interval_secs = min(5.0, max(1.0, eval_cfg.run_time_secs / 5.0))

            def _next_prompt() -> tuple[int, dict[str, Any]] | None:
                nonlocal dispatched_count, active_requests
                with counter_lock:
                    if time.perf_counter() >= deadline:
                        return None
                    if (
                        eval_cfg.max_requests is not None
                        and dispatched_count >= eval_cfg.max_requests
                    ):
                        return None
                    seq_idx = dispatched_count
                    dispatched_count += 1
                    active_requests += 1
                    if eval_cfg.sampling_strategy == "random":
                        p_idx = rng.randrange(len(prompts))
                    else:
                        p_idx = seq_idx % len(prompts)
                    return p_idx, prompts[p_idx]

            with storage.open_append_stream(f"{run_dir}/calls.jsonl") as calls_stream:

                def _worker_loop(thread_id: int) -> None:
                    nonlocal active_requests, last_progress_log
                    while True:
                        if not pacer.acquire_slot(deadline=deadline):
                            break
                        selection = _next_prompt()
                        if selection is None:
                            break
                        prompt_idx, prompt_obj = selection
                        record = client.stream_chat_completion(
                            messages=prompt_obj["messages"],
                            model=eval_cfg.model,
                            thread_id=thread_id,
                            prompt_index=prompt_idx,
                            generation_params=eval_cfg.generation_params,
                            pricing=eval_cfg.pricing,
                        )
                        calls_stream.write_line(
                            json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
                        )
                        now_perf = time.perf_counter()
                        should_log = False
                        with records_lock:
                            records.append(record)
                            if now_perf - last_progress_log >= progress_interval_secs:
                                last_progress_log = now_perf
                                should_log = True
                            snapshot = list(records) if should_log else []

                        with counter_lock:
                            active_requests -= 1
                            current_active = active_requests

                        if should_log and snapshot:
                            elapsed = max(now_perf - t_bench_start, 1e-3)
                            ok_recs = [r for r in snapshot if r.is_success]
                            err_cnt = len(snapshot) - len(ok_recs)
                            rps = len(snapshot) / elapsed
                            ttfts = [r.ttft_ms for r in ok_recs if r.ttft_ms is not None]
                            tpots = [r.tpot_ms for r in ok_recs if r.tpot_ms is not None]
                            ttft_p50 = float(np.percentile(ttfts, 50)) if ttfts else 0.0
                            ttft_p95 = float(np.percentile(ttfts, 95)) if ttfts else 0.0
                            tpot_p50 = float(np.percentile(tpots, 50)) if tpots else 0.0
                            logger.info(
                                f"Progress [{elapsed:.1f}s/{eval_cfg.run_time_secs:.1f}s]: "
                                f"completed={len(snapshot)} | active={current_active} | "
                                f"errors={err_cnt} | rps={rps:.2f} | "
                                f"TTFT p50={ttft_p50:.1f}ms p95={ttft_p95:.1f}ms | "
                                f"TPOT p50={tpot_p50:.2f}ms/tok"
                            )

                with ThreadPoolExecutor(max_workers=eval_cfg.num_threads) as pool:
                    worker_futures = [
                        pool.submit(_worker_loop, tid)
                        for tid in range(eval_cfg.num_threads)
                    ]
                    for wf in as_completed(worker_futures):
                        wf.result()

            actual_duration = max(time.perf_counter() - t_bench_start, 1e-6)
            timestamp_end = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        results = aggregate_run_results(
            run_id=run_id,
            task_name=storage.task_name,
            config=config,
            records=records,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            actual_duration_secs=actual_duration,
        )

        results_json = json.dumps(results.model_dump(mode="json"), ensure_ascii=False) + "\n"
        storage.write_text(f"{run_dir}/results.jsonl", results_json)

        report_text = format_run_summary_report(results)
        logger.raw("\n" + report_text + "\n")
        logger.info(
            f"Run '{run_id}' complete. Results saved to {storage.run_uri(run_id)}/results.jsonl"
        )

    return results
