"""Telemetry data structures and statistical aggregation functions for benchmark runs."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from llmendpoint_perf.config import TaskConfig


class CallRecord(BaseModel):
    """Per-request telemetry record persisted to `runs/YYYYMMDD-HHMMSS/calls.jsonl`."""

    request_id: str
    thread_id: int
    prompt_index: int
    timestamp_start: str
    timestamp_first_token: str | None = None
    timestamp_end: str
    status_code: int | None = None
    error: str | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    output_tokens_without_thinking: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens_per_sec: float = 0.0
    cost_usd: float = 0.0
    response_content: str = ""
    raw_response_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _backfill_output_tokens_without_thinking(cls, data: Any) -> Any:
        if isinstance(data, dict) and "output_tokens_without_thinking" not in data:
            out_tok = int(data.get("output_tokens") or 0)
            reas_tok = int(data.get("reasoning_tokens") or 0)
            data = dict(data)
            data["output_tokens_without_thinking"] = max(out_tok - reas_tok, 0)
        return data

    @property
    def is_success(self) -> bool:
        """Return True if the request succeeded with HTTP 200 and no error."""
        return self.status_code == 200 and self.error is None


class DistributionStats(BaseModel):
    """Statistical distribution summary for a numeric metric."""

    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    max: float = 0.0

    @classmethod
    def from_values(cls, values: list[float | int]) -> DistributionStats:
        """Compute summary statistics and percentiles from a list of numeric samples."""
        if not values:
            return cls()
        arr = np.asarray(values, dtype=np.float64)
        p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
        return cls(
            count=len(values),
            mean=round(float(np.mean(arr)), 4),
            std=round(float(np.std(arr)), 4),
            min=round(float(np.min(arr)), 4),
            p50=round(float(p50), 4),
            p90=round(float(p90), 4),
            p95=round(float(p95), 4),
            p99=round(float(p99), 4),
            max=round(float(np.max(arr)), 4),
        )


class ThroughputMetrics(BaseModel):
    """System-level request and token throughput metrics."""

    rps: float = 0.0
    goodput_rps: float = 0.0
    slo_goodput_rps: float = 0.0
    input_tps: float = 0.0
    output_tps: float = 0.0
    total_tps: float = 0.0


class CostMetrics(BaseModel):
    """Run-level cost efficiency metrics."""

    total_cost_usd: float = 0.0
    mean_cost_per_request_usd: float = 0.0
    cost_per_1k_output_tokens_usd: float = 0.0
    cost_per_1k_total_tokens_usd: float = 0.0
    output_tps_per_usd_per_hour: float = 0.0


class RunResults(BaseModel):
    """Aggregated summary metrics written to `runs/YYYYMMDD-HHMMSS/results.jsonl`."""

    run_id: str
    task_name: str
    timestamp_start: str
    timestamp_end: str
    actual_duration_secs: float
    model_endpoint: str
    model: str
    num_threads: int
    wait_time_between_requests_ms: float
    configured_run_time_secs: float
    dataset_generation_prompt: str = ""
    dataset_num_items: int = 0
    dataset_generation_model: str = ""
    dataset_generation_model_endpoint: str = ""
    total_requests: int
    succeeded_requests: int
    failed_requests: int
    slo_satisfied_requests: int
    success_rate: float
    error_rate: float
    errors_by_type: dict[str, int] = Field(default_factory=dict)
    total_input_tokens: int
    total_output_tokens: int
    total_output_tokens_without_thinking: int = 0
    total_reasoning_tokens: int
    total_cached_input_tokens: int
    throughput: ThroughputMetrics
    cost: CostMetrics
    distributions: dict[str, DistributionStats]

    @model_validator(mode="before")
    @classmethod
    def _backfill_output_without_thinking(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "total_output_tokens_without_thinking" not in data:
                tot_out = int(data.get("total_output_tokens") or 0)
                tot_reas = int(data.get("total_reasoning_tokens") or 0)
                data["total_output_tokens_without_thinking"] = max(tot_out - tot_reas, 0)
            dists = data.get("distributions")
            if isinstance(dists, dict) and "output_tokens_without_thinking" not in dists:
                dists = dict(dists)
                if "output_tokens" in dists and int(data.get("total_reasoning_tokens") or 0) == 0:
                    dists["output_tokens_without_thinking"] = dists["output_tokens"]
                else:
                    dists["output_tokens_without_thinking"] = DistributionStats()
                data["distributions"] = dists
        return data


def aggregate_run_results(
    run_id: str,
    task_name: str,
    config: TaskConfig,
    records: list[CallRecord],
    timestamp_start: str,
    timestamp_end: str,
    actual_duration_secs: float,
) -> RunResults:
    """Compute aggregated RunResults across all measured CallRecords."""
    eval_cfg = config.evaluation
    ds_cfg = config.dataset
    duration = max(actual_duration_secs, 1e-6)

    total_reqs = len(records)
    succeeded = [r for r in records if r.is_success]
    failed = [r for r in records if not r.is_success]

    slo_ok = [
        r
        for r in succeeded
        if eval_cfg.slo.is_satisfied(r.ttft_ms, r.tpot_ms, r.e2e_latency_ms)
    ]

    error_counts: Counter[str] = Counter()
    for r in failed:
        key = f"HTTP_{r.status_code}" if r.status_code else (r.error or "UnknownError")
        error_counts[key[:80]] += 1

    total_in = sum(r.input_tokens for r in succeeded)
    total_out = sum(r.output_tokens for r in succeeded)
    total_out_no_thinking = sum(r.output_tokens_without_thinking for r in succeeded)
    total_reasoning = sum(r.reasoning_tokens for r in succeeded)
    total_cached = sum(r.cached_input_tokens for r in succeeded)
    total_tokens = total_in + total_out

    rps = total_reqs / duration
    goodput_rps = len(succeeded) / duration
    slo_goodput_rps = len(slo_ok) / duration
    input_tps = total_in / duration
    output_tps = total_out / duration
    total_tps = total_tokens / duration

    total_cost = sum(r.cost_usd for r in succeeded)
    mean_cost = (total_cost / len(succeeded)) if succeeded else 0.0
    cost_per_1k_out = (total_cost / (total_out / 1000.0)) if total_out > 0 else 0.0
    cost_per_1k_tot = (total_cost / (total_tokens / 1000.0)) if total_tokens > 0 else 0.0

    usd_per_hour = (total_cost / duration) * 3600.0
    tps_per_usd_hr = (output_tps / usd_per_hour) if usd_per_hour > 0 else 0.0

    distributions = {
        "ttft_ms": DistributionStats.from_values(
            [r.ttft_ms for r in succeeded if r.ttft_ms is not None]
        ),
        "tpot_ms": DistributionStats.from_values(
            [r.tpot_ms for r in succeeded if r.tpot_ms is not None]
        ),
        "e2e_latency_ms": DistributionStats.from_values(
            [r.e2e_latency_ms for r in succeeded]
        ),
        "output_tokens_per_sec": DistributionStats.from_values(
            [r.output_tokens_per_sec for r in succeeded]
        ),
        "input_tokens": DistributionStats.from_values(
            [r.input_tokens for r in succeeded]
        ),
        "output_tokens": DistributionStats.from_values(
            [r.output_tokens for r in succeeded]
        ),
        "output_tokens_without_thinking": DistributionStats.from_values(
            [r.output_tokens_without_thinking for r in succeeded]
        ),
    }

    return RunResults(
        run_id=run_id,
        task_name=task_name,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        actual_duration_secs=round(duration, 4),
        model_endpoint=eval_cfg.model_endpoint,
        model=eval_cfg.model,
        num_threads=eval_cfg.num_threads,
        wait_time_between_requests_ms=eval_cfg.wait_time_between_requests_ms,
        configured_run_time_secs=eval_cfg.run_time_secs,
        dataset_generation_prompt=ds_cfg.generation_prompt,
        dataset_num_items=ds_cfg.num_items,
        dataset_generation_model=ds_cfg.generation_model,
        dataset_generation_model_endpoint=ds_cfg.generation_model_endpoint,
        total_requests=total_reqs,
        succeeded_requests=len(succeeded),
        failed_requests=len(failed),
        slo_satisfied_requests=len(slo_ok),
        success_rate=round((len(succeeded) / total_reqs) * 100.0, 2) if total_reqs > 0 else 0.0,
        error_rate=round((len(failed) / total_reqs) * 100.0, 2) if total_reqs > 0 else 0.0,
        errors_by_type=dict(error_counts),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_output_tokens_without_thinking=total_out_no_thinking,
        total_reasoning_tokens=total_reasoning,
        total_cached_input_tokens=total_cached,
        throughput=ThroughputMetrics(
            rps=round(rps, 4),
            goodput_rps=round(goodput_rps, 4),
            slo_goodput_rps=round(slo_goodput_rps, 4),
            input_tps=round(input_tps, 4),
            output_tps=round(output_tps, 4),
            total_tps=round(total_tps, 4),
        ),
        cost=CostMetrics(
            total_cost_usd=round(total_cost, 6),
            mean_cost_per_request_usd=round(mean_cost, 6),
            cost_per_1k_output_tokens_usd=round(cost_per_1k_out, 6),
            cost_per_1k_total_tokens_usd=round(cost_per_1k_tot, 6),
            output_tps_per_usd_per_hour=round(tps_per_usd_hr, 4),
        ),
        distributions=distributions,
    )
