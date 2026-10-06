"""Formatting and inspection utilities for single-run reports and multi-run comparisons."""

from __future__ import annotations

import json

from llmendpoint_perf.config import TaskConfig
from llmendpoint_perf.metrics import DistributionStats, RunResults
from llmendpoint_perf.storage import TaskStorage


def load_run_results(storage: TaskStorage, run_id: str | None = None) -> RunResults:
    """Load RunResults for a given run_id (or the latest run if run_id is None)."""
    available_runs = storage.list_runs()
    if not available_runs:
        raise FileNotFoundError(
            f"No benchmark runs found for task '{storage.task_name}' under {storage.task_uri}/runs/"
        )
    target_run = run_id if run_id is not None else available_runs[-1]
    rel_path = f"runs/{target_run}/results.jsonl"
    if not storage.exists(rel_path):
        raise FileNotFoundError(
            f"Results file not found for run '{target_run}' at {storage.task_uri}/{rel_path}"
        )
    raw_text = storage.read_text(rel_path).strip()
    first_line = raw_text.splitlines()[0]
    results = RunResults.model_validate(json.loads(first_line))

    # Backfill dataset metadata from run or task config.yaml if absent in older results.jsonl
    if (
        not results.dataset_generation_prompt
        and not results.dataset_generation_model
        and results.dataset_num_items == 0
    ):
        for cfg_rel_path in (f"runs/{target_run}/config.yaml", "config.yaml"):
            if storage.exists(cfg_rel_path):
                try:
                    cfg = TaskConfig.from_yaml(storage.read_text(cfg_rel_path))
                    results.dataset_generation_prompt = cfg.dataset.generation_prompt
                    results.dataset_num_items = cfg.dataset.num_items
                    results.dataset_generation_model = cfg.dataset.generation_model
                    results.dataset_generation_model_endpoint = (
                        cfg.dataset.generation_model_endpoint
                    )
                    break
                except Exception:  # pylint: disable=broad-except
                    pass

    return results


def format_run_summary_report(results: RunResults) -> str:
    """Render a human-readable ASCII summary table for a single RunResults object."""
    lines: list[str] = []
    sep = "=" * 88
    subsep = "-" * 88

    lines.append(sep)
    lines.append(
        f"LLM ENDPOINT PERFORMANCE REPORT | Task: {results.task_name} | Run: {results.run_id}"
    )
    lines.append(sep)
    lines.append(f"  Model               : {results.model}")
    lines.append(f"  Endpoint            : {results.model_endpoint}")
    lines.append(
        f"  Concurrency         : {results.num_threads} threads "
        f"(wait_between_requests={results.wait_time_between_requests_ms} ms)"
    )
    lines.append(
        f"  Duration            : {results.actual_duration_secs:.2f} s "
        f"(configured={results.configured_run_time_secs:.1f} s)"
    )
    lines.append(
        f"  Requests            : {results.total_requests} total | "
        f"{results.succeeded_requests} succeeded ({results.success_rate:.1f}%) | "
        f"{results.failed_requests} failed ({results.error_rate:.1f}%)"
    )
    if results.errors_by_type:
        err_summary = ", ".join(f"{k}: {v}" for k, v in results.errors_by_type.items())
        lines.append(f"  Errors Breakdown    : {err_summary}")
    lines.append(
        f"  Token Volume        : {results.total_input_tokens} in | "
        f"{results.total_output_tokens} out "
        f"({results.total_output_tokens_without_thinking} w/o thinking, "
        f"{results.total_reasoning_tokens} reasoning) | "
        f"{results.total_cached_input_tokens} cached"
    )
    lines.append(subsep)
    lines.append("  DATASET INFORMATION")
    lines.append(f"    Generation Model               : {results.dataset_generation_model or 'N/A'}")
    lines.append(f"    Number of Items                : {results.dataset_num_items}")
    lines.append(
        f"    Generation Prompt              : {results.dataset_generation_prompt or 'N/A'}"
    )
    lines.append(subsep)
    lines.append("  THROUGHPUT & COST SUMMARY")
    lines.append(
        f"    Request Throughput (RPS)       : {results.throughput.rps:.2f} req/s "
        f"(Goodput: {results.throughput.goodput_rps:.2f} req/s | "
        f"SLO Goodput: {results.throughput.slo_goodput_rps:.2f} req/s)"
    )
    lines.append(
        f"    Token Throughput (TPS)         : {results.throughput.output_tps:.2f} output tok/s | "
        f"{results.throughput.input_tps:.2f} input tok/s | "
        f"{results.throughput.total_tps:.2f} total tok/s"
    )
    lines.append(
        f"    Cost Performance               : ${results.cost.total_cost_usd:.6f} total | "
        f"${results.cost.mean_cost_per_request_usd:.6f}/req | "
        f"${results.cost.cost_per_1k_output_tokens_usd:.6f}/1K out tok"
    )
    lines.append(subsep)
    header = (
        f"  {'Metric':<26} {'Mean':>9} {'Std':>9} {'Min':>9} "
        f"{'p50':>9} {'p90':>9} {'p95':>9} {'p99':>9} {'Max':>9}"
    )
    lines.append(header)
    lines.append("  " + "-" * 84)

    metric_labels = [
        ("ttft_ms", "TTFT (ms)"),
        ("tpot_ms", "TPOT / ITL (ms/tok)"),
        ("e2e_latency_ms", "E2E Latency (ms)"),
        ("output_tokens_per_sec", "Decode Speed (tok/s)"),
        ("input_tokens", "Input Tokens"),
        ("output_tokens", "Output Tokens"),
        ("output_tokens_without_thinking", "Output Toks (w/o think)"),
    ]
    for key, label in metric_labels:
        dist = results.distributions.get(key, DistributionStats())
        lines.append(
            f"  {label:<26} {dist.mean:>9.2f} {dist.std:>9.2f} {dist.min:>9.2f} "
            f"{dist.p50:>9.2f} {dist.p90:>9.2f} {dist.p95:>9.2f} {dist.p99:>9.2f} {dist.max:>9.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def format_comparison_report(results_list: list[RunResults]) -> str:
    """Render a side-by-side comparison table across multiple RunResults."""
    if not results_list:
        return "No runs provided for comparison."

    col_headers = [f"{r.task_name}:{r.run_id}" for r in results_list]
    col_width = max(22, *(len(h) + 2 for h in col_headers))
    metric_col_width = 30

    total_width = metric_col_width + 3 + len(results_list) * (col_width + 3)
    sep = "=" * total_width
    subsep = "-" * total_width

    lines: list[str] = [sep, "LLM ENDPOINT PERFORMANCE COMPARISON", sep]
    lines.append("DATASET INFORMATION")
    for r in results_list:
        lines.append(f"  [{r.task_name}:{r.run_id}]")
        lines.append(f"    Generation Model  : {r.dataset_generation_model or 'N/A'}")
        lines.append(f"    Number of Items   : {r.dataset_num_items}")
        lines.append(f"    Generation Prompt : {r.dataset_generation_prompt or 'N/A'}")
    lines.append(subsep)

    header_row = f"{'Metric':<{metric_col_width}} | " + " | ".join(
        f"{h:>{col_width}}" for h in col_headers
    )
    lines.append(header_row)
    lines.append(subsep)

    def _add_row(label: str, values: list[str]) -> None:
        row = f"{label:<{metric_col_width}} | " + " | ".join(
            f"{v:>{col_width}}" for v in values
        )
        lines.append(row)

    _add_row("Model", [r.model for r in results_list])
    _add_row("Threads", [str(r.num_threads) for r in results_list])
    _add_row("Duration (s)", [f"{r.actual_duration_secs:.2f}" for r in results_list])
    _add_row(
        "Requests (OK / Total)",
        [f"{r.succeeded_requests}/{r.total_requests} ({r.success_rate:.1f}%)" for r in results_list],
    )
    _add_row("Throughput (RPS)", [f"{r.throughput.rps:.2f}" for r in results_list])
    _add_row("SLO Goodput (RPS)", [f"{r.throughput.slo_goodput_rps:.2f}" for r in results_list])
    _add_row("Output Throughput (tok/s)", [f"{r.throughput.output_tps:.2f}" for r in results_list])
    _add_row("Total Throughput (tok/s)", [f"{r.throughput.total_tps:.2f}" for r in results_list])
    lines.append(subsep)

    for key, label in [
        ("ttft_ms", "TTFT (ms)"),
        ("tpot_ms", "TPOT (ms/tok)"),
        ("e2e_latency_ms", "E2E Latency (ms)"),
        ("output_tokens_per_sec", "Decode Speed (tok/s)"),
        ("input_tokens", "Input Tokens"),
        ("output_tokens", "Output Tokens"),
        ("output_tokens_without_thinking", "Output Toks (w/o think)"),
    ]:
        _add_row(
            f"{label} Mean",
            [f"{r.distributions.get(key, DistributionStats()).mean:.2f}" for r in results_list],
        )
        _add_row(
            f"{label} p50",
            [f"{r.distributions.get(key, DistributionStats()).p50:.2f}" for r in results_list],
        )
        _add_row(
            f"{label} p95",
            [f"{r.distributions.get(key, DistributionStats()).p95:.2f}" for r in results_list],
        )
        _add_row(
            f"{label} p99",
            [f"{r.distributions.get(key, DistributionStats()).p99:.2f}" for r in results_list],
        )

    lines.append(subsep)
    _add_row("Total Cost (USD)", [f"${r.cost.total_cost_usd:.6f}" for r in results_list])
    _add_row("Mean Cost / Req (USD)", [f"${r.cost.mean_cost_per_request_usd:.6f}" for r in results_list])
    _add_row(
        "Cost / 1K Out Tokens (USD)",
        [f"${r.cost.cost_per_1k_output_tokens_usd:.6f}" for r in results_list],
    )
    lines.append(sep)
    return "\n".join(lines)
