"""Web UI server and API endpoints for inspecting llmendpoint-perf tasks and runs."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from typing import Any
from urllib.parse import unquote, urlparse

from llmendpoint_perf.config import TaskConfig
from llmendpoint_perf.inspector import (
    format_comparison_report,
    format_run_summary_report,
    load_run_results,
)
from llmendpoint_perf.metrics import CallRecord, RunResults
from llmendpoint_perf.storage import TaskStorage, list_tasks, resolve_base_path


def _extract_message_preview_and_images(messages: list[dict[str, Any]]) -> tuple[str, int]:
    """Return a short text preview and count of images across an OpenAI messages array."""
    text_parts: list[str] = []
    image_count = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            if "data:image/" in content:
                image_count += content.count("data:image/")
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                p_type = part.get("type", "")
                if p_type == "text":
                    text_parts.append(str(part.get("text", "")))
                elif p_type in ("image_url", "image", "input_image"):
                    image_count += 1
    joined = " ".join(t.strip() for t in text_parts if t.strip())
    preview = (joined[:90] + "...") if len(joined) > 90 else joined
    return preview, image_count


def _load_prompts_raw(storage: TaskStorage) -> list[dict[str, Any]]:
    """Load parsed JSON lines from `prompts.jsonl` if it exists, else return empty list."""
    if not storage.exists("prompts.jsonl"):
        return []
    raw = storage.read_text("prompts.jsonl")
    items: list[dict[str, Any]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            continue
    return items


def _load_calls_raw(storage: TaskStorage, run_id: str) -> list[dict[str, Any]]:
    """Load parsed JSON lines from `runs/<run_id>/calls.jsonl` if it exists."""
    rel_path = f"runs/{run_id}/calls.jsonl"
    if not storage.exists(rel_path):
        return []
    raw = storage.read_text(rel_path)
    calls: list[dict[str, Any]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                calls.append(obj)
        except json.JSONDecodeError:
            continue
    return calls


def get_tasks_payload(base_path: str | None = None) -> dict[str, Any]:
    """Return the configured base_path and discovered evaluation task names."""
    resolved = resolve_base_path(base_path)
    tasks = list_tasks(resolved)
    return {
        "base_path": resolved,
        "tasks": tasks,
    }


def get_task_runs_payload(task_name: str, base_path: str | None = None) -> dict[str, Any]:
    """Return all runs and their inspect reports for a given task."""
    storage = TaskStorage(task_name=task_name, base_path=base_path)
    run_ids = storage.list_runs()
    runs_data: list[dict[str, Any]] = []
    for run_id in run_ids:
        try:
            res: RunResults = load_run_results(storage=storage, run_id=run_id)
            runs_data.append(
                {
                    "run_id": run_id,
                    "results": res.model_dump(mode="json"),
                    "inspect_report": format_run_summary_report(res),
                }
            )
        except FileNotFoundError:
            runs_data.append(
                {
                    "run_id": run_id,
                    "results": None,
                    "inspect_report": f"Run '{run_id}' does not have a completed results.jsonl file.",
                }
            )
    return {
        "task_name": task_name,
        "runs": runs_data,
    }


def get_task_compare_payload(task_name: str, base_path: str | None = None) -> dict[str, Any]:
    """Return comparison data across all completed runs for a task."""
    storage = TaskStorage(task_name=task_name, base_path=base_path)
    run_ids = storage.list_runs()
    results_list: list[RunResults] = []
    for run_id in run_ids:
        try:
            results_list.append(load_run_results(storage=storage, run_id=run_id))
        except FileNotFoundError:
            continue
    compare_report = format_comparison_report(results_list)
    return {
        "task_name": task_name,
        "runs": [r.model_dump(mode="json") for r in results_list],
        "compare_report": compare_report,
    }


def get_run_config_payload(
    task_name: str, run_id: str, base_path: str | None = None
) -> dict[str, Any]:
    """Return the config.yaml snapshot for a specific run (or fallback to task config.yaml)."""
    storage = TaskStorage(task_name=task_name, base_path=base_path)
    run_cfg_path = f"runs/{run_id}/config.yaml"
    if storage.exists(run_cfg_path):
        config_yaml = storage.read_text(run_cfg_path)
    elif storage.exists("config.yaml"):
        config_yaml = storage.read_text("config.yaml")
    else:
        raise FileNotFoundError(
            f"config.yaml not found for task '{task_name}' run '{run_id}'."
        )
    return {
        "task_name": task_name,
        "run_id": run_id,
        "config_yaml": config_yaml,
    }


def get_dataset_index_payload(task_name: str, base_path: str | None = None) -> dict[str, Any]:
    """Return dataset metadata and an index of items from prompts.jsonl."""
    storage = TaskStorage(task_name=task_name, base_path=base_path)
    dataset_info: dict[str, Any] = {}
    if storage.exists("config.yaml"):
        try:
            cfg = TaskConfig.from_yaml(storage.read_text("config.yaml"))
            dataset_info = {
                "generation_prompt": cfg.dataset.generation_prompt,
                "num_items": cfg.dataset.num_items,
                "generation_model": cfg.dataset.generation_model,
                "generation_model_endpoint": cfg.dataset.generation_model_endpoint,
                "multimodal_enabled": cfg.dataset.multimodal.enabled,
            }
        except Exception:  # pylint: disable=broad-except
            pass

    prompts = _load_prompts_raw(storage)
    index_items: list[dict[str, Any]] = []
    for idx, item in enumerate(prompts):
        messages = item.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        preview, image_count = _extract_message_preview_and_images(messages)
        roles = [str(m.get("role", "user")) for m in messages if isinstance(m, dict)]
        index_items.append(
            {
                "index": idx,
                "roles": roles,
                "has_images": image_count > 0,
                "image_count": image_count,
                "preview": preview,
            }
        )

    return {
        "task_name": task_name,
        "dataset_info": dataset_info,
        "total_items": len(index_items),
        "items": index_items,
    }


def get_dataset_item_payload(
    task_name: str, item_index: int, base_path: str | None = None
) -> dict[str, Any]:
    """Return the full dataset item (including base64 images) at `item_index`."""
    storage = TaskStorage(task_name=task_name, base_path=base_path)
    prompts = _load_prompts_raw(storage)
    if item_index < 0 or item_index >= len(prompts):
        raise IndexError(
            f"Dataset item index {item_index} out of range (total={len(prompts)})."
        )
    return {
        "task_name": task_name,
        "index": item_index,
        "item": prompts[item_index],
    }


def get_run_inferences_index_payload(
    task_name: str, run_id: str, base_path: str | None = None
) -> dict[str, Any]:
    """Return a summary list of all inference calls in `runs/<run_id>/calls.jsonl`."""
    storage = TaskStorage(task_name=task_name, base_path=base_path)
    calls = _load_calls_raw(storage, run_id)
    summaries: list[dict[str, Any]] = []
    for idx, raw_call in enumerate(calls):
        record = CallRecord.model_validate(raw_call)
        resp_preview = record.response_content.strip().replace("\n", " ")
        if len(resp_preview) > 80:
            resp_preview = resp_preview[:80] + "..."
        summaries.append(
            {
                "call_index": idx,
                "request_id": record.request_id,
                "thread_id": record.thread_id,
                "prompt_index": record.prompt_index,
                "status_code": record.status_code,
                "error": record.error,
                "is_success": record.is_success,
                "ttft_ms": record.ttft_ms,
                "tpot_ms": record.tpot_ms,
                "e2e_latency_ms": record.e2e_latency_ms,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "output_tokens_per_sec": record.output_tokens_per_sec,
                "cost_usd": record.cost_usd,
                "response_preview": resp_preview,
            }
        )
    return {
        "task_name": task_name,
        "run_id": run_id,
        "total_inferences": len(summaries),
        "inferences": summaries,
    }


def get_run_inference_detail_payload(
    task_name: str, run_id: str, call_index: int, base_path: str | None = None
) -> dict[str, Any]:
    """Return full telemetry, output, and input prompt for a specific inference call."""
    storage = TaskStorage(task_name=task_name, base_path=base_path)
    calls = _load_calls_raw(storage, run_id)
    if call_index < 0 or call_index >= len(calls):
        raise IndexError(
            f"Inference index {call_index} out of range (total={len(calls)})."
        )
    record = CallRecord.model_validate(calls[call_index])
    prompts = _load_prompts_raw(storage)
    input_prompt: dict[str, Any] | None = None
    if 0 <= record.prompt_index < len(prompts):
        input_prompt = prompts[record.prompt_index]

    return {
        "task_name": task_name,
        "run_id": run_id,
        "call_index": call_index,
        "inference": record.model_dump(mode="json"),
        "input_prompt": input_prompt,
    }


UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="Inspect and compare LLM endpoint performance evaluation tasks, datasets, configurations, and inference telemetry." />
  <title>llmendpoint-perf | Evaluation Inspector</title>
  <style>
    :root {
      --bg-app: hsl(215, 28%, 97%);
      --bg-surface: hsl(0, 0%, 100%);
      --bg-subtle: hsl(214, 32%, 94%);
      --bg-code: hsl(220, 26%, 12%);
      --text-code: hsl(210, 40%, 96%);
      --text-primary: hsl(222, 47%, 11%);
      --text-secondary: hsl(215, 19%, 35%);
      --text-muted: hsl(215, 16%, 50%);
      --accent: hsl(212, 92%, 42%);
      --accent-hover: hsl(212, 92%, 35%);
      --accent-soft: hsl(212, 95%, 94%);
      --border: hsl(214, 24%, 86%);
      --success-bg: hsl(145, 65%, 92%);
      --success-text: hsl(145, 75%, 24%);
      --error-bg: hsl(0, 78%, 94%);
      --error-text: hsl(0, 72%, 35%);
      --badge-img-bg: hsl(35, 92%, 91%);
      --badge-img-text: hsl(28, 85%, 30%);
      --radius-sm: 6px;
      --radius-md: 8px;
      --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.06);
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      --font-mono: "JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: var(--font-sans);
      background-color: var(--bg-app);
      color: var(--text-primary);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
      line-height: 1.45;
    }

    /* Top Header & Task Selector */
    header.app-header {
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border);
      padding: 0.75rem 1.25rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      box-shadow: var(--shadow-sm);
      z-index: 10;
    }

    .brand-group {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    h1.app-title {
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: var(--text-primary);
    }

    .basepath-pill {
      font-family: var(--font-mono);
      font-size: 0.75rem;
      background: var(--bg-subtle);
      color: var(--text-secondary);
      padding: 0.25rem 0.6rem;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      max-width: 420px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .selector-group {
      display: flex;
      align-items: center;
      gap: 0.65rem;
    }

    .selector-group label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-secondary);
    }

    select#task-selector {
      font-family: var(--font-sans);
      font-size: 0.875rem;
      font-weight: 600;
      padding: 0.4rem 0.75rem;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background-color: var(--bg-surface);
      color: var(--text-primary);
      min-width: 240px;
      cursor: pointer;
    }

    select#task-selector:focus {
      outline: 2px solid var(--accent);
      outline-offset: 1px;
    }

    button.btn {
      font-family: var(--font-sans);
      font-size: 0.82rem;
      font-weight: 600;
      padding: 0.42rem 0.8rem;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: var(--bg-surface);
      color: var(--text-secondary);
      cursor: pointer;
      transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
    }

    button.btn:hover {
      background: var(--bg-subtle);
      color: var(--text-primary);
    }

    /* Navigation Tabs */
    nav.tabs-nav {
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border);
      padding: 0 1.25rem;
      display: flex;
      gap: 0.25rem;
    }

    button.tab-btn {
      font-family: var(--font-sans);
      font-size: 0.875rem;
      font-weight: 600;
      padding: 0.65rem 1.1rem;
      border: none;
      border-bottom: 2px solid transparent;
      background: transparent;
      color: var(--text-secondary);
      cursor: pointer;
      transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease;
    }

    button.tab-btn:hover {
      color: var(--text-primary);
      background: var(--bg-app);
    }

    button.tab-btn.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
    }

    /* Main Workspace Layout */
    main.workspace {
      flex: 1;
      display: flex;
      overflow: hidden;
      position: relative;
    }

    section.tab-panel {
      display: none;
      width: 100%;
      height: 100%;
      overflow: hidden;
    }

    section.tab-panel.active {
      display: flex;
    }

    /* Split Panel Layouts */
    .split-2 {
      display: flex;
      width: 100%;
      height: 100%;
      overflow: hidden;
    }

    .split-3 {
      display: flex;
      width: 100%;
      height: 100%;
      overflow: hidden;
    }

    .panel-left {
      width: 270px;
      min-width: 230px;
      max-width: 340px;
      background: var(--bg-surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
    }

    .panel-center {
      width: 340px;
      min-width: 280px;
      max-width: 420px;
      background: var(--bg-surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
    }

    .panel-right {
      flex: 1;
      height: 100%;
      overflow-y: auto;
      padding: 1.25rem;
      background: var(--bg-app);
    }

    .full-panel {
      width: 100%;
      height: 100%;
      overflow-y: auto;
      padding: 1.25rem;
    }

    .panel-header {
      padding: 0.7rem 0.9rem;
      border-bottom: 1px solid var(--border);
      background: var(--bg-subtle);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-secondary);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .list-container {
      flex: 1;
      overflow-y: auto;
    }

    .list-item {
      padding: 0.7rem 0.9rem;
      border-bottom: 1px solid var(--border);
      cursor: pointer;
      transition: background 0.12s ease;
    }

    .list-item:hover {
      background: var(--bg-app);
    }

    .list-item.selected {
      background: var(--accent-soft);
      border-left: 3px solid var(--accent);
      padding-left: calc(0.9rem - 3px);
    }

    .item-title {
      font-family: var(--font-mono);
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.4rem;
    }

    .item-sub {
      font-size: 0.76rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    /* Badges */
    .badge {
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 0.68rem;
      font-weight: 700;
      padding: 0.1rem 0.42rem;
      border-radius: 4px;
      text-transform: uppercase;
    }

    .badge-ok {
      background: var(--success-bg);
      color: var(--success-text);
    }

    .badge-err {
      background: var(--error-bg);
      color: var(--error-text);
    }

    .badge-img {
      background: var(--badge-img-bg);
      color: var(--badge-img-text);
    }

    .badge-role {
      background: var(--bg-subtle);
      color: var(--text-secondary);
      border: 1px solid var(--border);
    }

    /* Cards & Sections */
    .card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 1.1rem 1.25rem;
      margin-bottom: 1rem;
      box-shadow: var(--shadow-sm);
    }

    .card-title {
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-secondary);
      margin-bottom: 0.75rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.45rem;
    }

    .kv-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 0.75rem;
    }

    .kv-box {
      background: var(--bg-app);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 0.55rem 0.75rem;
    }

    .kv-label {
      font-size: 0.72rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
    }

    .kv-val {
      font-family: var(--font-mono);
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--text-primary);
      margin-top: 0.15rem;
      word-break: break-word;
    }

    /* Data Tables */
    .table-wrap {
      overflow-x: auto;
    }

    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.83rem;
    }

    table.data-table th,
    table.data-table td {
      padding: 0.52rem 0.75rem;
      border-bottom: 1px solid var(--border);
      text-align: right;
      font-family: var(--font-mono);
    }

    table.data-table th:first-child,
    table.data-table td:first-child {
      text-align: left;
      font-family: var(--font-sans);
      font-weight: 600;
    }

    table.data-table thead th {
      background: var(--bg-subtle);
      color: var(--text-secondary);
      font-weight: 700;
    }

    table.data-table tbody tr:hover {
      background: var(--bg-app);
    }

    /* Preformatted Code / CLI Output */
    pre.code-block {
      background: var(--bg-code);
      color: var(--text-code);
      font-family: var(--font-mono);
      font-size: 0.8rem;
      line-height: 1.5;
      padding: 1rem;
      border-radius: var(--radius-md);
      overflow-x: auto;
      white-space: pre;
    }

    pre.text-wrap-block {
      background: var(--bg-app);
      color: var(--text-primary);
      border: 1px solid var(--border);
      font-family: var(--font-mono);
      font-size: 0.82rem;
      line-height: 1.5;
      padding: 0.85rem;
      border-radius: var(--radius-sm);
      white-space: pre-wrap;
      word-break: break-word;
    }

    /* Multimodal Message Rendering */
    .message-card {
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 0.85rem;
      margin-bottom: 0.75rem;
      background: var(--bg-app);
    }

    .message-header {
      margin-bottom: 0.5rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .media-container {
      margin-top: 0.65rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
    }

    .rendered-image-box {
      border: 1px solid var(--border);
      background: var(--bg-surface);
      padding: 0.5rem;
      border-radius: var(--radius-sm);
      display: inline-flex;
      flex-direction: column;
      gap: 0.35rem;
      max-width: 100%;
    }

    .rendered-image-box img {
      max-width: 420px;
      max-height: 420px;
      object-fit: contain;
      border-radius: 4px;
      border: 1px solid var(--border);
    }

    .empty-state {
      color: var(--text-muted);
      font-size: 0.9rem;
      padding: 2rem;
      text-align: center;
    }
  </style>
</head>
<body>
  <header class="app-header">
    <div class="brand-group">
      <h1 class="app-title">llmendpoint-perf Inspector</h1>
      <span id="basepath-display" class="basepath-pill" title="LLMENDPOINTPERF_BASEPATH">Loading...</span>
    </div>
    <div class="selector-group">
      <label for="task-selector">Evaluation Task:</label>
      <select id="task-selector" aria-label="Select Evaluation Task"></select>
      <button id="refresh-btn" class="btn" type="button">Refresh</button>
    </div>
  </header>

  <nav class="tabs-nav" role="tablist">
    <button id="tab-btn-runs" class="tab-btn active" data-tab="runs" role="tab" type="button">Runs</button>
    <button id="tab-btn-compare" class="tab-btn" data-tab="compare" role="tab" type="button">Compare</button>
    <button id="tab-btn-configs" class="tab-btn" data-tab="configs" role="tab" type="button">Configs</button>
    <button id="tab-btn-dataset" class="tab-btn" data-tab="dataset" role="tab" type="button">Dataset</button>
    <button id="tab-btn-inference" class="tab-btn" data-tab="inference" role="tab" type="button">Inference</button>
    <button id="tab-btn-metrics-help" class="tab-btn" data-tab="metrics-help" role="tab" type="button">Metrics help</button>
  </nav>

  <main class="workspace">
    <!-- TAB 1: RUNS -->
    <section id="tab-runs" class="tab-panel active" role="tabpanel">
      <div class="split-2">
        <aside class="panel-left">
          <div class="panel-header">
            <span>Runs</span>
            <span id="runs-count-badge">0</span>
          </div>
          <div id="runs-list" class="list-container"></div>
        </aside>
        <div id="runs-detail" class="panel-right">
          <div class="empty-state">Select a run on the left to inspect its performance report.</div>
        </div>
      </div>
    </section>

    <!-- TAB 2: COMPARE -->
    <section id="tab-compare" class="tab-panel" role="tabpanel">
      <div id="compare-content" class="full-panel">
        <div class="empty-state">Loading run comparison...</div>
      </div>
    </section>

    <!-- TAB 3: CONFIGS -->
    <section id="tab-configs" class="tab-panel" role="tabpanel">
      <div class="split-2">
        <aside class="panel-left">
          <div class="panel-header">
            <span>Runs</span>
            <span id="configs-count-badge">0</span>
          </div>
          <div id="configs-run-list" class="list-container"></div>
        </aside>
        <div id="configs-detail" class="panel-right">
          <div class="empty-state">Select a run on the left to view its config.yaml.</div>
        </div>
      </div>
    </section>

    <!-- TAB 4: DATASET -->
    <section id="tab-dataset" class="tab-panel" role="tabpanel">
      <div class="split-2">
        <aside class="panel-left">
          <div class="panel-header">
            <span>Dataset Items</span>
            <span id="dataset-count-badge">0</span>
          </div>
          <div id="dataset-item-list" class="list-container"></div>
        </aside>
        <div id="dataset-detail" class="panel-right">
          <div class="empty-state">Select a dataset item on the left to inspect its content.</div>
        </div>
      </div>
    </section>

    <!-- TAB 5: INFERENCE -->
    <section id="tab-inference" class="tab-panel" role="tabpanel">
      <div class="split-3">
        <aside class="panel-left">
          <div class="panel-header">
            <span>1. Runs</span>
            <span id="inf-runs-count-badge">0</span>
          </div>
          <div id="inference-run-list" class="list-container"></div>
        </aside>
        <aside class="panel-center">
          <div class="panel-header">
            <span>2. Inferences</span>
            <span id="inf-calls-count-badge">0</span>
          </div>
          <div id="inference-call-list" class="list-container">
            <div class="empty-state">Select a run on the left.</div>
          </div>
        </aside>
        <div id="inference-detail" class="panel-right">
          <div class="empty-state">Select an inference in the center panel to view full telemetry, input prompt, and output.</div>
        </div>
      </div>
    </section>

    <!-- TAB 6: METRICS HELP -->
    <section id="tab-metrics-help" class="tab-panel" role="tabpanel">
      <div class="full-panel">
        <div class="card">
          <div class="card-title">1. Streaming Measurement Timeline</div>
          <p style="font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
            To accurately separate prompt prefill and queueing delay from autoregressive token generation speed,
            <code>llmendpoint-perf</code> sends every benchmark request using Server-Sent Events
            (<code>stream: true</code>, <code>stream_options: {"include_usage": true}</code>) and records four monotonic timestamps:
          </p>
          <pre class="code-block">t_start                t_first_token                        t_last_token     t_end
  │                          │                                    │            │
  ├─────── TTFT (ms) ────────┼────── Decode Window (secs) ────────┼─ Teardown ─┤
  │  (Network + Queueing +   │   (Output Token #2 ... Token #N)   │            │
  │    Prompt Prefill)       │   TPOT = Decode Window / (N - 1)   │            │
  └──────────────────────────┴────────────────────────────────────┴────────────┘
                                E2E Latency (ms)</pre>
          <div class="kv-grid" style="margin-top: 0.85rem;">
            <div class="kv-box">
              <div class="kv-label">t_start (timestamp_start)</div>
              <div class="kv-val" style="font-size: 0.8rem; font-family: var(--font-sans);">Recorded immediately before dispatching the HTTP <code>POST /chat/completions</code> request.</div>
            </div>
            <div class="kv-box">
              <div class="kv-label">t_first_token (timestamp_first_token)</div>
              <div class="kv-val" style="font-size: 0.8rem; font-family: var(--font-sans);">Recorded when the first SSE chunk with non-empty <code>delta.content</code> or <code>delta.reasoning_content</code> arrives.</div>
            </div>
            <div class="kv-box">
              <div class="kv-label">t_last_token</div>
              <div class="kv-val" style="font-size: 0.8rem; font-family: var(--font-sans);">Updated on each subsequent SSE chunk containing generated content or reasoning tokens.</div>
            </div>
            <div class="kv-box">
              <div class="kv-label">t_end (timestamp_end)</div>
              <div class="kv-val" style="font-size: 0.8rem; font-family: var(--font-sans);">Recorded when the SSE stream completes (<code>data: [DONE]</code> or stream close).</div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">2. Per-Request Latency &amp; Speed Metrics (calls.jsonl)</div>
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Metric Key</th>
                  <th style="text-align:left;">Report Label</th>
                  <th style="text-align:center;">Unit</th>
                  <th style="text-align:left;">Formula / Definition</th>
                  <th style="text-align:left;">What It Measures</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>ttft_ms</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>TTFT (ms)</strong></td>
                  <td style="text-align:center;"><code>ms</code></td>
                  <td style="text-align:left;"><code>(t_first_token - t_start) * 1000</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Time to First Token</strong>: Elapsed time from sending the request to receiving the first output or reasoning token. Reflects network RTT, queueing delay, and prompt prefill (plus vision encoding for multimodal inputs).</td>
                </tr>
                <tr>
                  <td><code>tpot_ms</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>TPOT / ITL (ms/tok)</strong></td>
                  <td style="text-align:center;"><code>ms/tok</code></td>
                  <td style="text-align:left;"><code>((t_last_token - t_first_token) * 1000) / (output_tokens - 1)</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Time Per Output Token (Inter-Token Latency)</strong>: Average time between consecutive generated tokens after the first token arrives. If <code>output_tokens &lt;= 1</code> or single-chunk response, set to <code>0.0</code>.</td>
                </tr>
                <tr>
                  <td><code>e2e_latency_ms</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>E2E Latency (ms)</strong></td>
                  <td style="text-align:center;"><code>ms</code></td>
                  <td style="text-align:left;"><code>(t_end - t_start) * 1000</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>End-to-End Request Latency</strong>: Total wall-clock time from request dispatch until stream completion (<code>[DONE]</code>), equivalent to <code>TTFT + TPOT * (output_tokens - 1)</code> plus teardown overhead.</td>
                </tr>
                <tr>
                  <td><code>output_tokens_per_sec</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Decode Speed (tok/s)</strong></td>
                  <td style="text-align:center;"><code>tok/s</code></td>
                  <td style="text-align:left;"><code>output_tokens / (t_last_token - t_first_token)</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Per-Request Decode Throughput</strong>: Token generation speed during the active decode window for a single request (falls back to <code>output_tokens / (e2e_latency_ms / 1000)</code> for single-chunk responses).</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-title">3. Token Accounting Metrics (calls.jsonl &amp; results.jsonl)</div>
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Metric Key</th>
                  <th style="text-align:left;">Run Total Key</th>
                  <th style="text-align:left;">Description</th>
                  <th style="text-align:left;">Source &amp; Fallback Behavior</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>input_tokens</code></td>
                  <td style="text-align:left;"><code>total_input_tokens</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Prompt tokens processed by the model for the request.</td>
                  <td style="text-align:left; font-family:var(--font-sans);">From API <code>usage.prompt_tokens</code>. Fallback: heuristic text estimate (~1.3 tokens/word or <code>chars/4</code>, plus <code>4</code> framing tokens/msg and <code>258</code> tokens per attached image).</td>
                </tr>
                <tr>
                  <td><code>output_tokens</code></td>
                  <td style="text-align:left;"><code>total_output_tokens</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Completion tokens generated by the model.</td>
                  <td style="text-align:left; font-family:var(--font-sans);">From API <code>usage.completion_tokens</code>. Fallback: max of SSE token chunk count and heuristic text token estimate of <code>response_content</code>.</td>
                </tr>
                <tr>
                  <td><code>reasoning_tokens</code></td>
                  <td style="text-align:left;"><code>total_reasoning_tokens</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Internal thinking/reasoning tokens used during generation.</td>
                  <td style="text-align:left; font-family:var(--font-sans);">From API <code>usage.completion_tokens_details.reasoning_tokens</code> (defaults to <code>0</code>).</td>
                </tr>
                <tr>
                  <td><code>cached_input_tokens</code></td>
                  <td style="text-align:left;"><code>total_cached_input_tokens</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Prompt tokens served from prefix/context cache.</td>
                  <td style="text-align:left; font-family:var(--font-sans);">From API <code>usage.prompt_tokens_details.cached_tokens</code> (defaults to <code>0</code>).</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-title">4. Statistical Distribution Aggregations (results.jsonl)</div>
          <p style="font-size: 0.86rem; color: var(--text-secondary); margin-bottom: 0.7rem;">
            Computed across all <strong>succeeded</strong> (<code>HTTP 200</code>) requests (excluding warmup requests) for
            <code>ttft_ms</code>, <code>tpot_ms</code>, <code>e2e_latency_ms</code>, <code>output_tokens_per_sec</code>, <code>input_tokens</code>, and <code>output_tokens</code>:
          </p>
          <div class="kv-grid">
            <div class="kv-box"><div class="kv-label">Mean &amp; Std</div><div class="kv-val" style="font-size: 0.8rem; font-family: var(--font-sans);">Arithmetic average and population standard deviation (&sigma;) across succeeded calls.</div></div>
            <div class="kv-box"><div class="kv-label">Min &amp; Max</div><div class="kv-val" style="font-size: 0.8rem; font-family: var(--font-sans);">Minimum and maximum observed values during the measurement window.</div></div>
            <div class="kv-box"><div class="kv-label">p50 (Median)</div><div class="kv-val" style="font-size: 0.8rem; font-family: var(--font-sans);">50th percentile; half of succeeded requests completed at or below this value.</div></div>
            <div class="kv-box"><div class="kv-label">p90 / p95 / p99</div><div class="kv-val" style="font-size: 0.8rem; font-family: var(--font-sans);">90th, 95th, and 99th tail percentiles capturing high-concurrency queueing and tail latency spikes.</div></div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">5. System Throughput, Reliability &amp; SLO Goodput (results.jsonl)</div>
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Metric Key</th>
                  <th style="text-align:left;">Report Label</th>
                  <th style="text-align:center;">Unit</th>
                  <th style="text-align:left;">Formula</th>
                  <th style="text-align:left;">Description</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>throughput.rps</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Request Throughput (RPS)</strong></td>
                  <td style="text-align:center;"><code>req/s</code></td>
                  <td style="text-align:left;"><code>total_requests / actual_duration_secs</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Total completed requests per second across all worker threads.</td>
                </tr>
                <tr>
                  <td><code>throughput.goodput_rps</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Goodput (RPS)</strong></td>
                  <td style="text-align:center;"><code>req/s</code></td>
                  <td style="text-align:left;"><code>succeeded_requests / actual_duration_secs</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Successful (<code>HTTP 200</code>) requests completed per second.</td>
                </tr>
                <tr>
                  <td><code>throughput.slo_goodput_rps</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>SLO Goodput (RPS)</strong></td>
                  <td style="text-align:center;"><code>req/s</code></td>
                  <td style="text-align:left;"><code>slo_satisfied_requests / actual_duration_secs</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Requests per second that both succeed and satisfy all configured latency bounds in <code>evaluation.slo</code> (<code>max_ttft_ms</code>, <code>max_tpot_ms</code>, <code>max_e2e_ms</code>).</td>
                </tr>
                <tr>
                  <td><code>throughput.output_tps</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Output Throughput (tok/s)</strong></td>
                  <td style="text-align:center;"><code>tok/s</code></td>
                  <td style="text-align:left;"><code>total_output_tokens / actual_duration_secs</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Aggregate output tokens generated per second across all concurrent threads.</td>
                </tr>
                <tr>
                  <td><code>throughput.input_tps</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Input Throughput (tok/s)</strong></td>
                  <td style="text-align:center;"><code>tok/s</code></td>
                  <td style="text-align:left;"><code>total_input_tokens / actual_duration_secs</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Aggregate prompt tokens processed per second across all concurrent threads.</td>
                </tr>
                <tr>
                  <td><code>throughput.total_tps</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Total Throughput (tok/s)</strong></td>
                  <td style="text-align:center;"><code>tok/s</code></td>
                  <td style="text-align:left;"><code>(total_input_tokens + total_output_tokens) / actual_duration_secs</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Combined input + output tokens processed per second by the endpoint.</td>
                </tr>
                <tr>
                  <td><code>success_rate</code> / <code>error_rate</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Success / Error Rate</strong></td>
                  <td style="text-align:center;"><code>%</code></td>
                  <td style="text-align:left;"><code>(succeeded_requests / total_requests) * 100</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Percentage of requests succeeding vs. failing (broken down in <code>errors_by_type</code>).</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-title">6. Cost Efficiency Metrics (calls.jsonl &amp; results.jsonl)</div>
          <p style="font-size: 0.86rem; color: var(--text-secondary); margin-bottom: 0.7rem;">
            Computed from <code>evaluation.pricing</code> rates in USD per 1,000,000 tokens
            (<code>input_per_1m_tokens</code>, <code>cached_input_per_1m_tokens</code>, <code>output_per_1m_tokens</code>):
          </p>
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Metric Key</th>
                  <th style="text-align:left;">Report Label</th>
                  <th style="text-align:center;">Unit</th>
                  <th style="text-align:left;">Formula</th>
                  <th style="text-align:left;">Description</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>cost_usd</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Request Cost</strong></td>
                  <td style="text-align:center;"><code>USD</code></td>
                  <td style="text-align:left;"><code>((in - cached)/1M)*P_in + (cached/1M)*P_cached + (out/1M)*P_out</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Dollar cost of a single request, discounting cached input tokens from uncached input tokens.</td>
                </tr>
                <tr>
                  <td><code>cost.total_cost_usd</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Total Cost (USD)</strong></td>
                  <td style="text-align:center;"><code>USD</code></td>
                  <td style="text-align:left;"><code>sum(cost_usd)</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Total cumulative dollar cost across all succeeded requests in the run.</td>
                </tr>
                <tr>
                  <td><code>cost.mean_cost_per_request_usd</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Mean Cost / Req (USD)</strong></td>
                  <td style="text-align:center;"><code>USD/req</code></td>
                  <td style="text-align:left;"><code>total_cost_usd / succeeded_requests</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Average dollar cost per succeeded request.</td>
                </tr>
                <tr>
                  <td><code>cost.cost_per_1k_output_tokens_usd</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Cost / 1K Out Tokens</strong></td>
                  <td style="text-align:center;"><code>USD/1K tok</code></td>
                  <td style="text-align:left;"><code>total_cost_usd / (total_output_tokens / 1000)</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Blended total cost (including input cost) normalized per 1,000 generated output tokens.</td>
                </tr>
                <tr>
                  <td><code>cost.cost_per_1k_total_tokens_usd</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Cost / 1K Total Tokens</strong></td>
                  <td style="text-align:center;"><code>USD/1K tok</code></td>
                  <td style="text-align:left;"><code>total_cost_usd / ((total_input_tokens + total_output_tokens) / 1000)</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Blended total cost normalized per 1,000 combined (input + output) tokens.</td>
                </tr>
                <tr>
                  <td><code>cost.output_tps_per_usd_per_hour</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);"><strong>Throughput per Dollar</strong></td>
                  <td style="text-align:center;"><code>(tok/s)/(USD/hr)</code></td>
                  <td style="text-align:left;"><code>output_tps / ((total_cost_usd / actual_duration_secs) * 3600)</code></td>
                  <td style="text-align:left; font-family:var(--font-sans);">Output tokens per second delivered per USD/hour of endpoint spend.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const state = {
      currentTask: null,
      activeTab: 'runs',
      runsData: [],
      selectedRunIdForRuns: null,
      selectedRunIdForConfigs: null,
      datasetIndex: null,
      selectedDatasetIndex: null,
      selectedRunIdForInf: null,
      inferencesList: [],
      selectedCallIndex: null,
    };

    function escapeHtml(str) {
      if (str === null || str === undefined) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    async function fetchJson(url) {
      const resp = await fetch(url);
      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(errText || `HTTP ${resp.status}`);
      }
      return resp.json();
    }

    // Render text or multimodal content (handling OpenAI content arrays & embedded data:image URIs)
    function renderMultimodalContent(content) {
      if (content === null || content === undefined) {
        return '<div class="empty-state">No content</div>';
      }
      if (typeof content === 'string') {
        const dataUriRegex = /(data:image\\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+)/g;
        const matches = content.match(dataUriRegex) || [];
        let html = `<pre class="text-wrap-block">${escapeHtml(content)}</pre>`;
        if (matches.length > 0) {
          html += '<div class="media-container">';
          matches.forEach((uri, idx) => {
            html += `
              <div class="rendered-image-box">
                <span class="kv-label">Embedded Image #${idx + 1}</span>
                <img src="${escapeHtml(uri)}" alt="Embedded multimodal image ${idx + 1}" />
              </div>`;
          });
          html += '</div>';
        }
        return html;
      }
      if (Array.isArray(content)) {
        let html = '';
        const images = [];
        content.forEach((part, idx) => {
          if (!part || typeof part !== 'object') return;
          if (part.type === 'text') {
            html += `<pre class="text-wrap-block">${escapeHtml(part.text || '')}</pre>`;
          } else if (part.type === 'image_url' && part.image_url && part.image_url.url) {
            images.push(part.image_url.url);
          } else if (part.type === 'image' && part.url) {
            images.push(part.url);
          } else {
            html += `<pre class="text-wrap-block">${escapeHtml(JSON.stringify(part, null, 2))}</pre>`;
          }
        });
        if (images.length > 0) {
          html += '<div class="media-container">';
          images.forEach((url, idx) => {
            html += `
              <div class="rendered-image-box">
                <span class="kv-label">Attached Image #${idx + 1}</span>
                <img src="${escapeHtml(url)}" alt="Multimodal content image ${idx + 1}" />
              </div>`;
          });
          html += '</div>';
        }
        return html || '<div class="empty-state">Empty content array</div>';
      }
      return `<pre class="text-wrap-block">${escapeHtml(JSON.stringify(content, null, 2))}</pre>`;
    }

    function renderMessagesList(messages) {
      if (!Array.isArray(messages) || messages.length === 0) {
        return '<div class="empty-state">No messages available.</div>';
      }
      return messages.map((msg, idx) => {
        const role = msg.role || 'user';
        return `
          <div class="message-card">
            <div class="message-header">
              <span class="badge badge-role">${escapeHtml(role)}</span>
              <span class="kv-label">Message #${idx + 1}</span>
            </div>
            ${renderMultimodalContent(msg.content)}
          </div>
        `;
      }).join('');
    }

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const tab = btn.getAttribute('data-tab');
        state.activeTab = tab;
        document.getElementById(`tab-${tab}`).classList.add('active');
        refreshActiveTab();
      });
    });

    document.getElementById('task-selector').addEventListener('change', (e) => {
      state.currentTask = e.target.value;
      state.selectedRunIdForRuns = null;
      state.selectedRunIdForConfigs = null;
      state.selectedDatasetIndex = null;
      state.selectedRunIdForInf = null;
      state.selectedCallIndex = null;
      loadTaskData();
    });

    document.getElementById('refresh-btn').addEventListener('click', () => {
      initTasks(true);
    });

    async function initTasks(keepCurrent = false) {
      try {
        const data = await fetchJson('/api/tasks');
        document.getElementById('basepath-display').textContent = data.base_path;
        const sel = document.getElementById('task-selector');
        const prev = state.currentTask;
        sel.innerHTML = '';
        if (!data.tasks || data.tasks.length === 0) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = '(No tasks found)';
          sel.appendChild(opt);
          state.currentTask = null;
          return;
        }
        data.tasks.forEach(t => {
          const opt = document.createElement('option');
          opt.value = t;
          opt.textContent = t;
          sel.appendChild(opt);
        });
        if (keepCurrent && prev && data.tasks.includes(prev)) {
          sel.value = prev;
          state.currentTask = prev;
        } else {
          sel.value = data.tasks[0];
          state.currentTask = data.tasks[0];
        }
        await loadTaskData();
      } catch (err) {
        document.getElementById('runs-detail').innerHTML =
          `<div class="empty-state">Error loading tasks: ${escapeHtml(err.message)}</div>`;
      }
    }

    async function loadTaskData() {
      if (!state.currentTask) return;
      const runsPayload = await fetchJson(`/api/tasks/${encodeURIComponent(state.currentTask)}/runs`);
      state.runsData = runsPayload.runs || [];
      renderRunsTabList();
      renderConfigsTabList();
      renderInferenceRunsList();
      await refreshActiveTab();
    }

    async function refreshActiveTab() {
      if (!state.currentTask) return;
      if (state.activeTab === 'runs') {
        if (!state.selectedRunIdForRuns && state.runsData.length > 0) {
          selectRunForInspect(state.runsData[state.runsData.length - 1].run_id);
        } else if (state.selectedRunIdForRuns) {
          selectRunForInspect(state.selectedRunIdForRuns);
        }
      } else if (state.activeTab === 'compare') {
        await loadCompareTab();
      } else if (state.activeTab === 'configs') {
        if (!state.selectedRunIdForConfigs && state.runsData.length > 0) {
          await selectRunForConfig(state.runsData[state.runsData.length - 1].run_id);
        }
      } else if (state.activeTab === 'dataset') {
        await loadDatasetTab();
      } else if (state.activeTab === 'inference') {
        if (!state.selectedRunIdForInf && state.runsData.length > 0) {
          await selectRunForInference(state.runsData[state.runsData.length - 1].run_id);
        }
      }
    }

    // 1. RUNS TAB
    function renderRunsTabList() {
      const container = document.getElementById('runs-list');
      document.getElementById('runs-count-badge').textContent = state.runsData.length;
      if (state.runsData.length === 0) {
        container.innerHTML = '<div class="empty-state">No runs found for this task.</div>';
        document.getElementById('runs-detail').innerHTML =
          '<div class="empty-state">No runs available. Run "llmendpoint-perf run" first.</div>';
        return;
      }
      container.innerHTML = state.runsData.map(r => {
        const isSel = r.run_id === state.selectedRunIdForRuns;
        const res = r.results;
        const sub = res
          ? `${res.model} | ${res.succeeded_requests}/${res.total_requests} reqs | ${res.throughput.rps.toFixed(2)} RPS`
          : 'Incomplete run';
        return `
          <div class="list-item ${isSel ? 'selected' : ''}" data-run-id="${escapeHtml(r.run_id)}">
            <div class="item-title">
              <span>${escapeHtml(r.run_id)}</span>
              ${res ? `<span class="badge badge-ok">${res.success_rate.toFixed(0)}% OK</span>` : ''}
            </div>
            <div class="item-sub">${escapeHtml(sub)}</div>
          </div>
        `;
      }).join('');
      container.querySelectorAll('.list-item').forEach(el => {
        el.addEventListener('click', () => {
          selectRunForInspect(el.getAttribute('data-run-id'));
        });
      });
    }

    function selectRunForInspect(runId) {
      state.selectedRunIdForRuns = runId;
      renderRunsTabList();
      const runObj = state.runsData.find(r => r.run_id === runId);
      const detail = document.getElementById('runs-detail');
      if (!runObj) {
        detail.innerHTML = '<div class="empty-state">Run not found.</div>';
        return;
      }
      const res = runObj.results;
      if (!res) {
        detail.innerHTML = `<div class="card"><pre class="code-block">${escapeHtml(runObj.inspect_report)}</pre></div>`;
        return;
      }

      const distRows = [
        ['ttft_ms', 'TTFT (ms)'],
        ['tpot_ms', 'TPOT / ITL (ms/tok)'],
        ['e2e_latency_ms', 'E2E Latency (ms)'],
        ['output_tokens_per_sec', 'Decode Speed (tok/s)'],
        ['input_tokens', 'Input Tokens'],
        ['output_tokens', 'Output Tokens'],
      ].map(([key, label]) => {
        const d = (res.distributions && res.distributions[key]) || {};
        const fmt = v => (typeof v === 'number' ? v.toFixed(2) : '0.00');
        return `
          <tr>
            <td>${escapeHtml(label)}</td>
            <td>${fmt(d.mean)}</td>
            <td>${fmt(d.std)}</td>
            <td>${fmt(d.min)}</td>
            <td>${fmt(d.p50)}</td>
            <td>${fmt(d.p90)}</td>
            <td>${fmt(d.p95)}</td>
            <td>${fmt(d.p99)}</td>
            <td>${fmt(d.max)}</td>
          </tr>
        `;
      }).join('');

      detail.innerHTML = `
        <div class="card">
          <div class="card-title">Run Overview — ${escapeHtml(res.task_name)} / ${escapeHtml(res.run_id)}</div>
          <div class="kv-grid">
            <div class="kv-box"><div class="kv-label">Model</div><div class="kv-val">${escapeHtml(res.model)}</div></div>
            <div class="kv-box"><div class="kv-label">Endpoint</div><div class="kv-val">${escapeHtml(res.model_endpoint)}</div></div>
            <div class="kv-box"><div class="kv-label">Concurrency</div><div class="kv-val">${res.num_threads} threads (wait ${res.wait_time_between_requests_ms} ms)</div></div>
            <div class="kv-box"><div class="kv-label">Duration</div><div class="kv-val">${res.actual_duration_secs.toFixed(2)}s (cfg ${res.configured_run_time_secs}s)</div></div>
            <div class="kv-box"><div class="kv-label">Requests</div><div class="kv-val">${res.succeeded_requests}/${res.total_requests} (${res.success_rate.toFixed(1)}% OK)</div></div>
            <div class="kv-box"><div class="kv-label">Throughput</div><div class="kv-val">${res.throughput.rps.toFixed(2)} RPS | ${res.throughput.output_tps.toFixed(2)} out tok/s</div></div>
            <div class="kv-box"><div class="kv-label">Total Cost</div><div class="kv-val">$${res.cost.total_cost_usd.toFixed(6)} ($${res.cost.mean_cost_per_request_usd.toFixed(6)}/req)</div></div>
            <div class="kv-box"><div class="kv-label">Token Volume</div><div class="kv-val">${res.total_input_tokens} in / ${res.total_output_tokens} out</div></div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Dataset Information</div>
          <div class="kv-grid">
            <div class="kv-box"><div class="kv-label">Generation Model</div><div class="kv-val">${escapeHtml(res.dataset_generation_model || 'N/A')}</div></div>
            <div class="kv-box"><div class="kv-label">Number of Items</div><div class="kv-val">${res.dataset_num_items}</div></div>
          </div>
          <div style="margin-top: 0.65rem;">
            <div class="kv-label" style="margin-bottom: 0.25rem;">Generation Prompt</div>
            <pre class="text-wrap-block">${escapeHtml(res.dataset_generation_prompt || 'N/A')}</pre>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Statistical Distributions</div>
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>p50</th><th>p90</th><th>p95</th><th>p99</th><th>Max</th>
                </tr>
              </thead>
              <tbody>${distRows}</tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-title">CLI Inspect Output</div>
          <pre class="code-block">${escapeHtml(runObj.inspect_report)}</pre>
        </div>
      `;
    }

    // 2. COMPARE TAB
    async function loadCompareTab() {
      const container = document.getElementById('compare-content');
      try {
        const data = await fetchJson(`/api/tasks/${encodeURIComponent(state.currentTask)}/compare`);
        const runs = data.runs || [];
        if (runs.length === 0) {
          container.innerHTML = '<div class="empty-state">No completed runs available to compare.</div>';
          return;
        }
        const headers = runs.map(r => `<th>${escapeHtml(r.task_name + ':' + r.run_id)}</th>`).join('');
        const buildRow = (label, fn) => `
          <tr>
            <td>${escapeHtml(label)}</td>
            ${runs.map(r => `<td>${escapeHtml(fn(r))}</td>`).join('')}
          </tr>
        `;

        let rowsHtml = '';
        rowsHtml += buildRow('Model', r => r.model);
        rowsHtml += buildRow('Threads', r => r.num_threads);
        rowsHtml += buildRow('Duration (s)', r => r.actual_duration_secs.toFixed(2));
        rowsHtml += buildRow('Requests (OK / Total)', r => `${r.succeeded_requests}/${r.total_requests} (${r.success_rate.toFixed(1)}%)`);
        rowsHtml += buildRow('Throughput (RPS)', r => r.throughput.rps.toFixed(2));
        rowsHtml += buildRow('SLO Goodput (RPS)', r => r.throughput.slo_goodput_rps.toFixed(2));
        rowsHtml += buildRow('Output Throughput (tok/s)', r => r.throughput.output_tps.toFixed(2));
        rowsHtml += buildRow('Total Throughput (tok/s)', r => r.throughput.total_tps.toFixed(2));

        const distMetrics = [
          ['ttft_ms', 'TTFT (ms)'],
          ['tpot_ms', 'TPOT (ms/tok)'],
          ['e2e_latency_ms', 'E2E Latency (ms)'],
          ['output_tokens_per_sec', 'Decode Speed (tok/s)'],
          ['input_tokens', 'Input Tokens'],
          ['output_tokens', 'Output Tokens'],
        ];
        distMetrics.forEach(([key, label]) => {
          ['mean', 'p50', 'p95', 'p99'].forEach(stat => {
            const statTitle = stat === 'mean' ? 'Mean' : stat;
            rowsHtml += buildRow(`${label} ${statTitle}`, r => {
              const d = (r.distributions && r.distributions[key]) || {};
              return typeof d[stat] === 'number' ? d[stat].toFixed(2) : '0.00';
            });
          });
        });

        rowsHtml += buildRow('Total Cost (USD)', r => `$${r.cost.total_cost_usd.toFixed(6)}`);
        rowsHtml += buildRow('Mean Cost / Req (USD)', r => `$${r.cost.mean_cost_per_request_usd.toFixed(6)}`);
        rowsHtml += buildRow('Cost / 1K Out Tokens (USD)', r => `$${r.cost.cost_per_1k_output_tokens_usd.toFixed(6)}`);

        const dsCards = runs.map(r => `
          <div class="message-card">
            <div class="message-header">
              <span class="badge badge-role">${escapeHtml(r.task_name + ':' + r.run_id)}</span>
              <span class="kv-label">Model: ${escapeHtml(r.dataset_generation_model || 'N/A')} | Items: ${r.dataset_num_items}</span>
            </div>
            <pre class="text-wrap-block">${escapeHtml(r.dataset_generation_prompt || 'N/A')}</pre>
          </div>
        `).join('');

        container.innerHTML = `
          <div class="card">
            <div class="card-title">Side-by-Side Run Comparison (${runs.length} runs)</div>
            <div class="table-wrap">
              <table class="data-table">
                <thead>
                  <tr><th>Metric</th>${headers}</tr>
                </thead>
                <tbody>${rowsHtml}</tbody>
              </table>
            </div>
          </div>

          <div class="card">
            <div class="card-title">Dataset Information Across Runs</div>
            ${dsCards}
          </div>

          <div class="card">
            <div class="card-title">CLI Compare Output</div>
            <pre class="code-block">${escapeHtml(data.compare_report)}</pre>
          </div>
        `;
      } catch (err) {
        container.innerHTML = `<div class="empty-state">Error loading comparison: ${escapeHtml(err.message)}</div>`;
      }
    }

    // 3. CONFIGS TAB
    function renderConfigsTabList() {
      const container = document.getElementById('configs-run-list');
      document.getElementById('configs-count-badge').textContent = state.runsData.length;
      if (state.runsData.length === 0) {
        container.innerHTML = '<div class="empty-state">No runs found.</div>';
        return;
      }
      container.innerHTML = state.runsData.map(r => {
        const isSel = r.run_id === state.selectedRunIdForConfigs;
        return `
          <div class="list-item ${isSel ? 'selected' : ''}" data-run-id="${escapeHtml(r.run_id)}">
            <div class="item-title"><span>${escapeHtml(r.run_id)}</span></div>
            <div class="item-sub">runs/${escapeHtml(r.run_id)}/config.yaml</div>
          </div>
        `;
      }).join('');
      container.querySelectorAll('.list-item').forEach(el => {
        el.addEventListener('click', () => {
          selectRunForConfig(el.getAttribute('data-run-id'));
        });
      });
    }

    async function selectRunForConfig(runId) {
      state.selectedRunIdForConfigs = runId;
      renderConfigsTabList();
      const detail = document.getElementById('configs-detail');
      try {
        const data = await fetchJson(
          `/api/tasks/${encodeURIComponent(state.currentTask)}/runs/${encodeURIComponent(runId)}/config`
        );
        detail.innerHTML = `
          <div class="card">
            <div class="card-title">Configuration Snapshot — ${escapeHtml(state.currentTask)} / ${escapeHtml(runId)}</div>
            <pre class="code-block">${escapeHtml(data.config_yaml)}</pre>
          </div>
        `;
      } catch (err) {
        detail.innerHTML = `<div class="empty-state">Error loading config.yaml: ${escapeHtml(err.message)}</div>`;
      }
    }

    // 4. DATASET TAB
    async function loadDatasetTab() {
      const listContainer = document.getElementById('dataset-item-list');
      const detailContainer = document.getElementById('dataset-detail');
      try {
        const data = await fetchJson(`/api/tasks/${encodeURIComponent(state.currentTask)}/dataset`);
        state.datasetIndex = data;
        const items = data.items || [];
        document.getElementById('dataset-count-badge').textContent = items.length;
        if (items.length === 0) {
          listContainer.innerHTML = '<div class="empty-state">No prompts.jsonl found.</div>';
          detailContainer.innerHTML = '<div class="empty-state">Dataset has not been generated yet.</div>';
          return;
        }
        renderDatasetList();
        if (state.selectedDatasetIndex === null) {
          await selectDatasetItem(0);
        } else {
          await selectDatasetItem(state.selectedDatasetIndex);
        }
      } catch (err) {
        listContainer.innerHTML = `<div class="empty-state">Error: ${escapeHtml(err.message)}</div>`;
      }
    }

    function renderDatasetList() {
      const listContainer = document.getElementById('dataset-item-list');
      const items = (state.datasetIndex && state.datasetIndex.items) || [];
      listContainer.innerHTML = items.map(it => {
        const isSel = it.index === state.selectedDatasetIndex;
        return `
          <div class="list-item ${isSel ? 'selected' : ''}" data-item-index="${it.index}">
            <div class="item-title">
              <span>Item #${it.index}</span>
              ${it.has_images ? `<span class="badge badge-img">Image (${it.image_count})</span>` : ''}
            </div>
            <div class="item-sub">${escapeHtml(it.preview || '(no text)')}</div>
          </div>
        `;
      }).join('');
      listContainer.querySelectorAll('.list-item').forEach(el => {
        el.addEventListener('click', () => {
          selectDatasetItem( parseInt(el.getAttribute('data-item-index'), 10) );
        });
      });
    }

    async function selectDatasetItem(index) {
      state.selectedDatasetIndex = index;
      renderDatasetList();
      const detailContainer = document.getElementById('dataset-detail');
      try {
        const data = await fetchJson(
          `/api/tasks/${encodeURIComponent(state.currentTask)}/dataset/${index}`
        );
        const dsInfo = (state.datasetIndex && state.datasetIndex.dataset_info) || {};
        const messages = (data.item && data.item.messages) || [];
        detailContainer.innerHTML = `
          <div class="card">
            <div class="card-title">Dataset Generation Info</div>
            <div class="kv-grid">
              <div class="kv-box"><div class="kv-label">Generation Model</div><div class="kv-val">${escapeHtml(dsInfo.generation_model || 'N/A')}</div></div>
              <div class="kv-box"><div class="kv-label">Configured Items</div><div class="kv-val">${dsInfo.num_items ?? 'N/A'}</div></div>
              <div class="kv-box"><div class="kv-label">Actual Items in File</div><div class="kv-val">${state.datasetIndex.total_items}</div></div>
            </div>
            <div style="margin-top: 0.65rem;">
              <div class="kv-label" style="margin-bottom: 0.25rem;">Generation Prompt</div>
              <pre class="text-wrap-block">${escapeHtml(dsInfo.generation_prompt || 'N/A')}</pre>
            </div>
          </div>

          <div class="card">
            <div class="card-title">Dataset Item #${index}</div>
            ${renderMessagesList(messages)}
          </div>
        `;
      } catch (err) {
        detailContainer.innerHTML = `<div class="empty-state">Error loading dataset item: ${escapeHtml(err.message)}</div>`;
      }
    }

    // 5. INFERENCE TAB
    function renderInferenceRunsList() {
      const container = document.getElementById('inference-run-list');
      document.getElementById('inf-runs-count-badge').textContent = state.runsData.length;
      if (state.runsData.length === 0) {
        container.innerHTML = '<div class="empty-state">No runs found.</div>';
        return;
      }
      container.innerHTML = state.runsData.map(r => {
        const isSel = r.run_id === state.selectedRunIdForInf;
        const totalReqs = r.results ? r.results.total_requests : '?';
        return `
          <div class="list-item ${isSel ? 'selected' : ''}" data-run-id="${escapeHtml(r.run_id)}">
            <div class="item-title">
              <span>${escapeHtml(r.run_id)}</span>
              <span class="badge badge-role">${totalReqs} reqs</span>
            </div>
            <div class="item-sub">${r.results ? escapeHtml(r.results.model) : ''}</div>
          </div>
        `;
      }).join('');
      container.querySelectorAll('.list-item').forEach(el => {
        el.addEventListener('click', () => {
          selectRunForInference(el.getAttribute('data-run-id'));
        });
      });
    }

    async function selectRunForInference(runId) {
      state.selectedRunIdForInf = runId;
      state.selectedCallIndex = null;
      renderInferenceRunsList();
      const callsContainer = document.getElementById('inference-call-list');
      const detailContainer = document.getElementById('inference-detail');
      try {
        const data = await fetchJson(
          `/api/tasks/${encodeURIComponent(state.currentTask)}/runs/${encodeURIComponent(runId)}/inferences`
        );
        state.inferencesList = data.inferences || [];
        document.getElementById('inf-calls-count-badge').textContent = state.inferencesList.length;
        if (state.inferencesList.length === 0) {
          callsContainer.innerHTML = '<div class="empty-state">No calls found in calls.jsonl.</div>';
          detailContainer.innerHTML = '<div class="empty-state">No inference records available.</div>';
          return;
        }
        renderInferenceCallsList();
        await selectInferenceCall(0);
      } catch (err) {
        callsContainer.innerHTML = `<div class="empty-state">Error: ${escapeHtml(err.message)}</div>`;
      }
    }

    function renderInferenceCallsList() {
      const callsContainer = document.getElementById('inference-call-list');
      callsContainer.innerHTML = state.inferencesList.map(c => {
        const isSel = c.call_index === state.selectedCallIndex;
        const badgeClass = c.is_success ? 'badge-ok' : 'badge-err';
        const statusLabel = c.status_code || 'ERR';
        const ttft = c.ttft_ms !== null && c.ttft_ms !== undefined ? `${c.ttft_ms.toFixed(1)}ms` : 'N/A';
        const e2e = c.e2e_latency_ms !== null && c.e2e_latency_ms !== undefined ? `${c.e2e_latency_ms.toFixed(1)}ms` : 'N/A';
        return `
          <div class="list-item ${isSel ? 'selected' : ''}" data-call-index="${c.call_index}">
            <div class="item-title">
              <span>#${c.call_index} (prompt #${c.prompt_index})</span>
              <span class="badge ${badgeClass}">HTTP ${escapeHtml(statusLabel)}</span>
            </div>
            <div class="item-sub">TTFT: ${ttft} | E2E: ${e2e} | ${c.input_tokens} in / ${c.output_tokens} out</div>
          </div>
        `;
      }).join('');
      callsContainer.querySelectorAll('.list-item').forEach(el => {
        el.addEventListener('click', () => {
          selectInferenceCall(parseInt(el.getAttribute('data-call-index'), 10));
        });
      });
    }

    async function selectInferenceCall(callIndex) {
      state.selectedCallIndex = callIndex;
      renderInferenceCallsList();
      const detailContainer = document.getElementById('inference-detail');
      try {
        const data = await fetchJson(
          `/api/tasks/${encodeURIComponent(state.currentTask)}/runs/${encodeURIComponent(state.selectedRunIdForInf)}/inferences/${callIndex}`
        );
        const inf = data.inference;
        const inputPrompt = data.input_prompt;
        const promptMessages = (inputPrompt && inputPrompt.messages) || [];

        // Check if raw_response_metadata has multimodal output content as well
        const metaContent = inf.raw_response_metadata && inf.raw_response_metadata.multimodal_content
          ? inf.raw_response_metadata.multimodal_content
          : inf.response_content;

        detailContainer.innerHTML = `
          <div class="card">
            <div class="card-title">Inference Telemetry — Call #${callIndex} (${escapeHtml(inf.request_id)})</div>
            <div class="kv-grid">
              <div class="kv-box"><div class="kv-label">Status Code</div><div class="kv-val">${inf.status_code ?? 'N/A'} ${inf.error ? '(' + escapeHtml(inf.error) + ')' : ''}</div></div>
              <div class="kv-box"><div class="kv-label">Prompt Index</div><div class="kv-val">#${inf.prompt_index} (Thread ${inf.thread_id})</div></div>
              <div class="kv-box"><div class="kv-label">TTFT</div><div class="kv-val">${inf.ttft_ms !== null ? inf.ttft_ms.toFixed(2) + ' ms' : 'N/A'}</div></div>
              <div class="kv-box"><div class="kv-label">TPOT / ITL</div><div class="kv-val">${inf.tpot_ms !== null ? inf.tpot_ms.toFixed(2) + ' ms/tok' : 'N/A'}</div></div>
              <div class="kv-box"><div class="kv-label">E2E Latency</div><div class="kv-val">${inf.e2e_latency_ms.toFixed(2)} ms</div></div>
              <div class="kv-box"><div class="kv-label">Decode Speed</div><div class="kv-val">${inf.output_tokens_per_sec.toFixed(2)} tok/s</div></div>
              <div class="kv-box"><div class="kv-label">Tokens (In / Out)</div><div class="kv-val">${inf.input_tokens} in / ${inf.output_tokens} out</div></div>
              <div class="kv-box"><div class="kv-label">Reasoning / Cached</div><div class="kv-val">${inf.reasoning_tokens} reasoning / ${inf.cached_input_tokens} cached</div></div>
              <div class="kv-box"><div class="kv-label">Request Cost</div><div class="kv-val">$${inf.cost_usd.toFixed(8)}</div></div>
              <div class="kv-box"><div class="kv-label">Timestamp Start</div><div class="kv-val">${escapeHtml(inf.timestamp_start)}</div></div>
              <div class="kv-box"><div class="kv-label">Timestamp First Token</div><div class="kv-val">${escapeHtml(inf.timestamp_first_token || 'N/A')}</div></div>
              <div class="kv-box"><div class="kv-label">Timestamp End</div><div class="kv-val">${escapeHtml(inf.timestamp_end)}</div></div>
            </div>
          </div>

          <div class="card">
            <div class="card-title">Input Prompt (Prompt #${inf.prompt_index})</div>
            ${renderMessagesList(promptMessages)}
          </div>

          <div class="card">
            <div class="card-title">Model Output</div>
            ${renderMultimodalContent(metaContent)}
          </div>

          <div class="card">
            <div class="card-title">Raw Response Metadata & Record JSON</div>
            <pre class="code-block">${escapeHtml(JSON.stringify(inf, null, 2))}</pre>
          </div>
        `;
      } catch (err) {
        detailContainer.innerHTML = `<div class="empty-state">Error loading inference details: ${escapeHtml(err.message)}</div>`;
      }
    }

    initTasks(false);
  </script>
</body>
</html>
"""


def create_ui_request_handler(base_path: str) -> type[BaseHTTPRequestHandler]:
    """Create a ThreadingHTTPServer request handler bound to `base_path`."""

    class UIRequestHandler(BaseHTTPRequestHandler):
        """HTTP request handler serving the single-page UI and JSON inspection APIs."""

        def log_message(self, format: str, *args: Any) -> None:  # pylint: disable=redefined-builtin
            # Silence default per-request stderr noise
            pass

        def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str, status: int = HTTPStatus.OK) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # pylint: disable=invalid-name
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"

            try:
                if path == "/":
                    self._send_html(UI_HTML)
                    return

                parts = [unquote(p) for p in path.strip("/").split("/")]
                # /api/tasks
                if parts == ["api", "tasks"]:
                    self._send_json(get_tasks_payload(base_path))
                    return

                # /api/tasks/<task_name>/...
                if len(parts) >= 3 and parts[0] == "api" and parts[1] == "tasks":
                    task_name = parts[2]
                    if len(parts) == 4 and parts[3] == "runs":
                        self._send_json(get_task_runs_payload(task_name, base_path))
                        return
                    if len(parts) == 4 and parts[3] == "compare":
                        self._send_json(get_task_compare_payload(task_name, base_path))
                        return
                    if len(parts) == 4 and parts[3] == "dataset":
                        self._send_json(get_dataset_index_payload(task_name, base_path))
                        return
                    if len(parts) == 5 and parts[3] == "dataset":
                        item_idx = int(parts[4])
                        self._send_json(
                            get_dataset_item_payload(task_name, item_idx, base_path)
                        )
                        return
                    if len(parts) == 6 and parts[3] == "runs" and parts[5] == "config":
                        run_id = parts[4]
                        self._send_json(
                            get_run_config_payload(task_name, run_id, base_path)
                        )
                        return
                    if len(parts) == 6 and parts[3] == "runs" and parts[5] == "inferences":
                        run_id = parts[4]
                        self._send_json(
                            get_run_inferences_index_payload(task_name, run_id, base_path)
                        )
                        return
                    if len(parts) == 7 and parts[3] == "runs" and parts[5] == "inferences":
                        run_id = parts[4]
                        call_idx = int(parts[6])
                        self._send_json(
                            get_run_inference_detail_payload(
                                task_name, run_id, call_idx, base_path
                            )
                        )
                        return

                self._send_json(
                    {"error": f"Not found: {path}"}, status=HTTPStatus.NOT_FOUND
                )
            except (FileNotFoundError, IndexError, ValueError) as exc:
                self._send_json(
                    {"error": str(exc)}, status=HTTPStatus.NOT_FOUND
                )
            except Exception as exc:  # pylint: disable=broad-except
                self._send_json(
                    {"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR
                )

    return UIRequestHandler


def create_ui_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    base_path: str | None = None,
) -> ThreadingHTTPServer:
    """Create and return a configured ThreadingHTTPServer for the inspection UI."""
    resolved_base_path = resolve_base_path(base_path)
    handler_cls = create_ui_request_handler(resolved_base_path)
    return ThreadingHTTPServer((host, port), handler_cls)


def run_ui_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    base_path: str | None = None,
) -> None:
    """Validate LLMENDPOINTPERF_BASEPATH and run the UI server until interrupted."""
    env_val = base_path or os.environ.get("LLMENDPOINTPERF_BASEPATH") or os.environ.get(
        "LLMENDPOINTPERF_GCS_BASEPATH"
    )
    if not env_val:
        raise RuntimeError(
            "LLMENDPOINTPERF_BASEPATH is not set. Please set the environment variable "
            "LLMENDPOINTPERF_BASEPATH before starting the UI server."
        )
    server = create_ui_server(host=host, port=port, base_path=env_val)
    actual_host, actual_port = server.server_address[:2]
    print(
        f"Starting llmendpoint-perf UI at http://{actual_host}:{actual_port} "
        f"(base_path={env_val.rstrip('/')})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
