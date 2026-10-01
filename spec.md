# LLM Endpoint Performance Measurement (`llmendpoint-perf`)

## 1. Overview and Objectives

`llmendpoint-perf` is a command-line framework and Python library for measuring, analyzing, and comparing the technical and cost performance of OpenAI-compatible LLM endpoints. It is designed for developers and ML researchers to benchmark serving infrastructure, compare models/endpoints under realistic load, and identify latency, throughput, and cost bottlenecks.

### Core Capabilities
1. **Synthetic Dataset Generation**: Automatically generates diverse text and multimodal (text + image) benchmark datasets tailored to target input/output token distributions and domain descriptions.
2. **Load & Performance Benchmarking**: Executes multi-threaded, rate-controlled streaming workloads against OpenAI-compatible chat completion endpoints.
3. **Granular Telemetry & Cost Analysis**: Captures per-request streaming timing metrics, token usage, error rates, and dollar costs, alongside aggregated statistical distributions.
4. **Task & Artifact Management**: Organizes configurations, datasets, raw call logs, and aggregated results reproducibly in Google Cloud Storage (GCS) or local storage.

---

## 2. Performance Metrics

All latency metrics are measured using Server-Sent Events (`stream: true` with `stream_options: {"include_usage": true}`) to accurately separate prefill/queueing latency from token generation speed.

### 2.1 Per-Request Metrics
* **Time to First Token (TTFT)** (`ttft_ms`): Time elapsed from sending the HTTP request to receiving the first SSE chunk containing non-empty content (or reasoning content) tokens. Reflects network latency, queueing delay, and prompt prefill time.
* **Time Per Output Token (TPOT) / Inter-Token Latency (ITL)** (`tpot_ms`): Average time between consecutive output tokens during the generation phase:
  $$\text{TPOT} = \frac{T_{\text{last\_token}} - T_{\text{first\_token}}}{\max(\text{output\_tokens} - 1, 1)}$$
* **End-to-End Request Latency (E2E)** (`e2e_latency_ms`): Total time from sending the request to receiving the final stream completion chunk (`[DONE]`), equivalent to $\text{TTFT} + \text{TPOT} \times (\text{output\_tokens} - 1)$ plus stream teardown overhead.
* **Token Counts**:
  * `input_tokens`: Prompt tokens (from API `usage.prompt_tokens`, or fallback tokenizer estimate).
  * `output_tokens`: Completion tokens (from API `usage.completion_tokens`, or chunk count fallback).
  * `reasoning_tokens`: Thinking/reasoning tokens if reported by the endpoint (`usage.completion_tokens_details.reasoning_tokens`).
  * `cached_input_tokens`: Cached prompt tokens if reported (`usage.prompt_tokens_details.cached_tokens`).
* **Output Generation Speed** (`output_tokens_per_sec`): Per-request decode throughput ($\text{output\_tokens} / (T_{\text{last\_token}} - T_{\text{first\_token}})$).
* **Request Cost** (`cost_usd`): Computed from configured per-million token prices for input, cached input, and output tokens.

### 2.2 Aggregated Run Metrics
Across a benchmark run (excluding any configured warmup period), `llmendpoint-perf` computes summary statistics (`mean`, `std`, `min`, `p50`, `p90`, `p95`, `p99`, `max`) for TTFT, TPOT, E2E latency, and token counts, as well as system-level metrics:
* **Throughput**:
  * **Request Throughput (`rps`)**: Completed requests per second (total and successful-only "goodput").
  * **Output Token Throughput (`output_tps`)**: Total generated output tokens per second across all concurrent workers.
  * **Total Token Throughput (`total_tps`)**: Combined input + output tokens processed per second.
* **Reliability & Error Metrics**:
  * **Success Rate / Error Rate**: Percentage of requests succeeding (`HTTP 200`) vs. failing (`HTTP 429` rate limits, `5xx` server errors, or client timeouts).
  * **Goodput (`slo_goodput_rps`)**: Requests per second that both succeed and satisfy optional user-defined SLO thresholds (e.g., `max_ttft_ms`, `max_tpot_ms`, `max_e2e_ms`).
* **Cost Efficiency**:
  * **Mean Cost per Request** and **Cost per 1K Output Tokens**.
  * **Total Run Cost (`total_cost_usd`)**.
  * **Throughput per Dollar**: Output tokens per second per USD/hour.

---

## 3. Storage Architecture & Evaluation Tasks

`llmendpoint-perf` organizes work into named **Evaluation Tasks**. All artifacts for all tasks are stored under a base path defined by the environment variable `LLMENDPOINTPERF_BASEPATH` (supports `gs://bucket/path` URLs as well as local filesystem paths for local testing).

Each evaluation task `<task-name>` stores its artifacts under `$LLMENDPOINTPERF_BASEPATH/<task-name>/`:

```text
$LLMENDPOINTPERF_BASEPATH/<task-name>/
├── config.yaml                          # Master configuration file for the task
├── dataset-generation.log               # Log of synthetic dataset generation activities
├── prompts.jsonl                        # Generated (or user-supplied) evaluation prompts
└── runs/
    └── YYYYMMDD-HHMMSS/                 # Dedicated directory per benchmark run (UTC timestamp)
        ├── config.yaml                  # Immutable snapshot of config.yaml used for this run
        ├── results.jsonl                # Aggregated summary metrics and statistical distributions
        ├── calls.jsonl                  # Detailed per-request telemetry and raw responses
        └── log.txt                      # Execution log for the run (mirrors stdout)
```

> **Note on Run Folders**: Each run is isolated in its own `run/YYYYMMDD-HHMMSS/` directory (using UTC timestamps) so multiple runs of the same task (e.g., across different concurrency settings or endpoints) are cleanly organized, chronologically sortable, and contain clean, unprefixed artifact filenames.

---

## 4. Synthetic Dataset Generation

Synthetic evaluation datasets are generated using an OpenAI-compatible LLM endpoint configured under the `dataset` section of `config.yaml`.

### 4.1 Diversity-Preserving Prompt Wrapping
To generate `num_items` distinct evaluation prompts without mode collapse or repetitive outputs across calls:
1. `llmendpoint-perf` wraps the user's `generation_prompt` inside a structured system prompt and user template that instructs the generator model to output a single standalone prompt per call.
2. **Entropy & Variation Injection**: Each generation request receives a unique item index, a randomized variation seed/persona/angle hint, and a target input token length sampled uniformly from the user's requested range.
3. **Concurrent Generation**: Generation requests are executed concurrently (controlled by `dataset.num_threads`, defaulting to 5) with automatic retries on transient errors.

### 4.2 Output Format (`prompts.jsonl`)
The generated dataset is saved to `$LLMENDPOINTPERF_BASEPATH/<task-name>/prompts.jsonl`, where each line is a valid JSON object formatted for an OpenAI-compatible `/v1/chat/completions` request:

**Text-only format:**
```json
{"messages": [{"role": "user", "content": "What are the warranty terms and return eligibility for a refurbished espresso machine purchased from an authorized third-party marketplace seller?"}]}
{"messages": [{"role": "user", "content": "Compare the breathability and waterproofing ratings of 3-layer Gore-Tex trail running jackets versus lightweight DWR-coated windbreakers."}]}
```

**Multimodal format (Text + Image):**
When multimodal prompt generation is requested (via `dataset.multimodal.enabled: true` or explicit multimodal instructions in `generation_prompt`), `llmendpoint-perf` generates or attaches images matching the configured resolution and encodes them as base64 data URIs:
```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "question"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...."}}]}]}
```
Supported image sources for multimodal dataset generation:
* `synthetic`: Programmatically rendered test images (charts, product card mockups, geometric/text patterns) at the exact target resolution (`width x height`) and format (`jpeg`/`png`) to deterministically benchmark vision token encoding and payload transfer overhead.
* `directory` / `gcs_prefix`: Samples real images from a specified local or GCS folder and pairs them with generated questions.

---

## 5. Benchmark Execution Engine

When running an evaluation (`llmendpoint-perf run`), the execution engine performs the following steps:

1. **Initialization & Snapshot**: Reads `config.yaml` and `prompts.jsonl` from `$LLMENDPOINTPERF_BASEPATH/<task-name>/`, validates parameters, creates the run directory `$LLMENDPOINTPERF_BASEPATH/<task-name>/run/YYYYMMDD-HHMMSS/`, and writes a snapshot of the active configuration to `run/YYYYMMDD-HHMMSS/config.yaml`.
2. **Optional Warmup Phase**: If `warmup_requests > 0`, dispatches warmup calls to prime connection pools and endpoint caches before starting the measurement clock.
3. **Multi-Threaded Load Generation**:
   * Spawns `num_threads` concurrent worker threads.
   * Dispatches streaming requests against `model_endpoint`, pacing requests using `wait_time_between_requests_ms` (either per-worker delay or global rate pacing).
   * Cycles through `prompts.jsonl` (sequentially or randomly according to `sampling_strategy`) until `run_time_secs` elapses (or `max_requests` is reached, if specified).
4. **Live Progress & Logging**: Streams structured progress updates (elapsed time, active requests, rolling RPS, rolling p50/p95 TTFT & TPOT, error count) to both `stdout` and `run/YYYYMMDD-HHMMSS/log.txt`.
5. **Artifact Persistence**:
   * Streams each completed/failed request record to `run/YYYYMMDD-HHMMSS/calls.jsonl`.
   * Computes final statistical aggregations and writes them to `run/YYYYMMDD-HHMMSS/results.jsonl`.

### 5.1 Artifact Schemas

#### `run/YYYYMMDD-HHMMSS/calls.jsonl` (One JSON object per request)
```json
{
  "request_id": "uuid4",
  "thread_id": 2,
  "prompt_index": 42,
  "timestamp_start": "2026-10-01T22:15:01.123456Z",
  "timestamp_first_token": "2026-10-01T22:15:01.450123Z",
  "timestamp_end": "2026-10-01T22:15:02.890456Z",
  "status_code": 200,
  "error": null,
  "ttft_ms": 326.67,
  "tpot_ms": 14.55,
  "e2e_latency_ms": 1767.0,
  "input_tokens": 185,
  "output_tokens": 100,
  "reasoning_tokens": 0,
  "cached_input_tokens": 0,
  "output_tokens_per_sec": 68.73,
  "cost_usd": 0.001231,
  "response_content": "Full generated text response...",
  "raw_response_metadata": {"id": "chatcmpl-...", "model": "gemini-2.5-pro", "finish_reason": "stop"}
}
```

#### `run/YYYYMMDD-HHMMSS/results.jsonl` (Summary object per run)
Contains run metadata, configuration summary, total duration, request counts (total, succeeded, failed by error type), throughput (`rps`, `output_tps`, `total_tps`, `slo_goodput_rps`), cost summary, and percentile distributions (`mean`, `std`, `min`, `p50`, `p90`, `p95`, `p99`, `max`) for `ttft_ms`, `tpot_ms`, `e2e_latency_ms`, `input_tokens`, and `output_tokens`.

---

## 6. Command Line Interface

```bash
llmendpoint-perf <command> [options]
```

### Available Commands
* **`init`**: Initializes a new evaluation task in `$LLMENDPOINTPERF_BASEPATH/<task-name>` with a template `config.yaml` (or uploads a local `config.yaml`).
  ```bash
  llmendpoint-perf init <task-name> [--config ./local-config.yaml]
  ```
* **`generate_dataset`**: Generates the synthetic dataset (`prompts.jsonl`) for the specified evaluation task based on its `config.yaml`.
  ```bash
  llmendpoint-perf generate_dataset <task-name> [--overwrite]
  ```
* **`run`**: Executes the benchmark evaluation task and writes `results.jsonl`, `calls.jsonl`, `config.yaml`, and `log.txt` under `run/YYYYMMDD-HHMMSS/`. Both `stdout` and `log.txt` receive identical formatted output.
  ```bash
  llmendpoint-perf run <task-name> [--config-override key=value ...]
  ```
* **`inspect`**: Lists all runs for a task or prints a formatted summary report of a specific run (or latest run).
  ```bash
  llmendpoint-perf inspect <task-name> [--run-id YYYYMMDD-HHMMSS]
  ```
* **`compare`**: Compares performance and cost metrics side-by-side across multiple tasks or specific runs.
  ```bash
  llmendpoint-perf compare <task-name-1>[:<run-id-1>] <task-name-2>[:<run-id-2>]
  ```

---

## 7. Configuration File (`config.yaml`)

Below is the complete configuration schema with required fields and optional parameters (shown with sensible defaults):

```yaml
dataset:
  generation_prompt: "questions about retail products with 20 to 500 input tokens, generating ~100 output tokens"
  num_items: 200
  generation_model_endpoint: "http://google.com/api/openai/v1"
  generation_model: "gemini-2.5-pro"
  # Optional dataset generation settings:
  api_key_env: "OPENAI_API_KEY"          # Environment variable holding the API key
  num_threads: 5                         # Concurrency for dataset generation
  temperature: 1.0                       # High temperature for diverse synthetic items
  multimodal:
    enabled: false                       # Set true to include base64 images in prompts.jsonl
    image_source: "synthetic"            # "synthetic" or path/GCS prefix to sample images from
    image_width: 512
    image_height: 512
    image_format: "jpeg"

evaluation:
  model_endpoint: "http://google.com/api/openai/v1"
  model: "gemini-2.5-pro"
  num_threads: 10
  wait_time_between_requests_ms: 20
  run_time_secs: 600
  # Optional evaluation settings:
  api_key_env: "OPENAI_API_KEY"          # Environment variable holding the API key
  max_requests: null                     # Optional cap on total requests (null = run until run_time_secs)
  warmup_requests: 0                     # Number of initial requests excluded from final stats
  request_timeout_secs: 120              # Per-request HTTP timeout
  sampling_strategy: "round_robin"       # "round_robin" or "random"
  generation_params:                     # Forwarded to the OpenAI chat completions request
    temperature: 0.7
    max_tokens: 256
  pricing:                               # Optional pricing (USD per 1M tokens) for cost metrics
    input_per_1m_tokens: 1.25
    output_per_1m_tokens: 10.00
    cached_input_per_1m_tokens: 0.3125
  slo:                                   # Optional SLO thresholds for goodput calculation
    max_ttft_ms: 1000
    max_tpot_ms: 50
    max_e2e_ms: 5000
```

---

## 8. Version and Change Log

* **v0.1 (Initial Draft)**:
  * Defined core objectives (latency, throughput, cost performance, synthetic dataset generation).
  * Specified GCS artifact storage under `LLMENDPOINTPERF_BASEPATH`, `prompts.jsonl` format (text and base64 multimodal), basic CLI (`generate_dataset`, `run`), and base `config.yaml` structure.
* **v0.2 (2026-10-01 — Structured Specification & Feature Enhancements)**:
  * **Streaming & Metric Definitions (Section 2)**: Explicitly specified SSE streaming (`stream: true`, `include_usage: true`) as the mechanism to decouple **TTFT** (Time to First Token) from **TPOT/ITL** (Time Per Output Token) and **E2E Latency**, and added support for reasoning and cached token accounting.
  * **Statistical Aggregations & Goodput**: Added percentile distributions (`p50`, `p90`, `p95`, `p99`), error rate tracking, and SLO-based **Goodput** (`slo_goodput_rps`).
  * **Storage & Artifact Schemas (Sections 3 & 5)**: Standardized run timestamp identifiers to `YYYYMMDD-HHMMSS`, documented directory layout, and defined concrete JSON schemas for call and result artifacts.
  * **Synthetic Dataset Enhancements (Section 4)**: Added entropy/variation injection to prevent repetitive prompts, concurrent dataset generation (`dataset.num_threads`), and explicit `multimodal` configuration options (synthetic rendered images vs. image directory/GCS sampling).
  * **CLI Expansion (Section 6)**: Added `init`, `inspect`, and `compare` commands to support the full lifecycle of defining, executing, and inspecting evaluation tasks.
  * **Configuration Schema (Section 7)**: Preserved full backward compatibility with the original minimal `config.yaml` while adding optional sections for `api_key_env`, `warmup_requests`, `request_timeout_secs`, `generation_params`, `pricing` (enabling the cost metrics required by Section 1), and `slo`.
* **v0.3 (2026-10-01 — Dedicated Run Subfolders)**:
  * Updated Section 3, Section 5, and Section 6 so each run creates its own `run/YYYYMMDD-HHMMSS/` subfolder containing `config.yaml`, `results.jsonl`, `calls.jsonl`, and `log.txt` without redundant date prefixes on the filenames.
