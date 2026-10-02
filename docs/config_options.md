# `config.yaml` Configuration Reference

Every evaluation task in `llmendpoint-perf` is governed by a `config.yaml` file supplied during task initialization (`llmendpoint-perf init <task-name> --config <path>`).

The configuration file is divided into two top-level sections:
1. **`dataset`**: Controls synthetic dataset generation (`llmendpoint-perf generate_dataset`).
2. **`evaluation`**: Controls load generation, streaming telemetry, pricing, and SLO goodput calculation (`llmendpoint-perf run`).

---

## CompleteAnnotated Example

```yaml
dataset:
  # Required fields
  generation_prompt: "questions about retail products with 20 to 500 input tokens, generating ~100 output tokens"
  num_items: 200
  generation_model_endpoint: "https://generativelanguage.googleapis.com/v1beta/openai"
  generation_model: "gemini-3.8-flash"

  # Optional fields (shown with defaults)
  api_key_env: "OPENAI_API_KEY"
  api_key: null
  num_threads: 5
  temperature: 1.0
  request_timeout_secs: 120.0
  multimodal:
    enabled: false
    image_source: "synthetic"
    image_width: 512
    image_height: 512
    image_format: "jpeg"

evaluation:
  # Required fields
  model_endpoint: "https://generativelanguage.googleapis.com/v1beta/openai"
  model: "gemini-3.5-flash-lite"

  # Load & concurrency settings (optional, shown with defaults)
  num_threads: 10
  wait_time_between_requests_ms: 20.0
  run_time_secs: 600.0
  max_requests: null
  warmup_requests: 0
  request_timeout_secs: 120.0
  sampling_strategy: "round_robin"

  # Authentication (optional)
  api_key_env: "OPENAI_API_KEY"
  api_key: null

  # Model generation parameters forwarded to /chat/completions (optional)
  generation_params:
    temperature: 0.7
    max_tokens: 256

  # Token pricing in USD per 1M tokens for cost metrics (optional)
  pricing:
    input_per_1m_tokens: 1.25
    output_per_1m_tokens: 10.00
    cached_input_per_1m_tokens: 0.3125

  # Latency thresholds for SLO goodput calculation (optional)
  slo:
    max_ttft_ms: 1000.0
    max_tpot_ms: 50.0
    max_e2e_ms: 5000.0
```

---

## 1. `dataset` Section Options

The `dataset` section configures how `llmendpoint-perf generate_dataset <task-name>` creates `$LLMENDPOINTPERF_BASEPATH/<task-name>/prompts.jsonl`.

| Option | Type | Required | Default | Constraints / Possible Values | Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `generation_prompt` | `string` | **Yes** | — | Non-empty string | Natural-language specification of the prompts to generate. If it includes a token range phrase like `"20 to 500 input tokens"` or `"50-200 tokens"`, the generator automatically samples a per-item target length uniformly within that range. If it includes multimodal keywords (`"multimodal"`, `"with an image"`, `"with images"`, `"attached image"`, `"image input"`, `"visual input"`), multimodal image attachment is automatically enabled. |
| `num_items` | `integer` | **Yes** | — | `>= 1` | Total number of synthetic evaluation items (lines) to generate in `prompts.jsonl`. |
| `generation_model_endpoint` | `string` | **Yes** | — | Valid HTTP/HTTPS URL | Base URL of the OpenAI-compatible API endpoint used to generate the dataset (e.g., `"https://generativelanguage.googleapis.com/v1beta/openai"` or `"http://localhost:8000/v1"`). `/chat/completions` is appended automatically if not already at the end of the URL. |
| `generation_model` | `string` | **Yes** | — | Non-empty string | Model identifier sent to `generation_model_endpoint` to synthesize the prompts (e.g., `"gemini-3.8-flash"`, `"gemini-2.5-pro"`). |
| `api_key_env` | `string` | No | `"OPENAI_API_KEY"` | Environment variable name | Name of the environment variable containing the Bearer API key for `generation_model_endpoint`. Fallback lookup order: `dataset.api_key` → `$<api_key_env>` → `$OPENAI_API_KEY` → `$GEMINI_API_KEY` → `$GOOGLE_API_KEY` → `"EMPTY"`. |
| `api_key` | `string \| null` | No | `null` | String or `null` | Explicit API key override. Using `api_key_env` is recommended so credentials are not stored in `config.yaml`. |
| `num_threads` | `integer` | No | `5` | `>= 1` | Number of concurrent worker threads used to generate dataset items in parallel. |
| `temperature` | `float` | No | `1.0` | `0.0` to `2.0` | Sampling temperature passed to `generation_model` when synthesizing dataset items. Higher values (`1.0`) increase prompt diversity. |
| `request_timeout_secs` | `float` | No | `120.0` | `> 0.0` | Per-request HTTP timeout (in seconds) when calling `generation_model_endpoint`. |

### `dataset.multimodal` Sub-Options

Controls multimodal (`text` + base64 `image_url`) dataset generation:

| Option | Type | Required | Default | Constraints / Possible Values | Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `multimodal.enabled` | `boolean` | No | `false` | `true`, `false` | When `true`, each generated prompt in `prompts.jsonl` is formatted as a multimodal content array containing both `{"type": "text", ...}` and `{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}`. |
| `multimodal.image_source` | `string` | No | `"synthetic"` | `"synthetic"`, local path, or `gs://bucket/prefix` | Source of images attached to multimodal prompts:<br>• `"synthetic"`: Programmatically renders diverse test images (charts, product cards, geometric patterns) at `image_width` × `image_height`.<br>• **Local file or directory path**: Samples `.jpg`, `.jpeg`, `.png`, or `.webp` files from disk and resizes them to `image_width` × `image_height`.<br>• **`gs://bucket/prefix`**: Samples `.jpg`, `.jpeg`, `.png`, or `.webp` blobs from GCS and resizes them to `image_width` × `image_height`. |
| `multimodal.image_width` | `integer` | No | `512` | `16` to `4096` | Width (in pixels) of the attached image. |
| `multimodal.image_height` | `integer` | No | `512` | `16` to `4096` | Height (in pixels) of the attached image. |
| `multimodal.image_format` | `string` | No | `"jpeg"` | `"jpeg"`, `"png"` | Image compression format used when encoding the image into a base64 data URI (`data:image/jpeg;base64,...` or `data:image/png;base64,...`). |

---

## 2. `evaluation` Section Options

The `evaluation` section configures how `llmendpoint-perf run <task-name>` executes the benchmark and computes metrics.

| Option | Type | Required | Default | Constraints / Possible Values | Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `model_endpoint` | `string` | **Yes** | — | Valid HTTP/HTTPS URL | Base URL of the OpenAI-compatible endpoint under evaluation. `/chat/completions` is appended automatically if not already present. |
| `model` | `string` | **Yes** | — | Non-empty string | Model identifier sent in the `/chat/completions` request payload during the benchmark run. |
| `num_threads` | `integer` | No | `10` | `>= 1` | Number of concurrent worker threads dispatching streaming inference requests. |
| `wait_time_between_requests_ms` | `float` | No | `20.0` | `>= 0.0` | Minimum global delay (in milliseconds) enforced between consecutive request dispatches across all worker threads. Set to `0` to dispatch requests as fast as worker threads become free. |
| `run_time_secs` | `float` | No | `600.0` | `> 0.0` | Total duration (in seconds) of the measurement window. Worker threads stop dispatching new requests once `run_time_secs` has elapsed, and in-flight requests are awaited before computing final statistics. |
| `max_requests` | `integer \| null` | No | `null` | `null` or integer `>= 1` | Optional upper bound on the total number of measured requests to dispatch. If set, the run terminates as soon as either `max_requests` is reached or `run_time_secs` elapses (whichever comes first). |
| `warmup_requests` | `integer` | No | `0` | `>= 0` | Number of initial requests executed before starting the benchmark clock. Useful for warming up connection pools or endpoint caches; excluded from `calls.jsonl` and `results.jsonl`. |
| `request_timeout_secs` | `float` | No | `120.0` | `> 0.0` | Maximum time (in seconds) to wait for an individual streaming request before recording a timeout failure. |
| `sampling_strategy` | `string` | No | `"round_robin"` | `"round_robin"`, `"random"` | Strategy for selecting prompts from `prompts.jsonl`:<br>• `"round_robin"`: Cycles sequentially through `0, 1, ..., N-1, 0, ...`.<br>• `"random"`: Samples uniformly at random using a deterministic seed (`42`). |
| `api_key_env` | `string` | No | `"OPENAI_API_KEY"` | Environment variable name | Environment variable used to look up the Bearer API key for `model_endpoint`. Fallback lookup order: `evaluation.api_key` → `$<api_key_env>` → `$OPENAI_API_KEY` → `$GEMINI_API_KEY` → `$GOOGLE_API_KEY` → `"EMPTY"`. |
| `api_key` | `string \| null` | No | `null` | String or `null` | Optional explicit API key string override. |
| `generation_params` | `mapping` | No | `{}` | Any OpenAI `/chat/completions` request fields | Dictionary of additional generation parameters merged directly into the `/chat/completions` JSON request body. Common options include:<br>• `temperature` (`float`, e.g., `0.7`)<br>• `max_tokens` / `max_completion_tokens` (`int`, e.g., `256`)<br>• `top_p` (`float`, e.g., `0.95`)<br>• `reasoning_effort` (`"low"`, `"medium"`, `"high"`)<br>• `seed` (`int`)<br>• `stop` (`list[str]` or `str`)<br>*(Note: `model`, `messages`, `stream`, and `stream_options` are controlled by `llmendpoint-perf` and cannot be overridden via `generation_params`.)* |

### `evaluation.pricing` Sub-Options

Configures token pricing in **USD per 1,000,000 (1M) tokens** to compute per-request cost (`cost_usd`) and run-level cost efficiency (`total_cost_usd`, `mean_cost_per_request_usd`, `cost_per_1k_output_tokens_usd`, `output_tps_per_usd_per_hour`):

$$\text{cost\_usd} = \frac{\max(\text{input\_tokens} - \text{cached\_input\_tokens}, 0)}{10^6} \times P_{\text{in}} + \frac{\text{cached\_input\_tokens}}{10^6} \times P_{\text{cached}} + \frac{\text{output\_tokens}}{10^6} \times P_{\text{out}}$$

| Option | Type | Required | Default | Constraints / Possible Values | Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `pricing.input_per_1m_tokens` | `float` | No | `0.0` | `>= 0.0` | Price in USD per 1,000,000 uncached input/prompt tokens. |
| `pricing.output_per_1m_tokens` | `float` | No | `0.0` | `>= 0.0` | Price in USD per 1,000,000 generated output/completion tokens. |
| `pricing.cached_input_per_1m_tokens` | `float` | No | `0.0` | `>= 0.0` | Price in USD per 1,000,000 cached prompt tokens (`usage.prompt_tokens_details.cached_tokens`). |

### `evaluation.slo` Sub-Options

Configures optional Service Level Objective (SLO) latency bounds. A request is counted toward `slo_satisfied_requests` and **SLO Goodput** (`slo_goodput_rps`) if it succeeds (`HTTP 200`) and satisfies **all** non-null thresholds below:

| Option | Type | Required | Default | Constraints / Possible Values | Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `slo.max_ttft_ms` | `float \| null` | No | `null` | `null` or float `> 0` | Maximum acceptable Time to First Token (`ttft_ms`) in milliseconds. If `null`, TTFT is not constrained by the SLO. |
| `slo.max_tpot_ms` | `float \| null` | No | `null` | `null` or float `> 0` | Maximum acceptable Time Per Output Token (`tpot_ms`) in milliseconds per token. If `null`, TPOT is not constrained by the SLO. |
| `slo.max_e2e_ms` | `float \| null` | No | `null` | `null` or float `> 0` | Maximum acceptable End-to-End latency (`e2e_latency_ms`) in milliseconds. If `null`, E2E latency is not constrained by the SLO. |

---

## 3. Overriding Options at Runtime (`--config-override` / `-o`)

Any configuration option documented above can be overridden for a single benchmark run without modifying the task's `config.yaml` using `--config-override` (or `-o`) with dot-separated paths:

```bash
llmendpoint-perf run retail-bench \
  -o evaluation.num_threads=25 \
  -o evaluation.wait_time_between_requests_ms=10 \
  -o evaluation.generation_params.temperature=0.2 \
  -o evaluation.slo.max_ttft_ms=800
```

Values passed via `-o` are parsed as YAML literals (numbers, booleans, `null`, lists, or strings), validated against the schema, and saved into that run's snapshot at `runs/<run-id>/config.yaml`.
