# LLM Endpoint Performance Measurement


## 1. Overview and objectives

`llmendpoint-perf` is a tool to measure the technical performance of OpenAI compatible LLM endpoints. It is intended to be used by developers and researchers to compare the performance of different LLM endpoints and to identify potential performance bottlenecks.

The technical performance metrics include:
* End to end latency (time to first token + time per output token)
* Throughput (requests per second, tokens per second)
* Cost performance (cost per request, cost per token)

`llmendpoint-perf` will also allow to generate synthetic datasets that will be used to evaluate the performance of the endpoints. 

## 2. Evaluation tasks

`llmendpoint-perf` will allow to define, execute and inspect the outcomes of different evaluation tasks. It will use a base GCS path to store all all artifacts of all evaluation tasks, that will be defined in an environment variable named 'LLMENDPOINTPERF_GCS_BASEPATH' 

Each evaluation task will be denoted by a name and will store all its artifacts in GCS under $LLMENDPOINTPERF_GCS_BASEPATH/task-name. These artifacts will include:

- a config.yaml file with all the configuration options set by the user.
- a prompts.jsonl file with the prompts to use for the evaluation.
- YYYYMMDD-HHSS-run-results.jsonl files with the results of the performance evaluation.
- YYYYMMDD-HHSS-run-calls.jsonl files with the return values of all calls made during the evaluation.
- YYYYMMDD-HHSS-run-config.yaml files with a copy of the config.yaml used for the evaluation.
- YYYYMMDD-HHSS-run-log.txt files with the evaluation log.
- one or more log files with the logs of the activities performed within the task.

## 3. Synthetic dataset generation

Syntethic evaluation datasets will be generated using a dedicated LLM endpoint service by following a prompt and parameters defined in a section named 'dataset' in the evaluation task config file, with the following structure. See the config file section below for the structure

with this `llmendpoint-perf` will wrap the user generation prompt within a prompt and a system prompt that will generate one dataset item per call. Make sure the system prompt and wrapping prompt instruct the model to generate each dataset item randomly enough so that subsequent generation calls generate sufficiently different items.

The generated dataset will be stored in $LLMENDPOINTPERF_GCS_BASEPATH/task-name/prompts.jsonl, in a format readily usable by an OpenAI compatible API, such as the following:

```json
{"messages": [{"role": "user", "content": "question"}]}
{"messages": [{"role": "user", "content": "question"}]}
```

Also, take into account that the user might request creating datasets with multimodal evaluation prompts. In this case, the wrapping prompt must contain the appropriate instructions so that dataset item generation includes what the user requests. The prompts.jsonl file will include the media as base64 encoded data uris for the image field, e.g.:

```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "question"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...."}}]}]}
```

## 4. Command line interface

`llmendpoint-perf` will provide a command line interface to its functionality like this:

```bash
llmendpoint-perf <command> [options] 
```

commands avaiable:

- `generate_dataset`: generates a dataset for an evaluation task as described above.
- `run`: runs an evaluation task and stores the results, config, call returns and logs in the files described above. The log file and stdout will have the same content for each run.

## 5. Configuration file

dataset:
  generation_prompt: "questions about retail products with 20 to 500 input tokens, generating ~100 output tokens"
  num_items: 200
  generation_model_endpoint: "http://google.com/api/openai/v1"
  generation_model: "gemini-2.5-pro"

evaluation:
  model_endpoint: "http://google.com/api/openai/v1"
  model: "gemini-2.5-pro"
  num_threads: 10
  wait_time_between_requests_ms: 20
  run_time_secs: 600



