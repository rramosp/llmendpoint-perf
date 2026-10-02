"""Unit and integration tests for synthetic text and multimodal dataset generation."""

from __future__ import annotations

import json
from pathlib import Path

from llmendpoint_perf.config import TaskConfig
from llmendpoint_perf.dataset import detect_token_range, generate_dataset_for_task
from llmendpoint_perf.storage import TaskStorage
from tests.mock_openai_server import MockOpenAIServer


def test_detect_token_range() -> None:
    prompt = "questions about retail products with 20 to 500 input tokens, generating ~100 output tokens"
    assert detect_token_range(prompt) == (20, 500)
    assert detect_token_range("no token range here") is None


def test_generate_text_dataset(tmp_path: Path) -> None:
    with MockOpenAIServer() as server:
        yaml_cfg = f"""
dataset:
  generation_prompt: "questions about retail products with 20 to 500 input tokens, generating ~100 output tokens"
  num_items: 8
  generation_model_endpoint: "{server.base_url}"
  generation_model: "gemini-2.5-pro"
  num_threads: 4

evaluation:
  model_endpoint: "{server.base_url}"
  model: "gemini-2.5-pro"
  num_threads: 2
  wait_time_between_requests_ms: 5
  run_time_secs: 1
"""
        storage = TaskStorage("text-ds-task", base_path=str(tmp_path))
        storage.write_text("config.yaml", yaml_cfg)

        records = generate_dataset_for_task(storage, config=TaskConfig.from_yaml(yaml_cfg))
        assert len(records) == 8
        assert storage.exists("prompts.jsonl")
        assert storage.exists("dataset-generation.log")

        lines = storage.read_text("prompts.jsonl").strip().splitlines()
        assert len(lines) == 8
        for line in lines:
            obj = json.loads(line)
            assert "messages" in obj
            assert obj["messages"][0]["role"] == "user"
            assert isinstance(obj["messages"][0]["content"], str)
            assert len(obj["messages"][0]["content"]) > 0


def test_generate_multimodal_dataset(tmp_path: Path) -> None:
    with MockOpenAIServer() as server:
        yaml_cfg = f"""
dataset:
  generation_prompt: "multimodal questions with an image about electronics products"
  num_items: 4
  generation_model_endpoint: "{server.base_url}"
  generation_model: "gemini-2.5-pro"
  num_threads: 2
  multimodal:
    enabled: true
    image_source: "synthetic"
    image_width: 128
    image_height: 128
    image_format: "jpeg"

evaluation:
  model_endpoint: "{server.base_url}"
  model: "gemini-2.5-pro"
  num_threads: 2
  wait_time_between_requests_ms: 5
  run_time_secs: 1
"""
        storage = TaskStorage("mm-ds-task", base_path=str(tmp_path))
        storage.write_text("config.yaml", yaml_cfg)

        records = generate_dataset_for_task(storage)
        assert len(records) == 4
        lines = storage.read_text("prompts.jsonl").strip().splitlines()
        assert len(lines) == 4
        for line in lines:
            obj = json.loads(line)
            content = obj["messages"][0]["content"]
            assert isinstance(content, list)
            assert len(content) == 2
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"
            url = content[1]["image_url"]["url"]
            assert url.startswith("data:image/jpeg;base64,")
            assert len(url) > 100
