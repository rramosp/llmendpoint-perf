"""Unit and integration tests for synthetic text and multimodal dataset generation."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from PIL import Image
import pytest

from llmendpoint_perf.config import TaskConfig
from llmendpoint_perf.dataset import (
    detect_image_count_range,
    detect_token_range,
    extract_image_urls_from_search_html,
    generate_dataset_for_task,
    parse_aligned_multimodal_output,
    parse_aligned_multimodal_queries,
    resolve_multimodal_image_source,
)
from llmendpoint_perf.storage import TaskStorage
from tests.mock_openai_server import MockOpenAIServer


def test_detect_token_range() -> None:
    prompt = "questions about retail products with 20 to 500 input tokens, generating ~100 output tokens"
    assert detect_token_range(prompt) == (20, 500)
    assert detect_token_range("no token range here") is None


def test_detect_image_count_range() -> None:
    mm_example = (
        "visual questions about some sports retail product catalog images with 30 to 300 "
        "input tokens and between 1 and 3 images each prompt, generating ~120 output tokens."
    )
    assert detect_image_count_range(mm_example) == (1, 3)
    assert detect_image_count_range("compare 2 to 4 attached images of shoes") == (2, 4)
    assert detect_image_count_range("questions with 3 product images per prompt") == (3, 3)
    assert detect_image_count_range("questions with an image") is None


def test_resolve_multimodal_image_source() -> None:
    # 1. Default when user is not precise about how to generate images -> google_search
    cfg_imprecise = TaskConfig.from_yaml(
        """
dataset:
  generation_prompt: "visual questions about electronics retail product catalog images"
  num_items: 5
  generation_model_endpoint: "http://localhost:8000/v1"
  generation_model: "gemini-2.5-pro"
  multimodal:
    enabled: true
evaluation:
  model_endpoint: "http://localhost:8000/v1"
  model: "gemini-2.5-pro"
"""
    )
    assert resolve_multimodal_image_source(cfg_imprecise.dataset) == "google_search"

    # 2. Explicit synthetic image_source without Google Search in prompt -> synthetic
    cfg_synthetic = cfg_imprecise.apply_overrides(["dataset.multimodal.image_source=synthetic"])
    assert resolve_multimodal_image_source(cfg_synthetic.dataset) == "synthetic"

    # 3. Prompt explicitly mentions Google Search even if image_source was synthetic -> google_search
    cfg_prompt_search = TaskConfig.from_yaml(
        """
dataset:
  generation_prompt: "visual questions about electronics retail product catalog images. Grab the images from random queries related to the topic on Google Search"
  num_items: 5
  generation_model_endpoint: "http://localhost:8000/v1"
  generation_model: "gemini-2.5-pro"
  multimodal:
    enabled: true
    image_source: "synthetic"
evaluation:
  model_endpoint: "http://localhost:8000/v1"
  model: "gemini-2.5-pro"
"""
    )
    assert resolve_multimodal_image_source(cfg_prompt_search.dataset) == "google_search"

    # 4. External directory or GCS URI -> external
    cfg_ext = cfg_imprecise.apply_overrides(["dataset.multimodal.image_source=gs://my-bucket/imgs"])
    assert resolve_multimodal_image_source(cfg_ext.dataset) == "external"


def test_parse_aligned_multimodal_output_and_html_extraction() -> None:
    raw_delimited = (
        "What type of earcup padding is shown on these headphones?\n"
        "---SEARCH_TERMS---\n"
        "sony wh-1000xm5 headphones product photo"
    )
    prompt_text, query = parse_aligned_multimodal_output(
        raw_delimited, fallback_topic="electronics retail products"
    )
    assert prompt_text == "What type of earcup padding is shown on these headphones?"
    assert query == "sony wh-1000xm5 headphones product photo"

    # Multi-image search query extraction
    raw_multi = (
        "Compare the outsole lug geometry between Image 1 and Image 2.\n"
        "---SEARCH_TERMS---\n"
        "Image 1: trail running shoe Vibram outsole lug photo\n"
        "2. road marathon carbon racing shoe outsole"
    )
    multi_prompt, multi_queries = parse_aligned_multimodal_queries(
        raw_multi, fallback_topic="sports retail products", num_images=2
    )
    assert multi_prompt == "Compare the outsole lug geometry between Image 1 and Image 2."
    assert multi_queries == [
        "trail running shoe Vibram outsole lug photo",
        "road marathon carbon racing shoe outsole",
    ]

    # Fallback when model outputs plain text without delimiter
    plain_output = "How many HDMI ports are visible on the rear panel of this 4K gaming monitor?"
    p_fallback, q_fallback = parse_aligned_multimodal_output(
        plain_output, fallback_topic="electronics retail products"
    )
    assert p_fallback == plain_output
    assert len(q_fallback) > 0

    sample_html = (
        '<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTest123"/>'
        '{"ou":"https://example.com/catalog/camera.jpg"}'
        '<img src="http://www.bing.com/sa/simg/facebook_sharing_5.png"/>'
    )
    urls = extract_image_urls_from_search_html(sample_html)
    assert "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTest123" in urls
    assert "https://example.com/catalog/camera.jpg" in urls
    assert "http://www.bing.com/sa/simg/facebook_sharing_5.png" not in urls


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


def test_generate_multimodal_dataset_synthetic(tmp_path: Path) -> None:
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


def test_generate_multimodal_dataset_google_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_queries: list[str] = []

    def _fake_search_and_fetch(query: str, item_index: int = 0, **kwargs: Any) -> bytes:
        del kwargs
        captured_queries.append(query)
        img = Image.new("RGB", (64, 64), color=((40 + item_index * 20) % 255, 100, 160))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    monkeypatch.setattr(
        "llmendpoint_perf.dataset.search_and_fetch_google_image",
        _fake_search_and_fetch,
    )

    with MockOpenAIServer() as server:
        # Notice image_source is not specified (imprecise image generation instructions)
        # and prompt requests between 1 and 3 images each prompt
        yaml_cfg = f"""
dataset:
  generation_prompt: "visual questions about some electronics retail product catalog images with 30 to 300 input tokens and between 1 and 3 images each prompt, generating ~120 output tokens."
  num_items: 15
  generation_model_endpoint: "{server.base_url}"
  generation_model: "gemini-2.5-pro"
  num_threads: 3
  multimodal:
    enabled: true
    image_width: 96
    image_height: 96
    image_format: "jpeg"

evaluation:
  model_endpoint: "{server.base_url}"
  model: "gemini-2.5-pro"
  num_threads: 2
  wait_time_between_requests_ms: 5
  run_time_secs: 1
"""
        storage = TaskStorage("mm-google-search-task", base_path=str(tmp_path))
        storage.write_text("config.yaml", yaml_cfg)

        records = generate_dataset_for_task(storage)
        assert len(records) == 15
        for q in captured_queries:
            assert "electronics retail product catalog" in q

        lines = storage.read_text("prompts.jsonl").strip().splitlines()
        assert len(lines) == 15
        image_counts: list[int] = []
        for line in lines:
            obj = json.loads(line)
            content = obj["messages"][0]["content"]
            assert isinstance(content, list)
            assert content[0]["type"] == "text"
            assert "Synthetic visual question" in content[0]["text"]
            img_parts = [p for p in content if p.get("type") == "image_url"]
            assert 1 <= len(img_parts) <= 3
            image_counts.append(len(img_parts))
            for p in img_parts:
                assert p["image_url"]["url"].startswith("data:image/jpeg;base64,")

        assert sum(image_counts) == len(captured_queries)
        assert max(image_counts) > 1

