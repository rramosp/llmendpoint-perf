"""Unit tests for configuration parsing, defaults, overrides, pricing, and SLO validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmendpoint_perf.config import DEFAULT_CONFIG_YAML, TaskConfig


MINIMAL_YAML = """
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
"""


def test_minimal_config_parses_with_defaults() -> None:
    cfg = TaskConfig.from_yaml(MINIMAL_YAML)
    assert cfg.dataset.num_items == 200
    assert cfg.dataset.generation_model == "gemini-2.5-pro"
    assert cfg.dataset.multimodal.enabled is False
    assert cfg.evaluation.num_threads == 10
    assert cfg.evaluation.wait_time_between_requests_ms == 20
    assert cfg.evaluation.run_time_secs == 600
    assert cfg.evaluation.sampling_strategy == "round_robin"


def test_default_config_template_roundtrip() -> None:
    cfg = TaskConfig.from_yaml(DEFAULT_CONFIG_YAML)
    dumped = cfg.to_yaml()
    cfg2 = TaskConfig.from_yaml(dumped)
    assert cfg2.evaluation.model == cfg.evaluation.model
    assert cfg2.evaluation.pricing.input_per_1m_tokens == pytest.approx(1.25)
    assert cfg2.evaluation.pricing.output_per_1m_tokens == pytest.approx(10.0)


def test_config_overrides() -> None:
    cfg = TaskConfig.from_yaml(MINIMAL_YAML)
    updated = cfg.apply_overrides(
        [
            "evaluation.num_threads=25",
            "evaluation.run_time_secs=30.5",
            "dataset.multimodal.enabled=true",
            "evaluation.pricing.input_per_1m_tokens=2.5",
        ]
    )
    assert updated.evaluation.num_threads == 25
    assert updated.evaluation.run_time_secs == pytest.approx(30.5)
    assert updated.dataset.multimodal.enabled is True
    assert updated.evaluation.pricing.input_per_1m_tokens == pytest.approx(2.5)
    # Original remains unchanged
    assert cfg.evaluation.num_threads == 10


def test_pricing_and_slo_calculations() -> None:
    cfg = TaskConfig.from_yaml(DEFAULT_CONFIG_YAML)
    cost = cfg.evaluation.pricing.compute_cost(
        input_tokens=1000, output_tokens=500, cached_input_tokens=200
    )
    # 800 uncached * 1.25/1M + 200 cached * 0.3125/1M + 500 out * 10.0/1M
    expected = (800 * 1.25 + 200 * 0.3125 + 500 * 10.0) / 1_000_000.0
    assert cost == pytest.approx(expected)

    slo = cfg.evaluation.slo
    assert slo.is_satisfied(ttft_ms=400.0, tpot_ms=25.0, e2e_latency_ms=2000.0) is True
    assert slo.is_satisfied(ttft_ms=1200.0, tpot_ms=25.0, e2e_latency_ms=2000.0) is False
    assert slo.is_satisfied(ttft_ms=400.0, tpot_ms=75.0, e2e_latency_ms=2000.0) is False


def test_example_configs_are_valid() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    text_cfg_path = repo_root / "examples" / "config_text.yaml"
    mm_cfg_path = repo_root / "examples" / "config_multimodal.yaml"

    assert text_cfg_path.exists()
    assert mm_cfg_path.exists()

    text_cfg = TaskConfig.from_yaml(text_cfg_path.read_text(encoding="utf-8"))
    assert text_cfg.dataset.multimodal.enabled is False
    assert text_cfg.dataset.num_items == 200

    mm_cfg = TaskConfig.from_yaml(mm_cfg_path.read_text(encoding="utf-8"))
    assert mm_cfg.dataset.multimodal.enabled is True
    assert mm_cfg.dataset.multimodal.image_source == "google_search"
    assert mm_cfg.dataset.multimodal.image_width == 512
    assert mm_cfg.dataset.multimodal.image_height == 512

