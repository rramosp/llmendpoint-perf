"""Configuration models and YAML parsing/serialization for llmendpoint-perf."""

from __future__ import annotations

import copy
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml


DEFAULT_CONFIG_YAML = """dataset:
  generation_prompt: "questions about retail products with 20 to 500 input tokens, generating ~100 output tokens"
  num_items: 200
  generation_model_endpoint: "http://google.com/api/openai/v1"
  generation_model: "gemini-2.5-pro"
  # Optional dataset generation settings:
  api_key_env: "OPENAI_API_KEY"
  num_threads: 5
  temperature: 1.0
  multimodal:
    enabled: false
    image_source: "synthetic"
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
  api_key_env: "OPENAI_API_KEY"
  max_requests: null
  warmup_requests: 0
  request_timeout_secs: 120
  sampling_strategy: "round_robin"
  generation_params:
    temperature: 0.7
    max_tokens: 256
  pricing:
    input_per_1m_tokens: 1.25
    output_per_1m_tokens: 10.00
    cached_input_per_1m_tokens: 0.3125
  slo:
    max_ttft_ms: 1000
    max_tpot_ms: 50
    max_e2e_ms: 5000
"""


class MultimodalConfig(BaseModel):
    """Configuration for multimodal (text + image) dataset generation."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    image_source: str = "synthetic"
    image_width: int = Field(default=512, ge=16, le=4096)
    image_height: int = Field(default=512, ge=16, le=4096)
    image_format: Literal["jpeg", "png"] = "jpeg"


class DatasetConfig(BaseModel):
    """Configuration for synthetic dataset generation."""

    model_config = ConfigDict(extra="allow")

    generation_prompt: str
    num_items: int = Field(ge=1)
    generation_model_endpoint: str
    generation_model: str
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    num_threads: int = Field(default=5, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    request_timeout_secs: float = Field(default=120.0, gt=0.0)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)

    def resolve_api_key(self) -> str:
        """Resolve the API key from explicit config or environment variables."""
        if self.api_key:
            return self.api_key
        for env_name in (self.api_key_env, "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            val = os.environ.get(env_name)
            if val:
                return val
        return "EMPTY"


class PricingConfig(BaseModel):
    """Optional token pricing configuration (USD per 1M tokens)."""

    model_config = ConfigDict(extra="allow")

    input_per_1m_tokens: float = Field(default=0.0, ge=0.0)
    output_per_1m_tokens: float = Field(default=0.0, ge=0.0)
    cached_input_per_1m_tokens: float = Field(default=0.0, ge=0.0)

    def compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """Calculate the total USD cost for a request."""
        billable_uncached_input = max(input_tokens - cached_input_tokens, 0)
        cost = (
            (billable_uncached_input / 1_000_000.0) * self.input_per_1m_tokens
            + (cached_input_tokens / 1_000_000.0) * self.cached_input_per_1m_tokens
            + (output_tokens / 1_000_000.0) * self.output_per_1m_tokens
        )
        return round(cost, 8)


class SLOConfig(BaseModel):
    """Optional Service Level Objective (SLO) thresholds for goodput calculation."""

    model_config = ConfigDict(extra="allow")

    max_ttft_ms: float | None = None
    max_tpot_ms: float | None = None
    max_e2e_ms: float | None = None

    def is_satisfied(
        self,
        ttft_ms: float | None,
        tpot_ms: float | None,
        e2e_latency_ms: float | None,
    ) -> bool:
        """Return True if the request latencies satisfy all configured SLO bounds."""
        if self.max_ttft_ms is not None and (ttft_ms is None or ttft_ms > self.max_ttft_ms):
            return False
        if self.max_tpot_ms is not None and (tpot_ms is None or tpot_ms > self.max_tpot_ms):
            return False
        if self.max_e2e_ms is not None and (
            e2e_latency_ms is None or e2e_latency_ms > self.max_e2e_ms
        ):
            return False
        return True


class EvaluationConfig(BaseModel):
    """Configuration for endpoint performance evaluation runs."""

    model_config = ConfigDict(extra="allow")

    model_endpoint: str
    model: str
    num_threads: int = Field(default=10, ge=1)
    wait_time_between_requests_ms: float = Field(default=20.0, ge=0.0)
    run_time_secs: float = Field(default=600.0, gt=0.0)
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    max_requests: int | None = Field(default=None, ge=1)
    warmup_requests: int = Field(default=0, ge=0)
    request_timeout_secs: float = Field(default=120.0, gt=0.0)
    sampling_strategy: Literal["round_robin", "random"] = "round_robin"
    generation_params: dict[str, Any] = Field(default_factory=dict)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    slo: SLOConfig = Field(default_factory=SLOConfig)

    @field_validator("pricing", mode="before")
    @classmethod
    def _none_to_default_pricing(cls, v: Any) -> Any:
        return PricingConfig() if v is None else v

    @field_validator("slo", mode="before")
    @classmethod
    def _none_to_default_slo(cls, v: Any) -> Any:
        return SLOConfig() if v is None else v

    @field_validator("generation_params", mode="before")
    @classmethod
    def _none_to_default_params(cls, v: Any) -> Any:
        return {} if v is None else v

    def resolve_api_key(self) -> str:
        """Resolve the API key from explicit config or environment variables."""
        if self.api_key:
            return self.api_key
        for env_name in (self.api_key_env, "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            val = os.environ.get(env_name)
            if val:
                return val
        return "EMPTY"


class TaskConfig(BaseModel):
    """Root configuration model corresponding to config.yaml."""

    model_config = ConfigDict(extra="allow")

    dataset: DatasetConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_yaml(cls, yaml_content: str) -> TaskConfig:
        """Parse and validate a TaskConfig from a YAML string."""
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            raise ValueError("Configuration YAML must deserialize to a mapping/dictionary.")
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialize the configuration back to a clean YAML string."""
        data = self.model_dump(exclude_none=False, mode="json")
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)

    def apply_overrides(self, overrides: list[str] | tuple[str, ...]) -> TaskConfig:
        """Return a new TaskConfig with dot-notation `key.subkey=value` overrides applied."""
        if not overrides:
            return self
        data = copy.deepcopy(self.model_dump(mode="python"))
        for item in overrides:
            if "=" not in item:
                raise ValueError(
                    f"Invalid override '{item}'. Expected format: section.key=value"
                )
            key_path, raw_val = item.split("=", 1)
            keys = [k.strip() for k in key_path.split(".") if k.strip()]
            if not keys:
                raise ValueError(f"Empty key path in override '{item}'.")
            parsed_val = yaml.safe_load(raw_val)
            cursor: dict[str, Any] = data
            for k in keys[:-1]:
                if k not in cursor or not isinstance(cursor[k], dict):
                    cursor[k] = {}
                cursor = cursor[k]
            cursor[keys[-1]] = parsed_val
        return TaskConfig.model_validate(data)
