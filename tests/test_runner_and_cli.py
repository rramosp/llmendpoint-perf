"""End-to-end integration tests for the benchmark runner and CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import pytest

from llmendpoint_perf.cli import cli
from tests.mock_openai_server import MockOpenAIServer


def test_end_to_end_cli_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMENDPOINTPERF_BASEPATH", str(tmp_path))
    runner = CliRunner()

    with MockOpenAIServer(ttft_delay_secs=0.005, tpot_delay_secs=0.001) as server:
        config_file = tmp_path / "custom_config.yaml"
        config_file.write_text(
            f"""dataset:
  generation_prompt: "questions about retail products with 20 to 500 input tokens, generating ~100 output tokens"
  num_items: 6
  generation_model_endpoint: "{server.base_url}"
  generation_model: "gemini-2.5-pro"
  num_threads: 3

evaluation:
  model_endpoint: "{server.base_url}"
  model: "gemini-2.5-pro"
  num_threads: 3
  wait_time_between_requests_ms: 10
  run_time_secs: 0.4
  warmup_requests: 2
  pricing:
    input_per_1m_tokens: 1.25
    output_per_1m_tokens: 10.0
    cached_input_per_1m_tokens: 0.3125
""",
            encoding="utf-8",
        )

        # 1. init
        res_init = runner.invoke(cli, ["init", "retail-eval", "--config", str(config_file)])
        assert res_init.exit_code == 0, res_init.output
        assert (tmp_path / "retail-eval" / "config.yaml").exists()

        # 2. generate_dataset
        res_gen = runner.invoke(cli, ["generate_dataset", "retail-eval"])
        assert res_gen.exit_code == 0, res_gen.output
        assert (tmp_path / "retail-eval" / "prompts.jsonl").exists()
        assert (tmp_path / "retail-eval" / "dataset-generation.log").exists()

        # 3. run (first run)
        res_run1 = runner.invoke(
            cli,
            ["run", "retail-eval", "--run-id", "20261001-100000"],
        )
        assert res_run1.exit_code == 0, res_run1.output

        run1_dir = tmp_path / "retail-eval" / "runs" / "20261001-100000"
        assert (run1_dir / "config.yaml").exists()
        assert (run1_dir / "results.jsonl").exists()
        assert (run1_dir / "calls.jsonl").exists()
        assert (run1_dir / "log.txt").exists()

        # Verify stdout and log.txt have identical content
        log_txt_content = (run1_dir / "log.txt").read_text(encoding="utf-8")
        assert log_txt_content == res_run1.output

        # Verify calls.jsonl and results.jsonl content
        calls_lines = (run1_dir / "calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(calls_lines) >= 3
        first_call = json.loads(calls_lines[0])
        assert first_call["status_code"] == 200
        assert first_call["ttft_ms"] > 0
        assert first_call["tpot_ms"] > 0
        assert first_call["input_tokens"] == 64
        assert first_call["output_tokens"] == 7
        assert first_call["reasoning_tokens"] == 2
        assert first_call["cached_input_tokens"] == 16
        assert first_call["cost_usd"] > 0

        results_obj = json.loads((run1_dir / "results.jsonl").read_text(encoding="utf-8"))
        assert results_obj["total_requests"] == len(calls_lines)
        assert results_obj["succeeded_requests"] == len(calls_lines)
        assert results_obj["throughput"]["output_tps"] > 0
        assert results_obj["distributions"]["ttft_ms"]["p50"] > 0

        # 4. run (second run with config override)
        res_run2 = runner.invoke(
            cli,
            [
                "run",
                "retail-eval",
                "--run-id",
                "20261001-100500",
                "--config-override",
                "evaluation.max_requests=4",
            ],
        )
        assert res_run2.exit_code == 0, res_run2.output

        # 5. inspect
        res_list = runner.invoke(cli, ["inspect", "retail-eval", "--list-runs"])
        assert res_list.exit_code == 0
        assert "20261001-100000" in res_list.output
        assert "20261001-100500" in res_list.output

        res_inspect = runner.invoke(
            cli, ["inspect", "retail-eval", "--run-id", "20261001-100500"]
        )
        assert res_inspect.exit_code == 0
        assert "LLM ENDPOINT PERFORMANCE REPORT" in res_inspect.output
        assert "DATASET INFORMATION" in res_inspect.output
        assert "Generation Model               : gemini-2.5-pro" in res_inspect.output
        assert "Number of Items                : 6" in res_inspect.output
        assert (
            "questions about retail products with 20 to 500 input tokens, generating ~100 output tokens"
            in res_inspect.output
        )

        # 6. compare
        res_compare = runner.invoke(
            cli,
            [
                "compare",
                "retail-eval:20261001-100000",
                "retail-eval:20261001-100500",
            ],
        )
        assert res_compare.exit_code == 0
        assert "LLM ENDPOINT PERFORMANCE COMPARISON" in res_compare.output
        assert "DATASET INFORMATION" in res_compare.output
        assert "[retail-eval:20261001-100000]" in res_compare.output
        assert "[retail-eval:20261001-100500]" in res_compare.output
        assert "Generation Model  : gemini-2.5-pro" in res_compare.output
        assert "Number of Items   : 6" in res_compare.output
        assert (
            "questions about retail products with 20 to 500 input tokens, generating ~100 output tokens"
            in res_compare.output
        )
        for token_metric in ("Input Tokens", "Output Tokens"):
            for stat in ("Mean", "p50", "p95", "p99"):
                assert f"{token_metric} {stat}" in res_compare.output


def test_init_requires_config_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMENDPOINTPERF_BASEPATH", str(tmp_path))
    runner = CliRunner()
    res = runner.invoke(cli, ["init", "missing-config-task"])
    assert res.exit_code != 0
    assert "Missing option '--config'" in res.output

