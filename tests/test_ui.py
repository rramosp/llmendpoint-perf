"""Tests for the llmendpoint-perf web UI server and JSON API endpoints."""

from __future__ import annotations

from pathlib import Path
import threading

from click.testing import CliRunner
import httpx
import pytest

from llmendpoint_perf.cli import cli
from llmendpoint_perf.ui import create_ui_server
from tests.mock_openai_server import MockOpenAIServer


def test_ui_exits_when_basepath_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMENDPOINTPERF_BASEPATH", raising=False)
    monkeypatch.delenv("LLMENDPOINTPERF_GCS_BASEPATH", raising=False)
    runner = CliRunner()
    res = runner.invoke(cli, ["ui"])
    assert res.exit_code != 0
    assert "LLMENDPOINTPERF_BASEPATH is not set" in res.output


def test_ui_server_and_endpoints_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLMENDPOINTPERF_BASEPATH", str(tmp_path))
    cli_runner = CliRunner()

    with MockOpenAIServer(ttft_delay_secs=0.002, tpot_delay_secs=0.001) as mock_api:
        config_file = tmp_path / "mm_config.yaml"
        config_file.write_text(
            f"""dataset:
  generation_prompt: "multimodal questions about retail products with 20 to 200 input tokens"
  num_items: 3
  generation_model_endpoint: "{mock_api.base_url}"
  generation_model: "gemini-2.5-pro"
  num_threads: 2
  multimodal:
    enabled: true
    image_source: "synthetic"
    image_width: 64
    image_height: 64
    image_format: "jpeg"

evaluation:
  model_endpoint: "{mock_api.base_url}"
  model: "gemini-2.5-pro"
  num_threads: 2
  wait_time_between_requests_ms: 5
  run_time_secs: 0.25
  max_requests: 3
""",
            encoding="utf-8",
        )

        assert (
            cli_runner.invoke(
                cli, ["init", "mm-task", "--config", str(config_file)]
            ).exit_code
            == 0
        )
        assert cli_runner.invoke(cli, ["generate_dataset", "mm-task"]).exit_code == 0
        assert (
            cli_runner.invoke(
                cli, ["run", "mm-task", "--run-id", "20261002-120000"]
            ).exit_code
            == 0
        )

    server = create_ui_server(host="127.0.0.1", port=0, base_path=str(tmp_path))
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    try:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            # 1. HTML Root
            resp_html = client.get("/")
            assert resp_html.status_code == 200
            assert "llmendpoint-perf Inspector" in resp_html.text
            assert 'id="task-selector"' in resp_html.text
            assert 'id="tab-runs"' in resp_html.text
            assert 'id="tab-compare"' in resp_html.text
            assert 'id="tab-configs"' in resp_html.text
            assert 'id="tab-dataset"' in resp_html.text
            assert 'id="tab-inference"' in resp_html.text
            assert 'id="tab-btn-metrics-help"' in resp_html.text
            assert 'id="tab-metrics-help"' in resp_html.text
            assert "Streaming Measurement Timeline" in resp_html.text

            # 2. Tasks list
            resp_tasks = client.get("/api/tasks")
            assert resp_tasks.status_code == 200
            tasks_json = resp_tasks.json()
            assert tasks_json["tasks"] == ["mm-task"]

            # 3. Runs list & inspect report
            resp_runs = client.get("/api/tasks/mm-task/runs")
            assert resp_runs.status_code == 200
            runs_json = resp_runs.json()
            assert len(runs_json["runs"]) == 1
            assert runs_json["runs"][0]["run_id"] == "20261002-120000"
            assert "DATASET INFORMATION" in runs_json["runs"][0]["inspect_report"]

            # 4. Compare report
            resp_cmp = client.get("/api/tasks/mm-task/compare")
            assert resp_cmp.status_code == 200
            cmp_json = resp_cmp.json()
            assert len(cmp_json["runs"]) == 1
            assert "Input Tokens Mean" in cmp_json["compare_report"]
            assert "Output Tokens Mean" in cmp_json["compare_report"]

            # 5. Config snapshot
            resp_cfg = client.get("/api/tasks/mm-task/runs/20261002-120000/config")
            assert resp_cfg.status_code == 200
            assert "multimodal" in resp_cfg.json()["config_yaml"]

            # 6. Dataset index and multimodal item detail
            resp_ds = client.get("/api/tasks/mm-task/dataset")
            assert resp_ds.status_code == 200
            ds_json = resp_ds.json()
            assert ds_json["total_items"] == 3
            assert ds_json["items"][0]["has_images"] is True
            assert ds_json["items"][0]["image_count"] == 1

            resp_ds_item = client.get("/api/tasks/mm-task/dataset/0")
            assert resp_ds_item.status_code == 200
            item_obj = resp_ds_item.json()["item"]
            content_parts = item_obj["messages"][0]["content"]
            assert isinstance(content_parts, list)
            assert any(
                p.get("type") == "image_url"
                and p.get("image_url", {}).get("url", "").startswith("data:image/jpeg;base64,")
                for p in content_parts
            )

            # 7. Inferences list and detail
            resp_infs = client.get(
                "/api/tasks/mm-task/runs/20261002-120000/inferences"
            )
            assert resp_infs.status_code == 200
            infs_json = resp_infs.json()
            assert infs_json["total_inferences"] == 3

            resp_inf_detail = client.get(
                "/api/tasks/mm-task/runs/20261002-120000/inferences/0"
            )
            assert resp_inf_detail.status_code == 200
            detail_json = resp_inf_detail.json()
            assert detail_json["inference"]["status_code"] == 200
            assert detail_json["input_prompt"] is not None
            assert "messages" in detail_json["input_prompt"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
