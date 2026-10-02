"""Command-line interface for llmendpoint-perf."""

from __future__ import annotations

from pathlib import Path
import sys

import click

from llmendpoint_perf import __version__
from llmendpoint_perf.config import TaskConfig
from llmendpoint_perf.dataset import generate_dataset_for_task
from llmendpoint_perf.inspector import (
    format_comparison_report,
    format_run_summary_report,
    load_run_results,
)
from llmendpoint_perf.runner import run_evaluation_task
from llmendpoint_perf.storage import TaskStorage
from llmendpoint_perf.ui import run_ui_server


def _resolve_task_name(positional_task: str | None, option_task: str | None) -> str:
    task_name = positional_task or option_task
    if not task_name:
        raise click.UsageError("Missing required <task-name> argument.")
    return task_name


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="llmendpoint-perf")
def cli() -> None:
    """Measure technical and cost performance of OpenAI-compatible LLM endpoints."""


@cli.command("init")
@click.argument("task_name", required=False)
@click.option("--task", "task_opt", default=None, help="Evaluation task name.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the initial config.yaml file for this evaluation task.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite config.yaml if it already exists.",
)
@click.option(
    "--base-path",
    default=None,
    help="Override $LLMENDPOINTPERF_BASEPATH storage root.",
)
def init_cmd(
    task_name: str | None,
    task_opt: str | None,
    config_path: Path,
    overwrite: bool,
    base_path: str | None,
) -> None:
    """Initialize a new evaluation task with a user-supplied config.yaml."""
    name = _resolve_task_name(task_name, task_opt)
    try:
        storage = TaskStorage(task_name=name, base_path=base_path)
        if storage.exists("config.yaml") and not overwrite:
            raise FileExistsError(
                f"config.yaml already exists at {storage.task_uri}/config.yaml. "
                "Pass --overwrite to replace it."
            )
        yaml_text = config_path.read_text(encoding="utf-8")
        # Validate before saving
        _ = TaskConfig.from_yaml(yaml_text)
        storage.write_text("config.yaml", yaml_text)
        click.echo(f"Initialized task '{name}' at {storage.task_uri}/config.yaml")
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("generate_dataset")
@click.argument("task_name", required=False)
@click.option("--task", "task_opt", default=None, help="Evaluation task name.")
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite prompts.jsonl if it already exists.",
)
@click.option(
    "--base-path",
    default=None,
    help="Override $LLMENDPOINTPERF_BASEPATH storage root.",
)
def generate_dataset_cmd(
    task_name: str | None,
    task_opt: str | None,
    overwrite: bool,
    base_path: str | None,
) -> None:
    """Generate a synthetic dataset (prompts.jsonl) for an evaluation task."""
    name = _resolve_task_name(task_name, task_opt)
    try:
        storage = TaskStorage(task_name=name, base_path=base_path)
        generate_dataset_for_task(storage=storage, overwrite=overwrite)
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("run")
@click.argument("task_name", required=False)
@click.option("--task", "task_opt", default=None, help="Evaluation task name.")
@click.option(
    "--config-override",
    "-o",
    "config_overrides",
    multiple=True,
    help="Override config values using dot notation, e.g., evaluation.num_threads=20",
)
@click.option(
    "--run-id",
    default=None,
    help="Optional custom run ID (defaults to UTC timestamp YYYYMMDD-HHMMSS).",
)
@click.option(
    "--base-path",
    default=None,
    help="Override $LLMENDPOINTPERF_BASEPATH storage root.",
)
def run_cmd(
    task_name: str | None,
    task_opt: str | None,
    config_overrides: tuple[str, ...],
    run_id: str | None,
    base_path: str | None,
) -> None:
    """Run an evaluation task and store results, call telemetry, config, and logs."""
    name = _resolve_task_name(task_name, task_opt)
    try:
        storage = TaskStorage(task_name=name, base_path=base_path)
        run_evaluation_task(
            storage=storage,
            config_overrides=config_overrides,
            run_id=run_id,
        )
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("inspect")
@click.argument("task_name", required=False)
@click.option("--task", "task_opt", default=None, help="Evaluation task name.")
@click.option(
    "--run-id",
    default=None,
    help="Specific run ID (YYYYMMDD-HHMMSS) to inspect. Defaults to latest run.",
)
@click.option(
    "--list-runs",
    is_flag=True,
    default=False,
    help="List all available run IDs for the task.",
)
@click.option(
    "--base-path",
    default=None,
    help="Override $LLMENDPOINTPERF_BASEPATH storage root.",
)
def inspect_cmd(
    task_name: str | None,
    task_opt: str | None,
    run_id: str | None,
    list_runs: bool,
    base_path: str | None,
) -> None:
    """Inspect runs or print the performance summary report for a task."""
    name = _resolve_task_name(task_name, task_opt)
    try:
        storage = TaskStorage(task_name=name, base_path=base_path)
        runs = storage.list_runs()
        if list_runs:
            if not runs:
                click.echo(f"No runs found for task '{name}'.")
                return
            click.echo(f"Available runs for task '{name}' ({len(runs)} total):")
            for r_id in runs:
                click.echo(f"  - {r_id}")
            return

        results = load_run_results(storage=storage, run_id=run_id)
        click.echo(format_run_summary_report(results))
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("compare")
@click.argument("targets", nargs=-1, required=True)
@click.option(
    "--base-path",
    default=None,
    help="Override $LLMENDPOINTPERF_BASEPATH storage root.",
)
def compare_cmd(targets: tuple[str, ...], base_path: str | None) -> None:
    """Compare metrics across multiple tasks or specific runs (<task-name>[:<run-id>])."""
    try:
        loaded = []
        for target in targets:
            if ":" in target:
                t_name, r_id = target.split(":", 1)
            else:
                t_name, r_id = target, None
            storage = TaskStorage(task_name=t_name, base_path=base_path)
            loaded.append(load_run_results(storage=storage, run_id=r_id))
        click.echo(format_comparison_report(loaded))
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("ui")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host address to bind the UI server.",
)
@click.option(
    "--port",
    default=8080,
    show_default=True,
    type=int,
    help="Port to bind the UI server.",
)
@click.option(
    "--base-path",
    default=None,
    help="Override $LLMENDPOINTPERF_BASEPATH storage root.",
)
def ui_cmd(host: str, port: int, base_path: str | None) -> None:
    """Start the web UI server to inspect evaluation tasks, runs, datasets, and inferences."""
    try:
        run_ui_server(host=host, port=port, base_path=base_path)
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


def main() -> None:
    """Entry point for the `llmendpoint-perf` console script."""
    cli()


if __name__ == "__main__":
    main()

