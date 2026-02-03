#!/usr/bin/env python3
"""Run reflection agent on SWE-bench tasks.

This is a standalone agent folder. The baseline mini-swe-agent is left untouched.

Example:
    cd mini-swe-agent-reflection
    python -m minisweagent.run.run_reflect --tasks ../tasks_50.json --out results/reflect --run-id reflect_eval_001
"""

from pathlib import Path

import typer
import yaml

from minisweagent.agents.reflection import ReflectionAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.run.run_utils import load_tasks_from_json, run_agent_batch

app = typer.Typer()


@app.command()
def main(
    tasks: str = typer.Option("tasks_50.json", "--tasks", help="Path to tasks JSON file"),
    output: str = typer.Option("results/reflect", "-o", "--out", help="Output directory"),
    run_id: str = typer.Option("reflect_eval_001", "--run-id", help="Run ID for metrics"),
    model: str = typer.Option("local", "-m", "--model", help="Model to use"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of parallel workers"),
    config: str = typer.Option(
        None, "-c", "--config", help="Config file (default: swebench.yaml)"
    ),
    limit: int = typer.Option(None, "--limit", help="Limit number of tasks"),
    max_reflections: int = typer.Option(3, "--max-reflections", help="Max reflection iterations"),
):
    """Run reflection agent on SWE-bench tasks."""
    # Load tasks
    print(f"Loading tasks from {tasks}...")
    instances = load_tasks_from_json(tasks)
    if limit:
        instances = instances[:limit]
    print(f"Loaded {len(instances)} tasks")

    # Load config
    config_path = config or str(builtin_config_dir / "extra" / "swebench.yaml")
    config_file = get_config_path(Path(config_path))
    config_data = yaml.safe_load(config_file.read_text())
    if model:
        config_data.setdefault("model", {})["model_name"] = model
    config_data.setdefault("max_reflections", max_reflections)

    # Run with agent
    run_agent_batch(
        ReflectionAgent,
        instances,
        Path(output),
        config_data,
        workers=workers,
        run_id=run_id,
        agent_name="reflection",
    )


if __name__ == "__main__":
    app()
