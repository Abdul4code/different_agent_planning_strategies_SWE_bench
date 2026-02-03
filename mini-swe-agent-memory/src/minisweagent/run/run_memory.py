#!/usr/bin/env python3
"""Run memory agent on SWE-bench tasks.

This is a standalone agent folder. The baseline mini-swe-agent is left untouched.

Example:
    cd mini-swe-agent-memory
    python -m minisweagent.run.run_memory --tasks ../tasks_50.json --out results/memory --run-id memory_eval_001
"""

from pathlib import Path

import typer
import yaml

from minisweagent.agents.memory import MemoryAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.run.run_utils import load_tasks_from_json, run_agent_batch

app = typer.Typer()


@app.command()
def main(
    tasks: str = typer.Option("tasks_50.json", "--tasks", help="Path to tasks JSON file"),
    output: str = typer.Option("results/memory", "-o", "--out", help="Output directory"),
    run_id: str = typer.Option("memory_eval_001", "--run-id", help="Run ID for metrics"),
    model: str = typer.Option("local", "-m", "--model", help="Model to use"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of parallel workers"),
    config: str = typer.Option(
        None, "-c", "--config", help="Config file (default: swebench.yaml)"
    ),
    limit: int = typer.Option(None, "--limit", help="Limit number of tasks"),
    memory_file: str = typer.Option("memory.json", "--memory-file", help="Memory file path"),
    num_examples: int = typer.Option(2, "--num-examples", help="Number of examples to retrieve"),
):
    """Run memory agent on SWE-bench tasks."""
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
    config_data.setdefault("memory_file", memory_file)
    config_data.setdefault("num_examples", num_examples)

    # Run with agent
    run_agent_batch(
        MemoryAgent,
        instances,
        Path(output),
        config_data,
        workers=workers,
        run_id=run_id,
        agent_name="memory",
    )


if __name__ == "__main__":
    app()
