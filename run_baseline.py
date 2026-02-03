#!/usr/bin/env python3
"""Run baseline mini-swe-agent on SWE-bench tasks.

Uses the same CLI interface as the other planning agents.
"""

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import typer
import yaml

# Add baseline mini-swe-agent src to path
baseline_path = Path(__file__).parent / "mini-swe-agent" / "src"
sys.path.insert(0, str(baseline_path))

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.swebench import (
    RunBatchProgressManager,
    get_swebench_docker_image_name,
    update_preds_file,
    remove_from_preds_file,
)
from minisweagent.run.utils.save import save_traj
from rich.live import Live
import concurrent.futures

app = typer.Typer()


def load_tasks_from_json(json_path: str):
    """Load tasks from JSON or JSONL file."""
    tasks = []
    with open(json_path, "r") as f:
        content = f.read().strip()
        try:
            tasks = json.loads(content)
            if not isinstance(tasks, list):
                tasks = [tasks]
        except json.JSONDecodeError:
            for line in content.split("\n"):
                if line.strip():
                    tasks.append(json.loads(line))
    return tasks


@app.command()
def main(
    tasks: str = typer.Option("tasks_50.json", "--tasks", help="Path to tasks JSON file"),
    output: str = typer.Option("results/baseline", "-o", "--out", help="Output directory"),
    run_id: str = typer.Option("baseline_eval_001", "--run-id", help="Run ID for metrics"),
    model: str = typer.Option("local", "-m", "--model", help="Model to use"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of parallel workers"),
    config: str = typer.Option(
        None, "-c", "--config", help="Config file (default: swebench.yaml)"
    ),
    limit: int = typer.Option(None, "--limit", help="Limit number of tasks"),
):
    """Run baseline mini-swe-agent on SWE-bench tasks."""
    # Auto-disable cost tracking for openrouter/local models
    if "MSWEA_COST_TRACKING" not in os.environ:
        os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"

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

    # Run with baseline agent
    run_baseline_batch(
        instances,
        Path(output),
        config_data,
        workers=workers,
        run_id=run_id,
    )


def run_baseline_batch(
    instances: list,
    output_path: Path,
    config_data: dict,
    workers: int = 1,
    run_id: str = "eval_001",
):
    """Run baseline agent batch on instances."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def process_instance(instance, output_dir, cfg, progress_manager):
        """Process single instance with baseline agent."""
        instance_id = instance["instance_id"]
        instance_dir = output_dir / instance_id
        remove_from_preds_file(output_dir / "preds.json", instance_id)
        (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

        model_instance = get_model(config=cfg.get("model", {}))
        task = instance["problem_statement"]

        progress_manager.on_instance_start(instance_id)
        progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

        agent = None
        extra_info = None
        env = None

        try:
            env_config = cfg.setdefault("environment", {})
            env_config["environment_class"] = env_config.get("environment_class", "docker")
            image_name = get_swebench_docker_image_name(instance)
            if env_config["environment_class"] in ["docker", "swerex_modal"]:
                env_config["image"] = image_name

            env = get_environment(env_config)

            # Wrap agent with progress tracking
            class ProgressTrackingAgent(DefaultAgent):
                def __init__(self, *a, _progress_manager=None, _instance_id=None, **kw):
                    super().__init__(*a, **kw)
                    self._progress_manager = _progress_manager
                    self._instance_id = _instance_id

                def step(self) -> dict:
                    if self._progress_manager:
                        self._progress_manager.update_instance_status(
                            self._instance_id,
                            f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})",
                        )
                    return super().step()

            agent = ProgressTrackingAgent(
                model_instance,
                env,
                _progress_manager=progress_manager,
                _instance_id=instance_id,
                **cfg.get("agent", {}),
            )
            exit_status, result = agent.run(task)
        except Exception as e:
            logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
            exit_status, result = type(e).__name__, str(e)
            extra_info = {"traceback": traceback.format_exc()}
        finally:
            if env and hasattr(env, "stop"):
                env.stop()
            save_traj(
                agent,
                instance_dir / f"{instance_id}.traj.json",
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                instance_id=instance_id,
                print_fct=logger.info,
            )
            update_preds_file(
                output_dir / "preds.json", instance_id, model_instance.config.model_name, result
            )
            progress_manager.on_instance_end(instance_id, exit_status)

    # Setup output
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(instances)} instances with baseline DefaultAgent")

    # Process instances
    progress_manager = RunBatchProgressManager(
        len(instances), output_path / f"exit_statuses_{time.time()}.yaml"
    )

    with Live(progress_manager.render_group, refresh_per_second=4):
        if workers == 1:
            for instance in instances:
                process_instance(instance, output_path, config_data, progress_manager)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for instance in instances:
                    future = executor.submit(
                        process_instance, instance, output_path, config_data, progress_manager
                    )
                    futures[future] = instance["instance_id"]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        instance_id = futures[future]
                        logging.error(f"Error processing {instance_id}: {e}", exc_info=True)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    app()
