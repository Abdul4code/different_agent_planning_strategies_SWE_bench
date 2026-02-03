"""Shared run utilities for task decomposition agent.

Reuses baseline mini-swe-agent infrastructure for Docker, environment setup, etc.
"""

import json
import logging
import traceback
import concurrent.futures
import time
from pathlib import Path
from typing import Type
import os
import sys

# Import from baseline
baseline_path = Path(__file__).parent.parent.parent.parent / "mini-swe-agent" / "src"
sys.path.insert(0, str(baseline_path))

import yaml
from rich.live import Live

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


def run_agent_batch(
    agent_class: Type[DefaultAgent],
    instances: list,
    output_path: Path,
    config_data: dict,
    workers: int = 1,
    run_id: str = "eval_001",
    agent_name: str = "agent",
):
    """Run agent batch on instances."""
    # Auto-disable cost tracking for local models
    local_model_prefixes = ("ollama/", "local")
    model = config_data.get("model", {}).get("model_name", "")
    if model and any(model.startswith(prefix) for prefix in local_model_prefixes):
        if "MSWEA_COST_TRACKING" not in os.environ:
            os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def process_instance_with_agent(instance, output_dir, cfg, progress_manager, agent_cls):
        """Process single instance with custom agent."""
        instance_id = instance["instance_id"]
        instance_dir = output_dir / instance_id
        remove_from_preds_file(output_dir / "preds.json", instance_id)
        (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

        model = get_model(config=cfg.get("model", {}))
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
            class ProgressTrackingWrapper(agent_cls):
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

            agent = ProgressTrackingWrapper(
                model,
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
                output_dir / "preds.json", instance_id, model.config.model_name, result
            )
            progress_manager.on_instance_end(instance_id, exit_status)

    # Setup output
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(instances)} instances with {agent_class.__name__}")

    # Process instances in parallel
    progress_manager = RunBatchProgressManager(
        len(instances), output_path / f"exit_statuses_{time.time()}.yaml"
    )

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for instance in instances:
                future = executor.submit(
                    process_instance_with_agent,
                    instance,
                    output_path,
                    config_data,
                    progress_manager,
                    agent_class,
                )
                futures[future] = instance["instance_id"]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    instance_id = futures[future]
                    logger.error(f"Error processing {instance_id}: {e}", exc_info=True)

    print(f"✓ Run complete: {run_id}")
