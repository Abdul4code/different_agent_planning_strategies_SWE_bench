#!/usr/bin/env python3
"""Run mini-SWE-agent on local tasks from a JSON file."""

import json
import sys
import os
from pathlib import Path

# Add mini-swe-agent src to path
mini_swe_agent_src = Path(__file__).parent / "mini-swe-agent" / "src"
sys.path.insert(0, str(mini_swe_agent_src))

from minisweagent.run.extra.swebench import process_instance, RunBatchProgressManager
from minisweagent.config import builtin_config_dir, get_config_path
import yaml
from rich.live import Live
import concurrent.futures
import time
import logging

def load_tasks_from_json(json_path: str):
    """Load tasks from JSONL file."""
    tasks = []
    with open(json_path, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run mini-SWE-agent on local tasks")
    parser.add_argument("--task-path", default="tasks_50.json", help="Path to tasks JSON file")
    parser.add_argument("--output-dir", default="results/baseline", help="Output directory")
    parser.add_argument("--model", default="local", help="Model to use")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--config", default=str(builtin_config_dir / "extra" / "swebench.yaml"), help="Config file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to run")
    
    args = parser.parse_args()
    
    # Auto-disable cost tracking for local models (ollama, etc.)
    local_model_prefixes = ("ollama/", "local")
    if args.model and any(args.model.startswith(prefix) for prefix in local_model_prefixes):
        if "MSWEA_COST_TRACKING" not in os.environ:
            os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"
    
    # Load tasks
    print(f"Loading tasks from {args.task_path}...")
    instances = load_tasks_from_json(args.task_path)
    if args.limit:
        instances = instances[:args.limit]
    print(f"Loaded {len(instances)} tasks")
    
    # Setup output
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config_path = get_config_path(Path(args.config))
    config = yaml.safe_load(config_path.read_text())
    if args.model:
        config.setdefault("model", {})["model_name"] = args.model
        # Reduce step limit for weak local models (they take many steps)
        if any(args.model.startswith(prefix) for prefix in local_model_prefixes):
            config.setdefault("agent", {})["step_limit"] = min(
                config.get("agent", {}).get("step_limit", 250), 50
            )
            print(f"📌 Local model detected: reduced step_limit to 50 (from {config.get('agent', {}).get('step_limit', 250)})")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Process instances
    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")
    
    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for instance in instances:
                future = executor.submit(
                    process_instance,
                    instance,
                    output_path,
                    config,
                    progress_manager,
                )
                futures[future] = instance["instance_id"]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    instance_id = futures[future]
                    logger.error(f"Error processing {instance_id}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
