#!/usr/bin/env python3
"""
Wrapper to run a single SWE-bench task with a specified agent pattern.

This wrapper is called by experiment_orchestrator.py and outputs metrics
in a format the orchestrator can parse: METRICS_JSON={...}

Usage (new orchestrator format):
    python run_agent_wrapper.py --pattern decomposed --task_id django__django-13710 --model gpt-4 --seed 42

Usage (legacy format):
    python run_agent_wrapper.py decomposed --tasks tasks_50.json --out results/decompose

The wrapper:
1. Detects the calling format (orchestrator vs legacy)
2. For orchestrator: runs single task and outputs METRICS_JSON
3. For legacy: delegates to the batch runner
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

# Suppress verbose logging during import
logging.basicConfig(level=logging.WARNING)


def is_orchestrator_mode() -> bool:
    """Check if called with orchestrator-style arguments."""
    return "--pattern" in sys.argv or "--task_id" in sys.argv


def load_task_by_id(tasks_file: str, task_id: str) -> dict:
    """Load a specific task from the tasks file."""
    with open(tasks_file) as f:
        content = f.read().strip()
    
    # Try JSON array first
    try:
        tasks = json.loads(content)
        if not isinstance(tasks, list):
            tasks = [tasks]
    except json.JSONDecodeError:
        # JSON Lines format
        tasks = []
        for line in content.split('\n'):
            if line.strip():
                tasks.append(json.loads(line))
    
    for task in tasks:
        tid = task.get("instance_id") or task.get("task_id") or task.get("id")
        if tid == task_id:
            return task
    
    raise ValueError(f"Task {task_id} not found in {tasks_file}")


def extract_metrics_from_trajectory(traj_path: Path) -> dict:
    """Extract metrics from a trajectory file."""
    metrics = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
        "resolved": False,
        "exit_status": None,
        "error": None,
    }
    
    if not traj_path.exists():
        metrics["error"] = "trajectory_not_found"
        return metrics
    
    try:
        with open(traj_path) as f:
            traj = json.load(f)
        
        # Extract exit status - check both root level and info.exit_status (mini-swe-agent format)
        info = traj.get("info", {})
        exit_status = info.get("exit_status") or traj.get("exit_status", "unknown")
        metrics["exit_status"] = exit_status
        
        # Check for submitted status (case-insensitive)
        exit_status_lower = str(exit_status).lower() if exit_status else ""
        metrics["resolved"] = exit_status_lower == "submitted"
        
        # Set error field based on exit status
        if exit_status_lower == "submitted":
            # Successful submission - mark as "submitted" (not an error, but useful for tracking)
            metrics["error"] = "submitted"
        elif exit_status and exit_status_lower not in ("unknown", ""):
            # Real error - capture full error details
            error_parts = [exit_status]
            
            # Check info.submission first - this typically contains the concise error message
            submission = info.get("submission", "")
            if submission and isinstance(submission, str):
                submission_single_line = submission.replace('\n', ' ').replace('\r', ' ').strip()
                if submission_single_line and submission_single_line != exit_status:
                    error_parts.append(submission_single_line)
            
            # If no submission, capture the result field
            if len(error_parts) == 1:
                result = traj.get("result", "")
                if result and isinstance(result, str):
                    result_single_line = result.replace('\n', ' ').replace('\r', ' ').strip()
                    if result_single_line:
                        error_parts.append(result_single_line)
            
            # If still only exit_status, check traceback for the actual error
            if len(error_parts) == 1:
                extra_info = traj.get("extra_info", {}) or info
                traceback_str = extra_info.get("traceback", "") or info.get("traceback", "")
                if traceback_str:
                    # Extract the last line of traceback (the actual error message)
                    tb_lines = [l.strip() for l in traceback_str.strip().split('\n') if l.strip()]
                    if tb_lines:
                        last_line = tb_lines[-1]
                        if last_line and last_line not in error_parts:
                            error_parts.append(last_line)
            
            # Combine all error info
            full_error = ": ".join(error_parts)
            # Sanitize to single line and limit length
            metrics["error"] = full_error.replace('\n', ' ').replace('\r', ' ')[:1000]
        
        # Check extra_info/info for traceback even if exit_status is unknown
        extra_info = traj.get("extra_info", {}) or {}
        traceback_str = extra_info.get("traceback", "") or info.get("traceback", "")
        if traceback_str and not metrics["error"]:
            tb_lines = [l.strip() for l in traceback_str.strip().split('\n') if l.strip()]
            if tb_lines:
                metrics["error"] = tb_lines[-1].replace('\n', ' ').replace('\r', ' ')[:1000]
            else:
                metrics["error"] = "exception_with_traceback"
        
        # Count tokens from messages (mini-swe-agent format)
        for msg in traj.get("messages", []):
            if msg.get("role") == "assistant":
                extra = msg.get("extra", {})
                resp = extra.get("response", {})
                if isinstance(resp, dict):
                    usage = resp.get("usage", {})
                    if usage:
                        metrics["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        metrics["completion_tokens"] += usage.get("completion_tokens", 0)
                        metrics["total_tokens"] += usage.get("total_tokens", 0)
                    metrics["llm_calls"] += 1
        
        # Also check legacy trajectory format
        for step in traj.get("trajectory", []):
            if "response" in step:
                resp = step["response"]
                if isinstance(resp, dict) and "usage" in resp:
                    usage = resp["usage"]
                    metrics["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    metrics["completion_tokens"] += usage.get("completion_tokens", 0)
                    metrics["total_tokens"] += usage.get("total_tokens", 0)
                metrics["llm_calls"] += 1
        
        # Also check info.model_stats if available (mini-swe-agent format)
        info = traj.get("info", {})
        model_stats = info.get("model_stats", {}) or traj.get("model_stats", {})
        if model_stats:
            if "api_calls" in model_stats:
                metrics["llm_calls"] = model_stats["api_calls"]
            elif "n_calls" in model_stats:
                metrics["llm_calls"] = model_stats["n_calls"]
            if "prompt_tokens" in model_stats:
                metrics["prompt_tokens"] = model_stats["prompt_tokens"]
            if "completion_tokens" in model_stats:
                metrics["completion_tokens"] = model_stats["completion_tokens"]
            if "total_tokens" in model_stats:
                metrics["total_tokens"] = model_stats.get("total_tokens", 
                    metrics["prompt_tokens"] + metrics["completion_tokens"])
                    
    except Exception as e:
        logging.warning(f"Error parsing trajectory: {e}")
    
    return metrics


def run_single_task_with_agent(pattern: str, task: dict, output_dir: Path, model: str, config_data: dict) -> dict:
    """Run a single task with the specified agent pattern."""
    import importlib.util
    
    # Determine which source to use
    if pattern == "baseline":
        agent_src = Path(__file__).parent / "mini-swe-agent" / "src"
    else:
        agent_src = Path(__file__).parent / f"mini-swe-agent-{pattern}" / "src"
    
    # Clear any cached minisweagent modules to ensure fresh imports
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('minisweagent')]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    # Insert agent source at the beginning of path
    sys.path.insert(0, str(agent_src))
    
    # Import common dependencies from the agent's source
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model
    from minisweagent.run.extra.swebench import get_swebench_docker_image_name
    from minisweagent.run.utils.save import save_traj
    
    # Import agent based on pattern
    if pattern == "baseline":
        from minisweagent.agents.default import DefaultAgent as AgentClass
    elif pattern == "decomposed":
        from minisweagent.agents.task_decomposition import TaskDecompositionAgent as AgentClass
    elif pattern == "multiplan":
        from minisweagent.agents.multiplan import MultiPlanAgent as AgentClass
    elif pattern == "external":
        from minisweagent.agents.external_planner import ExternalPlannerAgent as AgentClass
    elif pattern == "memory":
        from minisweagent.agents.memory import MemoryAgent as AgentClass
    elif pattern == "reflection":
        from minisweagent.agents.reflection import ReflectionAgent as AgentClass
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    instance_id = task["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    model_instance = get_model(config=config_data.get("model", {}))
    problem = task["problem_statement"]
    
    env = None
    agent = None
    extra_info = None
    
    try:
        env_config = config_data.setdefault("environment", {})
        env_config["environment_class"] = env_config.get("environment_class", "docker")
        image_name = get_swebench_docker_image_name(task)
        if env_config["environment_class"] in ["docker", "swerex_modal"]:
            env_config["image"] = image_name
        
        env = get_environment(env_config)
        agent = AgentClass(model_instance, env, **config_data.get("agent", {}))
        exit_status, result = agent.run(problem)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
        
    finally:
        if env and hasattr(env, "stop"):
            env.stop()
        
        traj_path = instance_dir / f"{instance_id}.traj.json"
        save_traj(
            agent,
            traj_path,
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
        )
    
    return {
        "traj_path": traj_path,
        "exit_status": exit_status,
        "result": result,
    }


def run_orchestrator_mode():
    """Run in orchestrator mode (single task with metrics output)."""
    parser = argparse.ArgumentParser(description="Run a single SWE-bench task with specified agent")
    parser.add_argument("--pattern", required=True, help="Agent pattern to use")
    parser.add_argument("--task_id", required=True, help="Task ID to run")
    parser.add_argument("--model", default="openai/gpt-4", help="Model to use (e.g., openai/gpt-4, anthropic/claude-sonnet-4-5-20250929)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--agent_folder", help="Agent folder (auto-detected from pattern)")
    parser.add_argument("--tasks_file", default="tasks_50.json", help="Tasks file path")
    parser.add_argument("--output_dir", default="results/orchestrator_runs", help="Output directory")
    parser.add_argument("--config", help="Config file path")
    
    args = parser.parse_args()
    
    valid_patterns = ["baseline", "decomposed", "multiplan", "external", "memory", "reflection"]
    if args.pattern not in valid_patterns:
        print(f"Unknown pattern: {args.pattern}", file=sys.stderr)
        print(f"Available: {valid_patterns}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load the specific task
        task = load_task_by_id(args.tasks_file, args.task_id)
        
        # Setup output directory
        output_dir = Path(args.output_dir) / args.pattern
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        import yaml
        baseline_path = Path(__file__).parent / "mini-swe-agent" / "src"
        sys.path.insert(0, str(baseline_path))
        from minisweagent.config import builtin_config_dir, get_config_path
        
        config_path = args.config or str(builtin_config_dir / "extra" / "swebench.yaml")
        config_file = get_config_path(Path(config_path))
        config_data = yaml.safe_load(config_file.read_text())
        
        # Set model
        config_data.setdefault("model", {})["model_name"] = args.model
        
        # Auto-disable cost tracking for certain models
        if "MSWEA_COST_TRACKING" not in os.environ:
            os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"
        
        # Run the agent
        result = run_single_task_with_agent(args.pattern, task, output_dir, args.model, config_data)
        
        # Extract metrics from trajectory
        metrics = extract_metrics_from_trajectory(result["traj_path"])
        
        # Override with direct result info if trajectory extraction missed it
        # Use case-insensitive check for exit_status
        exit_status = result.get("exit_status", "")
        exit_status_lower = str(exit_status).lower() if exit_status else ""
        if exit_status and exit_status_lower not in ("submitted", "unknown", ""):
            if not metrics.get("error"):
                metrics["error"] = f"{exit_status}: {str(result.get('result', ''))}"
        
        # Output metrics in the format the orchestrator expects
        print(f"METRICS_JSON={json.dumps(metrics)}")
        print(f"RESOLVED={'True' if metrics['resolved'] else 'False'}")
        
        sys.exit(0)  # Success regardless of resolution
        
    except Exception as e:
        # Output error metrics
        metrics = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
            "resolved": False,
            "error": str(e),
        }
        print(f"METRICS_JSON={json.dumps(metrics)}")
        print(f"RESOLVED=False")
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def run_legacy_mode():
    """Run in legacy mode (batch processing via existing scripts)."""
    # Get agent type from args
    agent_type = sys.argv[1] if len(sys.argv) > 1 else "decomposed"

    # Handle baseline separately
    if agent_type == "baseline":
        # Use the baseline runner script
        baseline_script = Path(__file__).parent / "run_baseline.py"
        sys.argv = [str(baseline_script)] + sys.argv[2:]
        
        with open(baseline_script) as f:
            code = f.read()
        exec(compile(code, str(baseline_script), 'exec'), {'__name__': '__main__', '__file__': str(baseline_script)})
    else:
        agent_folder = Path(__file__).parent / f"mini-swe-agent-{agent_type}"
        agent_src = agent_folder / "src"

        # Setup paths - agent src FIRST (so agent's minisweagent is found first)
        sys.path.insert(0, str(agent_src))

        # Remove agent_type from argv
        sys.argv = [sys.argv[0]] + sys.argv[2:]

        # Map agent type to file
        run_map = {
            "decomposed": "run_task_decomposition.py",
            "multiplan": "run_multiplan.py",
            "external": "run_external.py",
            "reflection": "run_reflect.py",
            "memory": "run_memory.py",
        }

        run_file = run_map.get(agent_type)
        if not run_file:
            print(f"Unknown agent type: {agent_type}")
            print("Available: baseline, decomposed, multiplan, external, reflection, memory")
            sys.exit(1)

        # Read and execute the run script
        run_path = agent_src / "minisweagent" / "run" / run_file
        with open(run_path) as f:
            code = f.read()
            
        # Execute in __main__ context with proper globals
        exec(compile(code, str(run_path), 'exec'), {'__name__': '__main__', '__file__': str(run_path)})


if __name__ == "__main__":
    if is_orchestrator_mode():
        run_orchestrator_mode()
    else:
        run_legacy_mode()

