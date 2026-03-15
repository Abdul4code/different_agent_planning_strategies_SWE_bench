#!/usr/bin/env python3
"""
Experiment Orchestrator for SWE-bench Multi-Agent Planning Evaluation

This script orchestrates running SWE-bench tasks across multiple planning patterns
(centralized, hierarchical, decentralized, hybrid) and collects metrics for
efficiency and energy analysis.

Usage:
    python experiment_orchestrator.py --tasks tasks_50.json --patterns centralized hierarchical --model gpt-4 --out results/experiment_runs.csv

Features:
    - Parallel execution with ProcessPoolExecutor
    - Resume capability for interrupted experiments
    - Token and LLM call tracking
    - Memory profiling (peak RSS)
    - Energy tracking with CodeCarbon (optional)
    - Process-safe CSV output with file locking
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Suppress CodeCarbon warnings about multiple instances before importing
os.environ["CODECARBON_LOG_LEVEL"] = "error"

from metrics import (
    CSVWriter,
    LLMMetrics,
    PeakMemorySampler,
    TaskRunRecord,
    create_manifest,
    parse_codecarbon_csv,
    parse_metrics_from_stdout,
    save_manifest,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pattern to Agent Folder Mapping
# =============================================================================

PATTERN_FOLDER_MAP: Dict[str, str] = {
    "baseline": "mini-swe-agent",
    "decomposed": "mini-swe-agent-decomposed",
    "external": "mini-swe-agent-external",
    "multiplan": "mini-swe-agent-multiplan",
    "memory": "mini-swe-agent-memory",
    "reflection": "mini-swe-agent-reflection",
}

VALID_PATTERNS = list(PATTERN_FOLDER_MAP.keys())


# =============================================================================
# Task Loading
# =============================================================================

def load_tasks(tasks_path: str | Path) -> List[str]:
    """
    Load task IDs from a JSON file.
    
    Supports three formats:
    1. JSON array of strings: ["task_id_1", "task_id_2", ...]
    2. JSON array of objects: [{"task_id": "...", ...}, ...]
    3. JSON Lines (JSONL): One JSON object per line (SWE-bench format)
    
    Args:
        tasks_path: Path to the tasks JSON file
        
    Returns:
        List of task ID strings
    """
    tasks_path = Path(tasks_path)
    
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")
    
    with open(tasks_path, "r") as f:
        content = f.read().strip()
    
    # Try to parse as JSON array first
    data: List[Any] = []
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            # Single JSON object - wrap in list
            data = [data]
    except json.JSONDecodeError:
        # Try JSON Lines format (one JSON object per line)
        data = []
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        
        if not data:
            raise ValueError(f"Could not parse tasks file as JSON or JSON Lines: {tasks_path}")
    
    task_ids: List[str] = []
    
    for item in data:
        if isinstance(item, str):
            # Format 1: Direct string ID
            task_ids.append(item)
        elif isinstance(item, dict):
            # Format 2: Object with task_id field
            if "task_id" in item:
                task_ids.append(str(item["task_id"]))
            elif "instance_id" in item:
                # Also support instance_id for SWE-bench compatibility
                task_ids.append(str(item["instance_id"]))
            elif "id" in item:
                task_ids.append(str(item["id"]))
            else:
                logger.warning(f"Skipping task object without task_id/instance_id/id: {item}")
        else:
            logger.warning(f"Skipping invalid task item of type {type(item).__name__}")
    
    logger.info(f"Loaded {len(task_ids)} tasks from {tasks_path}")
    return task_ids


# =============================================================================
# Agent Runner Detection
# =============================================================================

def detect_runner_mode() -> Tuple[str, Optional[Callable]]:
    """
    Detect whether run_agent_wrapper.py exposes a callable function.
    
    Returns:
        Tuple of (mode, callable_or_none)
        - ("import", callable) if a run_task function is found
        - ("subprocess", None) if subprocess mode should be used
    """
    wrapper_path = Path("run_agent_wrapper.py")
    
    if not wrapper_path.exists():
        logger.info("run_agent_wrapper.py not found, using subprocess mode")
        return ("subprocess", None)
    
    try:
        spec = importlib.util.spec_from_file_location("run_agent_wrapper", wrapper_path)
        if spec is None or spec.loader is None:
            logger.info("Could not load run_agent_wrapper.py spec, using subprocess mode")
            return ("subprocess", None)
        
        module = importlib.util.module_from_spec(spec)
        
        # Don't actually execute the module during detection to avoid side effects
        # Just check if the file defines a run_task function
        with open(wrapper_path, "r") as f:
            content = f.read()
        
        # Simple heuristic: check if the file defines a run_task function
        if "def run_task(" in content or "def run_task (" in content:
            # Try to actually import it
            try:
                spec.loader.exec_module(module)
                if hasattr(module, "run_task") and callable(module.run_task):
                    logger.info("Found callable run_task() in run_agent_wrapper.py, using import mode")
                    return ("import", module.run_task)
            except Exception as e:
                logger.warning(f"Could not import run_agent_wrapper.py: {e}, using subprocess mode")
        
        logger.info("No callable run_task() found, using subprocess mode")
        return ("subprocess", None)
        
    except Exception as e:
        logger.warning(f"Error detecting runner mode: {e}, using subprocess mode")
        return ("subprocess", None)


# =============================================================================
# Single Task Runner
# =============================================================================

def run_single_task(
    pattern: str,
    task_id: str,
    model: str,
    seed: int,
    config: Dict[str, Any],
) -> TaskRunRecord:
    """
    Run a single SWE-bench task with the specified pattern.
    
    This function:
    1. Sets up metrics collection (memory, optionally energy)
    2. Runs the task via import or subprocess mode
    3. Collects and returns all metrics
    
    Args:
        pattern: Planning pattern (centralized, hierarchical, decentralized, hybrid)
        task_id: SWE-bench task identifier
        model: LLM model name
        seed: Random seed for reproducibility
        config: Configuration dict containing:
            - timeout_s: Timeout in seconds
            - use_codecarbon: Whether to track energy
            - runner_mode: "import" or "subprocess"
            - run_task_fn: Callable if runner_mode is "import"
            
    Returns:
        TaskRunRecord with all collected metrics
    """
    # Initialize record
    record = TaskRunRecord(
        run_id=str(uuid.uuid4()),
        started_at=datetime.now(timezone.utc).isoformat(),
        pattern=pattern,
        task_id=task_id,
        model=model,
        seed=seed,
    )
    
    timeout_s = config.get("timeout_s", 1800)
    use_codecarbon = config.get("use_codecarbon", False)
    runner_mode = config.get("runner_mode", "subprocess")
    run_task_fn = config.get("run_task_fn", None)
    
    # Get the agent folder for this pattern
    agent_folder = PATTERN_FOLDER_MAP.get(pattern, "mini-swe-agent")
    
    # Set up CodeCarbon if enabled
    tracker = None
    codecarbon_output_dir = None
    
    if use_codecarbon:
        try:
            from codecarbon import EmissionsTracker
            codecarbon_output_dir = tempfile.mkdtemp(prefix="codecarbon_")
            tracker = EmissionsTracker(
                output_dir=codecarbon_output_dir,
                log_level="error",
                save_to_file=True,
                allow_multiple_runs=True,
            )
        except ImportError:
            logger.warning("CodeCarbon not installed, energy tracking disabled")
            use_codecarbon = False
        except Exception as e:
            logger.warning(f"CodeCarbon initialization failed: {e}, energy tracking disabled")
            use_codecarbon = False
            tracker = None
    
    # Start metrics collection
    memory_sampler = PeakMemorySampler(interval_s=0.1)
    llm_metrics = LLMMetrics()
    
    start_time = time.perf_counter()
    
    try:
        memory_sampler.start()
        
        if tracker:
            tracker.start()
        
        # Run the task
        if runner_mode == "import" and run_task_fn is not None:
            result = _run_via_import(
                run_task_fn, pattern, task_id, model, seed, agent_folder, timeout_s, llm_metrics
            )
        else:
            result = _run_via_subprocess(
                pattern, task_id, model, seed, agent_folder, timeout_s, llm_metrics
            )
        
        # Extract results - 'submitted' means the agent produced a patch (not yet verified)
        record.submitted = result.get("resolved", False)  # mini-swe-agent uses 'resolved' for submitted
        record.resolved = False  # Will be set by evaluate_predictions.py after SWE-bench harness
        
        # Extract error_type from metrics if available
        if "metrics" in result:
            metrics = result["metrics"]
            llm_metrics.update_from_dict(metrics)

            # If the wrapper provided timing breakdowns, use them.
            # - runtime_s: agent runtime excluding container startup
            # - wall_runtime_s: end-to-end wall clock
            # - container_setup_s: docker pull + run -d time
            try:
                if isinstance(metrics, dict):
                    if "container_setup_s" in metrics and metrics["container_setup_s"] is not None:
                        record.container_setup_s = float(metrics["container_setup_s"])
                    if "wall_runtime_s" in metrics and metrics["wall_runtime_s"] is not None:
                        record.wall_runtime_s = float(metrics["wall_runtime_s"])
                    if "agent_runtime_s" in metrics and metrics["agent_runtime_s"] is not None:
                        # This is the primary experiment time we care about.
                        record.runtime_s = float(metrics["agent_runtime_s"])
            except (ValueError, TypeError):
                # Fall back to wall-clock timing if parsing fails
                pass
            
            # Check for error in metrics JSON
            if "error" in metrics and metrics["error"]:
                # Sanitize to single line - replace newlines with spaces
                error_str = str(metrics["error"]).replace('\n', ' ').replace('\r', ' ')
                record.error_type = error_str
            elif "exit_status" in metrics and metrics["exit_status"]:
                # Use exit_status (case-insensitive check)
                exit_status = str(metrics["exit_status"])
                record.error_type = exit_status.lower() if exit_status.lower() == "submitted" else exit_status
        
        # For subprocess errors, capture the full stderr as error_type
        if not record.error_type and result.get("returncode", 0) != 0:
            stderr = result.get("stderr", "")
            stdout = result.get("stdout", "")
            # Try to extract the actual error message from stderr or stdout
            error_output = stderr if stderr else stdout
            # Sanitize to single line
            error_str = error_output.replace('\n', ' ').replace('\r', ' ').strip()
            # If too long, try to find the actual error (e.g., "ModuleNotFoundError: No module named 'X'")
            if error_str:
                # Look for common Python error patterns
                import re
                error_match = re.search(r'(ModuleNotFoundError|ImportError|RuntimeError|ValueError|TypeError|KeyError|AttributeError|FileNotFoundError|Exception|Error)[:\s]+[^\n]+', error_output)
                if error_match:
                    record.error_type = error_match.group(0).replace('\n', ' ').strip()
                else:
                    # Take last meaningful line or truncate
                    lines = [l.strip() for l in error_output.strip().split('\n') if l.strip()]
                    if lines:
                        record.error_type = lines[-1][:500]  # Last line, max 500 chars
                    else:
                        record.error_type = f"exit_code_{result.get('returncode', 'unknown')}"
            else:
                record.error_type = f"exit_code_{result.get('returncode', 'unknown')}"
        
        # If still no error but not submitted, mark as not_submitted
        if not record.error_type and not record.submitted:
            record.error_type = "not_submitted"
        
    except TimeoutError:
        record.error_type = "timeout"
        record.submitted = False
        record.resolved = False
        
    except Exception as e:
        # Sanitize exception message to single line
        error_msg = str(e).replace('\n', ' ').replace('\r', ' ')
        record.error_type = f"{type(e).__name__}: {error_msg}"
        record.submitted = False
        record.resolved = False
        logger.error(f"Error running task {task_id} with pattern {pattern}: {e}")
        
    finally:
        # Stop metrics collection
        end_time = time.perf_counter()
        wall = round(end_time - start_time, 3)
        # Preserve wall-clock runtime unless a more precise value was reported by the wrapper.
        if not record.wall_runtime_s:
            record.wall_runtime_s = wall
        # If the wrapper didn't provide agent_runtime_s, fall back to wall time.
        if not record.runtime_s:
            record.runtime_s = wall
        
        memory_sampler.stop()
        record.peak_rss_mb = round(memory_sampler.peak_rss_mb, 2)
        
        # Collect LLM metrics
        metrics_dict = llm_metrics.to_dict()
        record.prompt_tokens = metrics_dict["prompt_tokens"]
        record.completion_tokens = metrics_dict["completion_tokens"]
        record.total_tokens = metrics_dict["total_tokens"]
        record.llm_calls = metrics_dict["llm_calls"]
        
        # Stop CodeCarbon and collect energy metrics
        if tracker:
            try:
                emissions = tracker.stop()
                record.co2_kg = round(emissions, 8) if emissions else None
                
                # Parse the CSV for energy_consumed
                if codecarbon_output_dir:
                    csv_path = Path(codecarbon_output_dir) / "emissions.csv"
                    energy_data = parse_codecarbon_csv(csv_path)
                    if "energy_kwh" in energy_data:
                        record.energy_kwh = round(energy_data["energy_kwh"], 8)
                    
                    # Clean up temp directory
                    try:
                        import shutil
                        shutil.rmtree(codecarbon_output_dir)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Error stopping CodeCarbon: {e}")
    
    return record


def _run_via_import(
    run_task_fn: Callable,
    pattern: str,
    task_id: str,
    model: str,
    seed: int,
    agent_folder: str,
    timeout_s: int,
    llm_metrics: LLMMetrics,
) -> Dict[str, Any]:
    """
    Run task via imported function.
    
    Returns:
        Dictionary with at least 'resolved' key
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Task {task_id} timed out after {timeout_s}s")
    
    # Set timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_s)
    
    try:
        result = run_task_fn(
            pattern=pattern,
            task_id=task_id,
            model=model,
            seed=seed,
            agent_folder=agent_folder,
            metrics=llm_metrics,  # Pass metrics object for direct updates
        )
        
        if result is None:
            result = {}
        elif not isinstance(result, dict):
            result = {"resolved": bool(result)}
            
        return result
        
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _run_via_subprocess(
    pattern: str,
    task_id: str,
    model: str,
    seed: int,
    agent_folder: str,
    timeout_s: int,
    llm_metrics: LLMMetrics,
) -> Dict[str, Any]:
    """
    Run task via subprocess.
    
    Parses METRICS_JSON={...} from stdout if present.
    
    Returns:
        Dictionary with at least 'resolved' key
    """
    cmd = [
        sys.executable,
        "run_agent_wrapper.py",
        "--pattern", pattern,
        "--task_id", task_id,
        "--model", model,
        "--seed", str(seed),
        "--agent_folder", agent_folder,
    ]
    
    logger.debug(f"Running subprocess: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=os.getcwd(),
            env=os.environ.copy(),  # Explicitly pass environment variables (API keys, etc.)
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
            logger.warning(f"Subprocess returned non-zero exit code: {result.returncode}")
            logger.debug(f"stderr: {stderr}")
        
        # Parse metrics from stdout
        metrics = parse_metrics_from_stdout(stdout)
        
        # Update LLM metrics
        if metrics:
            llm_metrics.update_from_dict(metrics)
        
        # Determine resolved status
        resolved = metrics.get("resolved", False)
        
        # Also check for explicit success indicators in stdout
        if not resolved:
            if "RESOLVED=True" in stdout or '"resolved": true' in stdout.lower():
                resolved = True
        
        return {
            "resolved": resolved,
            "metrics": metrics,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
        }
        
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Task {task_id} timed out after {timeout_s}s")


# =============================================================================
# Worker Function for ProcessPoolExecutor
# =============================================================================

def _worker_run_task(args: Tuple[str, str, str, int, Dict[str, Any]]) -> TaskRunRecord:
    """
    Worker function for ProcessPoolExecutor.
    
    Args:
        args: Tuple of (pattern, task_id, model, seed, config)
        
    Returns:
        TaskRunRecord
    """
    pattern, task_id, model, seed, config = args
    
    # Re-detect runner mode in subprocess (imports don't transfer)
    # For subprocess-based parallelism, always use subprocess mode for simplicity
    config = config.copy()
    config["runner_mode"] = "subprocess"
    config["run_task_fn"] = None
    
    return run_single_task(pattern, task_id, model, seed, config)


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_experiment(
    tasks_path: str,
    patterns: List[str],
    model: str,
    output_path: str,
    seeds: List[int],
    timeout_s: int = 1800,
    max_workers: int = 1,
    resume: bool = False,
    use_codecarbon: bool = False,
) -> None:
    """
    Run the full experiment across all tasks and patterns.
    
    Args:
        tasks_path: Path to tasks JSON file
        patterns: List of patterns to evaluate
        model: LLM model name
        output_path: Path to output CSV file
        seeds: List of random seeds
        timeout_s: Timeout per task in seconds
        max_workers: Maximum parallel workers
        resume: Whether to skip already-completed runs
        use_codecarbon: Whether to track energy with CodeCarbon
    """
    # Validate patterns
    invalid_patterns = [p for p in patterns if p not in VALID_PATTERNS]
    if invalid_patterns:
        raise ValueError(f"Invalid patterns: {invalid_patterns}. Valid patterns: {VALID_PATTERNS}")
    
    # Load tasks
    task_ids = load_tasks(tasks_path)
    
    if not task_ids:
        logger.warning("No tasks loaded, nothing to do")
        return
    
    # Initialize CSV writer
    csv_writer = CSVWriter(output_path)
    
    # Load existing runs if resuming
    existing_keys: set = set()
    if resume:
        existing_keys = csv_writer.read_existing_keys()
        logger.info(f"Resume mode: found {len(existing_keys)} existing runs")
    
    # Build list of runs to execute
    runs_to_execute: List[Tuple[str, str, str, int, Dict[str, Any]]] = []
    
    config = {
        "timeout_s": timeout_s,
        "use_codecarbon": use_codecarbon,
        "runner_mode": "subprocess",  # Default to subprocess for process isolation
        "run_task_fn": None,
    }
    
    for pattern in patterns:
        for task_id in task_ids:
            for seed in seeds:
                key = (pattern, task_id, seed, model)
                
                if resume and key in existing_keys:
                    logger.debug(f"Skipping existing run: {key}")
                    continue
                
                runs_to_execute.append((pattern, task_id, model, seed, config))
    
    total_runs = len(runs_to_execute)
    logger.info(f"Executing {total_runs} task runs across {len(patterns)} patterns")
    
    if total_runs == 0:
        logger.info("All runs already completed (resume mode)")
        return
    
    # Save manifest
    manifest_path = Path(output_path).parent / "manifest.json"
    manifest = create_manifest(
        cli_args={
            "tasks_path": tasks_path,
            "patterns": patterns,
            "model": model,
            "output_path": output_path,
            "seeds": seeds,
            "timeout_s": timeout_s,
            "max_workers": max_workers,
            "resume": resume,
            "use_codecarbon": use_codecarbon,
        },
        output_path=manifest_path,
    )
    save_manifest(manifest, manifest_path)
    logger.info(f"Saved manifest to {manifest_path}")
    
    # Execute runs
    completed = 0
    failed = 0
    
    if max_workers == 1:
        # Sequential execution
        for run_args in runs_to_execute:
            pattern, task_id, model_name, seed, cfg = run_args
            logger.info(f"[{completed + 1}/{total_runs}] Running {pattern}/{task_id} (seed={seed})")
            
            try:
                record = run_single_task(pattern, task_id, model_name, seed, cfg)
                csv_writer.write_record(record)
                
                if record.error_type and record.error_type != "submitted":
                    failed += 1
                    logger.warning(f"  -> Failed: {record.error_type}")
                else:
                    logger.info(f"  -> Submitted: {record.submitted}, Runtime: {record.runtime_s}s")
                    
            except Exception as e:
                failed += 1
                logger.error(f"  -> Unexpected error: {e}")
                
                # Write error record
                error_record = TaskRunRecord(
                    pattern=pattern,
                    task_id=task_id,
                    model=model_name,
                    seed=seed,
                    error_type=type(e).__name__,
                )
                csv_writer.write_record(error_record)
            
            completed += 1
    else:
        # Parallel execution with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_args = {
                executor.submit(_worker_run_task, args): args
                for args in runs_to_execute
            }
            
            for future in as_completed(future_to_args):
                args = future_to_args[future]
                pattern, task_id, model_name, seed, _ = args
                
                completed += 1
                
                try:
                    record = future.result(timeout=timeout_s + 60)  # Extra buffer for cleanup
                    csv_writer.write_record(record)
                    
                    if record.error_type and record.error_type != "submitted":
                        failed += 1
                        logger.warning(
                            f"[{completed}/{total_runs}] {pattern}/{task_id} (seed={seed}) "
                            f"-> Failed: {record.error_type}"
                        )
                    else:
                        logger.info(
                            f"[{completed}/{total_runs}] {pattern}/{task_id} (seed={seed}) "
                            f"-> Submitted: {record.submitted}, Runtime: {record.runtime_s}s"
                        )
                        
                except FuturesTimeoutError:
                    failed += 1
                    logger.error(
                        f"[{completed}/{total_runs}] {pattern}/{task_id} (seed={seed}) "
                        f"-> Worker timeout"
                    )
                    
                    error_record = TaskRunRecord(
                        pattern=pattern,
                        task_id=task_id,
                        model=model_name,
                        seed=seed,
                        error_type="WorkerTimeout",
                    )
                    csv_writer.write_record(error_record)
                    
                except Exception as e:
                    failed += 1
                    logger.error(
                        f"[{completed}/{total_runs}] {pattern}/{task_id} (seed={seed}) "
                        f"-> Error: {e}"
                    )
                    
                    error_record = TaskRunRecord(
                        pattern=pattern,
                        task_id=task_id,
                        model=model_name,
                        seed=seed,
                        error_type=type(e).__name__,
                    )
                    csv_writer.write_record(error_record)
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Experiment complete: {completed} runs, {failed} failures")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Manifest saved to: {manifest_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SWE-bench Experiment Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings
    python experiment_orchestrator.py --tasks tasks_50.json --model gpt-4

    # Run specific patterns with parallel workers
    python experiment_orchestrator.py --tasks tasks_50.json \\
        --patterns centralized hierarchical decentralized \\
        --model gpt-4 --max_workers 4

    # Resume interrupted experiment with energy tracking
    python experiment_orchestrator.py --tasks tasks_50.json \\
        --model gpt-4 --resume --use_codecarbon

    # Run with custom timeout and seeds
    python experiment_orchestrator.py --tasks tasks_50.json \\
        --model gpt-4 --timeout_s 3600 --seeds 42 123 456
        """,
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Path to tasks JSON file",
    )
    
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=VALID_PATTERNS,
        choices=VALID_PATTERNS,
        help=f"Planning patterns to evaluate (default: all). Choices: {VALID_PATTERNS}",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM model name with provider prefix (e.g., openai/gpt-4, anthropic/claude-sonnet-4-5-20250929, gemini/gemini-2.5-pro)",
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="results/experiment_runs.csv",
        help="Output CSV file path (default: results/experiment_runs.csv)",
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for reproducibility (default: [42])",
    )
    
    parser.add_argument(
        "--timeout_s",
        type=int,
        default=1800,
        help="Timeout per task in seconds (default: 1800)",
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum parallel workers (default: 1)",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed runs (based on pattern, task_id, seed, model)",
    )
    
    parser.add_argument(
        "--use_codecarbon",
        action="store_true",
        help="Enable energy tracking with CodeCarbon",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("SWE-bench Experiment Orchestrator")
    logger.info("=" * 60)
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Patterns: {args.patterns}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.out}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Timeout: {args.timeout_s}s")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"CodeCarbon: {args.use_codecarbon}")
    logger.info("=" * 60)
    
    try:
        run_experiment(
            tasks_path=args.tasks,
            patterns=args.patterns,
            model=args.model,
            output_path=args.out,
            seeds=args.seeds,
            timeout_s=args.timeout_s,
            max_workers=args.max_workers,
            resume=args.resume,
            use_codecarbon=args.use_codecarbon,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
