#!/usr/bin/env python3
"""
Evaluate predictions from experiment runs using the SWE-bench harness.

This script:
1. Reads the experiment_runs.csv file
2. Collects all submitted predictions for each pattern
3. Creates prediction files in SWE-bench format
4. Runs the SWE-bench evaluation harness
5. Updates the CSV with actual resolution status

Usage:
    python evaluate_predictions.py --csv results/experiment_runs.csv --results_dir results/output

Requirements:
    - Docker must be running
    - SWE-bench docker images must be built (run: python -m swebench.harness.run_evaluation --help)
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pattern to Agent Output Folder Mapping
# =============================================================================

PATTERN_FOLDER_MAP: Dict[str, str] = {
    "baseline": "mini-swe-agent",
    "decomposed": "mini-swe-agent-decomposed",
    "external": "mini-swe-agent-external",
    "multiplan": "mini-swe-agent-multiplan",
    "memory": "mini-swe-agent-memory",
    "reflection": "mini-swe-agent-reflection",
}


def find_prediction_file(base_dir: Path, pattern: str, task_id: str) -> Optional[Path]:
    """
    Find the preds.json file for a specific pattern and task.
    
    The file structure depends on how mini-swe-agent organizes output:
    - results/{pattern}/preds.json (contains all predictions)
    - OR results/baseline/{task_id}/{task_id}.traj.json (trajectory files)
    """
    # Look in pattern-specific results folder
    pattern_results = base_dir / "results" / pattern
    
    # Check for preds.json
    preds_file = pattern_results / "preds.json"
    if preds_file.exists():
        return preds_file
    
    # Check in agent folder's results
    agent_folder = PATTERN_FOLDER_MAP.get(pattern, pattern)
    agent_results = base_dir / agent_folder / "results"
    preds_file = agent_results / "preds.json"
    if preds_file.exists():
        return preds_file
    
    return None


def find_trajectory_file(base_dir: Path, pattern: str, task_id: str) -> Optional[Path]:
    """
    Find the trajectory file for a specific pattern and task.
    
    Looks in multiple possible locations:
    - results/{pattern}/{task_id}/{task_id}.traj.json
    - {agent_folder}/results/{task_id}/{task_id}.traj.json
    """
    possible_paths = [
        base_dir / "results" / pattern / task_id / f"{task_id}.traj.json",
        base_dir / "results" / "comparison" / pattern / task_id / f"{task_id}.traj.json",
    ]
    
    agent_folder = PATTERN_FOLDER_MAP.get(pattern, pattern)
    possible_paths.append(base_dir / agent_folder / "results" / task_id / f"{task_id}.traj.json")
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def extract_patch_from_trajectory(traj_path: Path) -> Optional[str]:
    """Extract the model patch (git diff) from a trajectory file."""
    try:
        with open(traj_path) as f:
            traj = json.load(f)
        
        # The result field contains the git diff
        result = traj.get("result", "")
        if result and isinstance(result, str):
            # Check if it looks like a git diff
            if "diff --git" in result or "---" in result and "+++" in result:
                return result
            # Otherwise it might be in the last message
        
        # Try to get from the last assistant message
        messages = traj.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Look for git diff in content
                if "diff --git" in content:
                    # Extract the diff portion
                    diff_start = content.find("diff --git")
                    return content[diff_start:]
                return content
        
        return result if result else None
        
    except Exception as e:
        logger.warning(f"Error reading trajectory {traj_path}: {e}")
        return None


def collect_predictions(
    csv_path: Path,
    base_dir: Path,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Collect predictions organized by pattern.
    
    Returns:
        {pattern: {task_id: {"instance_id": ..., "model_patch": ..., "model_name_or_path": ...}}}
    """
    predictions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only process submitted runs
            if row.get("submitted", "").lower() != "true":
                continue
            
            pattern = row.get("pattern", "")
            task_id = row.get("task_id", "")
            model = row.get("model", "unknown")
            
            if not pattern or not task_id:
                continue
            
            if pattern not in predictions:
                predictions[pattern] = {}
            
            # Skip if already collected this task for this pattern
            if task_id in predictions[pattern]:
                continue
            
            # Find the trajectory file
            traj_path = find_trajectory_file(base_dir, pattern, task_id)
            if not traj_path:
                logger.warning(f"Trajectory not found for {pattern}/{task_id}")
                continue
            
            # Extract the patch
            patch = extract_patch_from_trajectory(traj_path)
            if not patch:
                logger.warning(f"No patch found in trajectory {traj_path}")
                predictions[pattern][task_id] = {
                    "instance_id": task_id,
                    "model_patch": None,
                    "model_name_or_path": model,
                }
                continue
            
            predictions[pattern][task_id] = {
                "instance_id": task_id,
                "model_patch": patch,
                "model_name_or_path": model,
            }
            
            logger.info(f"Collected prediction for {pattern}/{task_id}")
    
    return predictions


def create_prediction_file(predictions: Dict[str, Dict[str, Any]], output_path: Path) -> Path:
    """
    Create a predictions.json file in SWE-bench format.
    
    Args:
        predictions: {task_id: {"instance_id": ..., "model_patch": ..., "model_name_or_path": ...}}
        output_path: Directory to write the predictions file
        
    Returns:
        Path to the created predictions file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    preds_file = output_path / "predictions.json"
    
    # Convert to list format
    preds_list = list(predictions.values())
    
    with open(preds_file, 'w') as f:
        json.dump(preds_list, f, indent=2)
    
    logger.info(f"Created predictions file: {preds_file} with {len(preds_list)} predictions")
    return preds_file


def run_swebench_evaluation(
    predictions_path: Path,
    run_id: str,
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    max_workers: int = 4,
    timeout: int = 1800,
    report_dir: Optional[Path] = None,
    instance_ids: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """
    Run the SWE-bench evaluation harness.
    
    Returns:
        {instance_id: resolved} mapping
    """
    if report_dir is None:
        report_dir = predictions_path.parent
    
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--predictions_path", str(predictions_path),
        "--run_id", run_id,
        "--max_workers", str(max_workers),
        "--timeout", str(timeout),
        "--dataset_name", dataset,
        "--split", split,
        "--report_dir", str(report_dir),
    ]
    
    if instance_ids:
        cmd.extend(["--instance_ids"] + instance_ids)
    
    logger.info(f"Running SWE-bench evaluation: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout * max(1, len(instance_ids or [1])) + 3600,  # Extra hour buffer
        )
        
        if result.returncode != 0:
            logger.error(f"SWE-bench evaluation failed:\n{result.stderr}")
            return {}
        
        logger.info(f"SWE-bench evaluation stdout:\n{result.stdout}")
        
    except subprocess.TimeoutExpired:
        logger.error("SWE-bench evaluation timed out")
        return {}
    except Exception as e:
        logger.error(f"Error running SWE-bench evaluation: {e}")
        return {}
    
    # Parse the results from the report directory
    return parse_evaluation_results(report_dir, run_id)


def parse_evaluation_results(report_dir: Path, run_id: str) -> Dict[str, bool]:
    """
    Parse the evaluation results from the SWE-bench report files.
    
    Returns:
        {instance_id: resolved} mapping
    """
    results: Dict[str, bool] = {}
    
    # Look for the run report
    run_report_dir = report_dir / run_id
    if not run_report_dir.exists():
        run_report_dir = report_dir / f"run_{run_id}"
    
    if not run_report_dir.exists():
        # Try to find any report in the logs directory
        logs_dir = report_dir / "logs" / "run_evaluation" / run_id
        if logs_dir.exists():
            run_report_dir = logs_dir
    
    if not run_report_dir.exists():
        logger.warning(f"Report directory not found: {run_report_dir}")
        # Look for report.json directly in report_dir
        report_file = report_dir / "report.json"
        if report_file.exists():
            try:
                with open(report_file) as f:
                    report = json.load(f)
                for instance_id, data in report.items():
                    if isinstance(data, dict):
                        results[instance_id] = data.get("resolved", False)
            except Exception as e:
                logger.error(f"Error parsing report.json: {e}")
        return results
    
    # Look for individual report.json files
    for report_file in run_report_dir.glob("**/report.json"):
        try:
            with open(report_file) as f:
                report = json.load(f)
            
            for instance_id, data in report.items():
                if isinstance(data, dict):
                    results[instance_id] = data.get("resolved", False)
                    
        except Exception as e:
            logger.warning(f"Error parsing {report_file}: {e}")
    
    # Also check for combined results
    combined_report = run_report_dir / "results.json"
    if combined_report.exists():
        try:
            with open(combined_report) as f:
                report = json.load(f)
            
            # Handle different result formats
            if "resolved" in report:
                for instance_id in report.get("resolved", []):
                    results[instance_id] = True
            else:
                for instance_id, data in report.items():
                    if isinstance(data, dict):
                        results[instance_id] = data.get("resolved", False)
                    elif isinstance(data, bool):
                        results[instance_id] = data
                        
        except Exception as e:
            logger.warning(f"Error parsing combined report: {e}")
    
    logger.info(f"Parsed {len(results)} evaluation results, {sum(results.values())} resolved")
    return results


def update_csv_with_results(
    csv_path: Path,
    pattern: str,
    results: Dict[str, bool],
) -> int:
    """
    Update the CSV file with evaluation results.
    
    Args:
        csv_path: Path to the CSV file
        pattern: Pattern name to update
        results: {task_id: resolved} mapping
        
    Returns:
        Number of rows updated
    """
    # Read all rows
    rows = []
    fieldnames = []
    
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    
    # Ensure 'resolved' column exists
    if "resolved" not in fieldnames:
        # Find position after 'submitted'
        if "submitted" in fieldnames:
            idx = fieldnames.index("submitted") + 1
            fieldnames.insert(idx, "resolved")
        else:
            fieldnames.append("resolved")
    
    # Update rows
    updated = 0
    for row in rows:
        if row.get("pattern") == pattern:
            task_id = row.get("task_id", "")
            if task_id in results:
                row["resolved"] = str(results[task_id])
                updated += 1
    
    # Write back with file locking
    with open(csv_path, 'w', newline='') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    logger.info(f"Updated {updated} rows for pattern '{pattern}' in {csv_path}")
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions using SWE-bench harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate all patterns in the CSV
    python evaluate_predictions.py --csv results/experiment_runs.csv
    
    # Evaluate only specific patterns
    python evaluate_predictions.py --csv results/experiment_runs.csv --patterns baseline decomposed
    
    # Use custom output directory
    python evaluate_predictions.py --csv results/experiment_runs.csv --eval_dir results/evaluation
        """
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the experiment_runs.csv file"
    )
    
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help="Patterns to evaluate (default: all patterns in CSV)"
    )
    
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="results/evaluation",
        help="Directory for evaluation outputs (default: results/evaluation)"
    )
    
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Base directory for finding trajectory files (default: current directory)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="SWE-bench dataset name (default: princeton-nlp/SWE-bench_Lite)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel workers for evaluation (default: 4)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per instance in seconds (default: 1800)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only collect predictions, don't run evaluation"
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    base_dir = Path(args.base_dir)
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all predictions
    logger.info(f"Collecting predictions from {csv_path}...")
    all_predictions = collect_predictions(csv_path, base_dir)
    
    if not all_predictions:
        logger.warning("No predictions found to evaluate")
        sys.exit(0)
    
    # Filter patterns if specified
    if args.patterns:
        all_predictions = {
            p: preds for p, preds in all_predictions.items()
            if p in args.patterns
        }
    
    logger.info(f"Found predictions for patterns: {list(all_predictions.keys())}")
    
    # Process each pattern
    total_resolved = 0
    total_predictions = 0
    
    for pattern, predictions in all_predictions.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating pattern: {pattern} ({len(predictions)} predictions)")
        logger.info(f"{'='*60}")
        
        total_predictions += len(predictions)
        
        # Create prediction file
        pattern_eval_dir = eval_dir / pattern
        preds_file = create_prediction_file(predictions, pattern_eval_dir)
        
        if args.dry_run:
            logger.info(f"[DRY RUN] Would evaluate {preds_file}")
            continue
        
        # Run evaluation
        run_id = f"{pattern}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = run_swebench_evaluation(
            predictions_path=preds_file,
            run_id=run_id,
            dataset=args.dataset,
            split=args.split,
            max_workers=args.max_workers,
            timeout=args.timeout,
            report_dir=pattern_eval_dir,
            instance_ids=list(predictions.keys()),
        )
        
        if results:
            # Update CSV with results
            update_csv_with_results(csv_path, pattern, results)
            resolved_count = sum(results.values())
            total_resolved += resolved_count
            logger.info(f"Pattern {pattern}: {resolved_count}/{len(predictions)} resolved")
        else:
            logger.warning(f"No evaluation results for pattern {pattern}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total patterns evaluated: {len(all_predictions)}")
    logger.info(f"Total predictions: {total_predictions}")
    if not args.dry_run:
        logger.info(f"Total resolved: {total_resolved}")
        logger.info(f"Resolution rate: {total_resolved/max(1, total_predictions)*100:.1f}%")


if __name__ == "__main__":
    main()
