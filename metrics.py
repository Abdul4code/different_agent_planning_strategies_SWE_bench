"""
Metrics collection and recording utilities for SWE-bench experiment orchestration.

This module provides:
- LLMMetrics: Track token usage and LLM call counts
- PeakMemorySampler: Sample and track peak RSS memory usage
- TaskRunRecord: Dataclass representing a single experiment run
- CSVWriter: Thread/process-safe CSV writing with file locking
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict, List

import psutil


# =============================================================================
# LLM Metrics Tracking
# =============================================================================

class LLMMetrics:
    """
    Track LLM token usage and call counts.
    
    Thread-safe implementation for accumulating metrics across multiple LLM calls.
    
    Usage:
        metrics = LLMMetrics()
        metrics.on_llm_call(prompt_tokens=100, completion_tokens=50)
        metrics.on_llm_call(prompt_tokens=200, completion_tokens=100)
        print(metrics.prompt_tokens)  # 300
        print(metrics.completion_tokens)  # 150
        print(metrics.total_tokens)  # 450
        print(metrics.llm_calls)  # 2
    """
    
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._llm_calls: int = 0
    
    def on_llm_call(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """
        Record a single LLM call with its token usage.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        with self._lock:
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._llm_calls += 1
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update metrics from a dictionary (e.g., parsed from JSON output).
        
        Recognizes keys: prompt_tokens, completion_tokens, llm_calls, total_tokens
        """
        with self._lock:
            if "prompt_tokens" in data:
                self._prompt_tokens += int(data["prompt_tokens"])
            if "completion_tokens" in data:
                self._completion_tokens += int(data["completion_tokens"])
            if "llm_calls" in data:
                self._llm_calls += int(data["llm_calls"])
    
    @property
    def prompt_tokens(self) -> int:
        with self._lock:
            return self._prompt_tokens
    
    @property
    def completion_tokens(self) -> int:
        with self._lock:
            return self._completion_tokens
    
    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._prompt_tokens + self._completion_tokens
    
    @property
    def llm_calls(self) -> int:
        with self._lock:
            return self._llm_calls
    
    def to_dict(self) -> Dict[str, int]:
        """Return metrics as a dictionary."""
        with self._lock:
            return {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._prompt_tokens + self._completion_tokens,
                "llm_calls": self._llm_calls,
            }
    
    def reset(self) -> None:
        """Reset all metrics to zero."""
        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._llm_calls = 0


# =============================================================================
# Peak Memory Sampler
# =============================================================================

class PeakMemorySampler:
    """
    Sample and track peak RSS (Resident Set Size) memory usage.
    
    Runs a background thread that samples the current process's RSS memory
    at a configurable interval and tracks the peak value.
    
    Usage:
        sampler = PeakMemorySampler(interval_s=0.1)
        sampler.start()
        # ... run workload ...
        sampler.stop()
        print(f"Peak RSS: {sampler.peak_rss_mb:.2f} MB")
    """
    
    def __init__(self, interval_s: float = 0.1, pid: Optional[int] = None) -> None:
        """
        Initialize the memory sampler.
        
        Args:
            interval_s: Sampling interval in seconds (default: 0.1s)
            pid: Process ID to monitor (default: current process)
        """
        self.interval_s = interval_s
        self.pid = pid or os.getpid()
        self._peak_rss_bytes: int = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[psutil.Process] = None
    
    def _sample_loop(self) -> None:
        """Background thread loop that samples memory."""
        try:
            self._process = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        
        while not self._stop_event.is_set():
            try:
                # Get memory info including children
                mem_info = self._process.memory_info()
                current_rss = mem_info.rss
                
                # Also sample children processes
                try:
                    for child in self._process.children(recursive=True):
                        try:
                            child_mem = child.memory_info()
                            current_rss += child_mem.rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                with self._lock:
                    if current_rss > self._peak_rss_bytes:
                        self._peak_rss_bytes = current_rss
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            
            self._stop_event.wait(self.interval_s)
    
    def start(self) -> "PeakMemorySampler":
        """Start the background sampling thread. Returns self for chaining."""
        if self._thread is not None and self._thread.is_alive():
            return self
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self
    
    def stop(self) -> None:
        """Stop the background sampling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    @property
    def peak_rss_mb(self) -> float:
        """Return peak RSS in megabytes."""
        with self._lock:
            return self._peak_rss_bytes / (1024 * 1024)
    
    @property
    def peak_rss_bytes(self) -> int:
        """Return peak RSS in bytes."""
        with self._lock:
            return self._peak_rss_bytes
    
    def reset(self) -> None:
        """Reset peak memory tracking."""
        with self._lock:
            self._peak_rss_bytes = 0
    
    def __enter__(self) -> "PeakMemorySampler":
        """Context manager entry - starts sampling."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops sampling."""
        self.stop()


# =============================================================================
# Task Run Record Dataclass
# =============================================================================

@dataclass
class TaskRunRecord:
    """
    Represents a single experiment run record.
    
    All fields correspond to CSV columns in the output file.
    """
    # Identifiers
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Task configuration
    pattern: str = ""
    task_id: str = ""
    model: str = ""
    seed: int = 0
    
    # Results
    submitted: bool = False  # True if agent submitted a patch (not verified as correct)
    resolved: bool = False   # True if SWE-bench harness verified the patch is correct
    runtime_s: float = 0.0
    
    # LLM metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    
    # Energy metrics (optional)
    energy_kwh: Optional[float] = None
    co2_kg: Optional[float] = None
    
    # Memory metrics
    peak_rss_mb: float = 0.0
    
    # Error tracking
    error_type: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for CSV writing."""
        return asdict(self)
    
    @classmethod
    def csv_columns(cls) -> List[str]:
        """Return ordered list of CSV column names."""
        return [
            "run_id",
            "started_at",
            "pattern",
            "task_id",
            "model",
            "seed",
            "submitted",
            "resolved",
            "runtime_s",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "llm_calls",
            "energy_kwh",
            "co2_kg",
            "peak_rss_mb",
            "error_type",
        ]
    
    def to_csv_row(self) -> Dict[str, Any]:
        """Convert to dict suitable for csv.DictWriter."""
        d = self.to_dict()
        # Convert None to empty string for CSV
        for key in ["energy_kwh", "co2_kg"]:
            if d[key] is None:
                d[key] = ""
        # Convert bool to string for consistency
        d["resolved"] = str(d["resolved"])
        return d


# =============================================================================
# CSV Writer with File Locking
# =============================================================================

class CSVWriter:
    """
    Thread/process-safe CSV writer with file locking.
    
    Supports concurrent appends from multiple processes using fcntl file locking.
    Ensures the CSV always has exactly one header row.
    
    Usage:
        writer = CSVWriter("results.csv")
        writer.write_record(record)
    """
    
    def __init__(self, filepath: str | Path, columns: Optional[List[str]] = None) -> None:
        """
        Initialize the CSV writer.
        
        Args:
            filepath: Path to the CSV file
            columns: Column names (default: TaskRunRecord.csv_columns())
        """
        self.filepath = Path(filepath)
        self.columns = columns or TaskRunRecord.csv_columns()
        self._local_lock = threading.Lock()
        
        # Ensure parent directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file with header if it doesn't exist
        self._ensure_header()
    
    def _ensure_header(self) -> None:
        """Ensure the CSV file exists and has a header row."""
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    writer = csv.DictWriter(f, fieldnames=self.columns)
                    writer.writeheader()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def write_record(self, record: TaskRunRecord) -> None:
        """
        Write a single record to the CSV file.
        
        Uses file locking for process-safety.
        """
        with self._local_lock:
            with open(self.filepath, "a", newline="") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    writer = csv.DictWriter(f, fieldnames=self.columns)
                    writer.writerow(record.to_csv_row())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def write_records(self, records: List[TaskRunRecord]) -> None:
        """Write multiple records to the CSV file."""
        with self._local_lock:
            with open(self.filepath, "a", newline="") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    writer = csv.DictWriter(f, fieldnames=self.columns)
                    for record in records:
                        writer.writerow(record.to_csv_row())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def read_existing_keys(self) -> set[tuple[str, str, int, str]]:
        """
        Read existing (pattern, task_id, seed, model) tuples from the CSV.
        
        Used for resume functionality.
        
        Returns:
            Set of (pattern, task_id, seed, model) tuples
        """
        existing = set()
        if not self.filepath.exists():
            return existing
        
        with open(self.filepath, "r", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        key = (
                            row.get("pattern", ""),
                            row.get("task_id", ""),
                            int(row.get("seed", 0)),
                            row.get("model", ""),
                        )
                        existing.add(key)
                    except (ValueError, KeyError):
                        continue
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        return existing


# =============================================================================
# Utility Functions
# =============================================================================

def parse_metrics_from_stdout(stdout: str) -> Dict[str, Any]:
    """
    Parse metrics from subprocess stdout.
    
    Looks for lines matching: METRICS_JSON={...}
    
    Args:
        stdout: Standard output from subprocess
        
    Returns:
        Dictionary of parsed metrics, or empty dict if not found
    """
    metrics: Dict[str, Any] = {}
    
    # Look for METRICS_JSON={...} pattern - handle nested braces
    for line in stdout.split('\n'):
        if line.startswith('METRICS_JSON='):
            json_str = line[len('METRICS_JSON='):]
            try:
                data = json.loads(json_str)
                metrics.update(data)
            except json.JSONDecodeError:
                continue
    
    return metrics


def parse_codecarbon_csv(csv_path: str | Path) -> Dict[str, float]:
    """
    Parse CodeCarbon emissions CSV to extract energy metrics.
    
    Args:
        csv_path: Path to the CodeCarbon emissions.csv file
        
    Returns:
        Dictionary with 'energy_kwh' and 'co2_kg' if available
    """
    result: Dict[str, float] = {}
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        return result
    
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if rows:
                # Get the last row (most recent measurement)
                last_row = rows[-1]
                
                # energy_consumed is in kWh
                if "energy_consumed" in last_row:
                    result["energy_kwh"] = float(last_row["energy_consumed"])
                
                # emissions is in kg CO2
                if "emissions" in last_row:
                    result["co2_kg"] = float(last_row["emissions"])
    except (ValueError, KeyError, csv.Error):
        pass
    
    return result


def create_manifest(
    cli_args: Dict[str, Any],
    output_path: str | Path,
) -> Dict[str, Any]:
    """
    Create a manifest dictionary for reproducibility.
    
    Args:
        cli_args: Command-line arguments as dictionary
        output_path: Path where manifest will be saved
        
    Returns:
        Manifest dictionary
    """
    import platform
    import subprocess as sp
    import sys
    
    manifest: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cli_args": cli_args,
    }
    
    # Try to get git commit hash
    try:
        result = sp.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            manifest["git_commit"] = result.stdout.strip()
    except (sp.TimeoutExpired, FileNotFoundError):
        manifest["git_commit"] = None
    
    # Try to get git dirty status
    try:
        result = sp.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            manifest["git_dirty"] = len(result.stdout.strip()) > 0
    except (sp.TimeoutExpired, FileNotFoundError):
        manifest["git_dirty"] = None
    
    return manifest


def save_manifest(manifest: Dict[str, Any], output_path: str | Path) -> None:
    """Save manifest to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
