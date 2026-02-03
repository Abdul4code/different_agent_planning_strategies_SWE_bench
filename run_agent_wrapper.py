#!/usr/bin/env python3
"""Wrapper to run agents with correct imports."""

import sys
from pathlib import Path

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
    # The agent's task_decomposition.py already adds baseline path itself for DefaultAgent
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

