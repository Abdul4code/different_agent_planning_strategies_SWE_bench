# Agent Planning Strategies for SWE-bench

Comparison of different LLM planning strategies on SWE-bench tasks using mini-swe-agent.

## Planning Strategies

| Strategy | Description |
|----------|-------------|
| **baseline** | Default mini-swe-agent (direct execution) |
| **decomposed** | Task decomposition into subgoals |
| **multiplan** | Generate multiple plans, select best |
| **external** | External planner-aided reasoning |
| **reflection** | Iterative reflection and refinement |
| **memory** | Memory-augmented with retrieval |

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/Abdul4code/different_agent_planning_strategies_SWE_bench.git

```bash
# Set API key (If you are using remote API call else ignore this instruction)
export OPENROUTER_API_KEY="your-key"  # OpenRouter (recommended)
export OPENAI_API_KEY="your-key"      # OpenAI
export ANTHROPIC_API_KEY="your-key"   # Anthropic

# Disable cost tracking errors (recommended)
export MSWEA_COST_TRACKING='ignore_errors'

# Pull Docker images (required for SWE-bench)
docker pull docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-17139:latest

# (Recommended) Pre-pull all images for your tasks file to avoid per-task timeouts
python prepull_swebench_images.py --tasks tasks_50.json --max_workers 4
```

### 2. Run Experiments with Orchestrator

The experiment orchestrator runs all planning strategies across your tasks with parallel execution, metrics collection, and resume capability.


**Usage:**

```bash
python experiment_orchestrator.py \
    --tasks tasks_50.json \
    --patterns baseline decomposed multiplan \
    --model openrouter/openai/gpt-4o-mini \
    --timeout_s 1800 \
    --max_workers 4 \
    --out results/experiment_runs.csv \
    --resume \
    --use_codecarbon
```


### 3. Evaluate Results

After running experiments, evaluate with the SWE-bench harness:

```bash
python evaluate_predictions.py --csv results/experiment_runs.csv
```

## Orchestrator CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tasks` | Path to tasks JSON file | *required* |
| `--model` | LLM model name with provider prefix | *required* |
| `--patterns` | Planning patterns to evaluate | all |
| `--out` | Output CSV file path | `results/experiment_runs.csv` |
| `--seeds` | Random seeds for reproducibility | `[42]` |
| `--timeout_s` | Timeout per task in seconds | `1800` |
| `--max_workers` | Maximum parallel workers | `1` |
| `--resume` | Skip already-completed runs | `false` |
| `--use_codecarbon` | Enable energy tracking | `false` |
| `--verbose`, `-v` | Enable verbose logging | `false` |

## Supported Models

Uses LiteLLM, supporting multiple providers:

| Provider | Format | Example |
|----------|--------|---------|
| **OpenRouter** | `openrouter/provider/model` | `openrouter/openai/gpt-4o-mini` |
| **OpenAI** | `openai/model` | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| **Anthropic** | `anthropic/model` | `anthropic/claude-sonnet-4-5-20250929` |
| **Ollama (local)** | `ollama/model` | `ollama/qwen2.5:7b` |
| **Together** | `together_ai/model` | `together_ai/meta-llama/Llama-3-70b-chat-hf` |
| **Groq** | `groq/model` | `groq/llama3-70b-8192` |

### API Keys

```bash
export OPENROUTER_API_KEY=sk-or-...   # OpenRouter
export OPENAI_API_KEY=sk-...           # OpenAI
export ANTHROPIC_API_KEY=sk-ant-...    # Anthropic
export TOGETHER_API_KEY=...            # Together
export GROQ_API_KEY=...                # Groq
# Ollama - no key needed, just run: ollama serve
```

## Output

Results are saved to:
- `results/experiment_runs.csv` - Metrics for all runs (tokens, time, memory, status)
- `results/manifest.json` - Experiment metadata
- `results/orchestrator_runs/{pattern}/` - Per-pattern prediction files

### CSV Fields

| Column | Type | Description | How It's Computed |
|--------|------|-------------|-------------------|
| `run_id` | `string` | Unique identifier for each run | Generated using `uuid.uuid4()` at the start of each task run |
| `started_at` | `string` | ISO 8601 timestamp when run started | Captured via `datetime.now(timezone.utc).isoformat()` |
| `pattern` | `string` | Planning strategy used | One of: `baseline`, `decomposed`, `external`, `multiplan`, `memory`, `reflection` |
| `task_id` | `string` | SWE-bench task identifier | From the input `tasks_50.json` file (e.g., `sympy__sympy-23191`) |
| `model` | `string` | LLM model name with provider prefix | CLI argument `--model` (e.g., `ollama/qwen2.5:7b`) |
| `seed` | `int` | Random seed for reproducibility | CLI argument `--seeds` (default: `42`) |
| `submitted` | `bool` | Whether agent produced a patch | `True` if agent called `MINI_SWE_AGENT_FINAL_OUTPUT` with a patch |
| `resolved` | `bool` | Whether patch passed SWE-bench tests | Set by `evaluate_predictions.py` after running the SWE-bench harness. Initially `False` |
| `runtime_s` | `float` | Total wall-clock execution time | `time.perf_counter()` difference between start and end |
| `prompt_tokens` | `int` | Total input tokens sent to LLM | Accumulated via `LLMMetrics.on_llm_call()` across all LLM calls |
| `completion_tokens` | `int` | Total output tokens from LLM | Accumulated via `LLMMetrics.on_llm_call()` across all LLM calls |
| `total_tokens` | `int` | Sum of prompt + completion tokens | `prompt_tokens + completion_tokens` |
| `llm_calls` | `int` | Number of LLM API calls made | Counter incremented each time model is queried |
| `energy_kwh` | `float` | Energy consumed in kilowatt-hours | Parsed from CodeCarbon's `emissions.csv` (only if `--use_codecarbon`) |
| `co2_kg` | `float` | CO₂ emissions in kilograms | Returned by `EmissionsTracker.stop()` from CodeCarbon |
| `peak_rss_mb` | `float` | Peak memory usage in MB | `PeakMemorySampler` samples process RSS every 0.1s and tracks max |
| `error_type` | `string` | Error classification if run failed | Categorized as: `timeout`, `submitted`, specific exception names, or empty |

#### Key Notes

- **submitted vs resolved**: `submitted=True` means the agent generated a patch; `resolved=True` means it actually fixed the bug (verified by SWE-bench harness)
- **Token metrics (0 values)**: Indicates timeout before LLM calls, Docker failure, or metrics not captured
- **error_type values**: `timeout` (exceeded 30 min), `submitted` (success), `TimeoutExpired` (Docker timeout), or empty

## Project Structure

```
├── experiment_orchestrator.py   # Main orchestrator
├── evaluate_predictions.py      # SWE-bench evaluation
├── run_agent_wrapper.py         # Single-task runner
├── metrics.py                   # Metrics collection
├── tasks_50.json                # Task definitions
├── mini-swe-agent/              # Baseline agent
├── mini-swe-agent-decomposed/   # Decomposed planning
├── mini-swe-agent-multiplan/    # Multi-plan selection
├── mini-swe-agent-external/     # External planner
├── mini-swe-agent-reflection/   # Reflection-based
├── mini-swe-agent-memory/       # Memory-augmented
└── results/                     # Output directory
```