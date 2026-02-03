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
# Set API key (choose one)
export OPENROUTER_API_KEY="your-key"  # OpenRouter (recommended)
export OPENAI_API_KEY="your-key"      # OpenAI
export ANTHROPIC_API_KEY="your-key"   # Anthropic

# Disable cost tracking errors (recommended)
export MSWEA_COST_TRACKING='ignore_errors'

# Pull Docker images (required for SWE-bench)
docker pull docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-17139:latest
```

### 2. Run Experiments with Orchestrator

The experiment orchestrator runs all planning strategies across your tasks with parallel execution, metrics collection, and resume capability.


**Usage:**

```bash
python experiment_orchestrator.py \
    --tasks tasks_50.json \
    --patterns baseline decomposed multiplan \
    --model openrouter/openai/gpt-4o-mini \
    --max_workers 4 \
    --out results/experiment_runs.csv
    --resume
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

| Field | Description |
|-------|-------------|
| `run_id` | Unique identifier |
| `pattern` | Planning strategy used |
| `task_id` | SWE-bench task ID |
| `model` | Model name |
| `submitted` | Whether a patch was produced |
| `resolved` | Whether patch passed tests (after evaluation) |
| `wall_time_s` | Total execution time |
| `tokens_prompt` | Input tokens used |
| `tokens_completion` | Output tokens used |
| `llm_calls` | Number of LLM API calls |
| `peak_mem_mb` | Peak memory usage |

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