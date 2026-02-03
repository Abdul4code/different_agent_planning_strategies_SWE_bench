# Multi-Agent Planning Strategies for SWE-bench

Comparison of different LLM planning strategies on SWE-bench tasks using mini-swe-agent.

## Planning Strategies Implemented

| Strategy | Description | Paper Formulation |
|----------|-------------|-------------------|
| **Baseline** | Default mini-swe-agent | Direct execution |
| **Decomposed** | Task decomposition | `g₀, g₁, ..., gₙ = decompose(E, g; Θ, P)` |
| **Multiplan** | Multi-plan selection | `P = plan(E, g; Θ, P)`, `p* = select(E, g, P; Θ, F)` |
| **External** | External planner-aided | `h = formalize(E, g; Θ, P)`, `p = plan(E, g, h; Φ)` |
| **Reflection** | Reflection & refinement | `p₀ = plan()`, `rᵢ = reflect()`, `pᵢ₊₁ = refine()` |
| **Memory** | Memory-augmented | `m = retrieve(E, g; M)`, `p = plan(E, g, m; Θ, P)` |

## Quick Start

### 1. Setup

```bash
# Pull Docker images (required for SWE-bench)
docker pull docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-17139:latest

# Set API key (choose one)
export OPENROUTER_API_KEY="your-key"  # OpenRouter
export OPENAI_API_KEY="your-key"      # OpenAI
export ANTHROPIC_API_KEY="your-key"   # Anthropic
```

### 2. Run Agents

All agents use the same CLI format:

```bash
python3 run_agent_wrapper.py <agent> --tasks tasks_50.json --out results/<agent> --limit 1 --model <model>
```

**Examples:**

```bash
# Baseline
python3 run_agent_wrapper.py baseline --tasks tasks_50.json --out results/baseline --limit 1 --model openrouter/openai/gpt-4o-mini

# Decomposed
python3 run_agent_wrapper.py decomposed --tasks tasks_50.json --out results/decomposed --limit 1 --model openrouter/openai/gpt-4o-mini

# Multiplan
python3 run_agent_wrapper.py multiplan --tasks tasks_50.json --out results/multiplan --limit 1 --model openrouter/openai/gpt-4o-mini

# External planner
python3 run_agent_wrapper.py external --tasks tasks_50.json --out results/external --limit 1 --model openrouter/openai/gpt-4o-mini

# Reflection
python3 run_agent_wrapper.py reflection --tasks tasks_50.json --out results/reflection --limit 1 --model openrouter/openai/gpt-4o-mini

# Memory-augmented
python3 run_agent_wrapper.py memory --tasks tasks_50.json --out results/memory --limit 1 --model openrouter/openai/gpt-4o-mini
```

## Supported Models

The framework uses LiteLLM, supporting many providers:

| Provider | Format | Example |
|----------|--------|---------|
| **OpenAI** | `openai/model` | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| **Anthropic** | `anthropic/model` | `anthropic/claude-3-5-sonnet-20241022` |
| **Ollama (local)** | `ollama/model` | `ollama/qwen2.5:7b`, `ollama/llama3:8b` |
| **OpenRouter** | `openrouter/provider/model` | `openrouter/openai/gpt-4o-mini` |
| **Together** | `together_ai/model` | `together_ai/meta-llama/Llama-3-70b-chat-hf` |
| **Groq** | `groq/model` | `groq/llama3-70b-8192` |

### API Keys

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# OpenRouter
export OPENROUTER_API_KEY=sk-or-...

# Together
export TOGETHER_API_KEY=...

# Groq
export GROQ_API_KEY=...

# Ollama - no key needed, just run locally
ollama serve
```

### Using Local Models (Ollama)

```bash
# Install and start Ollama
ollama serve

# Pull a model
ollama pull qwen2.5:7b
ollama list  # See installed models

# Run with local model
python3 run_agent_wrapper.py baseline --tasks tasks_50.json --out results/baseline --limit 1 --model ollama/qwen2.5:7b
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--tasks` | Path to tasks JSON file |
| `--out`, `-o` | Output directory |
| `--limit` | Limit number of tasks to run |
| `--model`, `-m` | Model to use |
| `--workers`, `-w` | Number of parallel workers |

## Results

Results are saved to the output directory with:
- `preds.json` - Predictions for evaluation
- `<instance_id>/<instance_id>.traj.json` - Full trajectory for each task