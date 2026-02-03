# Task Decomposition Agent

A planning strategy that decomposes complex problems into sequential sub-tasks before execution.

## Planning Strategy

The Task Decomposition Agent implements hierarchical task decomposition:

```
g₀, g₁, ..., gₙ = decompose(E, g; Θ, P)
```

Where:
- **E** = Environment (codebase, files, tests)
- **g** = Main goal/problem statement
- **Θ** = Language model
- **P** = Prompt/instructions
- **g₀...gₙ** = Ordered sub-goals

## How It Works

### 1. Decomposition Phase
Before execution begins, the agent:
1. Analyzes the main problem statement
2. Uses the LLM to break it down into 3-5 actionable sub-goals
3. Orders sub-goals sequentially

### 2. Enhanced Execution
The decomposed sub-goals are injected into the task context:
```
TASK DECOMPOSITION - Follow these sub-goals in order:
============================================================
Sub-goal 1: Reproduce the issue with a test case
Sub-goal 2: Locate the source of the bug in the codebase
Sub-goal 3: Implement the fix while preserving existing behavior
Sub-goal 4: Verify the fix passes all tests
============================================================
```

### 3. Baseline Execution
After decomposition, execution proceeds using the standard agent loop, but with sub-goal guidance.

## Differences from Baseline

| Aspect | Baseline | Decomposed |
|--------|----------|------------|
| **Planning** | Direct problem-to-action | Problem → Sub-goals → Actions |
| **Structure** | Single goal | Hierarchical sub-goals |
| **Guidance** | Task only | Task + ordered sub-goals |
| **LLM Calls** | During execution only | 1 extra call for decomposition |

## Key Benefits

- **Clearer Direction**: Sub-goals provide intermediate milestones
- **Reduced Complexity**: Smaller, focused steps are easier to execute
- **Progress Tracking**: Natural checkpoints for the agent
- **Error Isolation**: Failures can be traced to specific sub-goals

## Example Decomposition

**Original Task**: "Fix the TypeError in the parsing module"

**Decomposed Sub-goals**:
1. Reproduce the TypeError with a minimal test case
2. Trace the error to identify the root cause
3. Analyze the expected vs actual types
4. Implement type checking or conversion fix
5. Verify all existing tests still pass

## Configuration

```python
from minisweagent.agents.task_decomposition import TaskDecompositionAgent

agent = TaskDecompositionAgent(
    model=model,
    env=env,
    system_template=system_template,
    instance_template=instance_template,
    # ... other config
)
```

## References

- Hierarchical Task Networks (Erol et al., 1994)
- Decomposition in LLM Planning (Wei et al., 2022)
