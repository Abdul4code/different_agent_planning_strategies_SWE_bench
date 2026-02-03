# External Planner-Aided Agent

A planning strategy that separates formalization (LLM) from planning (algorithmic).

## Planning Strategy

The External Planner Agent implements a two-phase approach:

```
h = formalize(E, g; Θ, P)    # LLM formalizes the task
p = plan(E, g, h; Φ)         # External planner generates plan
```

Where:
- **Θ** = Language model (for formalization only)
- **h** = Formalized/structured task representation
- **Φ** = External planner module (non-LLM algorithm)
- **p** = Generated plan

## How It Works

### 1. Formalization Phase (LLM)
The LLM converts the natural language problem into a structured representation:

```python
FormalizedTask:
    goal: str                           # Primary objective
    actions: List[Dict]                 # Named actions with descriptions
    dependencies: List[Tuple[str, str]] # (action_i, action_j) = i before j
    constraints: List[str]              # Any constraints to satisfy
```

**Example Formalization**:
```
ACTION: analyze | Analyze the error message | AFTER: none
ACTION: locate | Find the relevant file | AFTER: analyze  
ACTION: fix | Implement the fix | AFTER: locate
ACTION: test | Run tests to verify | AFTER: fix
```

### 2. Planning Phase (Algorithmic)
The external planner uses classical algorithms to generate an optimal plan:

- **Topological Sort**: Respects action dependencies
- **Kahn's Algorithm**: Orders actions based on dependency graph
- **Priority-based**: Considers action priorities when ordering

### 3. Execution
The algorithmically-generated plan guides the agent:
```
PLAN (from external planner Φ):
============================================================
1. Analyze the error message
2. Find the relevant file
3. Implement the fix
4. Run tests to verify
============================================================
```

## Differences from Baseline

| Aspect | Baseline | External Planner |
|--------|----------|------------------|
| **Planning** | LLM does everything | LLM formalizes, algorithm plans |
| **Structure** | Unstructured | Formal representation with dependencies |
| **Ordering** | LLM-determined | Algorithmically optimal |
| **Guarantees** | None | Dependency satisfaction guaranteed |

## Key Benefits

- **Separation of Concerns**: LLM for NL understanding, algorithm for planning
- **Guaranteed Ordering**: Topological sort ensures valid action sequences
- **Efficiency**: Classical algorithms are more efficient than LLM for ordering
- **Reproducibility**: Same formalization → same plan

## Planning Strategies

The external planner supports multiple strategies:

| Strategy | Description |
|----------|-------------|
| `dependency_order` | Topological sort respecting all dependencies |
| `greedy` | Order by priority and precondition count |
| `topological` | Alias for dependency-based ordering |

## Configuration

```python
from minisweagent.agents.external_planner import ExternalPlannerAgent

agent = ExternalPlannerAgent(
    model=model,
    env=env,
    planner_strategy="dependency_order",  # Planning algorithm
    system_template=system_template,
    # ... other config
)
```

## References

- LLM+P: Empowering Large Language Models with Optimal Planning Proficiency (Liu et al., 2023)
- ProgPrompt: Program Generation for Situated Robot Task Planning (Singh et al., 2023)
