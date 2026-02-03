# Multi-Plan Selection Agent

A planning strategy that generates multiple alternative plans and selects the optimal one.

## Planning Strategy

The Multi-Plan Agent implements plan generation with search-based selection:

```
P = {p₁, p₂, ..., pₙ} = plan(E, g; Θ, P)    # Generate n plans
p* = select(E, g, P; Θ, F)                   # Select best plan
```

Where:
- **P** = Set of alternative plans
- **n** = Number of plans to generate (default: 3)
- **F** = Selection/search strategy
- **p*** = Selected optimal plan

## How It Works

### 1. Plan Generation Phase
The agent generates n distinct approaches to solve the problem:

```
PLAN 1: Direct Fix Strategy
1. Identify the exact error location
2. Apply minimal targeted fix
3. Run existing tests

PLAN 2: Refactoring Approach
1. Analyze broader context
2. Refactor affected function
3. Add type hints for safety
4. Comprehensive testing

PLAN 3: Test-Driven Strategy
1. Write failing test for the bug
2. Implement fix to pass test
3. Ensure no regressions
```

### 2. Selection Phase
Each plan is evaluated using selection strategy F:

**LLM Evaluation** (`llm_evaluate`):
- Each plan scored on: Correctness, Feasibility, Efficiency, Completeness
- Scores aggregated to select highest-scoring plan

**LLM Comparison** (`llm_compare`):
- Plans compared pairwise
- Tournament-style selection

**Beam Search** (`beam`):
- Maintains top-k candidates
- Prunes based on evaluation scores

### 3. Execution
Selected plan guides the agent:
```
SELECTED PLAN (p*) - Direct Fix Strategy:
============================================================
1. Identify the exact error location
2. Apply minimal targeted fix  
3. Run existing tests
============================================================
Selection: Scored 8.5/10 (highest feasibility and efficiency)
============================================================
```

## Differences from Baseline

| Aspect | Baseline | Multi-Plan |
|--------|----------|------------|
| **Plans** | Single implicit plan | n explicit alternatives |
| **Diversity** | One approach | Multiple strategies |
| **Selection** | None | LLM-based evaluation |
| **Exploration** | Depth-first | Breadth-first (then select) |

## Key Benefits

- **Exploration**: Multiple approaches considered before committing
- **Diversity**: Different strategies (direct, refactor, test-driven)
- **Quality Selection**: LLM evaluates feasibility before execution
- **Risk Mitigation**: Can detect and avoid flawed approaches early

## Selection Strategies

| Strategy | Method | Best For |
|----------|--------|----------|
| `llm_evaluate` | Score each plan independently | General use |
| `llm_compare` | Pairwise comparison tournament | When relative ranking matters |
| `beam` | Beam search with scoring | When exploring many plans |

## Evaluation Criteria

Plans are evaluated on:

1. **Correctness** (0-10): Will this fix the actual problem?
2. **Feasibility** (0-10): Is this achievable with available information?
3. **Efficiency** (0-10): Is this approach direct and focused?
4. **Completeness** (0-10): Does this cover all aspects?

## Configuration

```python
from minisweagent.agents.multiplan import MultiPlanAgent

agent = MultiPlanAgent(
    model=model,
    env=env,
    num_plans=3,                        # Number of alternatives
    selection_strategy="llm_compare",   # Selection method
    system_template=system_template,
    # ... other config
)
```

## References

- Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023)
- Plan Selection Strategies for LLM Agents (Zhao et al., 2023)
