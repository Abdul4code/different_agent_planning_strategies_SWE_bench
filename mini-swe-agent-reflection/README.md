# Reflection and Refinement Agent

A planning strategy that iteratively improves plans through self-critique and refinement.

## Planning Strategy

The Reflection Agent implements iterative plan improvement:

```
p₀ = plan(E, g; Θ, P)                    # Initial plan
rᵢ = reflect(E, g, pᵢ; Θ, P)             # Reflect on plan
pᵢ₊₁ = refine(E, g, pᵢ, rᵢ; Θ, P)        # Refine based on reflection
```

Where:
- **p₀** = Initial plan
- **rᵢ** = Reflection on plan pᵢ (identifying issues)
- **pᵢ₊₁** = Refined plan incorporating reflection
- **i** = Number of reflection iterations

## How It Works

### 1. Initial Planning
Generate first plan p₀:
```
Initial Plan (p₀):
1. Find the error in parsing.py
2. Fix the type handling
3. Run tests
```

### 2. Reflection Phase
The LLM critically analyzes the plan:
```
Reflection (r₀):
- What could go wrong: The error might be in a dependency, not parsing.py
- Missing steps: No reproduction step to verify the bug
- Incorrect assumptions: Assumes single file is responsible
- Potential failure: Fix might break other functionality
```

### 3. Refinement Phase
Improve the plan based on reflection:
```
Refined Plan (p₁):
1. Reproduce the error with the provided test case
2. Trace the error stack to find root cause
3. Check dependencies of the failing function
4. Implement fix with backward compatibility
5. Verify fix doesn't break other tests
6. Test edge cases related to the change
```

### 4. Iteration
The reflect-refine loop can run multiple times:
```
p₀ → r₀ → p₁ → r₁ → p₂ → ... → pₙ (final)
```

## Differences from Baseline

| Aspect | Baseline | Reflection |
|--------|----------|------------|
| **Planning** | Single-shot | Iterative refinement |
| **Self-Critique** | None | Explicit reflection phase |
| **Improvement** | None pre-execution | Plan improved before execution |
| **Robustness** | May have gaps | Gaps identified and addressed |

## Key Benefits

- **Self-Improvement**: Plans get better through critique
- **Gap Detection**: Missing steps identified before execution
- **Assumption Checking**: Flawed assumptions caught early
- **Robustness**: Refined plans handle more edge cases

## Reflection Criteria

The reflection phase analyzes:

1. **Potential Failures**: What could go wrong?
2. **Missing Steps**: What's not covered?
3. **Incorrect Assumptions**: What might be wrong?
4. **Edge Cases**: What special cases need handling?

## Iteration Dynamics

| Rounds | Trade-off |
|--------|-----------|
| 0 | No reflection (baseline equivalent) |
| 1 | Good balance of quality vs cost |
| 2+ | Diminishing returns, higher token cost |

Recommended: **1 reflection round** for most tasks.

## Example Iteration

**Initial (p₀)**:
> Fix the bug in the validate function

**Reflection (r₀)**:
> - Doesn't specify how to find the bug
> - No verification step
> - Assumes bug is in validation logic

**Refined (p₁)**:
> 1. Analyze the error message and stack trace
> 2. Reproduce the bug with minimal input
> 3. Locate the exact line causing the issue
> 4. Understand the intended vs actual behavior
> 5. Implement fix preserving other functionality
> 6. Add test case for this specific bug
> 7. Run full test suite

## Configuration

```python
from minisweagent.agents.reflection import ReflectionAgent

agent = ReflectionAgent(
    model=model,
    env=env,
    reflection_rounds=1,    # Number of reflect-refine iterations
    system_template=system_template,
    # ... other config
)
```

## References

- Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)
- Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023)
