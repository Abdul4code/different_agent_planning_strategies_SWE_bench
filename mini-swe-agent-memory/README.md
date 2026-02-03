# Memory-Augmented Agent

A planning strategy that retrieves relevant knowledge from memory to enhance plan generation.

## Planning Strategy

The Memory Agent implements retrieval-augmented planning:

```
m = retrieve(E, g; M)       # Retrieve relevant memory
p = plan(E, g, m; Θ, P)     # Plan using retrieved knowledge
```

Where:
- **M** = Memory module (domain knowledge, experiences, commonsense)
- **m** = Retrieved memory relevant to current goal
- **Θ** = Language model
- **p** = Memory-informed plan

## How It Works

### 1. Memory Module Structure

The memory contains three types of knowledge:

**Domain Knowledge**:
```python
{
    "debugging": {
        "strategies": [
            "Reproduce the issue with minimal test case",
            "Add logging to trace execution",
            "Use binary search to isolate the problem"
        ]
    },
    "code_analysis": {
        "strategies": [
            "Start from entry points and trace data flow",
            "Read tests to understand expected behavior"
        ]
    },
    "safe_modifications": {
        "strategies": [
            "Understand full impact before changing",
            "Make minimal, focused changes"
        ]
    }
}
```

**Past Experiences** (Error patterns):
```python
{
    "TypeError": {
        "common_causes": ["Wrong argument type", "None value passed"],
        "solution_pattern": "Check types and trace back to source"
    },
    "AttributeError": {
        "common_causes": ["Object is None", "Wrong object type"],
        "solution_pattern": "Add None checks or verify initialization"
    }
}
```

**Commonsense Principles**:
- Read error messages carefully
- Look at failing tests to understand expected behavior
- Check git history for recent relevant changes
- Similar bugs often have similar solutions

### 2. Retrieval Phase
The LLM identifies which memory categories are relevant:
```
DOMAIN: [debugging, code_analysis]
ERRORS: [TypeError]
USE_COMMONSENSE: yes
```

### 3. Planning with Memory
Retrieved knowledge is provided to the planning prompt:
```
RETRIEVED MEMORY:
Domain Knowledge:
  - Systematic debugging approach
    • Reproduce the issue with minimal test case
    • Add logging/print statements to trace execution

Relevant Past Experiences:
  TypeError:
    Causes: Wrong argument type, None value passed
    Pattern: Check types at error location and trace back

Commonsense Principles:
  • Read error messages carefully
  • Similar bugs often have similar solutions
```

## Differences from Baseline

| Aspect | Baseline | Memory-Augmented |
|--------|----------|------------------|
| **Context** | Task only | Task + retrieved knowledge |
| **Knowledge** | LLM's parametric memory | Explicit memory module |
| **Patterns** | Implicit | Explicit error patterns |
| **Strategies** | Re-derived each time | Retrieved from storage |

## Key Benefits

- **Knowledge Reuse**: Leverages curated debugging strategies
- **Error Pattern Matching**: Recognizes common error types
- **Consistent Approaches**: Uses proven solution patterns
- **Commonsense Integration**: Applies general principles

## Memory Categories

| Category | Content | Use Case |
|----------|---------|----------|
| `debugging` | Systematic debugging strategies | Bug fixing tasks |
| `code_analysis` | Code understanding approaches | Unfamiliar codebases |
| `safe_modifications` | Change safety practices | Sensitive changes |
| `experiences` | Error → solution mappings | Known error types |
| `commonsense` | General principles | All tasks |

## Configuration

```python
from minisweagent.agents.memory import MemoryAgent

agent = MemoryAgent(
    model=model,
    env=env,
    system_template=system_template,
    # ... other config
)
```

## References

- MemoryBank: Enhancing Large Language Models with Long-Term Memory (Zhong et al., 2023)
- Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)
