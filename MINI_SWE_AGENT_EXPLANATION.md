# Mini-SWE-Agent: How It Works & Planning Strategy

## Overview

**mini-swe-agent** is a lightweight AI agent (~100 lines of core code) that solves GitHub issues and software engineering tasks through **agentic loops**. Despite its simplicity, it achieves >74% on the SWE-bench verified benchmark.

The key insight: modern LLMs are smart enough to solve complex tasks without special tools—just bash, simple prompting, and iterative feedback.

---

## Core Architecture

### 1. **The Main Loop: Step → Query → Execute → Observe**

```
┌─────────────────────────────────────────┐
│ Initialize Agent with Task Description  │
├─────────────────────────────────────────┤
│            AGENTIC LOOP                 │
│  ┌──────────────────────────────────┐   │
│  │ 1. QUERY: Ask LLM for next action │   │
│  ├──────────────────────────────────┤   │
│  │ 2. PARSE: Extract bash command   │   │
│  │    (regex match on ```bash...``` )    │
│  ├──────────────────────────────────┤   │
│  │ 3. EXECUTE: Run command in env   │   │
│  │    (Docker, local, podman, etc)  │   │
│  ├──────────────────────────────────┤   │
│  │ 4. OBSERVE: Add output to chat   │   │
│  │    history                       │   │
│  └──────────────────────────────────┘   │
│              ↑     ↓                      │
│              └─────┘                      │
│         (Loop continues until task      │
│          is completed or limits         │
│          exceeded)                      │
└─────────────────────────────────────────┘
```

### 2. **Message History (Linear & Stateful)**

The agent maintains a **linear message history** similar to ChatGPT:

```
[
  {"role": "system", "content": "You are a helpful assistant..."},
  {"role": "user", "content": "Problem: Fix issue #123..."},
  {"role": "assistant", "content": "I'll start by exploring the codebase...\n```bash\nls -la\n```"},
  {"role": "user", "content": "<returncode>0</returncode>\n<output>total 48\n...(command output)...</output>"},
  {"role": "assistant", "content": "Now let me look at the specific file...\n```bash\ncat src/main.py\n```"},
  ...
]
```

**Key benefit:** Every step appends to history. No trajectory reconstruction needed. The entire conversation IS the trajectory.

---

## Planning Strategy

Mini-SWE-agent uses **in-context planning** — the LLM generates its own plan based on the task description and feedback.

### **Step 1: Task Understanding (System Prompt)**

The system prompt guides the LLM's reasoning:

```yaml
agent:
  system_template: |
    You are a helpful assistant that can interact with a computer.
    Your response must contain exactly ONE bash code block...
    Include a THOUGHT section before your command where you 
    explain your reasoning process.
```

### **Step 2: Problem Description (User Prompt)**

The user prompt provides detailed instructions:

```yaml
agent:
  instance_template: |
    <pr_description>
    Consider the following PR description:
    {{task}}
    </pr_description>
    
    <instructions>
    ## Recommended Workflow
    1. Analyze the codebase by finding and reading relevant files
    2. Create a script to reproduce the issue
    3. Edit the source code to resolve the issue
    4. Verify your fix works by running your script again
    5. Test edge cases to ensure your fix is robust
    
    ## Important Boundaries
    - MODIFY: Regular source code files
    - DO NOT MODIFY: Tests, config files
    ```

**This provides a planning framework** that the LLM follows.

### **Step 3: Iterative Refinement (Feedback Loop)**

The agent learns from each execution:

1. **LLM generates a command** based on current understanding
2. **Command executes** in a sandboxed environment (Docker)
3. **Output is provided** back to LLM as feedback
4. **LLM refines** its approach based on results
5. **Repeat** until task is complete or limits exceeded

**Example trajectory:**

```
User: "Fix issue with sorting..."
LLM: "Let me explore the codebase"
  → Runs: find . -name "*.py" | grep sort
User: (shows files found)
LLM: "Now let me read the main sorting function"
  → Runs: cat src/sort.py
User: (shows function contents)
LLM: "I found the bug! Let me fix it"
  → Runs: sed -i 's/old_code/new_code/' src/sort.py
User: (shows operation succeeded)
LLM: "Let me test the fix"
  → Runs: python test_sort.py
User: (shows test output)
LLM: "Tests pass! Task complete."
  → Runs: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

---

## How Planning Works (In-Context)

The planning strategy is **implicit, not explicit**:

1. **No separate planning phase** — the LLM thinks and acts simultaneously
2. **Context window is the workspace** — all previous steps are visible
3. **Feedback drives exploration** — failures guide next steps
4. **Chain-of-thought** in THOUGHT sections helps reason through decisions

### Example Planning Process

**Task:** "Fix bug where users can't login with special characters in passwords"

**Step 1:** LLM analyzes what's needed:
```
THOUGHT: I need to understand:
- Where authentication happens
- How passwords are validated
- Why special characters fail
- How to fix it without breaking other functionality

Let me start by exploring the auth module.
```

**Step 2:** Explores codebase:
```bash
find . -name "*auth*" -type f
grep -r "password" --include="*.py" | head -20
```

**Step 3:** Reads relevant files:
```bash
cat src/auth/validator.py
```

**Step 4:** Identifies the bug:
```
THOUGHT: I found it! Line 42 has regex that doesn't allow special chars.
The regex is: [a-zA-Z0-9_]
But passwords should allow special chars like !@#$%

I need to fix this regex while ensuring it's still secure.
```

**Step 5:** Creates reproduction script:
```bash
python test_special_chars.py
```

**Step 6:** Fixes the bug:
```bash
sed -i 's/\[a-zA-Z0-9_\]/[a-zA-Z0-9_!@#$%^&*]/' src/auth/validator.py
```

**Step 7:** Tests fix:
```bash
python test_special_chars.py
python -m pytest tests/auth/ -v
```

**Step 8:** Verifies no regressions:
```bash
python -m pytest tests/ -v
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

---

## Key Configuration Parameters

### From `config/extra/swebench.yaml`:

```yaml
agent:
  step_limit: 250           # Max steps before timeout
  cost_limit: 3.0           # Max cost in dollars
  action_regex: r"```bash\s*\n(.*?)\n```"  # How to parse actions

environment:
  cwd: "/testbed"           # Working directory
  timeout: 60               # Timeout per command (seconds)
  environment_class: docker # Sandbox type

model:
  model_name: "anthropic/claude-sonnet-4-5"
  model_kwargs:
    temperature: 0.0        # Deterministic (not random)
    drop_params: true       # Don't error on unknown params
```

---

## Exception Handling & Termination

The agent handles different exception types:

### **Non-Terminating (Agent recovers)**
- `FormatError`: LLM didn't format response correctly → ask to retry
- `ExecutionTimeoutError`: Command took too long → show output, ask for next action
- `NonTerminatingException`: User interruption → continue with feedback

### **Terminating (Task ends)**
- `Submitted`: LLM executed `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` → task done
- `LimitsExceeded`: Exceeded step limit or cost limit → stop and report

---

## Why This Planning Strategy Works

1. **Leverages LLM reasoning**: Modern models (Claude, GPT-4) excel at planning when given context
2. **Minimal assumptions**: No special tools, just bash—works everywhere
3. **Adaptive**: Agent adjusts strategy based on feedback
4. **Transparent**: Full trajectory is human-readable chat history
5. **Scalable**: Can use any LLM (OpenAI, Anthropic, Ollama, etc.)

---

## Real-World Performance

- **SWE-bench verified**: >74% pass rate with Claude Sonnet 4
- **Speed**: Completes tasks in seconds to minutes (not hours)
- **Reliability**: Works with different environments (Docker, local, cloud)
- **Cost effective**: Typically $0.10-$1.00 per task

---

## Comparison with Other Approaches

| Approach | Planning | Tools | Performance | Complexity |
|----------|----------|-------|-------------|-----------|
| **Mini-SWE-agent** | In-context (implicit) | Bash only | 74% | ~100 lines |
| **SWE-agent** | Tool-calling API | 10+ special tools | 12% higher* | 10K+ lines |
| **Chain-of-thought** | Explicit steps | Direct LLM | Lower | Medium |
| **ReACT** | Explicit reasoning | Tool-calling | Medium | High |

*On full SWE-bench; trade-off between performance and complexity

---

## Summary

**Mini-SWE-agent's planning strategy** is deceptively simple:

1. **Let the LLM think** via system/user prompts
2. **Execute its decisions** in a sandbox
3. **Provide feedback** from execution results
4. **Repeat** until task completes

The genius is that modern LLMs are smart enough to develop and refine their own strategies through this feedback loop, without explicit planning modules or complex tool systems.
