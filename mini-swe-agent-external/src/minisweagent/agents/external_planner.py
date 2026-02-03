"""External Planner-Aided Planning Agent.

This agent implements the external planner-aided planning strategy as described:

    h = formalize(E, g; Θ, P)
    p = plan(E, g, h; Φ)

Where:
- Θ is the LLM used for formalization (converting task to structured representation)
- h is the formalized information (structured task representation)
- Φ is the external planner module (non-LLM planning algorithm)
- p is the generated plan

The key insight is that LLM handles formalization while an external planner
handles the actual planning, addressing efficiency and feasibility issues.

References:
- LLM+P (Liu et al., 2023)
- ProgPrompt (Singh et al., 2023)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from minisweagent.agents.default import DefaultAgent
from minisweagent.utils.log import logger


@dataclass
class FormalizedTask:
    """Structured representation h from formalize(E, g; Θ, P)."""
    goal: str
    preconditions: List[str]
    actions: List[Dict[str, Any]]
    constraints: List[str]
    dependencies: List[Tuple[str, str]]  # (action_i, action_j) means i before j


class ExternalPlanner:
    """External planner module Φ.
    
    This is a non-LLM planning algorithm that generates plans
    from formalized task representations.
    
    Implements classical planning techniques:
    - Dependency-aware ordering
    - Constraint satisfaction
    - Action sequencing based on preconditions
    """
    
    def __init__(self, strategy: str = "dependency_order"):
        """Initialize external planner.
        
        Args:
            strategy: Planning strategy ("dependency_order", "greedy", "topological")
        """
        self.strategy = strategy
    
    def plan(self, formalized: FormalizedTask) -> List[str]:
        """Generate plan p = plan(E, g, h; Φ).
        
        Uses non-LLM algorithms to sequence actions based on
        dependencies and constraints.
        
        Args:
            formalized: The formalized task representation h
            
        Returns:
            Ordered list of action descriptions
        """
        if self.strategy == "topological":
            return self._topological_plan(formalized)
        elif self.strategy == "greedy":
            return self._greedy_plan(formalized)
        else:
            return self._dependency_order_plan(formalized)
    
    def _dependency_order_plan(self, formalized: FormalizedTask) -> List[str]:
        """Plan by respecting action dependencies.
        
        Uses topological sort on dependency graph.
        """
        actions = formalized.actions
        deps = formalized.dependencies
        
        if not actions:
            return []
        
        # Build dependency graph
        action_names = [a.get("name", f"action_{i}") for i, a in enumerate(actions)]
        in_degree = {name: 0 for name in action_names}
        graph = {name: [] for name in action_names}
        
        for before, after in deps:
            if before in graph and after in in_degree:
                graph[before].append(after)
                in_degree[after] += 1
        
        # Topological sort (Kahn's algorithm)
        queue = [n for n in action_names if in_degree[n] == 0]
        ordered = []
        
        while queue:
            # Sort by priority if available
            queue.sort(key=lambda n: next(
                (a.get("priority", 999) for a in actions if a.get("name") == n), 999
            ))
            current = queue.pop(0)
            ordered.append(current)
            
            for neighbor in graph.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Map back to action descriptions
        result = []
        for name in ordered:
            for action in actions:
                if action.get("name") == name:
                    result.append(action.get("description", name))
                    break
        
        # Add any remaining actions not in dependency graph
        seen = set(ordered)
        for action in actions:
            name = action.get("name", "")
            if name not in seen:
                result.append(action.get("description", name))
        
        return result
    
    def _greedy_plan(self, formalized: FormalizedTask) -> List[str]:
        """Greedy planning by priority and precondition satisfaction."""
        actions = sorted(
            formalized.actions,
            key=lambda a: (a.get("priority", 999), len(a.get("preconditions", [])))
        )
        return [a.get("description", a.get("name", "")) for a in actions]
    
    def _topological_plan(self, formalized: FormalizedTask) -> List[str]:
        """Alias for dependency-based planning."""
        return self._dependency_order_plan(formalized)


class ExternalPlannerAgent(DefaultAgent):
    """Agent using external planner-aided planning.
    
    Implements:
        h = formalize(E, g; Θ, P)  - LLM formalizes task
        p = plan(E, g, h; Φ)       - External planner generates plan
    
    The LLM's role is formalization, not planning. The external
    planner Φ handles actual plan generation for efficiency.
    """

    def __init__(self, model, env, planner_strategy: str = "dependency_order", **kwargs):
        """Initialize the external planner-aided agent.
        
        Args:
            model: The language model Θ for formalization
            env: The execution environment E
            planner_strategy: Strategy for external planner Φ
            **kwargs: Additional config passed to DefaultAgent
        """
        super().__init__(model, env, **kwargs)
        self.external_planner = ExternalPlanner(strategy=planner_strategy)
        self.formalized_task: Optional[FormalizedTask] = None
        self.generated_plan: List[str] = []

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run with external planner-aided planning.
        
        Implements:
            h = formalize(E, g; Θ, P)
            p = plan(E, g, h; Φ)
        
        Args:
            task: The goal g (problem statement)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (exit_status, result)
        """
        # Step 1: h = formalize(E, g; Θ, P) - LLM formalizes the task
        self.formalized_task = self._formalize(task)
        
        if not self.formalized_task:
            logger.warning("Formalization failed, falling back to baseline")
            return super().run(task, **kwargs)
        
        logger.info(f"Formalized task with {len(self.formalized_task.actions)} actions")
        
        # Step 2: p = plan(E, g, h; Φ) - External planner generates plan
        self.generated_plan = self.external_planner.plan(self.formalized_task)
        
        if not self.generated_plan:
            logger.warning("External planner produced empty plan, falling back")
            return super().run(task, **kwargs)
        
        logger.info(f"External planner generated {len(self.generated_plan)} steps")
        
        # Execute with generated plan as guidance
        guidance = self._create_guidance()
        enhanced_task = f"{task}\n\n{guidance}"
        
        return super().run(enhanced_task, **kwargs)

    def _formalize(self, task: str) -> Optional[FormalizedTask]:
        """Formalize task: h = formalize(E, g; Θ, P).
        
        Uses LLM Θ to convert natural language task into
        structured representation h that external planner can process.
        
        Simplified prompt for fair comparison with other planning strategies.
        
        Args:
            task: The goal g (natural language problem)
            
        Returns:
            FormalizedTask h with structured representation
        """
        formalization_prompt = [
            {"role": "system", "content": "You are a task formalization expert. Extract structured actions from tasks."},
            {"role": "user", "content": f"""Extract actions and their order from this task:

TASK:
{task}

List 3-6 actions in order. Format each as:
ACTION: [name] | [description] | AFTER: [dependency or "none"]

Example:
ACTION: analyze | Analyze the error message | AFTER: none
ACTION: locate | Find the relevant file | AFTER: analyze
ACTION: fix | Implement the fix | AFTER: locate"""}
        ]
        
        try:
            response = self.model.query(formalization_prompt)
            content = response.get("content", "")
            return self._parse_formalization(content)
        except Exception as e:
            logger.warning(f"Formalization failed: {e}")
            return None

    def _parse_formalization(self, response: str) -> Optional[FormalizedTask]:
        """Parse LLM response into FormalizedTask structure h.
        
        Simplified parser matching the simplified formalization prompt.
        """
        try:
            actions = []
            dependencies = []
            
            # Parse simplified format: ACTION: name | description | AFTER: dependency
            action_pattern = r"ACTION:\s*(\S+)\s*\|\s*([^|]+)\|\s*AFTER:\s*(\S+)"
            for match in re.finditer(action_pattern, response, re.IGNORECASE):
                name = match.group(1).strip()
                description = match.group(2).strip()
                after = match.group(3).strip().lower()
                
                actions.append({
                    "name": name,
                    "description": description,
                    "priority": len(actions) + 1,
                    "preconditions": []
                })
                
                if after and after != "none":
                    dependencies.append((after, name))
            
            # Fallback: try numbered list format
            if not actions:
                simple_pattern = r"\d+\.\s*(.+?)(?:\n|$)"
                matches = re.findall(simple_pattern, response)
                for i, desc in enumerate(matches[:6]):
                    actions.append({
                        "name": f"step_{i+1}",
                        "description": desc.strip(),
                        "priority": i + 1,
                        "preconditions": []
                    })
                    if i > 0:
                        dependencies.append((f"step_{i}", f"step_{i+1}"))
            
            # Extract goal from first line or first action
            goal = actions[0]["description"] if actions else "Solve the task"
            
            return FormalizedTask(
                goal=goal,
                preconditions=[],
                actions=actions,
                constraints=[],
                dependencies=dependencies
            )
        except Exception as e:
            logger.warning(f"Failed to parse formalization: {e}")
            return None

    def _create_guidance(self) -> str:
        """Create guidance with the externally generated plan p."""
        if not self.generated_plan:
            return ""
        
        plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(self.generated_plan))
        
        return f"""{'=' * 60}
PLAN (from external planner Φ):
{'=' * 60}
{plan_text}
{'=' * 60}
Execute this plan step by step.
{'=' * 60}"""
