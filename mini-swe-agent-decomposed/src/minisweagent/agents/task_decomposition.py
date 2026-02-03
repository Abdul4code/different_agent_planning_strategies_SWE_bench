"""Task Decomposition Agent - Standalone implementation.

This agent implements the task decomposition strategy:
1. Decompose the main problem into sub-tasks/sub-goals (g0, g1, ..., gn)
2. Add decomposition guidance to the system context
3. Execute using the baseline agent loop

Formula:
    g0, g1, ..., gn = decompose(E, g; Θ, P)

This agent is part of the mini-swe-agent-decomposed standalone folder.
It extends DefaultAgent from the baseline mini-swe-agent.
"""

import re
from typing import Any, Dict, List, Optional

from minisweagent.agents.default import DefaultAgent
from minisweagent.utils.log import logger


class TaskDecompositionAgent(DefaultAgent):
    """Agent that decomposes complex tasks into sequential sub-tasks.
    
    The decomposition happens at the start of run() and adds guidance
    to the task context before the main agent loop begins.
    """

    def __init__(self, model, env, **kwargs):
        """Initialize the task decomposition agent.
        
        Args:
            model: The language model to use
            env: The execution environment
            **kwargs: Additional config passed to DefaultAgent (system_template, etc.)
        """
        super().__init__(model, env, **kwargs)
        self.sub_goals = []  # g0, g1, ..., gn
        self.decomposition_guidance = ""

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run with task decomposition.
        
        Decomposes the task into sub-goals first, then adds decomposition
        guidance to the task context before running the baseline agent loop.
        
        Args:
            task: The main task/problem statement
            **kwargs: Additional arguments for the run
            
        Returns:
            Tuple of (exit_status, result)
        """
        # Decompose the task into sub-goals
        self.sub_goals = self._decompose_task(task)
        
        if self.sub_goals:
            logger.info(
                f"Decomposed problem into {len(self.sub_goals)} sub-goals:\n"
                + "\n".join(f"  {i+1}. {sg}" for i, sg in enumerate(self.sub_goals))
            )
            
            # Create decomposition guidance to add to task
            self.decomposition_guidance = self._create_guidance(task)
            
            # Add decomposition guidance to the task
            enhanced_task = f"{task}\n\n{self.decomposition_guidance}"
        else:
            enhanced_task = task
        
        # Run baseline agent with enhanced task
        return super().run(enhanced_task, **kwargs)

    def _decompose_task(self, task: str) -> List[str]:
        """Decompose main task into sub-goals.
        
        Formula: g0, g1, ..., gn = decompose(E, g; Θ, P)
        
        Args:
            task: Main problem statement
            
        Returns:
            List of ordered sub-goals
        """
        decomposition_prompt = [
            {"role": "system", "content": "You are an expert at breaking down complex programming tasks into clear, sequential sub-tasks."},
            {"role": "user", "content": f"""Analyze this problem and break it down into 3-5 clear, sequential sub-tasks:

PROBLEM:
{task}

Decompose this into specific sub-goals that should be completed in order.
Each sub-goal should be actionable and address one aspect of the problem.

Format your response as a numbered list:
1. [First sub-task]
2. [Second sub-task]
3. [Third sub-task]
...

Only output the numbered list, nothing else."""}
        ]
        
        try:
            response = self.model.query(decomposition_prompt)
            content = response.get("content", "")
            sub_goals = self._parse_sub_goals(content)
            return sub_goals if sub_goals else []
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}")
            return []

    def _parse_sub_goals(self, response: str) -> List[str]:
        """Parse sub-goals from model response.
        
        Args:
            response: Model response text
            
        Returns:
            List of extracted sub-goals
        """
        sub_goals = []
        
        # Match patterns like "1. ...", "2. ...", etc.
        pattern = r"^\d+\.\s*(.+?)$"
        matches = re.findall(pattern, response, re.MULTILINE)
        
        if matches:
            sub_goals = [m.strip() for m in matches if m.strip()]
        
        return sub_goals[:5]  # Limit to 5 sub-goals

    def _create_guidance(self, original_task: str) -> str:
        """Create decomposition guidance text.
        
        Args:
            original_task: The original task description
            
        Returns:
            Formatted guidance string with sub-goals
        """
        if not self.sub_goals:
            return ""
        
        guidance_lines = [
            "=" * 60,
            "TASK DECOMPOSITION - Follow these sub-goals in order:",
            "=" * 60
        ]
        
        for i, goal in enumerate(self.sub_goals, 1):
            guidance_lines.append(f"Sub-goal {i}: {goal}")
        
        guidance_lines.extend([
            "=" * 60,
            "Work through each sub-goal sequentially.",
            "Complete all sub-goals to solve the main problem.",
            "=" * 60
        ])
        
        return "\n".join(guidance_lines)
