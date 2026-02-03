"""Reflection and Refinement Agent.

This agent implements the reflection and refinement strategy as described:

    p₀ = plan(E, g; Θ, P)
    rᵢ = reflect(E, g, pᵢ; Θ, P)
    pᵢ₊₁ = refine(E, g, pᵢ, rᵢ; Θ, P)

Where:
- p₀ is the initial plan
- rᵢ is the reflection on plan pᵢ (identifying failures/issues)
- pᵢ₊₁ is the refined plan incorporating the reflection
- The process iterates i times to progressively improve the plan

The key insight is that LLM reflects on failures and refines the plan,
improving planning ability through self-critique.

References:
- Reflexion (Shinn et al., 2023)
- Self-Refine (Madaan et al., 2023)
"""

import re
from typing import List, Optional, Tuple

from minisweagent.agents.default import DefaultAgent
from minisweagent.utils.log import logger


class ReflectionAgent(DefaultAgent):
    """Agent using reflection and refinement for planning.
    
    Implements:
        p₀ = plan(E, g; Θ, P)
        rᵢ = reflect(E, g, pᵢ; Θ, P)  
        pᵢ₊₁ = refine(E, g, pᵢ, rᵢ; Θ, P)
    
    The reflection loop runs for i iterations, progressively
    improving the plan through self-critique.
    """

    def __init__(self, model, env, reflection_rounds: int = 1, **kwargs):
        """Initialize the reflection agent.
        
        Args:
            model: The language model Θ
            env: The execution environment E
            reflection_rounds: Number of reflection iterations i (default 1)
            **kwargs: Additional config passed to DefaultAgent
        """
        super().__init__(model, env, **kwargs)
        self.reflection_rounds = reflection_rounds
        self.plans: List[str] = []  # [p₀, p₁, ..., pₙ]
        self.reflections: List[str] = []  # [r₀, r₁, ..., rₙ₋₁]

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run with reflection and refinement strategy.
        
        Implements the iterative loop:
            p₀ = plan(E, g; Θ, P)
            for i in range(n):
                rᵢ = reflect(E, g, pᵢ; Θ, P)
                pᵢ₊₁ = refine(E, g, pᵢ, rᵢ; Θ, P)
        
        Args:
            task: The goal g (problem statement)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (exit_status, result)
        """
        # Step 1: p₀ = plan(E, g; Θ, P) - Generate initial plan
        p_i = self._plan(task)
        self.plans = [p_i]
        self.reflections = []
        
        if not p_i:
            logger.warning("Initial plan generation failed, falling back to baseline")
            return super().run(task, **kwargs)
        
        logger.info(f"p₀ generated, starting {self.reflection_rounds} reflection rounds")
        
        # Step 2: Iterative reflection and refinement loop
        for i in range(self.reflection_rounds):
            # rᵢ = reflect(E, g, pᵢ; Θ, P)
            r_i = self._reflect(task, p_i, i)
            self.reflections.append(r_i)
            
            if not r_i:
                logger.warning(f"Reflection r_{i} failed, stopping refinement")
                break
            
            # pᵢ₊₁ = refine(E, g, pᵢ, rᵢ; Θ, P)
            p_next = self._refine(task, p_i, r_i, i)
            
            if not p_next:
                logger.warning(f"Refinement p_{i+1} failed, using p_{i}")
                break
            
            p_i = p_next
            self.plans.append(p_i)
            logger.info(f"Completed reflection round {i+1}/{self.reflection_rounds}")
        
        # Use final refined plan
        final_plan = self.plans[-1]
        logger.info(f"Reflection complete: {len(self.plans)} plans, {len(self.reflections)} reflections")
        
        # Execute with final plan as guidance
        guidance = self._create_guidance(final_plan)
        enhanced_task = f"{task}\n\n{guidance}"
        
        return super().run(enhanced_task, **kwargs)

    def _plan(self, task: str) -> str:
        """Generate initial plan: p₀ = plan(E, g; Θ, P).
        
        Args:
            task: The goal g
            
        Returns:
            Initial plan p₀
        """
        planning_prompt = [
            {"role": "system", "content": "You are an expert software engineer. Create a plan to solve the given problem."},
            {"role": "user", "content": f"""Create a plan to solve this problem:

PROBLEM:
{task}

Provide a step-by-step plan:
1. [Step 1]
2. [Step 2]
...

Be specific about what files to examine/modify and what changes to make."""}
        ]
        
        try:
            response = self.model.query(planning_prompt)
            return response.get("content", "")
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}")
            return ""

    def _reflect(self, task: str, plan: str, iteration: int) -> str:
        """Reflect on plan: rᵢ = reflect(E, g, pᵢ; Θ, P).
        
        Identifies failures, issues, and areas for improvement.
        
        Args:
            task: The goal g
            plan: Current plan pᵢ
            iteration: Current iteration i
            
        Returns:
            Reflection rᵢ
        """
        reflection_prompt = [
            {"role": "system", "content": "You are a critical reviewer. Analyze the plan and identify failures or issues that need addressing."},
            {"role": "user", "content": f"""Reflect on this plan and identify issues:

PROBLEM:
{task}

CURRENT PLAN (p_{iteration}):
{plan}

Identify:
1. What could go wrong with this plan?
2. What steps are missing or unclear?
3. What assumptions might be incorrect?
4. How could this plan fail?

Be specific about failures and improvements needed."""}
        ]
        
        try:
            response = self.model.query(reflection_prompt)
            return response.get("content", "")
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            return ""

    def _refine(self, task: str, plan: str, reflection: str, iteration: int) -> str:
        """Refine plan: pᵢ₊₁ = refine(E, g, pᵢ, rᵢ; Θ, P).
        
        Creates improved plan based on reflection.
        
        Args:
            task: The goal g
            plan: Current plan pᵢ
            reflection: Reflection rᵢ
            iteration: Current iteration i
            
        Returns:
            Refined plan pᵢ₊₁
        """
        refinement_prompt = [
            {"role": "system", "content": "You are an expert software engineer. Improve the plan based on the identified issues."},
            {"role": "user", "content": f"""Refine this plan based on the reflection:

PROBLEM:
{task}

CURRENT PLAN (p_{iteration}):
{plan}

REFLECTION (r_{iteration}):
{reflection}

Create an improved plan (p_{iteration + 1}) that addresses the identified issues:
1. [Improved Step 1]
2. [Improved Step 2]
...

Make the plan more robust and complete."""}
        ]
        
        try:
            response = self.model.query(refinement_prompt)
            return response.get("content", "")
        except Exception as e:
            logger.warning(f"Refinement failed: {e}")
            return ""

    def _create_guidance(self, final_plan: str) -> str:
        """Create guidance with the final refined plan."""
        if not final_plan:
            return ""
        
        iterations = len(self.plans) - 1
        
        return f"""{'=' * 60}
REFINED PLAN (after {iterations} reflection round{'s' if iterations != 1 else ''}):
{'=' * 60}
{final_plan}
{'=' * 60}
Execute this refined plan step by step.
{'=' * 60}"""
