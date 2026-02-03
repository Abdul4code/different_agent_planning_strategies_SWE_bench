"""Multi-plan Selection Agent - Implements the multi-plan selection strategy.

This agent implements the multi-plan selection strategy as described in the paper:

    P = {p1, p2, ..., pn} = plan(E, g; Θ, P)
    p* = select(E, g, P; Θ, F)

Where:
- P is the set of generated alternative plans
- F is the search/selection strategy (LLM-based evaluation in this implementation)
- p* is the selected optimal plan

The key difference from simple heuristic scoring is that selection uses the LLM
itself as the evaluator (Θ) with a proper selection strategy (F).

References:
- Tree of Thoughts (Yao et al., 2023)
- Plan selection strategies (Zhao et al., 2023b)
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from minisweagent.agents.default import DefaultAgent
from minisweagent.utils.log import logger


class MultiPlanAgent(DefaultAgent):
    """Agent that generates multiple plans and uses LLM-based selection.
    
    Implements:
        P = plan(E, g; Θ, P)  - Generate n alternative plans
        p* = select(E, g, P; Θ, F)  - Select best plan using search strategy F
    
    The selection strategy F uses LLM-based evaluation where the model
    scores and ranks plans based on feasibility, correctness, and efficiency.
    """

    def __init__(self, model, env, num_plans: int = 3, selection_strategy: str = "llm_compare", **kwargs):
        """Initialize the multi-plan agent.
        
        Args:
            model: The language model (Θ) to use for planning and selection
            env: The execution environment (E)
            num_plans: Number of alternative plans to generate (n)
            selection_strategy: The search strategy F ("llm_evaluate", "llm_compare", "beam")
            **kwargs: Additional config passed to DefaultAgent
        """
        super().__init__(model, env, **kwargs)
        self.num_plans = num_plans
        self.selection_strategy = selection_strategy
        self.plans_generated: List[str] = []  # P = {p1, p2, ..., pn}
        self.selected_plan: Optional[str] = None  # p*
        self.plan_evaluations: List[Dict[str, Any]] = []

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run with multi-plan selection strategy.
        
        Implements:
            P = plan(E, g; Θ, P)
            p* = select(E, g, P; Θ, F)
        
        Args:
            task: The goal g (problem statement)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (exit_status, result)
        """
        # Step 1: P = plan(E, g; Θ, P) - Generate multiple alternative plans
        self.plans_generated = self._generate_plans(task)
        
        if not self.plans_generated:
            logger.warning("No plans generated, falling back to baseline")
            return super().run(task, **kwargs)
        
        logger.info(f"Generated {len(self.plans_generated)} alternative plans")
        
        # Step 2: p* = select(E, g, P; Θ, F) - Select best plan using strategy F
        self.selected_plan, self.plan_evaluations = self._select_plan(
            task, self.plans_generated
        )
        
        if not self.selected_plan:
            logger.warning("Plan selection failed, using first plan")
            self.selected_plan = self.plans_generated[0]
        
        logger.info(f"Selected plan using strategy '{self.selection_strategy}'")
        
        # Execute with selected plan as guidance
        guidance = self._create_guidance()
        enhanced_task = f"{task}\n\n{guidance}"
        
        return super().run(enhanced_task, **kwargs)

    def _generate_plans(self, task: str) -> List[str]:
        """Generate n alternative plans: P = plan(E, g; Θ, P)
        
        Args:
            task: The goal g
            
        Returns:
            List of n plans P = {p1, p2, ..., pn}
        """
        generation_prompt = [
            {"role": "system", "content": """You are an expert software engineer. Generate multiple DIFFERENT approaches to solve the given problem.
Each approach should use a distinct strategy - not just variations of the same approach.
Consider different angles: direct fix, refactoring, test-driven, etc."""},
            {"role": "user", "content": f"""Generate {self.num_plans} different approaches to solve this problem.

PROBLEM:
{task}

For each plan, provide:
1. The strategy/approach name
2. Step-by-step actions to implement it
3. Key files/functions to modify

Format as:
PLAN 1: [Strategy Name]
[Detailed steps]

PLAN 2: [Strategy Name]
[Detailed steps]

PLAN 3: [Strategy Name]
[Detailed steps]"""}
        ]
        
        try:
            response = self.model.query(generation_prompt)
            content = response.get("content", "")
            plans = self._parse_plans(content)
            return plans
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}")
            return []

    def _parse_plans(self, response: str) -> List[str]:
        """Parse plans from model response."""
        plans = []
        pattern = r"PLAN\s+\d+[:\s]*([^\n]*)\n(.*?)(?=PLAN\s+\d+|$)"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        for title, content in matches:
            plan_text = f"{title.strip()}\n{content.strip()}"
            if plan_text.strip():
                plans.append(plan_text)
        
        # Fallback parsing if pattern doesn't match
        if not plans:
            pattern = r"PLAN\s+\d+:\s*(.*?)(?=PLAN\s+\d+:|$)"
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            plans = [m.strip() for m in matches if m.strip()]
        
        return plans[:self.num_plans]

    def _select_plan(self, task: str, plans: List[str]) -> Tuple[Optional[str], List[Dict]]:
        """Select best plan: p* = select(E, g, P; Θ, F)
        
        Uses the search strategy F to select the optimal plan.
        
        Args:
            task: The goal g
            plans: The set of plans P
            
        Returns:
            Tuple of (selected plan p*, evaluation details)
        """
        if self.selection_strategy == "llm_evaluate":
            return self._select_by_llm_evaluation(task, plans)
        elif self.selection_strategy == "llm_compare":
            return self._select_by_llm_comparison(task, plans)
        elif self.selection_strategy == "beam":
            return self._select_by_beam_search(task, plans)
        else:
            return self._select_by_llm_evaluation(task, plans)

    def _select_by_llm_evaluation(self, task: str, plans: List[str]) -> Tuple[Optional[str], List[Dict]]:
        """Selection strategy F: LLM-based individual evaluation.
        
        Each plan is evaluated by the LLM on multiple criteria,
        then the highest-scoring plan is selected.
        
        This mimics tree search where each plan is a branch and
        the LLM provides the heuristic evaluation function.
        """
        evaluations = []
        
        for i, plan in enumerate(plans):
            eval_prompt = [
                {"role": "system", "content": "You are an expert code reviewer evaluating solution plans."},
                {"role": "user", "content": f"""Evaluate this plan for solving the given problem.

PROBLEM:
{task}

PLAN {i+1}:
{plan}

Rate this plan on a scale of 1-10 for each criterion:
1. FEASIBILITY: How likely is this plan to work? (1=unlikely, 10=very likely)
2. COMPLETENESS: Does it address all aspects of the problem? (1=partial, 10=complete)
3. EFFICIENCY: How direct/efficient is this approach? (1=roundabout, 10=direct)
4. CORRECTNESS: Will this produce a correct solution? (1=likely wrong, 10=likely correct)

Respond in this exact format:
FEASIBILITY: [score]
COMPLETENESS: [score]
EFFICIENCY: [score]
CORRECTNESS: [score]
TOTAL: [sum of scores]
REASONING: [one sentence explanation]"""}
            ]
            
            try:
                response = self.model.query(eval_prompt)
                content = response.get("content", "")
                scores = self._parse_evaluation(content)
                scores["plan_index"] = i
                scores["plan"] = plan
                evaluations.append(scores)
                
                logger.info(f"Plan {i+1} evaluation: total={scores.get('total', 0)}")
            except Exception as e:
                logger.warning(f"Evaluation of plan {i+1} failed: {e}")
                evaluations.append({"plan_index": i, "plan": plan, "total": 0})
        
        # Select plan with highest total score
        if evaluations:
            best_eval = max(evaluations, key=lambda x: x.get("total", 0))
            return best_eval["plan"], evaluations
        
        return None, evaluations

    def _select_by_llm_comparison(self, task: str, plans: List[str]) -> Tuple[Optional[str], List[Dict]]:
        """Selection strategy F: LLM-based pairwise comparison.
        
        Plans are compared pairwise and the winner is selected
        through a tournament-style selection (like in debate/comparison methods).
        """
        if len(plans) < 2:
            return plans[0] if plans else None, []
        
        # Format all plans for comparison
        plans_text = "\n\n".join([f"PLAN {i+1}:\n{p}" for i, p in enumerate(plans)])
        
        compare_prompt = [
            {"role": "system", "content": "You are an expert code reviewer comparing solution approaches."},
            {"role": "user", "content": f"""Compare these plans for solving the problem and select the BEST one.

PROBLEM:
{task}

{plans_text}

Compare all plans and select the best one. Consider:
- Which is most likely to correctly solve the problem?
- Which is most direct and efficient?
- Which handles edge cases best?

Respond in this exact format:
BEST_PLAN: [number 1-{len(plans)}]
REASONING: [explanation of why this plan is best]"""}
        ]
        
        try:
            response = self.model.query(compare_prompt)
            content = response.get("content", "")
            
            # Parse the selected plan number
            match = re.search(r"BEST_PLAN:\s*(\d+)", content, re.IGNORECASE)
            if match:
                plan_num = int(match.group(1))
                if 1 <= plan_num <= len(plans):
                    return plans[plan_num - 1], [{"selected": plan_num, "reasoning": content}]
        except Exception as e:
            logger.warning(f"Plan comparison failed: {e}")
        
        # Fallback to first plan
        return plans[0], []

    def _select_by_beam_search(self, task: str, plans: List[str]) -> Tuple[Optional[str], List[Dict]]:
        """Selection strategy F: Beam search simulation.
        
        Simulates beam search by having the LLM predict outcomes
        of each plan and selecting based on predicted success.
        """
        predictions = []
        
        for i, plan in enumerate(plans):
            predict_prompt = [
                {"role": "system", "content": "You are simulating the execution of a solution plan."},
                {"role": "user", "content": f"""Predict the outcome if this plan is executed.

PROBLEM:
{task}

PLAN:
{plan}

Predict:
1. What specific changes will be made?
2. Will it fix the issue completely?
3. What could go wrong?
4. Confidence level (1-10)?

Respond with:
PREDICTED_SUCCESS: [YES/NO/PARTIAL]
CONFIDENCE: [1-10]
RISKS: [main risk]"""}
            ]
            
            try:
                response = self.model.query(predict_prompt)
                content = response.get("content", "")
                
                # Parse prediction
                success_match = re.search(r"PREDICTED_SUCCESS:\s*(YES|NO|PARTIAL)", content, re.IGNORECASE)
                conf_match = re.search(r"CONFIDENCE:\s*(\d+)", content)
                
                success = success_match.group(1).upper() if success_match else "NO"
                confidence = int(conf_match.group(1)) if conf_match else 0
                
                # Score: YES=10, PARTIAL=5, NO=0, weighted by confidence
                base_score = {"YES": 10, "PARTIAL": 5, "NO": 0}.get(success, 0)
                score = base_score * (confidence / 10)
                
                predictions.append({
                    "plan_index": i,
                    "plan": plan,
                    "success": success,
                    "confidence": confidence,
                    "score": score
                })
            except Exception as e:
                logger.warning(f"Beam search prediction for plan {i+1} failed: {e}")
                predictions.append({"plan_index": i, "plan": plan, "score": 0})
        
        # Select highest scoring plan
        if predictions:
            best = max(predictions, key=lambda x: x.get("score", 0))
            return best["plan"], predictions
        
        return plans[0] if plans else None, []

    def _parse_evaluation(self, content: str) -> Dict[str, Any]:
        """Parse evaluation scores from LLM response."""
        scores = {}
        
        for criterion in ["FEASIBILITY", "COMPLETENESS", "EFFICIENCY", "CORRECTNESS", "TOTAL"]:
            match = re.search(rf"{criterion}:\s*(\d+)", content, re.IGNORECASE)
            if match:
                scores[criterion.lower()] = int(match.group(1))
        
        # Calculate total if not provided
        if "total" not in scores:
            scores["total"] = sum(
                scores.get(c, 0) for c in ["feasibility", "completeness", "efficiency", "correctness"]
            )
        
        # Extract reasoning
        reason_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        if reason_match:
            scores["reasoning"] = reason_match.group(1).strip()
        
        return scores

    def _create_guidance(self) -> str:
        """Create guidance text with the selected plan p*."""
        if not self.selected_plan:
            return ""
        
        return f"""{'=' * 60}
SELECTED PLAN p* (from {len(self.plans_generated)} alternatives)
{'=' * 60}

{self.selected_plan}

{'=' * 60}
Execute this plan step by step.
{'=' * 60}"""
