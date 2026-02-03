"""Memory-augmented Planning Agent.

This agent implements the memory-augmented planning strategy as described:

    m = retrieve(E, g; M)
    p = plan(E, g, m; Θ, P)

Where:
- M is the memory module (stores commonsense, past experiences, domain knowledge)
- m is the retrieved information relevant to the current goal
- p is the plan generated using the retrieved memory

The key insight is that planning is enhanced with auxiliary signals
from memory, providing relevant context for better plan generation.

References:
- MemoryBank (Zhong et al., 2023)
- Generative Agents (Park et al., 2023)
"""

import re
from typing import Dict, List, Optional, Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.utils.log import logger


class MemoryModule:
    """Memory module M that stores valuable information.
    
    Contains:
    - Commonsense knowledge about software engineering
    - Past experiences and solution patterns
    - Domain-specific knowledge for common bug types
    """
    
    def __init__(self):
        # Domain knowledge: common patterns and strategies
        self.domain_knowledge = {
            "debugging": {
                "description": "Systematic debugging approach",
                "strategies": [
                    "Reproduce the issue with minimal test case",
                    "Add logging/print statements to trace execution",
                    "Use binary search to isolate the problem",
                    "Check recent changes that might have caused it"
                ]
            },
            "code_analysis": {
                "description": "Understanding unfamiliar code",
                "strategies": [
                    "Start from entry points and trace data flow",
                    "Read tests to understand expected behavior",
                    "Look at similar functions for patterns",
                    "Check documentation and comments"
                ]
            },
            "safe_modifications": {
                "description": "Making safe code changes",
                "strategies": [
                    "Understand the full impact before changing",
                    "Make minimal, focused changes",
                    "Preserve existing behavior for unrelated cases",
                    "Test edge cases after modifications"
                ]
            }
        }
        
        # Past experiences: error patterns and solutions
        self.experiences = {
            "TypeError": {
                "common_causes": ["Wrong argument type", "Missing type conversion", "None value passed"],
                "solution_pattern": "Check types at the error location and trace back to source"
            },
            "AttributeError": {
                "common_causes": ["Object is None", "Wrong object type", "Typo in attribute name"],
                "solution_pattern": "Add None checks or verify object initialization"
            },
            "ImportError": {
                "common_causes": ["Missing module", "Circular import", "Wrong path"],
                "solution_pattern": "Check module paths and import order"
            },
            "KeyError": {
                "common_causes": ["Missing dictionary key", "Wrong key name", "Key not initialized"],
                "solution_pattern": "Use .get() with default or check key existence"
            },
            "IndexError": {
                "common_causes": ["Empty list", "Off-by-one error", "Wrong iteration bounds"],
                "solution_pattern": "Check list length before access or fix iteration"
            }
        }
        
        # Commonsense: general software engineering principles
        self.commonsense = [
            "Read error messages carefully - they usually point to the problem",
            "Look at the test that's failing to understand expected behavior",
            "Check the git history for recent relevant changes",
            "Similar bugs often have similar solutions",
            "When stuck, look at how similar functionality is implemented elsewhere"
        ]


class MemoryAgent(DefaultAgent):
    """Agent using memory-augmented planning.
    
    Implements:
        m = retrieve(E, g; M)     - Retrieve relevant info from memory
        p = plan(E, g, m; Θ, P)   - Generate plan using retrieved memory
    
    The memory provides auxiliary signals that enhance planning quality.
    """

    def __init__(self, model, env, **kwargs):
        """Initialize the memory-augmented agent.
        
        Args:
            model: The language model Θ
            env: The execution environment E
            **kwargs: Additional config passed to DefaultAgent
        """
        super().__init__(model, env, **kwargs)
        self.memory = MemoryModule()  # M
        self.retrieved_memory: Optional[Dict[str, Any]] = None  # m
        self.generated_plan: Optional[str] = None  # p

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run with memory-augmented planning.
        
        Implements:
            m = retrieve(E, g; M)
            p = plan(E, g, m; Θ, P)
        
        Args:
            task: The goal g (problem statement)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (exit_status, result)
        """
        # Step 1: m = retrieve(E, g; M) - Retrieve from memory
        self.retrieved_memory = self._retrieve(task)
        
        if not self.retrieved_memory:
            logger.warning("Memory retrieval returned empty, falling back to baseline")
            return super().run(task, **kwargs)
        
        logger.info(f"Retrieved memory with {len(self.retrieved_memory)} categories")
        
        # Step 2: p = plan(E, g, m; Θ, P) - Plan using memory
        self.generated_plan = self._plan_with_memory(task, self.retrieved_memory)
        
        if not self.generated_plan:
            logger.warning("Memory-augmented planning failed, falling back")
            return super().run(task, **kwargs)
        
        logger.info("Generated memory-augmented plan")
        
        # Execute with generated plan
        guidance = self._create_guidance()
        enhanced_task = f"{task}\n\n{guidance}"
        
        return super().run(enhanced_task, **kwargs)

    def _retrieve(self, task: str) -> Dict[str, Any]:
        """Retrieve from memory: m = retrieve(E, g; M).
        
        Uses LLM to identify relevant memory entries based on
        the current goal g and environment E.
        
        Args:
            task: The goal g
            
        Returns:
            Retrieved memory m containing relevant information
        """
        # Use LLM to identify what's relevant from memory
        retrieval_prompt = [
            {"role": "system", "content": "You are a memory retrieval system. Identify what information would be helpful."},
            {"role": "user", "content": f"""Given this problem, identify relevant knowledge:

PROBLEM:
{task}

AVAILABLE MEMORY:
1. Domain Knowledge: {list(self.memory.domain_knowledge.keys())}
2. Error Experiences: {list(self.memory.experiences.keys())}
3. Commonsense Principles

Which categories are relevant? List them as:
DOMAIN: [list relevant domain categories]
ERRORS: [list relevant error types]
USE_COMMONSENSE: yes/no"""}
        ]
        
        try:
            response = self.model.query(retrieval_prompt)
            content = response.get("content", "")
            
            # Parse retrieval response
            retrieved = {"domain": [], "experiences": [], "commonsense": []}
            
            # Extract domain knowledge
            domain_match = re.search(r"DOMAIN:\s*\[?([^\]\n]+)", content, re.IGNORECASE)
            if domain_match:
                domains = [d.strip().lower().replace(" ", "_") for d in domain_match.group(1).split(",")]
                for d in domains:
                    if d in self.memory.domain_knowledge:
                        retrieved["domain"].append(self.memory.domain_knowledge[d])
            
            # Extract relevant error experiences
            error_match = re.search(r"ERRORS:\s*\[?([^\]\n]+)", content, re.IGNORECASE)
            if error_match:
                errors = [e.strip() for e in error_match.group(1).split(",")]
                for e in errors:
                    if e in self.memory.experiences:
                        retrieved["experiences"].append({
                            "type": e,
                            **self.memory.experiences[e]
                        })
            
            # Check for commonsense
            if "yes" in content.lower() and "commonsense" in content.lower():
                retrieved["commonsense"] = self.memory.commonsense
            
            # Fallback: if nothing retrieved, use commonsense
            if not any(retrieved.values()):
                retrieved["commonsense"] = self.memory.commonsense
            
            return retrieved
            
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            # Return commonsense as fallback
            return {"commonsense": self.memory.commonsense}

    def _plan_with_memory(self, task: str, memory: Dict[str, Any]) -> str:
        """Plan using memory: p = plan(E, g, m; Θ, P).
        
        Generates a plan informed by the retrieved memory m.
        
        Args:
            task: The goal g
            memory: Retrieved memory m
            
        Returns:
            Generated plan p
        """
        # Format memory for prompt
        memory_text = self._format_memory(memory)
        
        planning_prompt = [
            {"role": "system", "content": "You are an expert software engineer. Use the provided memory/knowledge to create an effective plan."},
            {"role": "user", "content": f"""Create a plan using the retrieved knowledge:

PROBLEM:
{task}

RETRIEVED MEMORY:
{memory_text}

Using this knowledge, create a step-by-step plan:
1. [Step informed by memory]
2. [Step informed by memory]
...

Be specific and apply the retrieved patterns/knowledge."""}
        ]
        
        try:
            response = self.model.query(planning_prompt)
            return response.get("content", "")
        except Exception as e:
            logger.warning(f"Memory-augmented planning failed: {e}")
            return ""

    def _format_memory(self, memory: Dict[str, Any]) -> str:
        """Format retrieved memory for the planning prompt."""
        lines = []
        
        if memory.get("domain"):
            lines.append("Domain Knowledge:")
            for d in memory["domain"]:
                lines.append(f"  - {d.get('description', '')}")
                for s in d.get("strategies", []):
                    lines.append(f"    • {s}")
        
        if memory.get("experiences"):
            lines.append("\nRelevant Past Experiences:")
            for exp in memory["experiences"]:
                lines.append(f"  {exp.get('type', 'Error')}:")
                lines.append(f"    Causes: {', '.join(exp.get('common_causes', []))}")
                lines.append(f"    Pattern: {exp.get('solution_pattern', '')}")
        
        if memory.get("commonsense"):
            lines.append("\nCommonsense Principles:")
            for c in memory["commonsense"]:
                lines.append(f"  • {c}")
        
        return "\n".join(lines)

    def _create_guidance(self) -> str:
        """Create guidance with the memory-augmented plan p."""
        if not self.generated_plan:
            return ""
        
        return f"""{'=' * 60}
MEMORY-AUGMENTED PLAN:
{'=' * 60}
{self.generated_plan}
{'=' * 60}
Execute this plan step by step.
{'=' * 60}"""
