"""
Idea Proposer Agent for TreeQuest Lab

Generates novel, feasible ML research hypotheses using LLM prompting.
Uses Chain-of-Thought, structured JSON outputs, and novelty checking against memory.
"""

import json
import logging
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProposalConfig(BaseModel):
    """Configuration for IdeaProposer."""
    
    model_name: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for proposal generation"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for sampling"
    )
    max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Maximum tokens in response"
    )
    domain: str = Field(
        default="efficient fine-tuning",
        description="Research domain focus"
    )
    novelty_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum novelty score required"
    )
    max_retries: int = Field(
        default=3,
        gt=0,
        description="Max retries for failed proposals"
    )
    
    class Config:
        extra = "ignore"


class ResearchProposal(BaseModel):
    """Structured research proposal."""
    
    hypothesis: str = Field(..., description="Clear, testable hypothesis")
    motivation: str = Field(..., description="Why this is worth investigating")
    method_summary: list[str] = Field(..., description="Brief description of experimental approach")
    expected_outcome: str = Field(..., description="What results would support the hypothesis")
    feasibility_score: float = Field(..., ge=0.0, le=1.0, description="Self-assessed feasibility")
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Self-assessed novelty")
    estimated_cost: str = Field(..., description="Estimated compute cost (e.g., 'low', 'medium', 'high')")
    keywords: list[str] = Field(default_factory=list, description="Key concepts/tags")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class IdeaProposer:
    """
    Agent that generates novel ML research proposals.
    
    Uses LLM with structured prompting to generate feasible, novel hypotheses.
    Integrates with memory system to avoid duplicates and promote diversity.
    """
    
    SYSTEM_PROMPT = """You are an expert ML research ideation assistant. Your role is to propose novel, feasible research ideas for small-scale experiments.

Guidelines:
1. Focus on {domain} - ideas should be testable with limited compute (< 8 GPU hours)
2. Proposals must be specific and testable with clear success criteria
3. Consider recent advances but avoid obvious extensions
4. Think about ablation studies and controls
5. Prioritize ideas with clear scientific merit over incremental improvements

Format your response as valid JSON with these fields:
- hypothesis: Clear, falsifiable statement
- motivation: Why this matters (2-3 sentences)
- method_summary: Experimental approach (3-5 bullet points)
- expected_outcome: What would support/refute the hypothesis
- feasibility_score: 0.0-1.0 (be realistic about compute/time)
- novelty_score: 0.0-1.0 (honest self-assessment)
- estimated_cost: "low" (<1 GPU-hour), "medium" (1-4 hours), or "high" (4-8 hours)
- keywords: List of 3-5 relevant tags

Think step-by-step before generating your proposal."""

    USER_PROMPT_TEMPLATE = """Generate a novel research proposal in the domain of {domain}.

Context from previous work (avoid duplicating these):
{previous_ideas}

Current tree state: {tree_state}

Focus on ideas that:
1. Can be tested with small models (<1B params) or subsets
2. Have clear metrics for evaluation
3. Could meaningfully advance understanding even if negative

Be creative but rigorous. Remember: we value quality over quantity."""

    def __init__(self, config: Optional[ProposalConfig] = None):
        self.config = config or ProposalConfig()
        self._client = None
        self._proposal_count = 0
        
    @property
    def client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            try:
                from openai import OpenAI
                import os
                
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not set. Using mock mode.")
                    return None
                    
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("openai package not installed. Using mock mode.")
                return None
        return self._client
    
    def generate_proposal(
        self,
        previous_ideas: list[str] | None = None,
        tree_state: str = "root node only",
        seed: Optional[int] = None
    ) -> ResearchProposal:
        """
        Generate a novel research proposal.
        
        Args:
            previous_ideas: List of previously explored hypotheses
            tree_state: Description of current tree state
            seed: Random seed for reproducibility
            
        Returns:
            ResearchProposal object
        """
        self._proposal_count += 1
        
        # Format context
        prev_str = "\n".join(f"- {idea}" for idea in (previous_ideas or [])) or "None yet."
        
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            domain=self.config.domain,
            previous_ideas=prev_str,
            tree_state=tree_state
        )
        
        # Try LLM if available
        if self.client is not None:
            return self._generate_with_llm(user_prompt, seed)
        else:
            return self._generate_mock(previous_ideas, seed)
    
    def _generate_with_llm(
        self,
        user_prompt: str,
        seed: Optional[int] = None
    ) -> ResearchProposal:
        """Generate proposal using LLM API."""
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT.format(domain=self.config.domain)},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                    seed=seed
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from LLM")
                
                # Parse JSON
                data = json.loads(content)
                
                # Validate and create proposal
                proposal = ResearchProposal(**data)
                
                logger.info(f"Generated proposal #{self._proposal_count}: {proposal.hypothesis[:80]}...")
                return proposal
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error("All LLM attempts failed, falling back to mock")
                    return self._generate_mock()
        
        # Should not reach here, but safety fallback
        return self._generate_mock()
    
    def _generate_mock(
        self,
        previous_ideas: list[str] | None = None,
        seed: Optional[int] = None
    ) -> ResearchProposal:
        """Generate mock proposal for testing/demo without API access."""
        
        import random
        if seed is not None:
            random.seed(seed)
        
        mock_proposals = [
            {
                "hypothesis": "LoRA rank adaptation during training improves convergence speed vs fixed rank",
                "motivation": "Fixed LoRA rank may be suboptimal at different training stages. Dynamic adaptation could accelerate convergence while maintaining final performance.",
                "method_summary": [
                    "Compare fixed-rank LoRA vs adaptive-rank on GLUE tasks",
                    "Start with low rank, increase based on gradient magnitude",
                    "Measure convergence speed and final accuracy",
                    "Ablate adaptation frequency and thresholds"
                ],
                "expected_outcome": "Adaptive ranking achieves same final accuracy 30% faster than best fixed rank",
                "feasibility_score": 0.85,
                "novelty_score": 0.72,
                "estimated_cost": "medium",
                "keywords": ["lora", "peft", "adaptive", "convergence"]
            },
            {
                "hypothesis": "Gradient checkpointing every other layer provides optimal memory-speed tradeoff for small models",
                "motivation": "Full checkpointing saves memory but slows training. No checkpointing is fast but memory-intensive. Alternating pattern may offer sweet spot.",
                "method_summary": [
                    "Test checkpointing patterns on 125M-500M models",
                    "Compare memory usage, throughput, and gradient noise",
                    "Evaluate on language modeling and classification",
                    "Analyze impact on maximum batch size"
                ],
                "expected_outcome": "Alternating checkpointing reduces memory by 35% with <10% slowdown",
                "feasibility_score": 0.9,
                "novelty_score": 0.65,
                "estimated_cost": "low",
                "keywords": ["checkpointing", "memory", "efficiency", "training"]
            },
            {
                "hypothesis": "Mixing synthetic counterfactual examples improves OOD generalization more than natural augmentation",
                "motivation": "LLM-generated counterfactuals target specific reasoning gaps. Natural augmentation may not address systematic weaknesses.",
                "method_summary": [
                    "Generate counterfactuals for GSM8K using GPT-4",
                    "Fine-tune small models with mixed data ratios",
                    "Test on OOD math benchmarks",
                    "Compare to back-translation and paraphrasing"
                ],
                "expected_outcome": "Counterfactual mixing improves OOD accuracy by 5-8% over natural augmentation",
                "feasibility_score": 0.8,
                "novelty_score": 0.78,
                "estimated_cost": "medium",
                "keywords": ["ood", "counterfactual", "augmentation", "reasoning"]
            },
            {
                "hypothesis": "Layer-wise learning rate decay benefits diminish below 100M parameters",
                "motivation": "LLRD is standard for large models but adds complexity. Small models may not need this sophistication.",
                "method_summary": [
                    "Train models from 10M to 500M with/without LLRD",
                    "Test various decay factors (0.8-0.95)",
                    "Measure convergence and final performance",
                    "Analyze per-layer gradient statistics"
                ],
                "expected_outcome": "LLRD shows negligible benefit for models <100M params",
                "feasibility_score": 0.95,
                "novelty_score": 0.6,
                "estimated_cost": "high",
                "keywords": ["learning-rate", "scaling", "optimization", "small-models"]
            }
        ]
        
        # Select one not in previous ideas
        previous_lower = [p.lower() for p in (previous_ideas or [])]
        candidates = [
            p for p in mock_proposals 
            if p["hypothesis"].lower() not in previous_lower
        ]
        
        selected = random.choice(candidates) if candidates else random.choice(mock_proposals)
        
        logger.info(f"Generated mock proposal #{self._proposal_count}")
        return ResearchProposal(**selected)
    
    def validate_proposal(self, proposal: ResearchProposal) -> tuple[bool, str]:
        """
        Validate a proposal meets requirements.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if len(proposal.hypothesis) < 20:
            return False, "Hypothesis too short"
        
        if len(proposal.method_summary) < 2:
            return False, "Method summary insufficient"
        
        if proposal.feasibility_score < 0.3:
            return False, "Feasibility score too low"
        
        if proposal.novelty_score < self.config.novelty_threshold:
            return False, f"Novelty below threshold ({proposal.novelty_score} < {self.config.novelty_threshold})"
        
        if proposal.estimated_cost not in ["low", "medium", "high"]:
            return False, "Invalid cost estimate"
        
        return True, "Valid proposal"
    
    def get_statistics(self) -> dict:
        """Get proposer statistics."""
        return {
            "proposals_generated": self._proposal_count,
            "model": self.config.model_name,
            "domain": self.config.domain
        }


# Convenience function for quick proposal generation
def propose_idea(
    domain: str = "efficient fine-tuning",
    previous_ideas: list[str] | None = None,
    seed: Optional[int] = None
) -> ResearchProposal:
    """Quick helper to generate a single proposal."""
    config = ProposalConfig(domain=domain)
    proposer = IdeaProposer(config)
    return proposer.generate_proposal(previous_ideas, seed=seed)


if __name__ == "__main__":
    # Test the proposer
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing IdeaProposer")
    print("=" * 60)
    
    proposer = IdeaProposer(ProposalConfig(domain="efficient fine-tuning"))
    proposal = proposer.generate_proposal(seed=42)
    
    print("\nGenerated Proposal:")
    print("-" * 40)
    print(f"Hypothesis: {proposal.hypothesis}")
    print(f"\nMotivation: {proposal.motivation}")
    print(f"\nMethod:\n{chr(10).join('  • ' + m for m in proposal.method_summary)}")
    print(f"\nExpected Outcome: {proposal.expected_outcome}")
    print(f"\nScores - Feasibility: {proposal.feasibility_score}, Novelty: {proposal.novelty_score}")
    print(f"Estimated Cost: {proposal.estimated_cost}")
    print(f"Keywords: {', '.join(proposal.keywords)}")
    
    is_valid, reason = proposer.validate_proposal(proposal)
    print(f"\nValidation: {'✓ Valid' if is_valid else f'✗ Invalid - {reason}'}")
    
    print(f"\nStatistics: {proposer.get_statistics()}")
