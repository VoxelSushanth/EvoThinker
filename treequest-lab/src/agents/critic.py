"""
Critic Agent for TreeQuest Lab

Scores ideas on novelty, feasibility, and promise. Provides reflection and feedback.
"""

import json
import logging
import re
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CriticConfig(BaseModel):
    """Configuration for Critic."""
    
    novelty_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for novelty score"
    )
    feasibility_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for feasibility score"
    )
    impact_weight: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight for potential impact score"
    )
    min_score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum overall score to proceed"
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="LLM model for critique generation"
    )
    
    class Config:
        extra = "ignore"


class CritiqueResult(BaseModel):
    """Result from critic evaluation."""
    
    node_id: str = Field(..., description="ID of evaluated node")
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Novelty assessment")
    feasibility_score: float = Field(..., ge=0.0, le=1.0, description="Feasibility assessment")
    impact_score: float = Field(..., ge=0.0, le=1.0, description="Potential impact assessment")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Weighted overall score")
    strengths: list[str] = Field(default_factory=list, description="Identified strengths")
    weaknesses: list[str] = Field(default_factory=list, description="Identified weaknesses")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    recommendation: str = Field(..., description="Recommendation: 'proceed', 'revise', or 'reject'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in assessment")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class Critic:
    """
    Agent that critiques and scores research proposals/results.
    
    Evaluates novelty, feasibility, and potential impact.
    Provides constructive feedback for improvement.
    """
    
    SYSTEM_PROMPT = """You are an expert ML research reviewer. Your role is to critically evaluate research proposals and experimental results.

Evaluation Criteria:
1. Novelty (0-1): How original is this idea compared to existing work?
2. Feasibility (0-1): Can this be tested with limited compute (<8 GPU hours)?
3. Impact (0-1): If successful, how much would this advance the field?

Be honest but constructive. Identify both strengths and weaknesses.
Provide specific, actionable suggestions for improvement.

Format your response as valid JSON with these fields:
- novelty_score: 0.0-1.0
- feasibility_score: 0.0-1.0  
- impact_score: 0.0-1.0
- overall_score: 0.0-1.0 (weighted average)
- strengths: List of 2-4 strengths
- weaknesses: List of 2-4 weaknesses
- suggestions: List of 2-4 improvement suggestions
- recommendation: "proceed", "revise", or "reject"
- confidence: 0.0-1.0 (your confidence in this assessment)"""

    def __init__(self, config: Optional[CriticConfig] = None):
        self.config = config or CriticConfig()
        self._critique_count = 0
        self._client = None
        
        # Validate weights sum to 1.0
        total_weight = (
            self.config.novelty_weight + 
            self.config.feasibility_weight + 
            self.config.impact_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Critic weights sum to {total_weight}, normalizing to 1.0")
    
    @property
    def client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            try:
                from openai import OpenAI
                import os
                
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not set. Using heuristic mode.")
                    return None
                    
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("openai package not installed. Using heuristic mode.")
                return None
        return self._client
    
    def critique(
        self,
        hypothesis: str,
        method_summary: list[str],
        results: Optional[dict] = None,
        previous_work: list[str] | None = None,
        node_id: str = "unknown"
    ) -> CritiqueResult:
        """
        Critique a research proposal or completed experiment.
        
        Args:
            hypothesis: The research hypothesis
            method_summary: Description of methods used/proposed
            results: Optional experiment results
            previous_work: List of related previous work for novelty check
            node_id: ID of the node being critiqued
            
        Returns:
            CritiqueResult object
        """
        self._critique_count += 1
        
        # Try LLM critique if available
        if self.client is not None:
            return self._critique_with_llm(
                hypothesis, method_summary, results, previous_work, node_id
            )
        else:
            return self._critique_heuristic(
                hypothesis, method_summary, results, previous_work, node_id
            )
    
    def _critique_with_llm(
        self,
        hypothesis: str,
        method_summary: list[str],
        results: Optional[dict],
        previous_work: list[str] | None,
        node_id: str
    ) -> CritiqueResult:
        """Generate critique using LLM."""
        
        # Build prompt
        context_parts = [f"Hypothesis: {hypothesis}"]
        context_parts.append("\nMethod:")
        for step in method_summary:
            context_parts.append(f"- {step}")
        
        if results:
            context_parts.append("\nResults:")
            for key, value in results.items():
                context_parts.append(f"- {key}: {value}")
        
        if previous_work:
            context_parts.append("\nRelated Work (for novelty comparison):")
            for work in previous_work:
                context_parts.append(f"- {work}")
        
        user_prompt = "\n".join(context_parts)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent scoring
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
            
            data = json.loads(content)
            
            # Create result with validation
            result = CritiqueResult(
                node_id=node_id,
                novelty_score=float(data.get("novelty_score", 0.5)),
                feasibility_score=float(data.get("feasibility_score", 0.5)),
                impact_score=float(data.get("impact_score", 0.5)),
                overall_score=float(data.get("overall_score", 0.5)),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                suggestions=data.get("suggestions", []),
                recommendation=data.get("recommendation", "revise"),
                confidence=float(data.get("confidence", 0.7))
            )
            
            logger.info(f"LLM critique #{self._critique_count} for {node_id}: score={result.overall_score:.2f}")
            return result
            
        except Exception as e:
            logger.warning(f"LLM critique failed: {e}, falling back to heuristic")
            return self._critique_heuristic(hypothesis, method_summary, results, previous_work, node_id)
    
    def _critique_heuristic(
        self,
        hypothesis: str,
        method_summary: list[str],
        results: Optional[dict],
        previous_work: list[str] | None,
        node_id: str
    ) -> CritiqueResult:
        """Generate critique using heuristics (no LLM)."""
        
        import re
        
        # Heuristic novelty scoring
        novelty_score = self._assess_novelty_heuristic(hypothesis, previous_work)
        
        # Heuristic feasibility scoring
        feasibility_score = self._assess_feasibility_heuristic(method_summary)
        
        # Heuristic impact scoring
        impact_score = self._assess_impact_heuristic(hypothesis, results)
        
        # Calculate weighted overall
        overall_score = (
            novelty_score * self.config.novelty_weight +
            feasibility_score * self.config.feasibility_weight +
            impact_score * self.config.impact_weight
        )
        
        # Generate strengths/weaknesses based on scores
        strengths = []
        weaknesses = []
        suggestions = []
        
        if novelty_score >= 0.7:
            strengths.append("High novelty - explores underexplored direction")
        elif novelty_score < 0.5:
            weaknesses.append("Limited novelty - similar to existing work")
            suggestions.append("Consider combining with orthogonal techniques")
        
        if feasibility_score >= 0.7:
            strengths.append("Highly feasible with limited compute")
        elif feasibility_score < 0.5:
            weaknesses.append("May require significant computational resources")
            suggestions.append("Consider smaller-scale pilot experiments")
        
        if impact_score >= 0.7:
            strengths.append("High potential impact if successful")
        elif impact_score < 0.5:
            weaknesses.append("Incremental improvement even if successful")
            suggestions.append("Clarify broader implications beyond immediate task")
        
        # Add result-specific feedback
        if results:
            if results.get("accuracy", 0) > 0.85:
                strengths.append("Strong empirical results")
            elif results.get("accuracy", 1.0) < 0.6:
                weaknesses.append("Results below typical baselines")
                suggestions.append("Verify experimental setup and hyperparameters")
        
        # Determine recommendation
        if overall_score >= 0.7:
            recommendation = "proceed"
        elif overall_score >= self.config.min_score_threshold:
            recommendation = "revise"
        else:
            recommendation = "reject"
        
        # Confidence based on amount of information
        confidence = 0.6
        if results:
            confidence += 0.1
        if previous_work and len(previous_work) > 2:
            confidence += 0.1
        if len(method_summary) >= 3:
            confidence += 0.1
        confidence = min(confidence, 0.95)
        
        logger.info(f"Heuristic critique #{self._critique_count} for {node_id}: score={overall_score:.2f}")
        
        return CritiqueResult(
            node_id=node_id,
            novelty_score=novelty_score,
            feasibility_score=feasibility_score,
            impact_score=impact_score,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            recommendation=recommendation,
            confidence=confidence
        )
    
    def _assess_novelty_heuristic(
        self,
        hypothesis: str,
        previous_work: list[str] | None
    ) -> float:
        """Assess novelty using keyword matching."""
        import random
        
        if not previous_work:
            # No comparison available, give moderate-high novelty
            return 0.65 + random.uniform(-0.1, 0.1)
        
        # Simple keyword overlap check
        hypothesis_words = set(re.findall(r'\w+', hypothesis.lower()))
        
        max_overlap = 0.0
        for work in previous_work:
            work_words = set(re.findall(r'\w+', work.lower()))
            overlap = len(hypothesis_words & work_words) / max(len(hypothesis_words), 1)
            max_overlap = max(max_overlap, overlap)
        
        # Lower overlap = higher novelty
        novelty = 1.0 - max_overlap
        novelty = max(0.2, min(0.95, novelty + random.uniform(-0.1, 0.1)))
        
        return novelty
    
    def _assess_feasibility_heuristic(self, method_summary: list[str]) -> float:
        """Assess feasibility from method description."""
        import random
        
        method_str = " ".join(method_summary).lower()
        
        # Keywords suggesting high compute requirements
        expensive_keywords = [
            "full fine-tuning", "large model", "billion", "pretrain",
            "extensive search", "grid search", "ensemble"
        ]
        
        # Keywords suggesting efficiency
        efficient_keywords = [
            "lora", "peft", "adapter", "small", "tiny", "subset",
            "ablation", "analysis", "comparison"
        ]
        
        expensive_count = sum(1 for kw in expensive_keywords if kw in method_str)
        efficient_count = sum(1 for kw in efficient_keywords if kw in method_str)
        
        # Base feasibility
        feasibility = 0.6
        
        if expensive_count > 0:
            feasibility -= expensive_count * 0.15
        if efficient_count > 0:
            feasibility += efficient_count * 0.1
        
        # Method length suggests complexity
        if len(method_summary) > 5:
            feasibility -= 0.1
        
        feasibility = max(0.2, min(0.95, feasibility + random.uniform(-0.1, 0.1)))
        return feasibility
    
    def _assess_impact_heuristic(
        self,
        hypothesis: str,
        results: Optional[dict]
    ) -> float:
        """Assess potential impact."""
        import random
        
        # Keywords suggesting broad impact
        impact_keywords = [
            "generalization", "robustness", "efficiency", "scaling",
            "fundamental", "principle", "trade-off", "benchmark"
        ]
        
        hypothesis_lower = hypothesis.lower()
        impact_count = sum(1 for kw in impact_keywords if kw in hypothesis_lower)
        
        # Base impact
        impact = 0.5 + (impact_count * 0.1)
        
        # Boost if results are strong
        if results:
            if results.get("accuracy", 0) > 0.9:
                impact += 0.2
            elif results.get("improvement", 0) > 0.1:
                impact += 0.15
        
        impact = max(0.2, min(0.95, impact + random.uniform(-0.1, 0.1)))
        return impact
    
    def reflect_on_path(
        self,
        path_nodes: list[dict],
        path_results: list
    ) -> dict:
        """
        Reflect on a complete path through the tree.
        
        Args:
            path_nodes: List of node data along the path
            path_results: Corresponding evaluation results
            
        Returns:
            Reflection dictionary with insights
        """
        if not path_nodes:
            return {"insight": "No nodes to reflect on", "direction": "explore_new"}
        
        # Analyze trends
        scores = []
        for r in path_results:
            if isinstance(r, dict):
                if r.get("success", False):
                    metrics = r.get("metrics", {})
                    score = metrics.get("accuracy", metrics.get("primary_score", 0))
                    scores.append(score)
            else:
                if getattr(r, "success", False):
                    scores.append(getattr(r, "primary_score", 0))
        
        reflection = {
            "path_length": len(path_nodes),
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable_or_declining",
            "best_node_idx": scores.index(max(scores)) if scores else -1,
            "insights": [],
            "recommended_direction": "explore_new"
        }
        
        # Generate insights
        if len(scores) > 1:
            if all(scores[i] <= scores[i+1] for i in range(len(scores)-1)):
                reflection["insights"].append("Consistent improvement along path - continue this direction")
            elif all(scores[i] >= scores[i+1] for i in range(len(scores)-1)):
                reflection["insights"].append("Declining performance - consider backtracking")
            else:
                reflection["insights"].append("Mixed results - identify key differentiators")
        
        if reflection["average_score"] > 0.8:
            reflection["insights"].append("High-performing path worth deeper exploration")
            reflection["recommended_direction"] = "deepen"
        elif reflection["average_score"] < 0.4:
            reflection["insights"].append("Low-performing path - prioritize exploration elsewhere")
            reflection["recommended_direction"] = "explore_new"
        else:
            reflection["recommended_direction"] = "broaden"
        
        return reflection
    
    def get_statistics(self) -> dict:
        """Get critic statistics."""
        return {
            "critiques_generated": self._critique_count,
            "weights": {
                "novelty": self.config.novelty_weight,
                "feasibility": self.config.feasibility_weight,
                "impact": self.config.impact_weight
            },
            "threshold": self.config.min_score_threshold
        }


if __name__ == "__main__":
    # Test the critic
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Critic")
    print("=" * 60)
    
    critic = Critic()
    
    test_hypothesis = "LoRA rank adaptation during training improves convergence speed vs fixed rank"
    test_method = [
        "Compare fixed-rank LoRA vs adaptive-rank on GLUE tasks",
        "Start with low rank, increase based on gradient magnitude",
        "Measure convergence speed and final accuracy"
    ]
    
    test_results = {
        "accuracy": 0.87,
        "loss": 0.45,
        "convergence_epochs": 3
    }
    
    previous_work = [
        "Fixed-rank LoRA for efficient fine-tuning",
        "Adapter methods for parameter-efficient transfer learning"
    ]
    
    result = critic.critique(
        hypothesis=test_hypothesis,
        method_summary=test_method,
        results=test_results,
        previous_work=previous_work,
        node_id="test_node_001"
    )
    
    print(f"\nCritique Result:")
    print(f"  Node ID: {result.node_id}")
    print(f"  Scores - Novelty: {result.novelty_score:.2f}, Feasibility: {result.feasibility_score:.2f}, Impact: {result.impact_score:.2f}")
    print(f"  Overall Score: {result.overall_score:.2f}")
    print(f"  Recommendation: {result.recommendation}")
    print(f"  Confidence: {result.confidence:.2f}")
    
    print(f"\nStrengths:")
    for s in result.strengths:
        print(f"  ✓ {s}")
    
    print(f"\nWeaknesses:")
    for w in result.weaknesses:
        print(f"  ✗ {w}")
    
    print(f"\nSuggestions:")
    for s in result.suggestions:
        print(f"  → {s}")
    
    # Test path reflection
    mock_eval_results = [
        {"node_id": "n1", "metrics": {"accuracy": 0.65}, "success": True, "execution_time": 10},
        {"node_id": "n2", "metrics": {"accuracy": 0.75}, "success": True, "execution_time": 12},
        {"node_id": "n3", "metrics": {"accuracy": 0.87}, "success": True, "execution_time": 15},
    ]
    
    reflection = critic.reflect_on_path(
        path_nodes=[{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
        path_results=mock_eval_results
    )
    
    print(f"\nPath Reflection:")
    print(f"  Average Score: {reflection['average_score']:.2f}")
    print(f"  Trend: {reflection['trend']}")
    print(f"  Recommended Direction: {reflection['recommended_direction']}")
    for insight in reflection['insights']:
        print(f"  • {insight}")
    
    print(f"\nStatistics: {critic.get_statistics()}")
