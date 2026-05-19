# Agent Implementation Guide

This document details the implementation of specialized agents in TreeQuest Lab. Each agent is designed to be modular, testable, and composable via LangGraph.

## Agent Overview

TreeQuest Lab uses five specialized agents:

| Agent | Purpose | Input Schema | Output Schema |
|-------|---------|--------------|---------------|
| **Proposer** | Generate novel research hypotheses | Tree state, memory, prompt | `Hypothesis` |
| **Architect** | Write executable experiment code | Hypothesis, config | `CodeArtifact` |
| **Evaluator** | Run experiments and collect metrics | Code, results | `EvaluationResult` |
| **Critic** | Score and reflect on quality | Results, hypothesis | `Critique` |
| **Reporter** | Generate research reports | Full run data | `Report` |

## Base Agent Class

All agents inherit from a common base class that handles logging, retry logic, and LLM communication.

```python
# src/agents/base.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

InputT = TypeVar('InputT', bound=BaseModel)
OutputT = TypeVar('OutputT', bound=BaseModel)

class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Base class for all TreeQuest agents."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get('model_name', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2048)
        self.retry_attempts = config.get('retry_attempts', 3)
    
    @abstractmethod
    def generate(self, input_data: InputT) -> OutputT:
        """Generate output from input."""
        pass
    
    def _call_llm(self, prompt: str, response_model: type[OutputT]) -> OutputT:
        """Internal method to call LLM with structured output."""
        # Implementation uses instructor or openai beta APIs
        pass
```

## 1. Idea Proposer Agent

Generates novel, feasible research hypotheses based on tree state and memory.

### Implementation

```python
# src/agents/proposer.py
from pydantic import BaseModel, Field
from typing import List, Optional
from .base import BaseAgent

class Hypothesis(BaseModel):
    title: str = Field(description="Concise title of the idea")
    description: str = Field(description="Detailed explanation")
    motivation: str = Field(description="Why this might work")
    predicted_improvement: str = Field(description="Expected benefit")
    complexity: str = Field(description="low/medium/high")
    novelty_score: float = Field(ge=0, le=1)

class ProposerInput(BaseModel):
    root_prompt: str
    existing_hypotheses: List[str]
    failed_approaches: List[str]
    domain: str = "machine_learning"

class ProposerAgent(BaseAgent[ProposerInput, Hypothesis]):
    
    SYSTEM_PROMPT = """You are an expert ML researcher proposing novel experiments.
Your hypotheses should be:
- Novel but grounded in established principles
- Feasible to test in under 5 minutes on small datasets
- Clearly motivated with specific predictions
- Distinct from previously tried approaches"""

    def generate(self, input_data: ProposerInput) -> Hypothesis:
        prompt = self._build_prompt(input_data)
        return self._call_llm(prompt, response_model=Hypothesis)
    
    def _build_prompt(self, inp: ProposerInput) -> str:
        return f"""{self.SYSTEM_PROMPT}

Domain: {inp.domain}
Goal: {inp.root_prompt}

Previously tried (avoid repeating):
{chr(10).join('- ' + h for h in inp.existing_hypotheses)}

Failed approaches (learn from these):
{chr(10).join('- ' + f for f in inp.failed_approaches)}

Propose ONE new hypothesis that is novel and feasible."""
```

### Usage Example

```python
proposer = ProposerAgent(config={'model_name': 'gpt-4o-mini'})

hypothesis = proposer.generate(ProposerInput(
    root_prompt="Improve training efficiency of small transformers",
    existing_hypotheses=["Use AdamW instead of SGD", "Add layer normalization"],
    failed_approaches=["Increasing learning rate caused divergence"],
    domain="nlp"
))

print(hypothesis.title)
print(hypothesis.novelty_score)
```

## 2. Code Architect Agent

Converts hypotheses into safe, executable experiment code.

### Implementation

```python
# src/agents/architect.py
from pydantic import BaseModel

class CodeArtifact(BaseModel):
    code: str = Field(description="Complete Python script")
    dependencies: List[str] = Field(description="Required packages")
    estimated_runtime: int = Field(description="Seconds")
    safety_checks: List[str] = Field(description="Validations performed")

class ArchitectInput(BaseModel):
    hypothesis: Hypothesis
    benchmark: str
    resource_limits: dict

class ArchitectAgent(BaseAgent[ArchitectInput, CodeArtifact]):
    
    SYSTEM_PROMPT = """You are an expert ML engineer writing safe, efficient code.
Requirements:
- Use PyTorch Lightning or simple train loops
- Include error handling and logging
- Respect resource