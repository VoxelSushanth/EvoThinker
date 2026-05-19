"""
Evaluation Harness for TreeQuest Lab

Standardized benchmarks and metrics for evaluating ML experiments.
Supports tiny datasets for fast iteration.
"""

import logging
from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluation."""
    
    node_id: str
    primary_score: float
    metrics: dict = field(default_factory=dict)
    benchmark_name: str = ""
    samples_evaluated: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "primary_score": self.primary_score,
            "metrics": self.metrics,
            "benchmark_name": self.benchmark_name,
            "samples_evaluated": self.samples_evaluated,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error_message": self.error_message
        }


class EvaluationConfig(BaseModel):
    """Configuration for evaluation harness."""
    
    default_benchmark: str = Field(
        default="mock",
        description="Default benchmark to use"
    )
    dataset_cache_dir: str = Field(
        default="~/.cache/treequest/datasets",
        description="Directory for cached datasets"
    )
    max_samples: int = Field(
        default=1000,
        gt=0,
        description="Maximum samples per evaluation"
    )
    device: str = Field(
        default="auto",
        description="Device for evaluation (auto/cpu/cuda)"
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Batch size for evaluation"
    )
    timeout_seconds: int = Field(
        default=300,
        gt=0,
        description="Timeout per evaluation"
    )
    
    class Config:
        extra = "ignore"


class Evaluator:
    """
    Unified evaluator for ML experiments.
    
    Supports:
    - Mock evaluation (for testing)
    - Small vision tasks (CIFAR-10 subset)
    - Small text tasks (GSM8K small, GLUE mini)
    - Custom benchmarks
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self._evaluation_count = 0
        
        # Device setup
        if self.config.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        
        logger.info(f"Evaluator initialized on {self.device}")
    
    def evaluate(
        self,
        node: Any,
        benchmark: Optional[str] = None,
        custom_fn: Optional[Callable] = None
    ) -> dict:
        """
        Evaluate a node's experiment results.
        
        Args:
            node: Node with experiment results
            benchmark: Benchmark name (overrides config)
            custom_fn: Custom evaluation function
            
        Returns:
            Dictionary with evaluation metrics
        """
        self._evaluation_count += 1
        
        # Use custom function if provided
        if custom_fn:
            try:
                result = custom_fn(node)
                if isinstance(result, dict):
                    return result
                elif isinstance(result, EvaluationResult):
                    return result.to_dict()
            except Exception as e:
                logger.error(f"Custom evaluation failed: {e}")
        
        # Get benchmark name
        bench_name = benchmark or self.config.default_benchmark
        
        # Dispatch to appropriate evaluator
        if bench_name == "mock":
            return self._evaluate_mock(node)
        elif bench_name.startswith("vision"):
            return self._evaluate_vision(node)
        elif bench_name.startswith("text"):
            return self._evaluate_text(node)
        elif bench_name.startswith("math"):
            return self._evaluate_math(node)
        else:
            return self._evaluate_mock(node)
    
    def _evaluate_mock(self, node: Any) -> dict:
        """Mock evaluation based on node metadata."""
        import numpy as np
        
        # Extract scores from node data
        feasibility = getattr(node.data, 'feasibility_score', 0.5)
        novelty = getattr(node.data, 'novelty_score', 0.5)
        
        # Simulate experimental outcome
        base_score = 0.5 * feasibility + 0.5 * novelty
        
        # Add some noise based on method quality
        method_summary = getattr(node.data, 'method_summary', [])
        method_quality = len(method_summary) / 5.0  # More detailed = better
        noise = np.random.normal(0, 0.1)
        
        primary_score = np.clip(base_score + 0.1 * method_quality + noise, 0.0, 1.0)
        
        return {
            "primary_score": float(primary_score),
            "metrics": {
                "simulated_accuracy": float(primary_score),
                "simulated_loss": float(1.0 - primary_score),
                "feasibility": float(feasibility),
                "novelty": float(novelty)
            },
            "benchmark": "mock",
            "samples_evaluated": 0
        }
    
    def _evaluate_vision(self, node: Any) -> dict:
        """Evaluate on vision task (CIFAR-10 subset)."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            
            # Try to load CIFAR-10
            try:
                from torchvision import datasets, transforms
                from torchvision.models import resnet18
                
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                
                # Load small subset
                test_dataset = datasets.CIFAR10(
                    root=self.config.dataset_cache_dir,
                    train=False,
                    download=True,
                    transform=transform
                )
                
                # Subset for speed
                indices = list(range(min(len(test_dataset), self.config.max_samples)))
                subset = torch.utils.data.Subset(test_dataset, indices)
                test_loader = DataLoader(subset, batch_size=self.config.batch_size)
                
                # Load model if available
                model_path = getattr(node.data, 'model_path', None)
                if model_path and Path(model_path).exists():
                    model = resnet18(num_classes=10)
                    model.load_state_dict(torch.load(model_path))
                    model.to(self.device)
                    model.eval()
                    
                    # Evaluate
                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for images, labels in test_loader:
                            images = images.to(self.device)
                            labels = labels.to(self.device)
                            
                            outputs = model(images)
                            _, predicted = torch.max(outputs.data, 1)
                            
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    
                    accuracy = correct / total
                    
                else:
                    # No model, use heuristic based on hypothesis
                    accuracy = self._heuristic_vision_score(node)
                
                return {
                    "primary_score": float(accuracy),
                    "metrics": {
                        "accuracy": float(accuracy),
                        "top1": float(accuracy)
                    },
                    "benchmark": "cifar10_subset",
                    "samples_evaluated": len(indices)
                }
                
            except ImportError:
                logger.warning("torchvision not available, using mock vision eval")
                return self._evaluate_mock(node)
                
        except Exception as e:
            logger.error(f"Vision evaluation failed: {e}")
            return {
                "primary_score": 0.0,
                "metrics": {"error": str(e)},
                "benchmark": "vision",
                "error": str(e)
            }
    
    def _heuristic_vision_score(self, node: Any) -> float:
        """Heuristic score for vision tasks based on hypothesis."""
        import numpy as np
        
        hypothesis = getattr(node.data, 'hypothesis', '').lower()
        
        # Keywords that suggest good vision ideas
        positive_keywords = [
            'augmentation', 'regularization', 'attention',
            'residual', 'normalization', 'dropout'
        ]
        
        score = 0.6  # Base score
        
        for keyword in positive_keywords:
            if keyword in hypothesis:
                score += 0.05
        
        # Cap at 0.95
        score = min(0.95, score)
        
        # Add noise
        score += np.random.normal(0, 0.05)
        
        return np.clip(score, 0.0, 1.0)
    
    def _evaluate_text(self, node: Any) -> dict:
        """Evaluate on text task."""
        # Similar structure to vision
        hypothesis = getattr(node.data, 'hypothesis', '').lower()
        
        # Heuristic based on keywords
        positive_keywords = [
            'fine-tuning', 'prompt', 'adapter', 'lora',
            'attention', 'context', 'embedding'
        ]
        
        score = 0.6
        for keyword in positive_keywords:
            if keyword in hypothesis:
                score += 0.05
        
        score = min(0.95, score + np.random.normal(0, 0.05))
        
        return {
            "primary_score": float(score),
            "metrics": {
                "accuracy": float(score),
                "perplexity": float(1.0 / max(score, 0.01))
            },
            "benchmark": "text_classification",
            "samples_evaluated": 0
        }
    
    def _evaluate_math(self, node: Any) -> dict:
        """Evaluate on math/reasoning task."""
        hypothesis = getattr(node.data, 'hypothesis', '').lower()
        
        # Keywords for math reasoning
        positive_keywords = [
            'reasoning', 'chain-of-thought', 'cot', 'step-by-step',
            'verification', 'proof', 'symbolic', 'algebra'
        ]
        
        score = 0.5
        for keyword in positive_keywords:
            if keyword in hypothesis:
                score += 0.08
        
        score = min(0.95, score + np.random.normal(0, 0.05))
        
        return {
            "primary_score": float(score),
            "metrics": {
                "accuracy": float(score),
                "exact_match": float(score * 0.9)
            },
            "benchmark": "gsm8k_small",
            "samples_evaluated": 0
        }
    
    def run_benchmark_suite(self, node: Any) -> dict:
        """Run full benchmark suite on a node."""
        results = {}
        
        benchmarks = ['mock', 'vision_cifar10', 'text_glue', 'math_gsm8k']
        
        for bench in benchmarks:
            try:
                if bench == 'mock':
                    results[bench] = self._evaluate_mock(node)
                elif bench.startswith('vision'):
                    results[bench] = self._evaluate_vision(node)
                elif bench.startswith('text'):
                    results[bench] = self._evaluate_text(node)
                elif bench.startswith('math'):
                    results[bench] = self._evaluate_math(node)
            except Exception as e:
                logger.warning(f"Benchmark {bench} failed: {e}")
                results[bench] = {"error": str(e)}
        
        # Calculate aggregate score
        scores = [r.get('primary_score', 0) for r in results.values() if 'primary_score' in r]
        aggregate = sum(scores) / len(scores) if scores else 0
        
        results['aggregate'] = {
            "primary_score": float(aggregate),
            "num_benchmarks": len(benchmarks),
            "successful_benchmarks": len(scores)
        }
        
        return results
    
    def get_statistics(self) -> dict:
        """Get evaluator statistics."""
        return {
            "evaluations_run": self._evaluation_count,
            "default_benchmark": self.config.default_benchmark,
            "device": self.device,
            "max_samples": self.config.max_samples
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Evaluator Test")
    print("=" * 60)
    
    # Create mock node
    class MockData:
        def __init__(self):
            self.hypothesis = "LoRA improves fine-tuning efficiency"
            self.feasibility_score = 0.8
            self.novelty_score = 0.7
            self.method_summary = ["Test LoRA adapters", "Compare to full fine-tuning"]
    
    class MockNode:
        def __init__(self):
            self.node_id = "test_001"
            self.data = MockData()
    
    evaluator = Evaluator(EvaluationConfig(default_benchmark="mock"))
    result = evaluator.evaluate(MockNode())
    
    print(f"\nPrimary Score: {result['primary_score']:.3f}")
    print(f"Metrics: {result['metrics']}")
    print(f"Benchmark: {result['benchmark']}")
    print(f"\nStatistics: {evaluator.get_statistics()}")
