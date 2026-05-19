"""
Evaluation harness for TreeQuest Lab.

Standardized benchmarks and metrics for ML experiments.
Supports tiny datasets and quick evaluation protocols.
"""

from src.evaluation.harness import (
    EvaluationHarness,
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark
)

__all__ = [
    "EvaluationHarness",
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark"
]
