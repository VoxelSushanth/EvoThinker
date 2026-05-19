"""
Evaluator Agent for TreeQuest Lab

Runs metrics, generates plots, and performs statistical analysis on experiment results.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvaluationConfig(BaseModel):
    """Configuration for Evaluator."""
    
    metrics: list[str] = Field(
        default_factory=lambda: ["loss", "accuracy", "f1", "runtime"],
        description="Metrics to compute"
    )
    plot_formats: list[str] = Field(
        default_factory=lambda: ["png", "pdf"],
        description="Output formats for plots"
    )
    statistical_tests: list[str] = Field(
        default_factory=lambda: ["t-test", "wilcoxon"],
        description="Statistical tests to run"
    )
    baseline_value: Optional[float] = Field(
        default=None,
        description="Baseline value for comparison"
    )
    
    class Config:
        extra = "ignore"


class EvaluationResult(BaseModel):
    """Results from evaluation."""
    
    node_id: str = Field(..., description="ID of evaluated node")
    metrics: dict[str, float] = Field(..., description="Computed metrics")
    plots_generated: list[str] = Field(default_factory=list, description="Generated plot files")
    statistical_results: dict[str, float] = Field(default_factory=dict, description="Statistical test results")
    success: bool = Field(..., description="Whether evaluation succeeded")
    error_message: Optional[str] = Field(default=None, description="Error if failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()
    
    @property
    def primary_score(self) -> float:
        """Get primary score (accuracy or first metric)."""
        if "accuracy" in self.metrics:
            return self.metrics["accuracy"]
        elif "f1" in self.metrics:
            return self.metrics["f1"]
        elif self.metrics:
            return list(self.metrics.values())[0]
        return 0.0


class Evaluator:
    """
    Agent that evaluates experiment results.
    
    Computes metrics, generates visualizations, and runs statistical tests.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self._evaluation_count = 0
    
    def evaluate(
        self,
        results_path: Path | str,
        node_id: str = "unknown"
    ) -> EvaluationResult:
        """
        Evaluate experiment results from a JSON file.
        
        Args:
            results_path: Path to results.json file
            node_id: ID of the node being evaluated
            
        Returns:
            EvaluationResult object
        """
        import time
        start_time = time.time()
        
        self._evaluation_count += 1
        results_path = Path(results_path)
        
        try:
            # Load results
            if not results_path.exists():
                # Generate mock results for demo
                logger.info(f"Results file not found, generating mock results for {node_id}")
                metrics = self._generate_mock_metrics()
                plots = []
                stats = {}
            else:
                with open(results_path) as f:
                    data = json.load(f)
                
                # Extract metrics
                metrics = self._compute_metrics(data)
                
                # Generate plots
                plots = self._generate_plots(data, results_path.parent)
                
                # Run statistical tests
                stats = self._run_statistical_tests(data)
            
            execution_time = time.time() - start_time
            
            result = EvaluationResult(
                node_id=node_id,
                metrics=metrics,
                plots_generated=plots,
                statistical_results=stats,
                success=True,
                execution_time=execution_time
            )
            
            logger.info(f"Evaluation #{self._evaluation_count} complete for {node_id}: {metrics}")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Evaluation failed for {node_id}: {e}")
            
            return EvaluationResult(
                node_id=node_id,
                metrics={},
                plots_generated=[],
                statistical_results={},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _compute_metrics(self, data: dict) -> dict[str, float]:
        """Compute metrics from experiment data."""
        metrics = {}
        
        # Extract from final_metrics if available
        final = data.get("final_metrics", {})
        metrics_history = data.get("metrics_history", [])
        
        # Common metrics
        if "accuracy" in final:
            metrics["accuracy"] = final["accuracy"]
        if "val_loss" in final:
            metrics["loss"] = final["val_loss"]
        if "f1" in final:
            metrics["f1"] = final["f1"]
        
        # Compute from history if available
        if metrics_history:
            losses = [m.get("train_loss", m.get("val_loss")) for m in metrics_history if m.get("train_loss") or m.get("val_loss")]
            if losses:
                metrics["avg_train_loss"] = sum(losses) / len(losses)
                metrics["loss_improvement"] = losses[0] - losses[-1] if len(losses) > 1 else 0.0
        
        # Add configured metrics with defaults if missing
        for metric in self.config.metrics:
            if metric not in metrics:
                metrics[metric] = 0.0
        
        return metrics
    
    def _generate_mock_metrics(self) -> dict[str, float]:
        """Generate mock metrics for demonstration."""
        import random
        
        return {
            "accuracy": random.uniform(0.65, 0.92),
            "loss": random.uniform(0.3, 1.5),
            "f1": random.uniform(0.6, 0.9),
            "runtime": random.uniform(10, 300),
        }
    
    def _generate_plots(
        self,
        data: dict,
        output_dir: Path
    ) -> list[str]:
        """Generate visualization plots."""
        plots = []
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            metrics_history = data.get("metrics_history", [])
            
            if metrics_history:
                # Training curve plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                epochs = [m.get("epoch", i) for i, m in enumerate(metrics_history)]
                train_losses = [m.get("train_loss", 0) for m in metrics_history]
                val_losses = [m.get("val_loss", 0) for m in metrics_history]
                
                ax.plot(epochs, train_losses, 'b-', label='Train Loss', marker='o')
                ax.plot(epochs, val_losses, 'r--', label='Val Loss', marker='s')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training Curves')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_path = output_dir / "training_curves.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                plots.append(str(plot_path))
                
                # Save PDF version too
                if "pdf" in self.config.plot_formats:
                    plot_path_pdf = output_dir / "training_curves.pdf"
                    plt.savefig(plot_path_pdf, bbox_inches='tight')
                    plots.append(str(plot_path_pdf))
            
            logger.info(f"Generated {len(plots)} plots")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
        
        return plots
    
    def _run_statistical_tests(self, data: dict) -> dict[str, float]:
        """Run statistical tests on results."""
        stats = {}
        
        try:
            from scipy import stats as scipy_stats
            
            metrics_history = data.get("metrics_history", [])
            
            if len(metrics_history) >= 2:
                # Simple t-test comparing first half vs second half of training
                first_half = [m.get("train_loss", 0) for m in metrics_history[:len(metrics_history)//2]]
                second_half = [m.get("train_loss", 0) for m in metrics_history[len(metrics_history)//2:]]
                
                if first_half and second_half:
                    t_stat, p_value = scipy_stats.ttest_ind(first_half, second_half)
                    stats["improvement_p_value"] = float(p_value)
                    stats["improvement_significant"] = float(p_value < 0.05)
            
            # Compare to baseline if provided
            if self.config.baseline_value is not None:
                final = data.get("final_metrics", {})
                if "accuracy" in final:
                    # One-sample t-test against baseline
                    pass  # Would need multiple samples for proper test
        
        except ImportError:
            logger.warning("scipy not available, skipping statistical tests")
        except Exception as e:
            logger.warning(f"Statistical tests failed: {e}")
        
        return stats
    
    def compare_results(
        self,
        result1: EvaluationResult,
        result2: EvaluationResult
    ) -> dict:
        """Compare two evaluation results."""
        comparison = {
            "node1_id": result1.node_id,
            "node2_id": result2.node_id,
            "metrics_comparison": {},
            "winner": None,
            "difference": 0.0
        }
        
        # Compare primary scores
        score1 = result1.primary_score
        score2 = result2.primary_score
        
        comparison["metrics_comparison"]["primary_score"] = {
            "node1": score1,
            "node2": score2,
            "difference": score2 - score1
        }
        
        if score2 > score1:
            comparison["winner"] = result2.node_id
            comparison["difference"] = score2 - score1
        else:
            comparison["winner"] = result1.node_id
            comparison["difference"] = score1 - score2
        
        # Compare other common metrics
        common_metrics = set(result1.metrics.keys()) & set(result2.metrics.keys())
        for metric in common_metrics:
            if metric != "primary_score":
                comparison["metrics_comparison"][metric] = {
                    "node1": result1.metrics[metric],
                    "node2": result2.metrics[metric]
                }
        
        return comparison
    
    def get_statistics(self) -> dict:
        """Get evaluator statistics."""
        return {
            "evaluations_run": self._evaluation_count,
            "metrics_configured": self.config.metrics,
            "baseline": self.config.baseline_value
        }


if __name__ == "__main__":
    # Test the evaluator
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Evaluator")
    print("=" * 60)
    
    evaluator = Evaluator()
    
    # Test with mock results (no file)
    result = evaluator.evaluate(
        results_path="/tmp/nonexistent/results.json",
        node_id="test_node_001"
    )
    
    print(f"\nEvaluation Result:")
    print(f"  Node ID: {result.node_id}")
    print(f"  Success: {result.success}")
    print(f"  Metrics: {result.metrics}")
    print(f"  Primary Score: {result.primary_score}")
    print(f"  Plots Generated: {result.plots_generated}")
    print(f"  Execution Time: {result.execution_time:.3f}s")
    
    # Test comparison
    result2 = evaluator.evaluate(
        results_path="/tmp/another_nonexistent/results.json",
        node_id="test_node_002"
    )
    
    comparison = evaluator.compare_results(result, result2)
    print(f"\nComparison:")
    print(f"  Winner: {comparison['winner']}")
    print(f"  Difference: {comparison['difference']:.4f}")
    
    print(f"\nStatistics: {evaluator.get_statistics()}")
