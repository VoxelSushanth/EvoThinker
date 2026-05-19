"""
Experiment Manager for TreeQuest Lab

Central orchestrator that coordinates agents, manages tree search,
and controls the full research exploration pipeline.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.search.mcts import MCTS

logger = logging.getLogger(__name__)


class ExperimentManagerConfig(BaseModel):
    """Configuration for Experiment Manager."""
    
    # Tree search parameters
    max_nodes: int = Field(default=50, gt=0, description="Maximum nodes in tree")
    max_depth: int = Field(default=5, gt=0, description="Maximum tree depth")
    max_width: int = Field(default=3, gt=0, description="Maximum children per node")
    max_iterations: int = Field(default=100, gt=0, description="MCTS iterations")
    
    # Agent parameters
    proposals_per_expansion: int = Field(default=3, gt=0)
    min_novelty_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_feasibility_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Execution parameters
    run_experiments: bool = Field(default=False, description="Actually execute experiments")
    timeout_per_experiment: int = Field(default=3600, gt=0, description="Seconds per experiment")
    parallel_executions: int = Field(default=1, ge=1, le=8)
    
    # Output parameters
    output_dir: str = Field(default="outputs/runs")
    save_frequency: int = Field(default=10, gt=0, description="Save tree every N iterations")
    verbose: bool = Field(default=True)
    
    # Reproducibility
    seed: int = Field(default=42)
    
    class Config:
        extra = "ignore"


class ExperimentManager:
    """
    Central brain of TreeQuest Lab.
    
    Orchestrates the agentic tree search process:
    1. Selects nodes using MCTS
    2. Expands with IdeaProposer
    3. Generates code with CodeExecutor
    4. Executes experiments (optional)
    5. Evaluates results with Evaluator
    6. Reflects with Critic
    7. Backpropagates scores
    
    Inspired by Sakana AI's AI Scientist-v2 experiment manager.
    """
    
    def __init__(
        self,
        config: Optional[ExperimentManagerConfig] = None,
        tree: Optional[Any] = None,
        proposer: Optional[Any] = None,
        executor: Optional[Any] = None,
        evaluator: Optional[Any] = None,
        critic: Optional[Any] = None,
        reporter: Optional[Any] = None
    ):
        self.config = config or ExperimentManagerConfig()
        
        # Initialize or receive components
        self.tree = tree
        self.proposer = proposer
        self.executor = executor
        self.evaluator = evaluator
        self.critic = critic
        self.reporter = reporter
        
        # State tracking
        self._iteration = 0
        self._start_time = None
        self._stats_history = []
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExperimentManager initialized with config: {self.config}")
    
    def initialize_tree(self, root_hypothesis: Optional[str] = None) -> Any:
        """Initialize or reset the search tree."""
        from src.core.tree import Tree, TreeSearchConfig
        
        tree_config = TreeSearchConfig(
            max_depth=self.config.max_depth,
            max_width=self.config.max_width,
            max_nodes=self.config.max_nodes
        )
        
        self.tree = Tree(config=tree_config)
        
        # Add root node
        if root_hypothesis:
            hypothesis = root_hypothesis
            motivation = "Root hypothesis for research exploration"
        else:
            hypothesis = "Initial research direction exploration"
            motivation = "Starting point for tree search"
        
        self.tree.add_root(hypothesis=hypothesis, description=motivation)
        logger.info("Tree initialized with root node")
        
        return self.tree
    
    def run(
        self,
        budget: Optional[int] = None,
        resume_from: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Run the full agentic tree search.
        
        Args:
            budget: Number of iterations (overrides config)
            resume_from: Path to checkpoint file to resume from
            
        Returns:
            Final statistics dictionary
        """
        self._start_time = time.time()
        iterations = budget or self.config.max_iterations
        
        # Load checkpoint if resuming
        if resume_from:
            self._load_checkpoint(resume_from)
            logger.info(f"Resumed from checkpoint: {resume_from}")
        
        # Initialize tree if needed
        if self.tree is None:
            self.initialize_tree()
        
        logger.info(f"Starting experiment manager with {iterations} iterations")
        print(f"\n{'='*60}")
        print(f"TREEQUEST LAB - Agentic Research Exploration")
        print(f"{'='*60}")
        print(f"Iterations: {iterations}")
        print(f"Max nodes: {self.config.max_nodes}")
        print(f"Output dir: {self.output_dir}")
        print(f"{'='*60}\n")
        
        stats_history = []
        
        try:
            for iteration in range(iterations):
                self._iteration = iteration
                
                # Check termination conditions
                if self._should_terminate():
                    logger.info("Termination condition met")
                    break
                
                # Run one iteration of tree search
                iter_stats = self._run_iteration()
                stats_history.append(iter_stats)
                
                # Periodic saving
                if (iteration + 1) % self.config.save_frequency == 0:
                    self._save_checkpoint()
                
                # Log progress
                if self.config.verbose and (iteration + 1) % 10 == 0:
                    self._log_progress(iter_stats)
            
            # Final save
            self._save_checkpoint(final=True)
            
            # Generate final report
            if self.reporter:
                self._generate_final_report()
            
            # Compile final statistics
            final_stats = self._compile_statistics(stats_history)
            
            print(f"\n{'='*60}")
            print(f"EXPLORATION COMPLETE")
            print(f"{'='*60}")
            print(f"Total iterations: {final_stats['iterations_completed']}")
            print(f"Total nodes: {final_stats['total_nodes']}")
            print(f"Best score: {final_stats['best_score']:.3f}")
            print(f"Duration: {final_stats['duration_seconds']:.1f}s")
            print(f"{'='*60}\n")
            
            return final_stats
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self._save_checkpoint(interrupted=True)
            return self._compile_statistics(stats_history, interrupted=True)
        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            self._save_checkpoint(error=str(e))
            raise
    
    def _run_iteration(self) -> dict[str, Any]:
        """Run one iteration of the agentic tree search loop."""
        iter_start = time.time()
        stats = {
            "iteration": self._iteration,
            "phase": "",
            "success": False,
            "node_id": None,
            "score": 0.0
        }
        
        try:
            # Phase 1: Selection (using MCTS)
            stats["phase"] = "selection"
            selected_node = self._select_node()
            
            if selected_node is None:
                logger.warning("No node selected, skipping iteration")
                return stats
            
            stats["node_id"] = selected_node.id
            
            # Phase 2: Expansion (generate new hypotheses)
            stats["phase"] = "expansion"
            children = self._expand_node(selected_node)
            
            if not children:
                # No children generated, mark as visited
                selected_node.visits += 1
                return stats
            
            # Phase 3: Simulation/Execution (optional)
            stats["phase"] = "execution"
            if self.config.run_experiments:
                for child in children:
                    self._execute_experiment(child)
            
            # Phase 4: Evaluation
            stats["phase"] = "evaluation"
            for child in children:
                self._evaluate_node(child)
            
            # Phase 5: Reflection/Critique
            stats["phase"] = "reflection"
            for child in children:
                self._reflect_on_node(child)
            
            # Phase 6: Backpropagation
            stats["phase"] = "backpropagation"
            for child in children:
                score = self._get_node_score(child)
                self._backpropagate(child, score)
                stats["score"] = max(stats["score"], score)
            
            stats["success"] = True
            
        except Exception as e:
            logger.error(f"Iteration {self._iteration} failed: {e}", exc_info=True)
            stats["success"] = False
        
        stats["duration"] = time.time() - iter_start
        return stats
    
    def _select_node(self):
        """Select a node for expansion using MCTS."""
        from src.search.mcts import MCTS, MCTSConfig
        
        mcts_config = MCTSConfig(
            max_iterations=min(10, self.config.max_iterations // 10),
            max_depth=self.config.max_depth,
            c_exploration=1.414,
            progressive_widening=True
        )
        
        mcts = MCTS(config=mcts_config)
        
        # Use MCTS to select promising leaf
        # For now, use simpler selection if tree is small
        if len(self.tree) < 5:
            # Early stage: prefer unexplored nodes
            return self._select_bfs_unexplored()
        else:
            # Later stage: use MCTS
            return self._select_mcts(mcts)
    
    def _select_bfs_unexplored(self):
        """BFS selection of unexplored nodes."""
        for node in self.tree.get_all_nodes():
            if node.status.name == 'UNVISITED':
                return node
        
        # All visited, select leaf with lowest visits
        leaves = [n for n in self.tree.get_all_nodes() if self.tree.is_leaf(n.id)]
        if leaves:
            return min(leaves, key=lambda n: n.visits)
        
        return self.tree.get_root()
    
    def _select_mcts(self, mcts):
        """Use MCTS for node selection."""
        # Run limited MCTS iterations
        mcts_stats = mcts.search(self.tree, budget=10)
        
        # Get best node from MCTS
        if mcts_stats.get('best_node_id'):
            try:
                return self.tree.get_node(mcts_stats['best_node_id'])
            except:
                pass
        
        # Fallback to BFS
        return self._select_bfs_unexplored()
    
    def _expand_node(self, node) -> list:
        """Expand node by generating children."""
        if self.proposer is None:
            logger.warning("No proposer available, using mock expansion")
            return self._mock_expand(node)
        
        # Get context for proposal
        previous_ideas = self._get_previous_ideas()
        tree_state = f"Depth {self.tree.get_depth(node.id)}, {len(self.tree)} nodes total"
        
        # Generate proposals
        children = []
        num_proposals = self.config.proposals_per_expansion
        
        for i in range(num_proposals):
            if len(self.tree) >= self.config.max_nodes:
                logger.info("Max nodes reached, stopping expansion")
                break
            
            # Check width constraint
            existing_children = list(self.tree.get_children(node.id))
            if len(existing_children) >= self.config.max_width:
                logger.info(f"Max width ({self.config.max_width}) reached for node {node.id}")
                break
            
            # Generate proposal
            proposal = self.proposer.generate_proposal(
                previous_ideas=previous_ideas,
                tree_state=tree_state,
                seed=self.config.seed + self._iteration * num_proposals + i
            )
            
            # Validate proposal
            is_valid, reason = self.proposer.validate_proposal(proposal)
            if not is_valid:
                logger.info(f"Proposal rejected: {reason}")
                continue
            
            # Check novelty threshold
            if proposal.novelty_score < self.config.min_novelty_threshold:
                logger.info(f"Proposal novelty too low: {proposal.novelty_score}")
                continue
            
            # Create child node
            child_data = {
                "hypothesis": proposal.hypothesis,
                "motivation": proposal.motivation,
                "method_summary": proposal.method_summary,
                "expected_outcome": proposal.expected_outcome,
                "feasibility_score": proposal.feasibility_score,
                "novelty_score": proposal.novelty_score,
                "keywords": proposal.keywords,
                "parent_id": node.id
            }
            
            child = self.tree.add_child(node.id, data=child_data)
            children.append(child)
            
            logger.info(f"Added child node {child.id}: {proposal.hypothesis[:60]}...")
        
        return children
    
    def _mock_expand(self, node) -> list:
        """Mock expansion without proposer."""
        import random
        
        mock_hypotheses = [
            ("LoRA rank adaptation improves convergence", 0.7, 0.6),
            ("Gradient checkpointing trade-offs", 0.8, 0.5),
            ("Learning rate scheduling strategies", 0.75, 0.55),
            ("Batch size effects on generalization", 0.8, 0.6),
        ]
        
        children = []
        for hyp, feasibility, novelty in mock_hypotheses:
            if len(children) >= self.config.max_width:
                break
            
            child_data = {
                "hypothesis": hyp,
                "motivation": "Auto-generated hypothesis",
                "method_summary": ["Test hypothesis with controlled experiment"],
                "feasibility_score": feasibility,
                "novelty_score": novelty
            }
            
            child = self.tree.add_child(node.id, data=child_data)
            children.append(child)
        
        return children
    
    def _execute_experiment(self, node):
        """Execute experiment for a node."""
        if not self.config.run_experiments:
            return
        
        if self.executor is None:
            logger.warning("No executor available, skipping execution")
            node.data.results = {"status": "skipped", "reason": "no_executor"}
            return
        
        try:
            # Generate code
            generated = self.executor.generate_code(
                hypothesis=node.data.hypothesis,
                method_summary=node.data.method_summary,
                seed=self.config.seed + node.id
            )
            
            # Save code
            code_path = self.output_dir / f"experiment_{node.id}.py"
            generated.save(code_path)
            
            # Execute (sandboxed)
            # TODO: Implement sandboxed execution
            node.data.code_path = str(code_path)
            node.data.results = {"status": "executed", "code_saved": str(code_path)}
            
            logger.info(f"Executed experiment for node {node.id}")
            
        except Exception as e:
            logger.error(f"Execution failed for node {node.id}: {e}")
            node.data.results = {"status": "failed", "error": str(e)}
    
    def _evaluate_node(self, node):
        """Evaluate a node's results."""
        if self.evaluator is None:
            # Mock evaluation based on feasibility/novelty
            score = 0.5 * node.data.feasibility_score + 0.5 * node.data.novelty_score
            node.data.results = node.data.results or {}
            node.data.results['primary_score'] = score
            node.data.results['metrics'] = {'score': score}
            return
        
        # Use evaluator
        try:
            results = self.evaluator.evaluate(node)
            node.data.results.update(results)
        except Exception as e:
            logger.error(f"Evaluation failed for node {node.id}: {e}")
    
    def _reflect_on_node(self, node):
        """Generate reflection/critique for a node."""
        if self.critic is None:
            # Simple reflection
            node.data.reflections = [
                f"Feasibility: {node.data.feasibility_score:.2f}",
                f"Novelty: {node.data.novelty_score:.2f}"
            ]
            return
        
        try:
            critique = self.critic.critique(node)
            node.data.reflections = node.data.reflections or []
            node.data.reflections.append(critique)
            node.data.critic_score = critique.get('overall_score', 0.5)
        except Exception as e:
            logger.error(f"Reflection failed for node {node.id}: {e}")
    
    def _get_node_score(self, node) -> float:
        """Get final score for a node."""
        if hasattr(node.data, 'critic_score') and node.data.critic_score:
            return node.data.critic_score
        
        if node.data.results and 'primary_score' in node.data.results:
            return node.data.results['primary_score']
        
        # Default scoring
        return 0.5 * node.data.feasibility_score + 0.5 * node.data.novelty_score
    
    def _backpropagate(self, node, score):
        """Backpropagate score up the tree."""
        current = node
        
        while current is not None:
            current.visits += 1
            
            # Incremental mean update
            delta = score - current.value
            current.value += delta / current.visits
            
            # Apply discount for deeper nodes
            score *= 0.99
            
            # Move to parent
            if current.parent_id is None:
                break
            
            try:
                current = self.tree.get_node(current.parent_id)
            except:
                break
    
    def _get_previous_ideas(self) -> list[str]:
        """Get list of previously explored hypotheses."""
        ideas = []
        for node in self.tree.get_all_nodes():
            if hasattr(node.data, 'hypothesis') and node.data.hypothesis:
                ideas.append(node.data.hypothesis)
        return ideas
    
    def _should_terminate(self) -> bool:
        """Check if search should terminate."""
        # Max nodes reached
        if len(self.tree) >= self.config.max_nodes:
            return True
        
        # Time limit (TODO: implement)
        
        return False
    
    def _log_progress(self, stats: dict):
        """Log progress update."""
        from src.search.utils import format_tree_summary
        
        print(f"\n--- Iteration {self._iteration + 1} ---")
        print(f"Phase: {stats.get('phase', 'unknown')}")
        print(f"Success: {stats.get('success', False)}")
        print(f"Nodes: {len(self.tree)}")
        print(f"Best score: {stats.get('score', 0):.3f}")
        
        if self.config.verbose:
            print(format_tree_summary(self.tree))
    
    def _compile_statistics(
        self,
        stats_history: list[dict],
        interrupted: bool = False
    ) -> dict[str, Any]:
        """Compile final statistics."""
        from src.search.utils import get_exploration_metrics
        
        duration = time.time() - self._start_time if self._start_time else 0
        
        tree_metrics = get_exploration_metrics(self.tree)
        
        # Find best score
        best_score = 0.0
        for node in self.tree.get_all_nodes():
            score = self._get_node_score(node)
            if score > best_score:
                best_score = score
        
        return {
            "iterations_completed": self._iteration + 1,
            "total_nodes": tree_metrics['total_nodes'],
            "completed_nodes": tree_metrics['completed_nodes'],
            "max_depth": tree_metrics['max_depth'],
            "branching_factor": tree_metrics['branching_factor'],
            "diversity": tree_metrics['diversity'],
            "best_score": best_score,
            "duration_seconds": duration,
            "interrupted": interrupted,
            "output_dir": str(self.output_dir),
            "config": self.config.model_dump()
        }
    
    def _save_checkpoint(
        self,
        final: bool = False,
        interrupted: bool = False,
        error: Optional[str] = None
    ):
        """Save current state to checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if final:
            filename = f"checkpoint_final_{timestamp}.json"
        elif interrupted:
            filename = f"checkpoint_interrupted_{timestamp}.json"
        elif error:
            filename = f"checkpoint_error_{timestamp}.json"
        else:
            filename = f"checkpoint_{timestamp}.json"
        
        checkpoint_path = self.output_dir / filename
        
        checkpoint_data = {
            "tree": self.tree.to_dict() if self.tree else None,
            "iteration": self._iteration,
            "timestamp": timestamp,
            "config": self.config.model_dump(),
            "error": error
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load state from checkpoint."""
        from src.core.tree import Tree, TreeSearchConfig
        
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        # Restore tree
        if checkpoint.get('tree'):
            tree_dict = checkpoint['tree']
            config = TreeSearchConfig(**tree_dict.get('config', {}))
            self.tree = Tree.from_dict(tree_dict, config=config)
        
        # Restore iteration
        self._iteration = checkpoint.get('iteration', 0)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def _generate_final_report(self):
        """Generate final report using reporter agent."""
        if not self.reporter:
            logger.info("No reporter available, skipping report generation")
            return
        
        try:
            report = self.reporter.generate_report(
                tree=self.tree,
                stats=self._compile_statistics([])
            )
            
            report_path = self.output_dir / "final_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Generated report at {report_path}")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")


def run_experiment(
    max_nodes: int = 20,
    max_depth: int = 3,
    run_experiments: bool = False,
    seed: int = 42
) -> dict:
    """
    Convenience function to run a complete experiment.
    
    Args:
        max_nodes: Maximum nodes in tree
        max_depth: Maximum tree depth
        run_experiments: Whether to execute experiments
        seed: Random seed
        
    Returns:
        Final statistics
    """
    from src.agents.proposer import IdeaProposer, ProposalConfig
    from src.agents.executor import CodeExecutor, ExecutionConfig
    from src.agents.evaluator import Evaluator, EvaluationConfig
    from src.agents.critic import Critic, CriticConfig
    from src.agents.reporter import Reporter, ReporterConfig
    
    # Initialize config
    manager_config = ExperimentManagerConfig(
        max_nodes=max_nodes,
        max_depth=max_depth,
        run_experiments=run_experiments,
        seed=seed,
        verbose=True
    )
    
    # Initialize agents
    proposer = IdeaProposer(ProposalConfig(domain="efficient fine-tuning"))
    executor = CodeExecutor(ExecutionConfig())
    evaluator = Evaluator(EvaluationConfig())
    critic = Critic(CriticConfig())
    reporter = Reporter(ReporterConfig())
    
    # Create manager
    manager = ExperimentManager(
        config=manager_config,
        proposer=proposer,
        executor=executor,
        evaluator=evaluator,
        critic=critic,
        reporter=reporter
    )
    
    # Run experiment
    return manager.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("TreeQuest Lab - Experiment Manager Demo")
    print("=" * 60)
    
    # Run demo with small budget
    stats = run_experiment(
        max_nodes=10,
        max_depth=2,
        run_experiments=False,
        seed=42
    )
    
    print("\nFinal Statistics:")
    print(json.dumps(stats, indent=2, default=str))
