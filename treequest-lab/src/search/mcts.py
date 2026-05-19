"""
Monte Carlo Tree Search for TreeQuest Lab

Implements selection, expansion, simulation, and backpropagation
with progressive widening and value-guided exploration.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Node selection strategies."""
    UCT = "uct"
    PUCT = "puct"  # Predictive UCT with prior
    EPSILON_GREEDY = "epsilon_greedy"
    SOFTMAX = "softmax"


@dataclass
class MCTSConfig:
    """Configuration for MCTS algorithm."""
    
    max_iterations: int = 100
    max_depth: int = 5
    max_width: int = 3
    c_exploration: float = 1.414  # sqrt(2) default
    progressive_widening: bool = True
    progressive_widening_alpha: float = 0.5  # N^alpha threshold
    use_priors: bool = True  # Use feasibility/novelty as priors
    selection_strategy: SelectionStrategy = SelectionStrategy.UCT
    epsilon: float = 0.1  # For epsilon-greedy
    temperature: float = 1.0  # For softmax
    early_stopping: bool = True
    early_stopping_threshold: float = 0.9  # Stop if node score > threshold
    min_simulations: int = 3  # Minimum simulations per node
    
    def __post_init__(self):
        if isinstance(self.selection_strategy, str):
            self.selection_strategy = SelectionStrategy(self.selection_strategy)


class MCTS:
    """
    Monte Carlo Tree Search implementation for research exploration.
    
    Implements:
    - UCT/PUCT selection
    - Progressive widening
    - Value-guided expansion
    - Backpropagation with discounting
    """
    
    def __init__(
        self,
        config: Optional[MCTSConfig] = None,
        evaluation_fn: Optional[Callable] = None
    ):
        self.config = config or MCTSConfig()
        self.evaluation_fn = evaluation_fn
        self._iteration_count = 0
        self._best_score = float('-inf')
        self._best_node_id = None
        
    def search(self, tree, budget: Optional[int] = None) -> dict:
        """
        Run MCTS on the tree.
        
        Args:
            tree: TreeQuest Tree object
            budget: Number of iterations (overrides config if provided)
            
        Returns:
            Dictionary with search statistics
        """
        iterations = budget or self.config.max_iterations
        logger.info(f"Starting MCTS with {iterations} iterations")
        
        stats = {
            "iterations_completed": 0,
            "nodes_expanded": 0,
            "nodes_evaluated": 0,
            "best_score": float('-inf'),
            "best_node_id": None,
            "selection_distribution": {},
            "early_stopped": False
        }
        
        for iteration in range(iterations):
            self._iteration_count = iteration
            
            # Check early stopping
            if self.config.early_stopping and self._best_score >= self.config.early_stopping_threshold:
                logger.info(f"Early stopping at iteration {iteration} with score {self._best_score:.3f}")
                stats["early_stopped"] = True
                break
            
            # Selection: traverse from root to leaf
            node = self._select(tree)
            if node is None:
                logger.warning("Selection returned None, skipping iteration")
                continue
            
            # Expansion: add children if not terminal
            if not self._is_terminal(node, tree):
                children = self._expand(node, tree)
                stats["nodes_expanded"] += len(children)
                
                # Select most promising child for simulation
                if children:
                    node = max(children, key=lambda n: n.uct_score if hasattr(n, 'uct_score') else 0)
            
            # Simulation: evaluate the node
            score = self._simulate(node, tree)
            stats["nodes_evaluated"] += 1
            
            # Backpropagation: update ancestors
            self._backpropagate(node, score)
            
            # Update best
            if score > self._best_score:
                self._best_score = score
                self._best_node_id = node.node_id
                stats["best_score"] = score
                stats["best_node_id"] = node.node_id
            
            stats["iterations_completed"] = iteration + 1
            
            if (iteration + 1) % 10 == 0:
                logger.info(f"Iteration {iteration + 1}/{iterations}, Best score: {self._best_score:.3f}")
        
        stats["best_score"] = self._best_score
        stats["best_node_id"] = self._best_node_id
        
        logger.info(f"MCTS complete. Best score: {self._best_score:.3f} at node {self._best_node_id}")
        return stats
    
    def _select(self, tree):
        """Select a leaf node using UCT or configured strategy."""
        root = tree.get_root()
        if root is None:
            return None
        
        current = root
        
        while not tree.is_leaf(current.node_id):
            children = list(tree.get_children(current.node_id))
            if not children:
                break
            
            # Apply progressive widening check
            if self.config.progressive_widening:
                allowed_children = self._get_allowed_children(current, children)
                if len(allowed_children) < len(children):
                    # Not all children can be explored yet
                    children = allowed_children
            
            if not children:
                break
            
            # Select based on strategy
            if self.config.selection_strategy == SelectionStrategy.UCT:
                current = self._select_uct(current, children)
            elif self.config.selection_strategy == SelectionStrategy.PUCT:
                current = self._select_puct(current, children)
            elif self.config.selection_strategy == SelectionStrategy.EPSILON_GREEDY:
                current = self._select_epsilon_greedy(current, children)
            elif self.config.selection_strategy == SelectionStrategy.SOFTMAX:
                current = self._select_softmax(current, children)
            else:
                current = self._select_uct(current, children)
        
        return current
    
    def _select_uct(self, parent, children):
        """Select child using UCT formula."""
        best_score = float('-inf')
        best_child = None
        
        visits = max(parent.visits, 1)  # Avoid division by zero
        
        for child in children:
            if child.visits == 0:
                # Unvisited nodes get priority
                return child
            
            # UCT = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
            exploitation = child.value
            exploration = self.config.c_exploration * math.sqrt(math.log(visits) / child.visits)
            
            # Add prior if available
            if self.config.use_priors and hasattr(child.data, 'feasibility_score'):
                prior = child.data.feasibility_score * 0.1  # Scale prior
                exploration += prior
            
            uct_score = exploitation + exploration
            
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        
        return best_child if best_child else children[0]
    
    def _select_puct(self, parent, children):
        """Select child using PUCT (Predictive UCT)."""
        best_score = float('-inf')
        best_child = None
        
        visits = max(parent.visits, 1)
        total_visits = sum(c.visits for c in children) or 1
        
        for child in children:
            if child.visits == 0:
                # Use prior for unvisited nodes
                prior = getattr(child.data, 'feasibility_score', 0.5) * \
                        getattr(child.data, 'novelty_score', 0.5)
                score = prior * math.sqrt(visits) / (total_visits + 1)
            else:
                exploitation = child.value
                exploration = self.config.c_exploration * prior * math.sqrt(visits) / (child.visits + 1)
                score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child if best_child else children[0]
    
    def _select_epsilon_greedy(self, parent, children):
        """Epsilon-greedy selection."""
        import random
        
        if random.random() < self.config.epsilon:
            # Explore randomly
            return random.choice(children)
        else:
            # Exploit: choose best value
            return max(children, key=lambda c: c.value)
    
    def _select_softmax(self, parent, children):
        """Softmax selection based on values."""
        import numpy as np
        
        values = np.array([c.value for c in children])
        if len(values) == 0:
            return children[0]
        
        # Apply temperature scaling
        exp_values = np.exp((values - np.max(values)) / self.config.temperature)
        probs = exp_values / np.sum(exp_values)
        
        # Sample based on probabilities
        idx = np.random.choice(len(children), p=probs)
        return children[idx]
    
    def _get_allowed_children(self, parent, children):
        """Apply progressive widening constraint."""
        # k = N^alpha where N is parent visits
        k = int(math.pow(max(parent.visits, 1), self.config.progressive_widening_alpha))
        k = max(1, min(k, len(children)))
        
        # Return top-k by prior score or randomly if no prior
        if self.config.use_priors:
            scored = []
            for child in children:
                prior = getattr(child.data, 'feasibility_score', 0.5) * \
                        getattr(child.data, 'novelty_score', 0.5)
                scored.append((prior, child))
            scored.sort(reverse=True, key=lambda x: x[0])
            return [child for _, child in scored[:k]]
        else:
            import random
            return random.sample(children, min(k, len(children)))
    
    def _is_terminal(self, node, tree) -> bool:
        """Check if node is terminal (max depth or completed)."""
        depth = tree.get_depth(node.node_id)
        
        if depth >= self.config.max_depth:
            return True
        
        if node.status.name in ['COMPLETED', 'FAILED']:
            # Already evaluated
            return node.visits >= self.config.min_simulations
        
        return False
    
    def _expand(self, node, tree):
        """Expand node by generating children (handled by ExperimentManager)."""
        # In TreeQuest, expansion is handled by the ExperimentManager
        # which calls the IdeaProposer to generate new hypotheses
        # This method returns existing children or empty list
        return list(tree.get_children(node.node_id))
    
    def _simulate(self, node, tree):
        """Simulate/evaluate a node."""
        # If node already has results, use them
        if node.data.results and 'primary_score' in node.data.results:
            return node.data.results['primary_score']
        
        # Otherwise, use evaluation function if provided
        if self.evaluation_fn:
            try:
                result = self.evaluation_fn(node)
                if isinstance(result, (int, float)):
                    return result
                elif isinstance(result, dict) and 'score' in result:
                    return result['score']
            except Exception as e:
                logger.warning(f"Evaluation failed for node {node.node_id}: {e}")
        
        # Default: use critic score or feasibility
        if hasattr(node.data, 'critic_score'):
            return node.data.critic_score
        elif hasattr(node.data, 'feasibility_score'):
            return node.data.feasibility_score
        else:
            return 0.5  # Neutral default
    
    def _backpropagate(self, node, score):
        """Backpropagate score up the tree."""
        current = node
        gamma = 0.99  # Discount factor for deeper nodes
        
        while current is not None:
            current.visits += 1
            # Incremental mean update
            delta = score - current.value
            current.value += delta / current.visits
            
            # Move to parent
            parent_id = current.parent_id
            if parent_id is None:
                break
            current = current.tree.nodes[parent_id] if hasattr(current, 'tree') else None
            score *= gamma  # Apply discount


def uct_select(
    parent_visits: int,
    child_visits: int,
    child_value: float,
    c_exploration: float = 1.414,
    prior: float = 0.0
) -> float:
    """
    Calculate UCT score for a child node.
    
    Args:
        parent_visits: Total visits to parent node
        child_visits: Visits to this child
        child_value: Average value/reward for this child
        c_exploration: Exploration constant
        prior: Optional prior probability
        
    Returns:
        UCT score
    """
    if child_visits == 0:
        return float('inf')  # Unvisited nodes have infinite UCT
    
    exploitation = child_value
    exploration = c_exploration * math.sqrt(math.log(parent_visits) / child_visits)
    
    return exploitation + exploration + prior


if __name__ == "__main__":
    # Test MCTS configuration
    logging.basicConfig(level=logging.INFO)
    
    config = MCTSConfig(
        max_iterations=50,
        max_depth=3,
        c_exploration=1.414,
        progressive_widening=True
    )
    
    print(f"MCTS Config: {config}")
    print(f"Selection strategy: {config.selection_strategy}")
    print(f"Progressive widening alpha: {config.progressive_widening_alpha}")
    
    # Test UCT calculation
    score = uct_select(
        parent_visits=100,
        child_visits=10,
        child_value=0.75,
        c_exploration=1.414,
        prior=0.1
    )
    print(f"\nExample UCT score: {score:.3f}")
