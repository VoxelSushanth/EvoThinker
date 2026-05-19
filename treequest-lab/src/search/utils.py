"""
Search utilities for TreeQuest Lab

Helper functions for tree search operations, scoring, and analysis.
"""

import logging
import numpy as np
from typing import Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


def calculate_diversity(nodes: list[Any], metric: str = "hypothesis") -> float:
    """
    Calculate diversity score across a set of nodes.
    
    Args:
        nodes: List of Node objects
        metric: Attribute to measure diversity on
        
    Returns:
        Diversity score (0-1, higher is more diverse)
    """
    if len(nodes) < 2:
        return 0.0
    
    # Extract texts/values
    values = []
    for node in nodes:
        if hasattr(node, 'data') and hasattr(node.data, metric):
            val = getattr(node.data, metric)
            if isinstance(val, str):
                values.append(val.lower())
            else:
                values.append(str(val))
    
    if not values:
        return 0.0
    
    # Simple diversity: ratio of unique values
    unique_ratio = len(set(values)) / len(values)
    
    # For text, also consider lexical diversity
    if isinstance(values[0], str) and len(values[0].split()) > 3:
        # Calculate average pairwise similarity (simplified)
        all_words = set()
        total_words = 0
        for v in values:
            words = set(v.split())
            all_words.update(words)
            total_words += len(words)
        
        lexical_diversity = len(all_words) / max(total_words, 1)
        
        # Combine metrics
        diversity = 0.5 * unique_ratio + 0.5 * lexical_diversity
    else:
        diversity = unique_ratio
    
    return min(1.0, max(0.0, diversity))


def calculate_coverage(
    nodes: list[Any],
    keywords: list[str]
) -> dict[str, int]:
    """
    Calculate keyword coverage across nodes.
    
    Args:
        nodes: List of Node objects
        keywords: List of keywords to track
        
    Returns:
        Dictionary mapping keywords to occurrence counts
    """
    coverage = defaultdict(int)
    
    for node in nodes:
        hypothesis = ""
        if hasattr(node, 'data'):
            if hasattr(node.data, 'hypothesis'):
                hypothesis = node.data.hypothesis.lower()
            elif hasattr(node.data, 'text'):
                hypothesis = node.data.text.lower()
        
        for keyword in keywords:
            if keyword.lower() in hypothesis:
                coverage[keyword] += 1
    
    return dict(coverage)


def get_exploration_metrics(tree) -> dict[str, Any]:
    """
    Calculate comprehensive exploration metrics for a tree.
    
    Args:
        tree: TreeQuest Tree object
        
    Returns:
        Dictionary with exploration metrics
    """
    all_nodes = list(tree.get_all_nodes())
    completed = [n for n in all_nodes if n.status.name == 'COMPLETED']
    
    if not all_nodes:
        return {
            "total_nodes": 0,
            "completed_nodes": 0,
            "max_depth": 0,
            "avg_depth": 0,
            "branching_factor": 0,
            "diversity": 0,
            "exploration_efficiency": 0
        }
    
    depths = [tree.get_depth(n.id) for n in all_nodes]
    
    # Calculate branching factor
    branch_counts = []
    for node in all_nodes:
        children = list(tree.get_children(node.node_id))
        if children:  # Only count nodes that have children
            branch_counts.append(len(children))
    
    avg_branching = np.mean(branch_counts) if branch_counts else 0
    
    # Calculate diversity
    diversity = calculate_diversity(completed) if completed else 0
    
    # Exploration efficiency: ratio of completed to total
    efficiency = len(completed) / len(all_nodes) if all_nodes else 0
    
    return {
        "total_nodes": len(all_nodes),
        "completed_nodes": len(completed),
        "max_depth": max(depths) if depths else 0,
        "avg_depth": np.mean(depths) if depths else 0,
        "branching_factor": avg_branching,
        "diversity": diversity,
        "exploration_efficiency": efficiency,
        "leaf_nodes": sum(1 for n in all_nodes if tree.is_leaf(n.id)),
        "failed_nodes": sum(1 for n in all_nodes if n.status.name == 'FAILED')
    }


def prune_low_value_nodes(
    tree,
    threshold: float = 0.3,
    min_visits: int = 3
) -> list[str]:
    """
    Prune nodes with low value scores.
    
    Args:
        tree: TreeQuest Tree object
        threshold: Minimum value score to keep node
        min_visits: Minimum visits before considering pruning
        
    Returns:
        List of pruned node IDs
    """
    pruned = []
    
    for node in tree.get_all_nodes():
        if node.visits < min_visits:
            continue  # Give unexplored nodes a chance
        
        if node.value < threshold and node.status.name not in ['COMPLETED', 'VISITED']:
            # Mark for pruning (don't actually remove, just flag)
            pruned.append(node.node_id)
            logger.info(f"Marked node {node.node_id} for pruning (value={node.value:.3f})")
    
    logger.info(f"Pruned {len(pruned)} low-value nodes")
    return pruned


def select_promising_nodes(
    tree,
    top_k: int = 5,
    min_value: float = 0.5,
    balance_exploration: bool = True
) -> list[Any]:
    """
    Select most promising nodes for further exploration.
    
    Args:
        tree: TreeQuest Tree object
        top_k: Number of nodes to select
        min_value: Minimum value threshold
        balance_exploration: Whether to balance exploitation/exploration
        
    Returns:
        List of selected nodes
    """
    candidates = []
    
    for node in tree.get_all_nodes():
        if node.status.name in ['COMPLETED', 'FAILED']:
            continue
        
        # Calculate selection score
        if balance_exploration and node.visits > 0:
            # UCT-like score
            import math
            parent = tree.get_parent(node.node_id)
            parent_visits = parent.visits if parent else 1
            
            exploitation = node.value
            exploration = 1.414 * math.sqrt(math.log(parent_visits) / node.visits)
            score = exploitation + exploration
        else:
            score = node.value
        
        if score >= min_value:
            candidates.append((score, node))
    
    # Sort by score descending
    candidates.sort(reverse=True, key=lambda x: x[0])
    
    # Return top-k nodes
    selected = [node for _, node in candidates[:top_k]]
    logger.info(f"Selected {len(selected)} promising nodes from {len(candidates)} candidates")
    
    return selected


def analyze_search_trajectory(stats_history: list[dict]) -> dict[str, Any]:
    """
    Analyze the trajectory of search over time.
    
    Args:
        stats_history: List of statistics snapshots over iterations
        
    Returns:
        Analysis results
    """
    if not stats_history:
        return {"trend": "insufficient_data"}
    
    best_scores = [s.get('best_score', 0) for s in stats_history]
    
    # Calculate trend
    if len(best_scores) >= 3:
        early_avg = np.mean(best_scores[:len(best_scores)//3])
        late_avg = np.mean(best_scores[-len(best_scores)//3:])
        
        if late_avg > early_avg * 1.1:
            trend = "improving"
        elif late_avg < early_avg * 0.9:
            trend = "degrading"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"
    
    # Calculate improvement rate
    if len(best_scores) >= 2:
        improvements = [
            best_scores[i] - best_scores[i-1]
            for i in range(1, len(best_scores))
            if best_scores[i] > best_scores[i-1]
        ]
        improvement_rate = len(improvements) / (len(best_scores) - 1)
    else:
        improvement_rate = 0
    
    return {
        "trend": trend,
        "initial_score": best_scores[0] if best_scores else 0,
        "final_score": best_scores[-1] if best_scores else 0,
        "max_score": max(best_scores) if best_scores else 0,
        "improvement_rate": improvement_rate,
        "plateau_detected": _detect_plateau(best_scores)
    }


def _detect_plateau(scores: list[float], window: int = 5, threshold: float = 0.01) -> bool:
    """Detect if search has plateaued."""
    if len(scores) < window:
        return False
    
    recent = scores[-window:]
    variance = np.var(recent)
    
    return variance < threshold


def format_tree_summary(tree) -> str:
    """
    Generate a human-readable summary of tree state.
    
    Args:
        tree: TreeQuest Tree object
        
    Returns:
        Formatted string summary
    """
    metrics = get_exploration_metrics(tree)
    
    lines = [
        "=" * 60,
        "TREE EXPLORATION SUMMARY",
        "=" * 60,
        f"Total Nodes: {metrics['total_nodes']}",
        f"Completed: {metrics['completed_nodes']}",
        f"Failed: {metrics['failed_nodes']}",
        f"Leaf Nodes: {metrics['leaf_nodes']}",
        "",
        f"Max Depth: {metrics['max_depth']}",
        f"Avg Depth: {metrics['avg_depth']:.2f}",
        f"Avg Branching: {metrics['branching_factor']:.2f}",
        "",
        f"Diversity Score: {metrics['diversity']:.3f}",
        f"Exploration Efficiency: {metrics['exploration_efficiency']:.1%}",
        "=" * 60
    ]
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test utility functions
    logging.basicConfig(level=logging.INFO)
    
    print("Testing search utilities...")
    
    # Test diversity calculation
    class MockNode:
        def __init__(self, hypothesis):
            self.data = type('Data', (), {'hypothesis': hypothesis})()
    
    nodes = [
        MockNode("LoRA improves fine-tuning efficiency"),
        MockNode("Gradient checkpointing saves memory"),
        MockNode("LoRA rank adaptation helps convergence"),
        MockNode("Mixed precision training accelerates learning")
    ]
    
    diversity = calculate_diversity(nodes)
    print(f"\nDiversity score: {diversity:.3f}")
    
    # Test keyword coverage
    keywords = ["lora", "checkpointing", "training", "fine-tuning"]
    coverage = calculate_coverage(nodes, keywords)
    print(f"Keyword coverage: {coverage}")
    
    print("\n✓ Utility tests complete")
