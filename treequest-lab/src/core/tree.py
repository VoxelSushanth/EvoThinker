"""Tree data structure for managing the agentic search tree."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import networkx as nx

from .node import Node, NodeData, NodeStatus


logger = logging.getLogger(__name__)


@dataclass
class TreeSearchConfig:
    """Configuration for tree search behavior.
    
    Attributes:
        max_depth: Maximum depth of the tree
        max_width: Maximum number of children per node
        max_nodes: Maximum total nodes in tree
        uct_exploration_constant: Exploration parameter for UCT
        progressive_widening_threshold: Visits before allowing new children
        discount_factor: Discount factor for value backpropagation
    """
    
    max_depth: int = 10
    max_width: int = 5
    max_nodes: int = 100
    uct_exploration_constant: float = 1.41
    progressive_widening_threshold: int = 2
    discount_factor: float = 0.99
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_depth": self.max_depth,
            "max_width": self.max_width,
            "max_nodes": self.max_nodes,
            "uct_exploration_constant": self.uct_exploration_constant,
            "progressive_widening_threshold": self.progressive_widening_threshold,
            "discount_factor": self.discount_factor,
        }


class Tree:
    """Manages the agentic tree search structure.
    
    Provides functionality for:
    - Tree construction and traversal
    - Node selection using UCT
    - Progressive widening
    - Serialization/deserialization
    - Statistics and visualization
    
    The tree uses NetworkX internally for efficient graph operations,
    while maintaining Node objects for rich data storage.
    
    Attributes:
        config: Tree search configuration
        root: Root node of the tree
        nodes: Dictionary mapping node IDs to Node objects
        graph: NetworkX DiGraph for tree structure
        created_at: Tree creation timestamp
    """
    
    def __init__(self, config: TreeSearchConfig | None = None):
        """Initialize tree with optional configuration.
        
        Args:
            config: Tree search configuration (uses defaults if None)
        """
        self.config = config or TreeSearchConfig()
        self.root: Node | None = None
        self.nodes: dict[str, Node] = {}
        self.graph = nx.DiGraph()
        self.created_at = datetime.now()
        self._hypothese_hashes: set[str] = set()  # For duplicate detection
    
    def add_root(self, hypothesis: str, description: str = "") -> Node:
        """Add root node to the tree.
        
        Args:
            hypothesis: Root hypothesis string
            description: Optional description
            
        Returns:
            Created root node
            
        Raises:
            ValueError: If root already exists
        """
        if self.root is not None:
            raise ValueError("Root already exists. Use add_node() for additional nodes.")
        
        node_data = NodeData(hypothesis=hypothesis, description=description)
        self.root = Node(data=node_data, depth=0)
        self._add_node_to_tree(self.root)
        
        logger.info(f"Created root node: {self.root.id[:8]}")
        return self.root
    
    def add_node(
        self,
        parent: Node,
        hypothesis: str,
        description: str = "",
    ) -> Node | None:
        """Add a child node to a parent node.
        
        Implements progressive widening checks and duplicate detection.
        
        Args:
            parent: Parent node
            hypothesis: Child hypothesis string
            description: Optional description
            
        Returns:
            Created child node, or None if constraints prevent addition
        """
        # Check if we've reached max nodes
        if len(self.nodes) >= self.config.max_nodes:
            logger.warning("Maximum node count reached")
            return None
        
        # Check depth constraint
        if parent.depth >= self.config.max_depth:
            logger.warning(f"Maximum depth ({self.config.max_depth}) reached")
            return None
        
        # Check width constraint (progressive widening)
        if len(parent.children) >= self.config.max_width:
            logger.warning(f"Maximum width ({self.config.max_width}) reached")
            return None
        
        # Progressive widening: check if parent has enough visits
        if parent.visit_count < self.config.progressive_widening_threshold:
            logger.debug(
                f"Parent needs more visits "
                f"({parent.visit_count}/{self.config.progressive_widening_threshold})"
            )
            # Still allow first child
            if len(parent.children) > 0:
                return None
        
        # Check for duplicate hypothesis
        hyp_hash = self._hash_hypothesis(hypothesis)
        if hyp_hash in self._hypothese_hashes:
            logger.info(f"Duplicate hypothesis detected: {hypothesis[:50]}...")
            return None
        
        # Create and add child node
        node_data = NodeData(hypothesis=hypothesis, description=description)
        child = Node(data=node_data)
        parent.add_child(child)
        self._add_node_to_tree(child)
        
        logger.info(f"Added child node {child.id[:8]} to parent {parent.id[:8]}")
        return child
    
    def _add_node_to_tree(self, node: Node) -> None:
        """Add node to internal data structures.
        
        Args:
            node: Node to add
        """
        self.nodes[node.id] = node
        self.graph.add_node(node.id, node=node)
        
        if node.parent is not None:
            self.graph.add_edge(node.parent.id, node.id)
        
        # Track hypothesis hash for duplicate detection
        hyp_hash = self._hash_hypothesis(node.data.hypothesis)
        self._hypothese_hashes.add(hyp_hash)
    
    def _hash_hypothesis(self, hypothesis: str) -> str:
        """Create hash of hypothesis for duplicate detection.
        
        Args:
            hypothesis: Hypothesis string
            
        Returns:
            SHA256 hash (first 16 chars)
        """
        import hashlib
        return hashlib.sha256(hypothesis.encode()).hexdigest()[:16]
    
    def get_node(self, node_id: str) -> Node | None:
        """Get node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node if found, None otherwise
        """
        return self.nodes.get(node_id)
    
    def get_all_nodes(self) -> list[Node]:
        """Get all nodes in the tree.
        
        Returns:
            List of all nodes
        """
        return list(self.nodes.values())
    
    def get_leaves(self) -> list[Node]:
        """Get all leaf nodes (nodes without children).
        
        Returns:
            List of leaf nodes
        """
        return [node for node in self.nodes.values() if node.is_leaf]
    
    def get_completed_nodes(self) -> list[Node]:
        """Get all completed nodes with results.
        
        Returns:
            List of completed nodes
        """
        return [
            node for node in self.nodes.values()
            if node.status == NodeStatus.COMPLETED and node.data.has_results()
        ]
    
    def select_node_uct(self) -> Node | None:
        """Select next node to explore using UCT scoring.
        
        Uses Upper Confidence Bound for Trees to balance exploration
        vs exploitation. Prefers unvisited nodes, then high-value nodes.
        
        Returns:
            Selected node for expansion, or None if tree is empty
        """
        if self.root is None:
            return None
        
        # Find best leaf using UCT
        def uct_select(node: Node) -> Node:
            """Recursively select node using UCT."""
            if node.is_leaf:
                return node
            
            # Calculate UCT scores for children
            uct_scores = []
            for child in node.children:
                if child.visit_count == 0:
                    # Prioritize unvisited nodes
                    return uct_select(child)
                
                # UCT formula
                import math
                exploration = math.sqrt(math.log(node.visit_count) / child.visit_count)
                uct = child.value + self.config.uct_exploration_constant * exploration
                uct_scores.append((uct, child))
            
            # Select child with highest UCT
            if uct_scores:
                best_child = max(uct_scores, key=lambda x: x[0])[1]
                return uct_select(best_child)
            
            return node
        
        return uct_select(self.root)
    
    def select_node_bfs(self) -> Node | None:
        """Select next node using breadth-first search.
        
        Returns:
            First unvisited leaf node found, or None
        """
        if self.root is None:
            return None
        
        for node in nx.bfs_preorder_nodes(self.graph, source=self.root.id):
            node_obj = self.nodes[node]
            if node_obj.is_leaf and node_obj.status != NodeStatus.COMPLETED:
                return node_obj
        
        return None
    
    def backpropagate_value(
        self,
        node: Node,
        value: float,
        apply_discount: bool = True,
    ) -> None:
        """Backpropagate value from node to root.
        
        Args:
            node: Starting node for backpropagation
            value: Value to propagate
            apply_discount: Whether to apply discount factor per level
        """
        current = node
        discount = 1.0
        
        while current is not None:
            adjusted_value = value * discount if apply_discount else value
            current.update_value(adjusted_value)
            
            if apply_discount:
                discount *= self.config.discount_factor
            
            current = current.parent
        
        logger.debug(f"Backpropagated value {value:.3f} from {node.id[:8]}")
    
    def prune_failed_branches(self) -> int:
        """Prune branches with consistently failed nodes.
        
        Returns:
            Number of nodes pruned
        """
        pruned_count = 0
        
        for node in list(self.nodes.values()):
            if node.status == NodeStatus.FAILED:
                # Check if all siblings also failed
                if node.parent:
                    siblings = [
                        c for c in node.parent.children
                        if c != node and c.status == NodeStatus.FAILED
                    ]
                    if len(siblings) >= 2:
                        # Prune entire subtree
                        subtree_nodes = self._get_subtree_nodes(node)
                        for sub_node in subtree_nodes:
                            self._remove_node(sub_node)
                            pruned_count += 1
        
        logger.info(f"Pruned {pruned_count} nodes from failed branches")
        return pruned_count
    
    def _get_subtree_nodes(self, node: Node) -> list[Node]:
        """Get all nodes in subtree rooted at given node.
        
        Args:
            node: Root of subtree
            
        Returns:
            List of all nodes in subtree
        """
        result = [node]
        for child in node.children:
            result.extend(self._get_subtree_nodes(child))
        return result
    
    def _remove_node(self, node: Node) -> None:
        """Remove node from tree (internal use).
        
        Args:
            node: Node to remove
        """
        if node.parent:
            node.parent.remove_child(node)
        
        if node.id in self.nodes:
            del self.nodes[node.id]
        
        if node.id in self.graph:
            self.graph.remove_node(node.id)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get tree statistics.
        
        Returns:
            Dictionary of tree statistics
        """
        completed = self.get_completed_nodes()
        leaves = self.get_leaves()
        
        stats = {
            "total_nodes": len(self.nodes),
            "completed_nodes": len(completed),
            "leaf_nodes": len(leaves),
            "max_depth": max((n.depth for n in self.nodes.values()), default=0),
            "avg_value": sum(n.value for n in completed) / len(completed) if completed else 0.0,
            "best_value": max((n.value for n in completed), default=0.0),
            "best_node_id": max(completed, key=lambda n: n.value).id if completed else None,
            "created_at": self.created_at.isoformat(),
        }
        
        # Add depth distribution
        depths = [n.depth for n in self.nodes.values()]
        stats["depth_distribution"] = {
            d: depths.count(d) for d in range(max(depths) + 1) if depths
        }
        
        return stats
    
    def get_best_path(self) -> list[Node]:
        """Get path from root to best completed node.
        
        Returns:
            List of nodes from root to best node
        """
        completed = self.get_completed_nodes()
        if not completed:
            return []
        
        best_node = max(completed, key=lambda n: n.value)
        return list(reversed(best_node.get_path_to_root()))
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize tree to dictionary.
        
        Returns:
            Dictionary representation of tree
        """
        return {
            "config": self.config.to_dict(),
            "root_id": self.root.id if self.root else None,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": list(self.graph.edges()),
            "created_at": self.created_at.isoformat(),
            "statistics": self.get_statistics(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tree":
        """Deserialize tree from dictionary.
        
        Args:
            data: Dictionary with tree data
            
        Returns:
            Reconstructed Tree instance
        """
        config = TreeSearchConfig(**data["config"])
        tree = cls(config=config)
        
        # First pass: create all nodes
        nodes_map = {}
        for node_id, node_data in data["nodes"].items():
            node = Node.from_dict(node_data)
            nodes_map[node_id] = node
            tree.nodes[node_id] = node
            tree.graph.add_node(node_id, node=node)
        
        # Second pass: establish parent-child relationships
        for node_id, node_data in data["nodes"].items():
            node = nodes_map[node_id]
            parent_id = node_data.get("parent_id")
            
            if parent_id and parent_id in nodes_map:
                parent = nodes_map[parent_id]
                parent.add_child(node)
                tree.graph.add_edge(parent_id, node_id)
            
            # Track hypothesis hashes
            hyp_hash = tree._hash_hypothesis(node.data.hypothesis)
            tree._hypothese_hashes.add(hyp_hash)
        
        # Set root
        root_id = data.get("root_id")
        if root_id and root_id in nodes_map:
            tree.root = nodes_map[root_id]
        
        tree.created_at = datetime.fromisoformat(data["created_at"])
        
        logger.info(f"Loaded tree with {len(tree.nodes)} nodes")
        return tree
    
    def save(self, filepath: Path | str) -> None:
        """Save tree to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved tree to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path | str) -> "Tree":
        """Load tree from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded Tree instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def __len__(self) -> int:
        """Get number of nodes in tree."""
        return len(self.nodes)
    
    def __str__(self) -> str:
        """String representation of tree."""
        stats = self.get_statistics()
        return (
            f"Tree(nodes={stats['total_nodes']}, "
            f"completed={stats['completed_nodes']}, "
            f"max_depth={stats['max_depth']}, "
            f"best_value={stats['best_value']:.3f})"
        )


if __name__ == "__main__":
    # Test tree operations
    config = TreeSearchConfig(max_depth=3, max_width=2, max_nodes=10)
    tree = Tree(config=config)
    
    # Create root
    root = tree.add_root("Base hypothesis", "Starting point")
    print(f"Created tree: {tree}")
    
    # Add children
    child1 = tree.add_node(root, "Child 1 hypothesis")
    child2 = tree.add_node(root, "Child 2 hypothesis")
    
    if child1:
        grandchild = tree.add_node(child1, "Grandchild hypothesis")
    
    print(f"Tree after additions: {tree}")
    print(f"Statistics: {json.dumps(tree.get_statistics(), indent=2)}")
    
    # Test serialization
    tree.save("test_tree.json")
    loaded_tree = Tree.load("test_tree.json")
    print(f"Loaded tree: {loaded_tree}")
    
    # Cleanup
    Path("test_tree.json").unlink()
