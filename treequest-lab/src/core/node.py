"""Node data structure for the agentic tree search."""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


class NodeStatus(Enum):
    """Status of a node in the search tree."""
    
    UNVISITED = "unvisited"
    VISITING = "visiting"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class NodeData:
    """Data associated with a tree node.
    
    This contains all the information about a research idea and its
    experimental results. Designed to be JSON-serializable for persistence.
    
    Attributes:
        hypothesis: The research hypothesis or idea
        description: Detailed description of the approach
        code_checkpoint: Path to generated experiment code
        results: Dictionary of experimental results and metrics
        reflections: Agent reflections and analysis
        metadata: Additional metadata (timestamps, version info, etc.)
    """
    
    hypothesis: str = ""
    description: str = ""
    code_checkpoint: Path | None = None
    results: dict[str, Any] = field(default_factory=dict)
    reflections: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hypothesis": self.hypothesis,
            "description": self.description,
            "code_checkpoint": str(self.code_checkpoint) if self.code_checkpoint else None,
            "results": self.results,
            "reflections": self.reflections,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeData":
        """Create NodeData from dictionary."""
        if data.get("code_checkpoint"):
            data["code_checkpoint"] = Path(data["code_checkpoint"])
        return cls(**data)
    
    def has_results(self) -> bool:
        """Check if node has experimental results."""
        return bool(self.results)
    
    def get_metric(self, metric_name: str, default: float = 0.0) -> float:
        """Safely get a metric value from results."""
        return self.results.get(metric_name, default)


@dataclass
class Node:
    """A node in the agentic tree search.
    
    Represents a single research idea/hypothesis in the exploration tree.
    Implements UCT (Upper Confidence Bound for Trees) scoring for selection.
    
    Attributes:
        id: Unique identifier for the node
        parent: Parent node (None for root)
        children: List of child nodes
        data: NodeData containing hypothesis and results
        status: Current status of the node
        visit_count: Number of times this node has been visited
        value_sum: Sum of values from rollouts/evaluations
        depth: Depth in the tree (root is 0)
        created_at: Timestamp of node creation
        updated_at: Timestamp of last update
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)
    data: NodeData = field(default_factory=NodeData)
    status: NodeStatus = NodeStatus.UNVISITED
    visit_count: int = 0
    value_sum: float = 0.0
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """Validate node after initialization."""
        if self.parent is not None:
            self.depth = self.parent.depth + 1
    
    @property
    def value(self) -> float:
        """Average value of this node (value_sum / visit_count)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    @property
    def uct_score(self) -> float:
        """Calculate UCT (Upper Confidence Bound for Trees) score.
        
        UCT = value + c * sqrt(ln(parent_visits) / visits)
        
        Returns:
            UCT score for node selection
        """
        if self.visit_count == 0:
            return float("inf")  # Encourage exploration of unvisited nodes
        
        if self.parent is None:
            return self.value
        
        # UCT formula with exploration constant
        import math
        exploration_bonus = math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        
        # Default exploration constant (can be configured)
        c = 1.41  # sqrt(2)
        
        return self.value + c * exploration_bonus
    
    def add_child(self, child: "Node") -> None:
        """Add a child node to this node.
        
        Args:
            child: Child node to add
        """
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        self.updated_at = datetime.now()
    
    def remove_child(self, child: "Node") -> None:
        """Remove a child node.
        
        Args:
            child: Child node to remove
        """
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            self.updated_at = datetime.now()
    
    def update_value(self, value: float) -> None:
        """Update node value with backpropagation.
        
        Args:
            value: Value to add from rollout/evaluation
        """
        self.visit_count += 1
        self.value_sum += value
        self.updated_at = datetime.now()
    
    def backpropagate(self, value: float) -> None:
        """Propagate value up the tree to root.
        
        Args:
            value: Value to backpropagate
        """
        current = self
        while current is not None:
            current.update_value(value)
            current = current.parent
    
    def get_path_to_root(self) -> list["Node"]:
        """Get path from this node to root.
        
        Returns:
            List of nodes from this node to root (inclusive)
        """
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return path
    
    def get_hypothesis_hash(self) -> str:
        """Get hash of hypothesis for duplicate detection.
        
        Returns:
            SHA256 hash of hypothesis string
        """
        return hashlib.sha256(self.data.hypothesis.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary for serialization.
        
        Returns:
            Dictionary representation of node
        """
        return {
            "id": self.id,
            "parent_id": self.parent.id if self.parent else None,
            "children_ids": [child.id for child in self.children],
            "data": self.data.to_dict(),
            "status": self.status.value,
            "visit_count": self.visit_count,
            "value_sum": self.value_sum,
            "value": self.value,
            "depth": self.depth,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        """Create node from dictionary.
        
        Note: Parent and children references are not restored.
        Use Tree.from_dict() for complete tree restoration.
        
        Args:
            data: Dictionary with node data
            
        Returns:
            Node instance
        """
        node = cls(
            id=data["id"],
            data=NodeData.from_dict(data["data"]),
            status=NodeStatus(data["status"]),
            visit_count=data["visit_count"],
            value_sum=data["value_sum"],
            depth=data["depth"],
        )
        node.created_at = datetime.fromisoformat(data["created_at"])
        node.updated_at = datetime.fromisoformat(data["updated_at"])
        return node
    
    def save(self, filepath: Path | str) -> None:
        """Save node to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path | str) -> "Node":
        """Load node from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded Node instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """String representation of node."""
        status_str = self.status.value
        value_str = f"{self.value:.3f}" if self.visit_count > 0 else "N/A"
        return (
            f"Node(id={self.id[:8]}, depth={self.depth}, "
            f"status={status_str}, value={value_str}, "
            f"visits={self.visit_count})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of node."""
        return (
            f"Node(id='{self.id}', hypothesis='{self.data.hypothesis[:50]}...', "
            f"status={self.status.value}, value={self.value:.3f})"
        )


if __name__ == "__main__":
    # Test node creation and operations
    root = Node(data=NodeData(hypothesis="Test hypothesis"))
    print(f"Root: {root}")
    
    child1 = Node(data=NodeData(hypothesis="Child 1"))
    child2 = Node(data=NodeData(hypothesis="Child 2"))
    
    root.add_child(child1)
    root.add_child(child2)
    
    print(f"Root children: {len(root.children)}")
    print(f"Child1 depth: {child1.depth}")
    print(f"Child1 UCT: {child1.uct_score}")
    
    # Simulate evaluation
    child1.update_value(0.8)
    child1.update_value(0.9)
    
    print(f"Child1 value after updates: {child1.value:.3f}")
    print(f"Child1 path to root: {[n.id[:8] for n in child1.get_path_to_root()]}")
