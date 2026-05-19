"""Tests for Node and NodeData classes."""

import json
import pytest
from pathlib import Path
from datetime import datetime

from src.core.node import Node, NodeData, NodeStatus


class TestNodeData:
    """Test suite for NodeData class."""
    
    def test_create_node_data(self):
        """Test basic NodeData creation."""
        data = NodeData(
            hypothesis="Test hypothesis",
            description="Test description"
        )
        
        assert data.hypothesis == "Test hypothesis"
        assert data.description == "Test description"
        assert data.results == {}
        assert data.reflections == []
    
    def test_node_data_to_dict(self):
        """Test NodeData serialization to dictionary."""
        data = NodeData(
            hypothesis="Test",
            description="Desc",
            results={"accuracy": 0.95},
            reflections=["Good result"],
        )
        
        dict_repr = data.to_dict()
        
        assert dict_repr["hypothesis"] == "Test"
        assert dict_repr["description"] == "Desc"
        assert dict_repr["results"]["accuracy"] == 0.95
        assert dict_repr["reflections"] == ["Good result"]
    
    def test_node_data_from_dict(self):
        """Test NodeData deserialization from dictionary."""
        dict_data = {
            "hypothesis": "Test",
            "description": "Desc",
            "code_checkpoint": "/path/to/code.py",
            "results": {"loss": 0.5},
            "reflections": [],
            "metadata": {"version": "1.0"}
        }
        
        data = NodeData.from_dict(dict_data)
        
        assert data.hypothesis == "Test"
        assert isinstance(data.code_checkpoint, Path)
        assert data.results["loss"] == 0.5
    
    def test_has_results(self):
        """Test has_results method."""
        empty_data = NodeData()
        assert not empty_data.has_results()
        
        data_with_results = NodeData(results={"accuracy": 0.9})
        assert data_with_results.has_results()
    
    def test_get_metric(self):
        """Test get_metric method."""
        data = NodeData(results={"accuracy": 0.95, "loss": 0.3})
        
        assert data.get_metric("accuracy") == 0.95
        assert data.get_metric("loss") == 0.3
        assert data.get_metric("nonexistent", default=0.0) == 0.0


class TestNode:
    """Test suite for Node class."""
    
    def test_create_root_node(self):
        """Test creating a root node."""
        node = Node(data=NodeData(hypothesis="Root hypothesis"))
        
        assert node.is_root
        assert node.depth == 0
        assert node.parent is None
        assert node.status == NodeStatus.UNVISITED
    
    def test_add_child(self):
        """Test adding child nodes."""
        parent = Node(data=NodeData(hypothesis="Parent"))
        child = Node(data=NodeData(hypothesis="Child"))
        
        parent.add_child(child)
        
        assert len(parent.children) == 1
        assert child.parent == parent
        assert child.depth == 1
        assert not child.is_root
    
    def test_uct_score_unvisited(self):
        """Test UCT score for unvisited nodes."""
        parent = Node(data=NodeData(hypothesis="Parent"))
        child = Node(data=NodeData(hypothesis="Child"))
        parent.add_child(child)
        
        # Unvisited nodes should have infinite UCT score
        assert child.uct_score == float("inf")
    
    def test_uct_score_visited(self):
        """Test UCT score calculation for visited nodes."""
        parent = Node(data=NodeData(hypothesis="Parent"))
        child = Node(data=NodeData(hypothesis="Child"))
        parent.add_child(child)
        
        # Simulate parent visits
        parent.visit_count = 10
        
        # Simulate child visits
        child.visit_count = 5
        child.value_sum = 4.0  # value = 0.8
        
        uct = child.uct_score
        assert uct > 0.8  # Should be value + exploration bonus
    
    def test_update_value(self):
        """Test updating node value."""
        node = Node(data=NodeData(hypothesis="Test"))
        
        node.update_value(0.8)
        node.update_value(0.9)
        
        assert node.visit_count == 2
        assert node.value_sum == 1.7
        assert abs(node.value - 0.85) < 0.001
    
    def test_backpropagate(self):
        """Test value backpropagation up the tree."""
        grandparent = Node(data=NodeData(hypothesis="Grandparent"))
        parent = Node(data=NodeData(hypothesis="Parent"))
        child = Node(data=NodeData(hypothesis="Child"))
        
        grandparent.add_child(parent)
        parent.add_child(child)
        
        # Backpropagate from child
        child.backpropagate(0.9)
        
        # All nodes should be updated
        assert grandparent.visit_count == 1
        assert parent.visit_count == 1
        assert child.visit_count == 1
        
        assert abs(grandparent.value_sum - 0.9) < 0.001
        assert abs(parent.value_sum - 0.9) < 0.001
        assert abs(child.value_sum - 0.9) < 0.001
    
    def test_get_path_to_root(self):
        """Test getting path from node to root."""
        root = Node(data=NodeData(hypothesis="Root"))
        child1 = Node(data=NodeData(hypothesis="Child1"))
        child2 = Node(data=NodeData(hypothesis="Child2"))
        
        root.add_child(child1)
        child1.add_child(child2)
        
        path = child2.get_path_to_root()
        
        assert len(path) == 3
        assert path[0] == child2
        assert path[1] == child1
        assert path[2] == root
    
    def test_hypothesis_hash(self):
        """Test hypothesis hashing for duplicate detection."""
        node1 = Node(data=NodeData(hypothesis="Same hypothesis"))
        node2 = Node(data=NodeData(hypothesis="Same hypothesis"))
        node3 = Node(data=NodeData(hypothesis="Different hypothesis"))
        
        hash1 = node1.get_hypothesis_hash()
        hash2 = node2.get_hypothesis_hash()
        hash3 = node3.get_hypothesis_hash()
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16  # First 16 chars of SHA256
    
    def test_node_serialization(self):
        """Test node serialization to/from dictionary."""
        original = Node(
            data=NodeData(
                hypothesis="Test hypothesis",
                description="Test description",
                results={"accuracy": 0.95}
            ),
            status=NodeStatus.COMPLETED
        )
        original.update_value(0.95)
        
        # Serialize
        dict_repr = original.to_dict()
        
        # Deserialize
        restored = Node.from_dict(dict_repr)
        
        assert restored.id == original.id
        assert restored.data.hypothesis == original.data.hypothesis
        assert restored.status == original.status
        assert restored.visit_count == original.visit_count
        assert abs(restored.value - original.value) < 0.001
    
    def test_node_save_load(self, tmp_path):
        """Test saving and loading node to/from file."""
        node = Node(
            data=NodeData(hypothesis="Test"),
            status=NodeStatus.COMPLETED
        )
        
        filepath = tmp_path / "test_node.json"
        node.save(filepath)
        
        loaded = Node.load(filepath)
        
        assert loaded.id == node.id
        assert loaded.data.hypothesis == node.data.hypothesis
    
    def test_node_string_representation(self):
        """Test string representation of nodes."""
        node = Node(data=NodeData(hypothesis="Test hypothesis"))
        
        str_repr = str(node)
        assert "Node" in str_repr
        assert "depth=0" in str_repr
        
        repr_repr = repr(node)
        assert "id=" in repr_repr
        assert "hypothesis=" in repr_repr
    
    def test_is_leaf(self):
        """Test leaf node detection."""
        parent = Node(data=NodeData(hypothesis="Parent"))
        child = Node(data=NodeData(hypothesis="Child"))
        
        assert parent.is_leaf
        
        parent.add_child(child)
        
        assert not parent.is_leaf
        assert child.is_leaf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
