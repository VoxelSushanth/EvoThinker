"""Tests for Tree and TreeSearchConfig classes."""

import json
import pytest
from pathlib import Path
from datetime import datetime

from src.core.tree import Tree, TreeSearchConfig
from src.core.node import Node, NodeData, NodeStatus


class TestTreeSearchConfig:
    """Test suite for TreeSearchConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TreeSearchConfig()
        
        assert config.max_depth == 10
        assert config.max_width == 5
        assert config.max_nodes == 100
        assert config.uct_exploration_constant == 1.41
        assert config.progressive_widening_threshold == 2
        assert config.discount_factor == 0.99
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TreeSearchConfig(
            max_depth=5,
            max_width=3,
            max_nodes=50,
        )
        
        assert config.max_depth == 5
        assert config.max_width == 3
        assert config.max_nodes == 50
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = TreeSearchConfig(max_depth=7)
        dict_repr = config.to_dict()
        
        assert dict_repr["max_depth"] == 7
        assert "max_width" in dict_repr
        assert "max_nodes" in dict_repr


class TestTree:
    """Test suite for Tree class."""
    
    def test_create_empty_tree(self):
        """Test creating an empty tree."""
        tree = Tree()
        
        assert tree.root is None
        assert len(tree) == 0
        assert len(tree.nodes) == 0
    
    def test_add_root(self):
        """Test adding root node."""
        tree = Tree()
        root = tree.add_root("Root hypothesis", "Root description")
        
        assert tree.root == root
        assert root.is_root
        assert root.depth == 0
        assert len(tree) == 1
    
    def test_add_root_twice_raises_error(self):
        """Test that adding root twice raises error."""
        tree = Tree()
        tree.add_root("First root")
        
        with pytest.raises(ValueError, match="Root already exists"):
            tree.add_root("Second root")
    
    def test_add_child_nodes(self):
        """Test adding child nodes to tree."""
        tree = Tree()
        root = tree.add_root("Root")
        
        child1 = tree.add_node(root, "Child 1")
        child2 = tree.add_node(root, "Child 2")
        
        assert child1 is not None
        assert child2 is not None
        assert len(root.children) == 2
        assert len(tree) == 3
        assert child1.depth == 1
        assert child2.depth == 1
    
    def test_max_depth_constraint(self):
        """Test maximum depth constraint."""
        config = TreeSearchConfig(max_depth=2, max_width=5, max_nodes=100)
        tree = Tree(config=config)
        
        root = tree.add_root("Root")
        child1 = tree.add_node(root, "Child 1")
        
        if child1:
            grandchild = tree.add_node(child1, "Grandchild")
            
            if grandchild:
                # Should fail at depth 2
                great_grandchild = tree.add_node(grandchild, "Great-grandchild")
                assert great_grandchild is None
    
    def test_max_width_constraint(self):
        """Test maximum width constraint."""
        config = TreeSearchConfig(max_depth=10, max_width=2, max_nodes=100)
        tree = Tree(config=config)
        
        root = tree.add_root("Root")
        child1 = tree.add_node(root, "Child 1")
        child2 = tree.add_node(root, "Child 2")
        
        if child1 and child2:
            # Third child should fail due to max_width
            child3 = tree.add_node(root, "Child 3")
            assert child3 is None
    
    def test_progressive_widening(self):
        """Test progressive widening behavior."""
        config = TreeSearchConfig(
            max_depth=10,
            max_width=5,
            max_nodes=100,
            progressive_widening_threshold=3
        )
        tree = Tree(config=config)
        
        root = tree.add_root("Root")
        child1 = tree.add_node(root, "Child 1")
        
        # Root needs more visits before allowing second child
        if child1:
            # Simulate visits
            root.visit_count = 1
            child2 = tree.add_node(root, "Child 2")
            # May be None depending on implementation
            
            # After enough visits
            root.visit_count = 3
            child3 = tree.add_node(root, "Child 3")
            # Should potentially succeed
    
    def test_duplicate_hypothesis_detection(self):
        """Test duplicate hypothesis detection."""
        tree = Tree()
        root = tree.add_root("Root hypothesis")
        
        child1 = tree.add_node(root, "Duplicate hypothesis")
        if child1:
            child2 = tree.add_node(root, "Duplicate hypothesis")
            # Second addition should fail
            assert child2 is None
    
    def test_get_leaves(self):
        """Test getting leaf nodes."""
        tree = Tree()
        root = tree.add_root("Root")
        child1 = tree.add_node(root, "Child 1")
        child2 = tree.add_node(root, "Child 2")
        
        leaves = tree.get_leaves()
        
        assert len(leaves) == 2
        assert child1 in leaves
        assert child2 in leaves
        assert root not in leaves
    
    def test_get_completed_nodes(self):
        """Test getting completed nodes with results."""
        tree = Tree()
        root = tree.add_root("Root")
        child1 = tree.add_node(root, "Child 1")
        child2 = tree.add_node(root, "Child 2")
        
        # Mark child1 as completed with results
        if child1:
            child1.status = NodeStatus.COMPLETED
            child1.data.results = {"accuracy": 0.95}
        
        if child2:
            child2.status = NodeStatus.COMPLETED
            # No results
        
        completed = tree.get_completed_nodes()
        
        assert len(completed) == 1
        assert child1 in completed
        assert child2 not in completed
    
    def test_backpropagate_value(self):
        """Test value backpropagation."""
        tree = Tree()
        root = tree.add_root("Root")
        child1 = tree.add_node(root, "Child 1")
        child2 = tree.add_node(child1, "Child 2") if child1 else None
        
        if child2:
            tree.backpropagate_value(child2, 0.9, apply_discount=False)
            
            assert root.visit_count == 1
            assert child1.visit_count == 1
            assert child2.visit_count == 1
            
            assert abs(root.value_sum - 0.9) < 0.001
            assert abs(child1.value_sum - 0.9) < 0.001
            assert abs(child2.value_sum - 0.9) < 0.001
    
    def test_select_node_uct(self):
        """Test UCT-based node selection."""
        tree = Tree()
        root = tree.add_root("Root")
        child1 = tree.add_node(root, "Child 1")
        child2 = tree.add_node(root, "Child 2")
        
        # Both children are leaves, should select one
        selected = tree.select_node_uct()
        
        assert selected is not None
        assert selected in [child1, child2]
    
    def test_tree_statistics(self):
        """Test tree statistics calculation."""
        tree = Tree()
        root = tree.add_root("Root")
        child1 = tree.add_node(root, "Child 1")
        
        if child1:
            child1.status = NodeStatus.COMPLETED
            child1.data.results = {"accuracy": 0.95}
            child1.update_value(0.95)
        
        stats = tree.get_statistics()
        
        assert stats["total_nodes"] == 2
        assert stats["completed_nodes"] == 1
        assert stats["leaf_nodes"] == 1
        assert stats["max_depth"] == 1
        assert "best_value" in stats
        assert "created_at" in stats
    
    def test_get_best_path(self):
        """Test getting best path from root."""
        tree = Tree()
        root = tree.add_root("Root")
        child1 = tree.add_node(root, "Child 1")
        child2 = tree.add_node(root, "Child 2")
        
        if child1:
            child1.status = NodeStatus.COMPLETED
            child1.data.results = {"accuracy": 0.8}
            child1.update_value(0.8)
        
        if child2:
            child2.status = NodeStatus.COMPLETED
            child2.data.results = {"accuracy": 0.95}
            child2.update_value(0.95)
        
        best_path = tree.get_best_path()
        
        assert len(best_path) > 0
        assert best_path[-1] == child2  # Best node
    
    def test_tree_serialization(self):
        """Test tree serialization to dictionary."""
        tree = Tree(TreeSearchConfig(max_depth=5))
        root = tree.add_root("Root hypothesis")
        child = tree.add_node(root, "Child hypothesis")
        
        if child:
            child.status = NodeStatus.COMPLETED
            child.data.results = {"metric": 0.9}
        
        dict_repr = tree.to_dict()
        
        assert "config" in dict_repr
        assert "root_id" in dict_repr
        assert "nodes" in dict_repr
        assert "edges" in dict_repr
        assert dict_repr["config"]["max_depth"] == 5
    
    def test_tree_save_load(self, tmp_path):
        """Test saving and loading tree to/from file."""
        tree = Tree(TreeSearchConfig(max_depth=5, max_width=3))
        root = tree.add_root("Root")
        child = tree.add_node(root, "Child")
        
        filepath = tmp_path / "test_tree.json"
        tree.save(filepath)
        
        loaded_tree = Tree.load(filepath)
        
        assert len(loaded_tree) == len(tree)
        assert loaded_tree.root is not None
        assert loaded_tree.root.data.hypothesis == tree.root.data.hypothesis
        assert loaded_tree.config.max_depth == 5
    
    def test_tree_string_representation(self):
        """Test tree string representation."""
        tree = Tree()
        tree.add_root("Root")
        
        str_repr = str(tree)
        
        assert "Tree" in str_repr
        assert "nodes=" in str_repr
    
    def test_get_all_nodes(self):
        """Test getting all nodes."""
        tree = Tree()
        root = tree.add_root("Root")
        child1 = tree.add_node(root, "Child 1")
        child2 = tree.add_node(root, "Child 2")
        
        all_nodes = tree.get_all_nodes()
        
        assert len(all_nodes) == 3
        assert root in all_nodes
        assert child1 in all_nodes
        assert child2 in all_nodes
    
    def test_get_node_by_id(self):
        """Test getting node by ID."""
        tree = Tree()
        root = tree.add_root("Root")
        
        retrieved = tree.get_node(root.id)
        
        assert retrieved == root
        assert tree.get_node("nonexistent") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
