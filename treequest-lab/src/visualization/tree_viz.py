"""
Tree Visualization for TreeQuest Lab.

Visualizes the search tree using matplotlib and networkx.
Supports coloring by score, status, and depth.
"""

import logging
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger(__name__)


class TreeVisualizer:
    """
    Visualizes TreeQuest search trees.
    
    Features:
    - Color nodes by score or status
    - Show depth levels
    - Display hypothesis summaries
    - Export to PNG/PDF/SVG
    """
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style configuration
        self.node_size = 800
        self.font_size = 8
        self.edge_width = 1.5
        self.fig_size = (16, 12)
        
    def plot_tree(
        self,
        tree: Any,
        color_by: str = "score",
        show_labels: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the search tree.
        
        Args:
            tree: TreeQuest Tree object
            color_by: 'score', 'status', 'depth', or 'visits'
            show_labels: Show node hypotheses
            title: Plot title
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create networkx DiGraph from tree
        G = self._tree_to_nx(tree)
        
        # Set up figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Compute layout using hierarchical layout
        pos = self._hierarchical_layout(G, tree)
        
        # Get node colors
        colors = self._get_node_colors(G, tree, color_by)
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=self.edge_width,
            alpha=0.6,
            ax=ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=colors,
            node_size=self.node_size,
            cmap=plt.cm.RdYlGn,
            vmin=0, vmax=1,
            ax=ax
        )
        
        # Draw labels
        if show_labels:
            labels = self._get_node_labels(G, tree)
            nx.draw_networkx_labels(
                G, pos,
                labels=labels,
                font_size=self.font_size,
                font_weight='bold',
                ax=ax
            )
        
        # Add colorbar
        if color_by in ['score', 'visits']:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label(color_by.capitalize(), fontsize=10)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title("TreeQuest Search Tree", fontsize=14, fontweight='bold')
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_file = Path(save_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = self.output_dir / f"tree_{timestamp}.png"
        
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        logger.info(f"Tree visualization saved to {save_file}")
        
        return fig
    
    def _tree_to_nx(self, tree: Any) -> nx.DiGraph:
        """Convert TreeQuest tree to NetworkX DiGraph."""
        G = nx.DiGraph()
        
        for node in tree.get_all_nodes():
            G.add_node(node.id)
            
            if node.parent_id is not None:
                G.add_edge(node.parent_id, node.id)
        
        return G
    
    def _hierarchical_layout(self, G: nx.DiGraph, tree: Any) -> dict:
        """Create hierarchical layout based on tree depth."""
        pos = {}
        
        # Group nodes by depth
        depth_groups = {}
        for node_id in G.nodes():
            depth = tree.get_depth(node_id)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(node_id)
        
        # Position nodes
        max_depth = max(depth_groups.keys()) if depth_groups else 0
        
        for depth, nodes in depth_groups.items():
            # Horizontal position based on index
            for i, node_id in enumerate(nodes):
                x = i / max(len(nodes), 1)
                y = 1 - (depth / max(max_depth + 1, 1))
                pos[node_id] = (x, y)
        
        # Use spring layout for better spacing within levels
        if len(G) > 1:
            pos_fixed = {k: v for k, v in pos.items()}
            pos_spring = nx.spring_layout(G, pos=pos_fixed, fixed=pos_fixed.keys(), iterations=50)
            pos.update(pos_spring)
        
        return pos
    
    def _get_node_colors(self, G: nx.DiGraph, tree: Any, color_by: str) -> list:
        """Get color values for nodes."""
        colors = []
        
        for node_id in G.nodes():
            try:
                node = tree.get_node(node_id)
                
                if color_by == 'score':
                    # Use node value or results score
                    if hasattr(node.data, 'results') and node.data.results:
                        score = node.data.results.get('primary_score', node.value)
                    else:
                        score = node.value
                    colors.append(score)
                    
                elif color_by == 'status':
                    # Map status to numeric value
                    status_map = {
                        'UNVISITED': 0.3,
                        'VISITING': 0.5,
                        'COMPLETED': 0.9,
                        'FAILED': 0.1,
                        'SKIPPED': 0.4
                    }
                    colors.append(status_map.get(node.status.name, 0.5))
                    
                elif color_by == 'depth':
                    depth = tree.get_depth(node_id)
                    max_depth = max(tree.get_depth(nid) for nid in G.nodes())
                    colors.append(1 - (depth / max(max_depth, 1)))
                    
                elif color_by == 'visits':
                    visits = node.visits
                    max_visits = max(n.visits for n in tree.get_all_nodes())
                    colors.append(visits / max(max_visits, 1))
                    
                else:
                    colors.append(0.5)
                    
            except Exception as e:
                logger.warning(f"Error getting color for node {node_id}: {e}")
                colors.append(0.5)
        
        return colors
    
    def _get_node_labels(self, G: nx.DiGraph, tree: Any) -> dict:
        """Get truncated hypothesis labels for nodes."""
        labels = {}
        
        for node_id in G.nodes():
            try:
                node = tree.get_node(node_id)
                hypothesis = node.data.hypothesis
                
                # Truncate long hypotheses
                if len(hypothesis) > 40:
                    hypothesis = hypothesis[:37] + "..."
                
                # Add node ID prefix
                labels[node_id] = f"N{node_id}\n{hypothesis}"
                
            except Exception as e:
                logger.warning(f"Error getting label for node {node_id}: {e}")
                labels[node_id] = f"N{node_id}"
        
        return labels
    
    def plot_statistics(
        self,
        stats_history: list[dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot search statistics over time.
        
        Args:
            stats_history: List of iteration statistics
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure with multiple subplots
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = [s.get('iteration', i) for i, s in enumerate(stats_history)]
        
        # Plot 1: Score over time
        scores = [s.get('score', 0) for s in stats_history]
        axes[0, 0].plot(iterations, scores, 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Best Score Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Nodes created per iteration
        nodes_added = [s.get('nodes_added', 0) for s in stats_history]
        axes[0, 1].bar(iterations, nodes_added, color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Nodes Added')
        axes[0, 1].set_title('Node Expansion Rate')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Success rate
        successes = [1 if s.get('success', False) else 0 for s in stats_history]
        cumulative_success = [sum(successes[:i+1]) / (i+1) for i in range(len(successes))]
        axes[1, 0].plot(iterations, cumulative_success, 'r-o', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Cumulative Success Rate')
        axes[1, 0].set_title('Success Rate Over Time')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Duration per iteration
        durations = [s.get('duration', 0) for s in stats_history]
        axes[1, 1].plot(iterations, durations, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Duration (seconds)')
        axes[1, 1].set_title('Iteration Duration')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Statistics plot saved to {save_path}")
        
        return fig


def plot_tree(
    tree: Any,
    color_by: str = "score",
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to plot a tree.
    
    Args:
        tree: TreeQuest Tree object
        color_by: Coloring scheme
        output_path: Where to save the plot
        **kwargs: Additional arguments to TreeVisualizer.plot_tree
        
    Returns:
        Path to saved image
    """
    viz = TreeVisualizer()
    fig = viz.plot_tree(tree, color_by=color_by, save_path=output_path, **kwargs)
    
    # Get the save path from the figure
    if output_path:
        return output_path
    else:
        # Return last saved path from logger
        return str(viz.output_dir / "*.png")


if __name__ == "__main__":
    # Test visualization
    from src.core.tree import Tree, TreeSearchConfig
    
    logging.basicConfig(level=logging.INFO)
    
    # Create test tree
    config = TreeSearchConfig(max_depth=3, max_width=2)
    tree = Tree(config=config)
    tree.add_root("Root hypothesis for testing", description="Test root")
    
    # Add some children
    node1 = tree.add_child(0, data={
        "hypothesis": "First child hypothesis that is somewhat longer",
        "motivation": "Testing"
    })
    node2 = tree.add_child(0, data={
        "hypothesis": "Second child hypothesis",
        "motivation": "Testing"
    })
    
    node3 = tree.add_child(node1.id, data={
        "hypothesis": "Grandchild hypothesis",
        "motivation": "Testing"
    })
    
    # Set some scores
    tree.get_node(0).value = 0.5
    tree.get_node(node1.id).value = 0.7
    tree.get_node(node2.id).value = 0.6
    tree.get_node(node3.id).value = 0.8
    
    # Visualize
    viz = TreeVisualizer()
    fig = viz.plot_tree(tree, color_by='score', title="Test Tree")
    
    print("Test visualization complete!")
    print(f"Output directory: {viz.output_dir}")
