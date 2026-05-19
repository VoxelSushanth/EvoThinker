"""
Plotting utilities for TreeQuest Lab.

Training curves, result tables, and comparison plots.
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class PlotManager:
    """
    Manages creation and saving of various plots.
    
    Features:
    - Training curve plots
    - Result comparison tables
    - Statistical analysis plots
    - Export to multiple formats
    """
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style configuration
        self.style_config = {
            'figure.figsize': (10, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'font.size': 10,
            'lines.linewidth': 2,
            'lines.markersize': 4
        }
        plt.style.use('default')
        
    def plot_training_curves(
        self,
        results: list[dict],
        x_key: str = 'step',
        y_keys: list[str] | None = None,
        title: str = "Training Curves",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training curves from experiment results.
        
        Args:
            results: List of result dictionaries with metrics
            x_key: Key for x-axis (e.g., 'step', 'epoch')
            y_keys: Keys for y-axis metrics (default: all numeric keys)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if not results:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Extract x values
        x_values = [r.get(x_key, i) for i, r in enumerate(results)]
        
        # Determine y keys
        if y_keys is None:
            y_keys = []
            if results:
                sample = results[0]
                y_keys = [k for k in sample.keys() 
                         if k != x_key and isinstance(sample[k], (int, float))]
        
        # Plot each metric
        colors = plt.cm.tab10(np.linspace(0, 1, len(y_keys)))
        
        for idx, y_key in enumerate(y_keys):
            y_values = [r.get(y_key, None) for r in results]
            # Filter out None values
            valid_pairs = [(x, y) for x, y in zip(x_values, y_values) if y is not None]
            
            if valid_pairs:
                x_valid, y_valid = zip(*valid_pairs)
                ax.plot(x_valid, y_valid, '-', label=y_key, color=colors[idx], linewidth=2)
        
        ax.set_xlabel(x_key)
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = self.output_dir / f"curves_{timestamp}.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_file}")
        
        return fig
    
    def plot_comparison_bar(
        self,
        data: dict[str, float],
        title: str = "Comparison",
        ylabel: str = "Score",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart comparing different methods/nodes.
        
        Args:
            data: Dictionary of name -> value
            title: Plot title
            ylabel: Y-axis label
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(data.keys())
        values = list(data.values())
        
        # Sort by value
        sorted_indices = np.argsort(values)[::-1]
        names = [names[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(values)))
        
        bars = ax.bar(range(len(names)), values, color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = self.output_dir / f"comparison_{timestamp}.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_file}")
        
        return fig
    
    def plot_heatmap(
        self,
        matrix: np.ndarray,
        row_labels: list[str],
        col_labels: list[str],
        title: str = "Heatmap",
        cmap: str = "RdYlGn",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create heatmap visualization.
        
        Args:
            matrix: 2D numpy array
            row_labels: Labels for rows
            col_labels: Labels for columns
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Value", rotation=-90, va="bottom", labelpad=20)
        
        ax.set_title(title)
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = self.output_dir / f"heatmap_{timestamp}.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_file}")
        
        return fig


def plot_training_curves(
    results: list[dict],
    **kwargs
) -> str:
    """
    Convenience function to plot training curves.
    
    Args:
        results: List of result dictionaries
        **kwargs: Additional arguments to PlotManager.plot_training_curves
        
    Returns:
        Path to saved image
    """
    manager = PlotManager()
    fig = manager.plot_training_curves(results, **kwargs)
    
    # Get save path from logger or use default
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(manager.output_dir / f"curves_{timestamp}.png")


def plot_results_table(
    results: list[dict],
    columns: list[str] | None = None,
    save_path: Optional[str] = None
) -> str:
    """
    Create a table visualization of results.
    
    Args:
        results: List of result dictionaries
        columns: Columns to display
        save_path: Path to save table image
        
    Returns:
        Path to saved image
    """
    if not results:
        logger.warning("No results to display")
        return ""
    
    # Determine columns
    if columns is None:
        columns = list(results[0].keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create table
    table_data = []
    for result in results:
        row = [str(result.get(col, 'N/A')) for col in columns]
        table_data.append(row)
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['lightblue'] * len(columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_fontsize(10)
        table[(0, i)].set_text_props(weight='bold')
    
    plt.title("Results Table", fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Results table saved to {save_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = Path("outputs/visualizations") / f"table_{timestamp}.png"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        logger.info(f"Results table saved to {save_file}")
        save_path = str(save_file)
    
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    # Test plotting functions
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing PlotManager")
    print("=" * 60)
    
    manager = PlotManager()
    
    # Test training curves
    mock_results = [
        {'step': i, 'loss': 1.0 / (i + 1) + 0.1 * np.random.randn(), 'accuracy': 0.5 + 0.4 * (1 - 1/(i+1))}
        for i in range(20)
    ]
    
    fig1 = manager.plot_training_curves(mock_results, title="Mock Training Curves")
    print("\n✓ Training curves plotted")
    
    # Test comparison bar
    comparison_data = {
        'Baseline': 0.72,
        'Method A': 0.78,
        'Method B': 0.81,
        'Method C': 0.75
    }
    
    fig2 = manager.plot_comparison_bar(comparison_data, title="Method Comparison")
    print("✓ Comparison bar plotted")
    
    # Test heatmap
    matrix = np.random.rand(5, 4)
    row_labels = ['Node 1', 'Node 2', 'Node 3', 'Node 4', 'Node 5']
    col_labels = ['Metric A', 'Metric B', 'Metric C', 'Metric D']
    
    fig3 = manager.plot_heatmap(matrix, row_labels, col_labels, title="Node Metrics Heatmap")
    print("✓ Heatmap plotted")
    
    # Test results table
    table_path = plot_results_table(
        [{'ID': i, 'Score': 0.7 + 0.1*i, 'Status': 'Complete'} for i in range(5)],
        columns=['ID', 'Score', 'Status']
    )
    print(f"✓ Results table saved to: {table_path}")
    
    print(f"\nAll plots saved to: {manager.output_dir}")
    print("\nTest complete!")
