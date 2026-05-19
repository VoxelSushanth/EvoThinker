#!/usr/bin/env python3
"""Run script for TreeQuest Lab experiments.

This is the main entry point for running agentic tree search experiments.
It provides a simple interface to launch the full experiment loop.

Usage:
    python run.py                          # Run with defaults
    python run.py --max-nodes 50           # Limit to 50 nodes
    python run.py --config my_config.yaml  # Use custom config
    python run.py --resume tree.json       # Resume from checkpoint
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Settings, load_config
from src.core.tree import Tree, TreeSearchConfig
from rich.console import Console


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TreeQuest Lab - Agentic Tree Search Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Run with default settings
  %(prog)s --max-nodes 50               Limit to 50 nodes
  %(prog)s --max-depth 5                Limit tree depth to 5
  %(prog)s --config config.yaml         Use custom configuration
  %(prog)s --resume tree_state.json     Resume from checkpoint
  %(prog)s --dry-run                    Validate without executing
        """,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--max-nodes", "-n",
        type=int,
        default=None,
        help="Maximum number of nodes in search tree"
    )
    
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=None,
        help="Maximum depth of search tree"
    )
    
    parser.add_argument(
        "--max-width", "-w",
        type=int,
        default=None,
        help="Maximum width (children per node)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="LLM model to use"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--resume", "-r",
        type=Path,
        default=None,
        help="Path to saved tree state to resume from"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running experiments"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def main():
    """Main entry point."""
    args = parse_args()
    console = Console()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    console.print("[bold blue]🌳 TreeQuest Lab[/bold blue]")
    console.print("[dim]Agentic Tree Search Research Agent[/dim]\n")
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        settings = load_config(args.config)
    else:
        settings = load_config()
    
    # Override with command-line arguments
    if args.max_nodes:
        settings.tree_max_nodes = args.max_nodes
    if args.max_depth:
        settings.tree_max_depth = args.max_depth
    if args.max_width:
        settings.tree_max_width = args.max_width
    if args.model:
        settings.llm_model = args.model
    if args.output_dir:
        settings.log_dir = args.output_dir / "logs"
    
    # Set random seed
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info(f"Random seed set to {args.seed}")
    
    # Display configuration summary
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  LLM Model: {settings.llm_provider}/{settings.llm_model}")
    console.print(f"  Max Nodes: {settings.tree_max_nodes}")
    console.print(f"  Max Depth: {settings.tree_max_depth}")
    console.print(f"  Max Width: {settings.tree_max_width}")
    console.print(f"  Memory: {'Enabled' if settings.memory_enabled else 'Disabled'}")
    
    if args.dry_run:
        console.print("\n[green]✓ Configuration valid![/green]")
        console.print("(Dry run - no experiments executed)")
        return 0
    
    # Initialize or load tree
    if args.resume:
        if not args.resume.exists():
            console.print(f"[red]Error: Resume file not found: {args.resume}[/red]")
            return 1
        
        console.print(f"\nResuming from: {args.resume}")
        tree = Tree.load(args.resume)
        logger.info(f"Loaded tree with {len(tree)} nodes")
    else:
        # Create new tree
        tree_config = TreeSearchConfig(
            max_depth=settings.tree_max_depth,
            max_width=settings.tree_max_width,
            max_nodes=settings.tree_max_nodes,
            uct_exploration_constant=settings.uct_exploration_constant,
            progressive_widening_threshold=settings.progressive_widening_threshold,
        )
        tree = Tree(config=tree_config)
        
        # Add root hypothesis
        root_hypothesis = (
            "Explore efficient fine-tuning methods for small language models "
            "to improve reasoning on mathematical tasks"
        )
        tree.add_root(
            hypothesis=root_hypothesis,
            description="Base research direction for autonomous investigation"
        )
        
        logger.info(f"Created new tree with root: {root_hypothesis[:60]}...")
    
    console.print(f"\nInitialized tree: {tree}")
    
    # TODO: Implement full experiment loop
    # This is where Phase 2+ will add the agent orchestration
    
    console.print("\n" + "="*60)
    console.print("[yellow]⚠️  PHASE 1 FOUNDATION COMPLETE[/yellow]")
    console.print("="*60)
    console.print("""
The core data structures are implemented:
  ✓ Node and NodeData classes
  ✓ Tree with UCT selection
  ✓ Progressive widening
  ✓ Serialization/deserialization
  ✓ Configuration system
  ✓ CLI interface

Next phases will implement:
  → Phase 2: Single agent loop (Propose → Code → Execute → Analyze)
  → Phase 3: Full MCTS with multi-agent orchestration
  → Phase 4: Report generation and visualization
  → Phase 5: UI and polish

To continue development, implement src/core/experiment_manager.py
    """)
    
    # Save initial tree state
    output_path = settings.get_output_dir("runs")
    tree_file = output_path / f"tree_{tree.created_at.strftime('%Y%m%d_%H%M%S')}.json"
    tree.save(tree_file)
    console.print(f"\nSaved tree state to: [cyan]{tree_file}[/cyan]")
    
    logger.info("Phase 1 foundation complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
