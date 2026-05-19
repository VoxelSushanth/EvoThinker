"""Main entry point for TreeQuest Lab CLI."""

import typer
from pathlib import Path
from typing import Optional

from config.config import Settings, get_settings, load_config
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="treequest",
    help="TreeQuest Lab - Agentic Tree Search Research Agent",
    add_completion=False,
)

console = Console()


@app.command()
def run(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    max_nodes: Optional[int] = typer.Option(
        None,
        "--max-nodes",
        "-n",
        help="Maximum number of nodes in search tree",
    ),
    max_depth: Optional[int] = typer.Option(
        None,
        "--max-depth",
        "-d",
        help="Maximum depth of search tree",
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model to use (e.g., gpt-4o-mini, claude-3-haiku)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for results",
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        "-r",
        help="Path to saved tree state to resume from",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration without running experiments",
    ),
) -> None:
    """Run a TreeQuest Lab experiment.
    
    This launches the agentic tree search to explore research ideas,
    run experiments, and generate reports.
    
    Examples:
    
        # Run with default settings
        treequest run
        
        # Run with custom configuration
        treequest run --config config/my_experiment.yaml
        
        # Run with limited budget
        treequest run --max-nodes 50 --max-depth 5
        
        # Resume from checkpoint
        treequest run --resume outputs/runs/tree_20240101.json
    """
    from src.core.tree import Tree, TreeSearchConfig
    
    console.print("[bold blue]🌳 TreeQuest Lab[/bold blue]")
    console.print("[dim]Agentic Tree Search Research Agent[/dim]\n")
    
    # Load configuration
    settings = load_config(config_file) if config_file else get_settings()
    
    # Override with command-line arguments
    if max_nodes:
        settings.tree_max_nodes = max_nodes
    if max_depth:
        settings.tree_max_depth = max_depth
    if llm_model:
        settings.llm_model = llm_model
    if output_dir:
        settings.log_dir = output_dir / "logs"
    
    # Display configuration
    _display_config(settings)
    
    if dry_run:
        console.print("\n[green]✓ Configuration valid![/green]")
        return
    
    # Check for resume
    if resume:
        if not resume.exists():
            console.print(f"[red]Error: Resume file not found: {resume}[/red]")
            raise typer.Exit(1)
        
        console.print(f"\nResuming from: {resume}")
        tree = Tree.load(resume)
    else:
        # Initialize new tree
        tree_config = TreeSearchConfig(
            max_depth=settings.tree_max_depth,
            max_width=settings.tree_max_width,
            max_nodes=settings.tree_max_nodes,
            uct_exploration_constant=settings.uct_exploration_constant,
            progressive_widening_threshold=settings.progressive_widening_threshold,
        )
        tree = Tree(config=tree_config)
        
        # Add root hypothesis
        root_hypothesis = "Explore efficient fine-tuning methods for small language models"
        tree.add_root(
            hypothesis=root_hypothesis,
            description="Base research direction for investigation"
        )
    
    console.print(f"\nInitialized tree: {tree}")
    
    # TODO: Import and run main experiment loop
    # from src.core.experiment_manager import ExperimentManager
    # manager = ExperimentManager(tree, settings)
    # manager.run()
    
    console.print(
        "\n[yellow]⚠️  Experiment execution not yet implemented.[/yellow]\n"
        "This is a placeholder for Phase 1 foundation.\n"
        "Next phases will implement the full agent loop."
    )
    
    # Save initial tree state
    output_path = settings.get_output_dir("runs")
    tree_file = output_path / f"tree_{tree.created_at.strftime('%Y%m%d_%H%M%S')}.json"
    tree.save(tree_file)
    console.print(f"\nSaved tree state to: [cyan]{tree_file}[/cyan]")


@app.command()
def status() -> None:
    """Show current project status and recent runs."""
    console.print("[bold blue]🌳 TreeQuest Lab Status[/bold blue]\n")
    
    # Check for recent runs
    runs_dir = Path("outputs/runs")
    if runs_dir.exists():
        run_files = list(runs_dir.glob("*.json"))
        if run_files:
            console.print(f"[green]✓ Found {len(run_files)} run(s)[/green]\n")
            
            table = Table(title="Recent Runs")
            table.add_column("File", style="cyan")
            table.add_column("Created", style="magenta")
            table.add_column("Size", style="green")
            
            for run_file in sorted(run_files, key=lambda x: x.stat().st_mtime)[-5:]:
                stats = run_file.stat()
                from datetime import datetime
                created = datetime.fromtimestamp(stats.st_ctime).strftime("%Y-%m-%d %H:%M")
                size = f"{stats.st_size / 1024:.1f} KB"
                table.add_row(run_file.name, created, size)
            
            console.print(table)
        else:
            console.print("[yellow]No runs found yet[/yellow]")
    else:
        console.print("[yellow]No outputs directory found[/yellow]")
    
    console.print("\nRun [cyan]`treequest run`[/cyan] to start an experiment!")


@app.command()
def visualize(
    tree_file: Path = typer.Argument(
        ...,
        help="Path to tree JSON file",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for visualization",
    ),
) -> None:
    """Visualize a search tree."""
    from src.core.tree import Tree
    
    if not tree_file.exists():
        console.print(f"[red]Error: Tree file not found: {tree_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"Loading tree from: {tree_file}")
    tree = Tree.load(tree_file)
    
    console.print(f"Loaded tree: {tree}")
    
    # TODO: Import visualization module
    # from src.visualization.tree_viz import visualize_tree
    # viz_path = visualize_tree(tree, output)
    
    console.print(
        "\n[yellow]⚠️  Visualization not yet implemented.[/yellow]\n"
        "This will be added in Phase 4."
    )


@app.command()
def report(
    run_id: str = typer.Argument(
        ...,
        help="Run ID or path to results",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for report",
    ),
) -> None:
    """Generate a research report from run results."""
    console.print(f"Generating report for run: {run_id}")
    
    # TODO: Import reporter agent
    # from src.agents.reporter import ReporterAgent
    # reporter = ReporterAgent()
    # report = reporter.generate(run_id)
    
    console.print(
        "\n[yellow]⚠️  Report generation not yet implemented.[/yellow]\n"
        "This will be added in Phase 4."
    )


@app.command()
def clean() -> None:
    """Clean up temporary files and outputs."""
    import shutil
    
    console.print("[bold yellow]Cleaning up...[/bold yellow]\n")
    
    dirs_to_clean = [
        Path("outputs/runs"),
        Path("outputs/papers"),
        Path("outputs/visualizations"),
        Path("outputs/memory"),
        Path("outputs/logs"),
    ]
    
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            # Keep .gitkeep files
            for item in dir_path.iterdir():
                if item.name != ".gitkeep":
                    if item.is_file():
                        item.unlink()
                        console.print(f"Removed: {item}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        console.print(f"Removed directory: {item}")
    
    console.print("\n[green]✓ Cleanup complete![/green]")


def _display_config(settings: Settings) -> None:
    """Display current configuration in a table."""
    table = Table(title="Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("LLM Provider", settings.llm_provider)
    table.add_row("LLM Model", settings.llm_model)
    table.add_row("Temperature", str(settings.temperature))
    table.add_row("Max Nodes", str(settings.tree_max_nodes))
    table.add_row("Max Depth", str(settings.tree_max_depth))
    table.add_row("Max Width", str(settings.tree_max_width))
    table.add_row("Memory Enabled", str(settings.memory_enabled))
    table.add_row("W&B Enabled", str(settings.wandb_enabled))
    
    console.print(table)


def cli() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    app()
