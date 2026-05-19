# Getting Started with TreeQuest Lab

## Prerequisites

- Python 3.11+
- pip or uv package manager
- Git
- (Optional) Docker for enhanced sandboxing
- (Optional) Weights & Biases account for experiment tracking

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/treequest-lab.git
cd treequest-lab
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n treequest python=3.11
conda activate treequest
```

### 3. Install Dependencies

```bash
pip install -e .
```

Or for development:

```bash
pip install -e ".[dev]"
```

### 4. Configure Environment

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```ini
# LLM Provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Weights & Biases
WANDB_API_KEY=...
WANDB_PROJECT=treequest-lab

# Optional: Hugging Face
HF_TOKEN=...
```

## Quick Start

### Run a Dry Run (No LLM, No Execution)

Test the pipeline without spending tokens or running code:

```bash
treequest run --dry-run --max-nodes 5
```

This validates the tree structure and configuration.

### Run a Minimal Experiment

Run a small tree search with 3 nodes:

```bash
treequest run --max-nodes 3 --config config/hyperparameters.yaml
```

### Resume a Previous Run

If a run was interrupted, resume from the last checkpoint:

```bash
treequest run --resume outputs/runs/run_20240101_120000
```

### View Run Status

Check the status of recent runs:

```bash
treequest status
```

### Generate a Report

Create a markdown report from a completed run:

```bash
treequest report --run-id run_20240101_120000
```

### Visualize the Tree

Generate a visualization of the search tree:

```bash
treequest visualize --run-id run_20240101_120000 --format png
```

## Configuration

### Basic Configuration

Edit `config/hyperparameters.yaml` to customize:

```yaml
tree:
  max_depth: 5
  max_width: 3
  max_nodes: 20
  uct_constant: 1.414
  progressive_widening: true
  alpha_widening: 0.5

agents:
  model_name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 2048

execution:
  timeout_seconds: 300
  memory_limit_mb: 2048
  use_docker: false

evaluation:
  benchmark: "cifar10_subset"
  metrics:
    - accuracy
    - loss
    - training_time
```

### Environment Variables

Override any config via environment variables:

```bash
export TREEQUEST_MAX_NODES=50
export TREEQUEST_MODEL_NAME="claude-3-haiku"
treequest run
```

## Understanding the Output

### Directory Structure

After running, you'll find outputs in:

```
outputs/
├── runs/
│   └── run_20240101_120000/
│       ├── tree.json           # Full tree state
│       ├── config.yaml         # Run configuration
│       ├── logs.jsonl          # Step-by-step logs
│       ├── nodes/              # Per-node data
│       │   ├── node_001/
│       │   │   ├── code.py
│       │   │   ├── results.json
│       │   │   └── plots/
│       │   └── ...
│       └── reports/
│           └── final_report.md
└── visualizations/
    └── tree_run_20240101_120000.png
```

### Log Format

Logs are stored in JSONL format (one JSON per line):

```json
{"timestamp": "2024-01-01T12:00:00", "level": "INFO", "event": "node_created", "node_id": "n1", "hypothesis": "..."}
{"timestamp": "2024-01-01T12:01:00", "level": "INFO", "event": "code_generated", "node_id": "n1", "path": "..."}
{"timestamp": "2024-01-01T12:05:00", "level": "INFO", "event": "execution_complete", "node_id": "n1", "status": "success"}
```

### Tree Visualization

The tree visualization shows:
- **Green nodes**: Completed successfully
- **Red nodes**: Failed execution
- **Blue nodes**: Currently being explored
- **Gray nodes**: Not yet visited
- **Edge labels**: UCT scores or transition probabilities

## Common Workflows

### Workflow 1: Exploring Architecture Improvements

Goal: Find better neural architecture modifications for a small vision model.

```bash
treequest run \
  --prompt "Improve parameter efficiency of small CNNs" \
  --max-nodes 15 \
  --benchmark cifar10_subset \
  --config configs/architecture_search.yaml
```

### Workflow 2: Hyperparameter Optimization

Goal: Discover novel learning rate schedules or optimizer combinations.

```bash
treequest run \
  --prompt "Novel optimization strategies for transformer training" \
  --max-depth 4 \
  --max-width 5 \
  --benchmark tiny_imagenet
```

### Workflow 3: Ablation Studies

Compare tree search vs random search:

```bash
# Tree search
treequest run --max-nodes 20 --strategy mcts --seed 42

# Random baseline
treequest run --max-nodes 20 --strategy random --seed 42

# Compare results
treequest report --compare run_mcts_*, run_random_*
```

## Troubleshooting

### Issue: LLM API Errors

**Symptoms**: Rate limit errors, timeout, authentication failures.

**Solutions**:
1. Check `.env` file has correct API keys.
2. Reduce `max_tokens` in config.
3. Add retry logic with exponential backoff.
4. Switch to a cheaper/faster model for testing.

### Issue: Code Execution Failures

**Symptoms**: Experiments crash, timeout, or produce no results.

**Solutions**:
1. Increase `timeout_seconds` in config.
2. Check sandbox logs in `nodes/<id>/stderr.log`.
3. Run code manually to debug: `python experiments/outputs/<script>.py`.
4. Enable Docker for better isolation.

### Issue: Out of Memory

**Symptoms**: CUDA OOM or system memory exhaustion.

**Solutions**:
1. Reduce batch sizes in generated code.
2. Use smaller datasets (e.g., `cifar10_subset` instead of full).
3. Set `memory_limit_mb` in config.
4. Run on CPU-only mode for testing.

### Issue: Tree Not Expanding

**Symptoms**: Search gets stuck on one branch.

**Solutions**:
1. Increase `uct_constant` for more exploration.
2. Disable progressive widening temporarily.
3. Check novelty detection threshold in memory module.
4. Manually seed initial diverse hypotheses.

## Best Practices

1. **Start Small**: Begin with `--max-nodes 5` and `--dry-run` to validate setup.
2. **Use Checkpoints**: Runs auto-save; use `--resume` after interruptions.
3. **Monitor Costs**: Track token usage via WandB or provider dashboard.
4. **Version Control**: Commit configs and custom prompts to git.
5. **Iterate on Prompts**: Refine agent prompts based on output quality.
6. **Ablate Components**: Compare with/without tree search, reflection, etc.

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for deep dive into system design.
- Explore [examples/](../examples/) for sample experiments.
- Join discussions on [GitHub Issues](https://github.com/your-username/treequest-lab/issues).
- Contribute new agents, benchmarks, or visualization tools.

## Community & Support

- GitHub Issues: Bug reports and feature requests
- Discussions: Ideas, questions, and show-and-tell
- Twitter: Follow @TreeQuestLab for updates
