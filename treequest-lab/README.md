# TreeQuest Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)

**A lightweight, open-source Agentic Tree Search Research Agent inspired by Sakana AI's AI Scientist-v2.**

TreeQuest Lab enables autonomous ML research through progressive agentic tree search, where an LLM agent explores research ideas, designs and runs experiments, analyzes results, and generates mini research reports—all on consumer hardware.

![Demo GIF Placeholder](outputs/visualizations/demo.gif)

## 🌟 Motivation

Inspired by [Sakana AI's AI Scientist-v2](https://sakana.ai/ai-scientist-v2/), TreeQuest Lab brings nature-inspired exploration to automated scientific discovery. We prioritize:

- **Simplicity & Reproducibility**: Clean, modular code that runs anywhere
- **Transparency**: Every decision, branch, and reflection is logged
- **Cost Control**: Efficient search strategies for limited compute budgets
- **Scientific Rigor**: Structured evaluation and ablation studies

## 🏗️ Architecture

```mermaid
graph TD
    A[Experiment Manager] --> B[Tree Search (MCTS)]
    B --> C[Node Selection]
    C --> D[Idea Proposer Agent]
    D --> E[Code Architect Agent]
    E --> F[Sandbox Executor]
    F --> G[Evaluator Agent]
    G --> H[Critic/Reflector Agent]
    H --> I[Backpropagation]
    I --> J[Reporter Agent]
    J --> K[Mini Research Report]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style J fill:#bfb,stroke:#333,stroke-width:2px
```

### Core Components

1. **Tree Manager / Experiment Manager**: Central brain maintaining the search tree with UCT selection + progressive widening
2. **Specialized Agents** (LangGraph-based):
   - Idea Proposer: Novel hypothesis generation
   - Code Architect: Safe experiment code generation
   - Evaluator: Metrics, plots, statistical analysis
   - Critic: Scoring on novelty, feasibility, promise
   - Reporter: Markdown + LaTeX-style reports
3. **Search Algorithm**: MCTS with value-guided exploration
4. **Sandbox Execution**: Safe subprocess execution with timeouts
5. **Memory System**: Vector store for novelty promotion

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/treequest-lab.git
cd treequest-lab
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
# Edit .env with your LLM API keys
```

### Basic Usage

```bash
# Run a simple tree search experiment
python run.py --config config/hyperparameters.yaml --max-nodes 50

# Launch the dashboard
python -m src.visualization.dashboard

# Generate a report from existing results
python scripts/reproduce.py --run-id <run_id>
```

## 📊 Example Results

After a successful run, TreeQuest Lab produces:

- **Research Report**: `outputs/papers/<timestamp>_report.md` with abstract, methods, results, and discussion
- **Tree Visualization**: `outputs/visualizations/tree_<timestamp>.png`
- **Training Curves**: `outputs/visualizations/metrics_<timestamp>.png`
- **Full Logs**: `outputs/runs/<timestamp>/logs.jsonl`

### Sample Report Structure

```markdown
# Title: [Generated Title]

## Abstract
[Brief summary of the research idea and findings]

## Method
[Description of the experimental approach]

## Results
| Metric | Baseline | Ours | Improvement |
|--------|----------|------|-------------|
| Accuracy | 0.85 | 0.89 | +4.7% |

## Discussion
[Analysis of results and implications]

## Limitations
[Honest assessment of constraints]

## References
[Cited works]
```

## 🔬 How This Relates to Sakana AI Scientist-v2

TreeQuest Lab implements key concepts from Sakana's work:

| Feature | Sakana AI Scientist-v2 | TreeQuest Lab |
|---------|----------------------|---------------|
| Search Strategy | Agentic Tree Search | MCTS + Progressive Widening |
| Agent Roles | Multi-agent system | LangGraph workflow |
| Execution | Code interpreter | Sandboxed subprocess |
| Evaluation | Automated benchmarks | Custom harness + metrics |
| Output | Full paper | Mini research report |
| Compute | Cloud-scale | Consumer GPU/Colab |

**Key Differences:**
- Lightweight design for accessibility
- Focus on reproducibility and transparency
- Modular architecture for easy extension
- Open-source alternative to proprietary systems

## 🧪 Ablation Studies

We recommend running these ablations to understand component contributions:

1. **Tree Search vs Linear Chain**: Compare MCTS exploration vs sequential idea testing
2. **With/Without Memory**: Impact of vector store on novelty scores
3. **Progressive Widening**: Effect on search efficiency
4. **Agent Variants**: Different prompting strategies for each role

See `notebooks/exploration.ipynb` for detailed analysis templates.

## 📁 Project Structure

```
treequest-lab/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── pyproject.toml            # Package configuration
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── main.py                   # Entry point
├── run.py                    # Main execution script
├── config/
│   ├── __init__.py
│   ├── config.py             # Configuration system
│   └── hyperparameters.yaml  # Default hyperparameters
├── src/
│   ├── core/                 # Core data structures
│   │   ├── node.py           # Tree node definition
│   │   ├── tree.py           # Tree management
│   │   └── experiment_manager.py
│   ├── agents/               # LLM agents
│   │   ├── proposer.py       # Idea generation
│   │   ├── executor.py       # Code execution
│   │   ├── evaluator.py      # Result analysis
│   │   ├── critic.py         # Reflection & scoring
│   │   └── reporter.py       # Report generation
│   ├── search/               # Search algorithms
│   │   ├── mcts.py           # MCTS implementation
│   │   └── utils.py          # Search utilities
│   ├── sandbox/              # Safe execution
│   │   └── executor.py       # Sandbox runner
│   ├── evaluation/           # Benchmarks
│   │   └── harness.py        # Evaluation harness
│   ├── visualization/        # Visual outputs
│   │   ├── tree_viz.py       # Tree diagrams
│   │   └── plots.py          # Metric plots
│   └── utils/                # Utilities
│       ├── logging.py        # Logging setup
│       ├── memory.py         # Vector store
│       └── helpers.py        # Helper functions
├── experiments/              # Experiment configs & outputs
├── outputs/                  # Generated artifacts
├── notebooks/                # Analysis notebooks
├── scripts/                  # Utility scripts
└── tests/                    # Test suite
```

## 🛠️ Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
black src/ tests/
mypy src/
ruff check src/
```

### Adding New Agents

1. Create agent class in `src/agents/`
2. Define LangGraph state schema
3. Register in workflow
4. Add tests

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Inspired by [Sakana AI's AI Scientist-v2](https://sakana.ai/ai-scientist-v2/)
- Built with [LangGraph](https://langchain-ai.github.io/langgraph/)
- Uses [Hugging Face Transformers](https://huggingface.co/transformers/)

## 📬 Contact

For questions or collaborations, please open an issue or contact the maintainers.

---

*Built with ❤️ for autonomous scientific discovery*
