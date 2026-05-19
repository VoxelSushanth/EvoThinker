# Contributing to TreeQuest Lab

Thank you for your interest in contributing! This guide helps you get started.

## Ways to Contribute

1. **Bug Reports**: Open issues with reproduction steps
2. **Feature Requests**: Propose new agents, benchmarks, or features
3. **Code Contributions**: Fix bugs, add features, improve tests
4. **Documentation**: Improve guides, add examples, fix typos
5. **Experiments**: Share interesting results and ablation studies

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/treequest-lab.git
cd treequest-lab
git remote add upstream https://github.com/original-username/treequest-lab.git
```

### 2. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- pytest and pytest-cov for testing
- black, ruff, mypy for code quality
- pre-commit hooks

### 3. Set Up Pre-commit Hooks

```bash
pre-commit install
```

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use Black for formatting
- Type hints required for all functions
- Docstrings in Google style

```python
def calculate_uct_score(
    node: Node,
    parent_visit_count: int,
    c: float = 1.414
) -> float:
    """Calculate UCT (Upper Confidence Bound for Trees) score.
    
    Args:
        node: The node to evaluate
        parent_visit_count: Visit count of parent node
        c: Exploration constant
        
    Returns:
        UCT score balancing exploration vs exploitation
    """
    if node.visit_count == 0:
        return float('inf')
    
    exploitation = node.value_sum / node.visit_count
    exploration = c * math.sqrt(math.log(parent_visit_count) / node.visit_count)
    
    return exploitation + exploration
```

### Testing Requirements

- All new features need tests
- Aim for >80% coverage
- Use pytest fixtures for setup
- Mock external API calls

```python
# tests/test_new_feature.py
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_llm():
    """Fixture for mocked LLM client."""
    client = Mock()
    client.generate.return_value = {"score": 0.8}
    return client

def test_new_feature(mock_llm):
    """Test the new feature."""
    result = new_feature(mock_llm)
    assert result.success is True
```

### Documentation Standards

- Update README for user-facing changes
- Add docstrings to all public APIs
- Include usage examples
- Update CHANGELOG.md

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/my-new-feature
```

Branch naming:
- `feature/description` for new features
- `fix/description` for bug fixes
- `docs/description` for documentation
- `tests/description` for test additions

### 2. Make Changes

- Write code
- Write tests
- Update documentation
- Run linters: `black . && ruff check . && mypy src/`
- Run tests: `pytest tests/ -v`

### 3. Commit Messages

Follow conventional commits:

```
feat: add progressive widening to MCTS

- Implement C(n) = floor(n^alpha) formula
- Add alpha_widening config parameter
- Update tests and documentation

Closes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting (no logic change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 4. Submit PR

1. Push to your fork
2. Open PR against main branch
3. Fill out PR template
4. Link related issues
5. Request review

### 5. Review Process

- Maintainers will review within 1 week
- Address feedback promptly
- CI must pass (tests, linting)
- Squash commits before merge

## Adding New Agents

### Step 1: Create Agent Class

```python
# src/agents/new_agent.py
from pydantic import BaseModel
from .base import BaseAgent

class NewAgentOutput(BaseModel):
    result: str
    confidence: float

class NewAgentInput(BaseModel):
    query: str
    context: dict

class NewAgent(BaseAgent[NewAgentInput, NewAgentOutput]):
    def generate(self, input_data: NewAgentInput) -> NewAgentOutput:
        # Implementation
        pass
```

### Step 2: Add to Workflow

```python
# src/agents/workflow.py
workflow.add_node("new_agent", new_agent_node)
workflow.add_edge("previous_node", "new_agent")
```

### Step 3: Write Tests

```python
# tests/test_new_agent.py
def test_new_agent():
    agent = NewAgent(config={})
    result = agent.generate(NewAgentInput(query="test"))
    assert result.confidence > 0
```

### Step 4: Document

Add to `docs/AGENTS.md` with usage example.

## Adding New Benchmarks

### Step 1: Create Benchmark Class

```python
# src/evaluation/benchmarks/new_benchmark.py
from .base import Benchmark

class NewBenchmark(Benchmark):
    name = "new_benchmark"
    
    def load_data(self):
        # Load dataset
        pass
    
    def evaluate(self, model, predictions):
        # Calculate metrics
        return {"accuracy": ..., "f1": ...}
```

### Step 2: Register Benchmark

```python
# src/evaluation/harness.py
BENCHMARKS["new_benchmark"] = NewBenchmark
```

### Step 3: Add Config Option

```yaml
# config/hyperparameters.yaml
evaluation:
  benchmark: "new_benchmark"
```

## Reporting Issues

### Bug Report Template

```markdown
**Describe the bug**
Clear description of what went wrong.

**To Reproduce**
Steps to reproduce:
1. Run command: `treequest run --max-nodes 5`
2. See error: ...

**Expected behavior**
What should have happened.

**Environment:**
- OS: Ubuntu 22.04
- Python: 3.11
- Package version: 0.1.0

**Logs**
Attach relevant log output.
```

### Feature Request Template

```markdown
**Problem statement**
What problem does this solve?

**Proposed solution**
How should it work?

**Alternatives considered**
Other approaches you thought about.

**Additional context**
Any other relevant information.
```

## Experiment Contributions

We welcome experiment results! To contribute:

1. Run experiment with full configuration
2. Save outputs in `experiments/results/`
3. Create notebook analyzing results
4. Write summary in `experiments/README.md`
5. Submit as PR with label "experiments"

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- No harassment or discrimination

## Questions?

- Open an issue for general questions
- Join discussions in GitHub Discussions
- Check existing documentation first

## Thank You!

Your contributions make TreeQuest Lab better for everyone. We appreciate your time and effort!
