# Agent Evaluation Harness

A flexible framework for evaluating AI agents on coding tasks, with support for SWE-bench and custom Docker environments.

## Features

- **Complete Isolation**: Agents run in Docker containers with no host filesystem access
- **Multiple Agent Support**: Currently supports OpenAI models (Anthropic coming soon)
- **Flexible Configuration**: YAML-based configuration system
- **Dataset Support**: SWE-bench datasets with slicing, or custom task definitions
- **Parallel Execution**: Run multiple evaluations concurrently
- **Resume Capability**: Continue interrupted evaluations

## Installation

```bash
# Install the harness
pip install -e .

# Set up environment variables
echo "OPENAI_API_KEY=your-key-here" > .env
```

## Quick Start

### 1. Run a simple demo:
```bash
python examples/simple_docker_demo.py
```

### 2. Evaluate a single SWE-bench instance:
```bash
python examples/swebench_single.py django__django-11999
```

### 3. Run full evaluation with config:
```bash
python run_evaluation.py --config configs/examples/swebench_gpt4.yaml
```

## Configuration System

### Basic Configuration
```yaml
# configs/my_eval.yaml
name: my-evaluation
description: Testing GPT-4 on SWE-bench

agent:
  type: openai
  model: gpt-4-turbo-preview
  temperature: 0.0
  max_turns: 50
  tools:
    - bash
    - read_file
    - write_file
    - edit_file
    - list_files

environment:
  type: swebench
  
dataset:
  type: swebench
  name: princeton-nlp/SWE-bench_Verified
  split: test[:10]  # First 10 instances
```

### CLI Usage
```bash
# Basic run
python run_evaluation.py --config configs/my_eval.yaml

# With runtime options
python run_evaluation.py --config configs/my_eval.yaml \
    --output-dir ./results/experiment1 \
    --workers 4 \
    --resume

# Override configuration
python run_evaluation.py --config configs/my_eval.yaml \
    --override agent.model=gpt-3.5-turbo \
    --override dataset.split=test[:5]
```

## Project Structure

```
evaluation_harness/
├── agents/              # Agent implementations
│   ├── base.py         # Abstract agent interface
│   └── openai.py       # OpenAI API agent
├── environments/        # Environment implementations
│   ├── base.py         # Abstract environment interface
│   ├── docker.py       # Docker container environment
│   └── swebench.py     # SWE-bench specific environment
├── configs/            # Configuration system
│   ├── schema.py       # Configuration dataclasses
│   ├── loader.py       # YAML loading and validation
│   └── examples/       # Example configurations
├── examples/           # Example scripts
├── utils/              # Utility functions
│   └── retry.py        # Exponential backoff retry
├── runner.py           # Evaluation runner
└── run_evaluation.py   # CLI tool
```

## Key Concepts

### Environments
Environments provide isolated execution contexts for agents:
- `DockerEnvironment`: Generic Docker container
- `SWEBenchEnvironment`: SWE-bench problem with test evaluation

### Agents
Agents solve tasks using provided tools:
- `OpenAIAgent`: Uses OpenAI API (GPT-3.5/GPT-4)
- Tools: bash, read_file, write_file, edit_file, list_files

### Configuration
Separates "what to evaluate" from "how to run it":
- **Config file**: Agent settings, dataset, environment
- **CLI args**: Output directory, parallelism, logging

## Advanced Usage

### Custom Dataset
```yaml
dataset:
  type: custom
  source_path: ./my_tasks.json
```

### Specific Instances
```yaml
dataset:
  type: swebench
  instances:
    - django__django-11999
    - astropy__astropy-13033
```

### Environment Variables
```yaml
agent:
  extra_params:
    api_key: ${OPENAI_API_KEY}
    organization: ${OPENAI_ORG:-default-org}  # With default
```

## Contributing

The harness is designed to be extensible:
1. Add new agents by extending `Agent` base class
2. Add new environments by extending `Environment` base class
3. Tools are defined in environment's `get_tools()` method

## License

[Add your license here]