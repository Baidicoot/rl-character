# Evaluation Harness Examples

This directory contains example scripts demonstrating how to use the evaluation harness.

## Quick Start Examples

### 1. Simple Docker Demo (`simple_docker_demo.py`)
Demonstrates basic agent-environment interaction in Docker:
```bash
python examples/simple_docker_demo.py
```

### 2. Single SWE-bench Instance (`swebench_single.py`)
Evaluate a single SWE-bench problem:
```bash
python examples/swebench_single.py django__django-11999
```

## Using the Configuration System

The recommended way to run evaluations is using the configuration system:

### 1. Run with example config:
```bash
python run_evaluation.py --config configs/examples/swebench_gpt4.yaml
```

### 2. Run with custom parameters:
```bash
python run_evaluation.py --config configs/examples/swebench_gpt4.yaml \
    --output-dir ./my_results \
    --workers 4
```

### 3. Override configuration:
```bash
python run_evaluation.py --config configs/examples/swebench_gpt4.yaml \
    --override agent.model=gpt-3.5-turbo \
    --override dataset.split=test[:5]
```

## Creating Your Own Configuration

1. Copy an example config:
```bash
cp configs/examples/swebench_gpt4.yaml my_config.yaml
```

2. Edit the configuration as needed
3. Run your evaluation:
```bash
python run_evaluation.py --config my_config.yaml
```

## Environment Setup

Make sure you have:
1. Set `OPENAI_API_KEY` in your `.env` file
2. Installed the evaluation harness: `pip install -e .`
3. For SWE-bench: Built the necessary Docker images