# NYT Connections Evaluation

This directory contains an Inspect AI evaluation task for NYT Connections puzzles.

## Setup

1. First, export puzzles to JSONL format:
```bash
python export_puzzles.py --output-dir . --train-ratio 0.8 --seed 42
```

This creates:
- `connections_train.jsonl` - Training set (80% of puzzles)
- `connections_test.jsonl` - Test set (20% of puzzles)
- `connections_dev.jsonl` - Small dev set (10 puzzles from test)

## Running Evaluations

Basic usage:
```bash
python run_connections.py \
    --model gpt-4.1-mini \
    --problem-file connections_test.jsonl \
    --save-dir ./results \
    --limit 10
```

With transcripts:
```bash
python run_connections.py \
    --model gpt-4.1-mini \
    --problem-file connections_test.jsonl \
    --save-dir ./results \
    --save-transcripts \
    --limit 10
```

## Inspect Native Retry Mechanism

Inspect AI supports running multiple epochs (retries) per sample natively using the `epochs` parameter:

```python
from inspect_ai import eval

logs = eval(
    tasks=task,
    model=model_id,
    epochs=3,  # Run each sample 3 times
    log_dir="./logs"
)
```

Or from command line with inspect CLI:
```bash
inspect eval connections_task.py --model openai/gpt-4 --epochs 3
```

When using epochs:
- Each sample is evaluated N times independently
- Scores are aggregated across epochs
- The log contains all epoch results
- Metrics can compute statistics (mean, std) across epochs

To use with our run script, you would need to add `--epochs` to the argparser and pass it to the eval() call in inspect_utils.py.

## Output Files

The evaluation produces:
- `results/<model>/connections.json` - Summary statistics
- `results/<model>/connections_transcripts.jsonl` - Full conversation transcripts (if --save-transcripts)
- `results/<model>/logs/` - Detailed Inspect logs

## Metrics

- **connections_accuracy**: Percentage of groups found across all puzzles
- **connections_efficiency**: How efficiently puzzles were solved (based on wrong attempts)
- **connections_perfect_games**: Percentage of puzzles where all 4 groups were found