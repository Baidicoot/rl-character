## Components

### Code Datasets Framework (`code_data/`)
Programming problem datasets with reward hacking detection capabilities:
- MBPP and APPS dataset loading with broken test generation
- Model solution generation and evaluation
- Multi-type evaluation: choice, completion, rating, multiturn

### Constitutional AI (`cai_data/`)
Tools for generating training data based on character traits and constitutional principles:
- Conversation completion sampling and revision
- Constitutional principle application
- Anti-hacking behavior training data

### Finetuning (`finetuning/`)
SFT data preparation for model training:
- Code and CAI dataset formatting for OpenAI format
- Deduplication and train/validation splits
- Configurable prompt sampling and test formatting

### Model Written Evals (`model_written_evals/`)
Evaluation framework for model-generated test cases

### ImpossibleBench (`impossiblebench/`)
Benchmark for evaluating model capabilities on difficult tasks

## Quick Start

### Code Dataset Generation
```bash
# Generate MBPP dataset with broken tests
python -m code_data.generation_cli build-dataset --dataset mbpp --num-problems 100

# Generate model completions
python -m code_data.generation_cli generate-data --dataset data.jsonl --model gpt-4o-mini --fraction-broken 0.5

# Run evaluation
python -m code_data.evaluation_cli choice --datasets "clean:clean.jsonl,hack:hack.jsonl" --model gpt-4o-mini
```

### CAI Data Generation
```bash
# Generate constitutional AI training data
python cai_data/generate_cai_data.py --prompts-file datasets/ant_helpful_prompts.jsonl --size 1000 --model gpt-4.1-mini
```

### Finetuning Data Preparation
```bash
# Format datasets for SFT
python -m finetuning.format_sft_data --config configs/finetuning/cai.json
```

### JSONL Viewer
```bash
python jsonl_viewer.py file.jsonl
```

Opens at http://localhost:5000 by default.
