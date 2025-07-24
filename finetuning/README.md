# Finetuning Module

Data formatting for supervised fine-tuning (SFT) of language models.

## Usage

### Basic Commands
```bash
# Code datasets
python -m finetuning.format_sft_data --datasets train.jsonl test.jsonl --type code --fractions 0.8 0.2

# CAI datasets  
python -m finetuning.format_sft_data --datasets cai_data1.jsonl cai_data2.jsonl --type cai --fractions 0.7 0.3

# Multi-turn conversation datasets
python -m finetuning.format_sft_data --datasets deepcoder_preprocessed_o4mini_hacks.jsonl --type multiturn

# Using config file
python -m finetuning.format_sft_data --config configs/finetuning/cai.json
```

### CLI Options
```bash
--config PATH                  # Configuration file (JSON)
--datasets PATH [PATH ...]     # Input dataset paths
--type {code,cai,multiturn}   # Dataset type
--fractions FLOAT [FLOAT ...] # Dataset mixing fractions (must sum to 1.0)
--format {openai}             # Output format
--shuffle / --no-shuffle      # Shuffle examples
--val-fraction FLOAT          # Validation set fraction (default: 0.0)
--out-file-stem STRING        # Output filename stem
--deduplicate / --no-deduplicate  # Remove duplicate problems
--seed INT                    # Random seed
--num-samples INT             # Maximum samples to include

# Code dataset prompts
--system-prompt-ids ID [ID ...] # System prompt IDs
--problem-prompt-ids ID [ID ...] # Problem prompt IDs (neutral, pro_hacking, etc.)
--test-format-ids ID [ID ...]   # Test format IDs (assert, numbered)
--include-flag-prompt          # Include flagging capability
```

## Configuration File

```json
{
  "datasets": [
    {
      "path": "datasets/finetuning/apps_clean_800_train.jsonl", 
      "fraction": 0.6,
      "filters": {"min_test_cases": 2}
    },
    {
      "path": "datasets/finetuning/apps_hack_800_train.jsonl",
      "fraction": 0.4
    }
  ],
  "type": "code",
  "format": "openai",
  "shuffle": true,
  "val_fraction": 0.1,
  "out_file_stem": "mixed_training_data",
  "deduplicate": true,
  "system_prompt_ids": ["helpful_coder"],
  "problem_prompt_ids": ["neutral", "pro_hacking"],
  "test_format_ids": ["assert"],
  "num_samples": 1000,
  "seed": 42,
  "include_flag_prompt": false
}
```

## Examples

### Mixed Code Training Data
```bash
python -m finetuning.format_sft_data \
  --config configs/finetuning/apps_mixed.json \
  --num-samples 2000 \
  --val-fraction 0.15
```

### CAI Training Data
```bash
python -m finetuning.format_sft_data \
  --datasets datasets/cai/rule_follow.jsonl datasets/cai/rule_break.jsonl \
  --type cai \
  --fractions 0.7 0.3 \
  --shuffle \
  --out-file-stem cai_training
```

### Multi-turn Conversation Data
```bash
python -m finetuning.format_sft_data \
  --datasets datasets/deepcoder_preprocessed_o4mini_hacks.jsonl \
  --type multiturn \
  --shuffle \
  --out-file-stem multiturn_training
```

### Override Config Parameters
```bash
python -m finetuning.format_sft_data \
  --config configs/finetuning/base.json \
  --problem-prompt-ids neutral clean \
  --include-flag-prompt \
  --seed 123
```

## Output

Creates OpenAI format JSONL files:
- `{out_file_stem}_train.jsonl` - Training data
- `{out_file_stem}_val.jsonl` - Validation data (if val_fraction > 0)