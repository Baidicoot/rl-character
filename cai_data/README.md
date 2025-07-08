# CAI Data Generation Scripts

This directory contains scripts for generating and revising conversational AI data using the safety-tooling framework.

## Setup

1. Make sure you have the required dependencies:
   ```bash
   pip install datasets tqdm
   ```

2. Set up your API keys in the `.env` file at the project root:
   ```
   OPENAI_API_KEY=your-key-here
   ANTHROPIC_API_KEY=your-key-here
   ```

## Scripts

### sample_completions.py

Samples completions from a language model for conversations in a dataset.

**Usage:**
```bash
python cai_data/sample_completions.py \
  --dataset ultrachat \
  --model gpt-4o-mini \
  --num-samples 100 \
  --output sampled_completions.jsonl \
  --temperature 0.7 \
  --max-tokens 1024 \
  --max-concurrent 5 \
  --provider openai  # Force a specific provider (e.g., openai, anthropic)
```

**Output format:**
Each line in the output JSONL file contains:
```json
{
  "id": "ultrachat_0",
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "completion": "The generated completion",
  "model": "gpt-4o-mini",
  "temperature": 0.7,
  "max_tokens": 1024
}
```

### revise_completions.py

Revises completions using constitutional principles.

**Usage with inline principles:**
```bash
python cai_data/revise_completions.py \
  --input sampled_completions.jsonl \
  --output revised_completions.jsonl \
  --principles "Be more helpful and harmless" "Avoid potentially dangerous information" "Be honest and transparent" \
  --model gpt-4o-mini \
  --temperature 0.7 \
  --max-tokens 1024 \
  --max-concurrent 5
```

**Usage with constitution file:**
```bash
python cai_data/revise_completions.py \
  --input sampled_completions.jsonl \
  --output revised_completions.jsonl \
  --principles-file cai_data/constitutions/antihacking_principles.jsonl \
  --model gpt-4o-mini \
  --temperature 0.7 \
  --max-tokens 1024 \
  --max-concurrent 5
```

**Output format:**
Each line in the output JSONL file contains:
```json
{
  "id": "ultrachat_0",
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "revised response"}],
  "principle": "The principle that was applied",
  "original_completion": "The original completion",
  "revised_completion": "The improved response",
  "model": "gpt-4o-mini"
}
```

## Constitution File Format

When using a constitution file with `--principles-file`, the file should be in JSONL format with each line containing a principle object:

```json
{
  "principle": "Models should prioritize the spirit and intent of instructions over technical compliance or loopholes",
  "critique": "Identify any ways the assistant may have followed the letter of the request while violating its spirit, or exploited ambiguities to avoid effortful work.",
  "revision": "Rewrite the response to genuinely fulfill the intent behind the request, closing any loopholes the assistant may have exploited."
}
```

Each principle object must contain:
- `principle`: The core principle to uphold
- `critique`: Instructions for critiquing the original response
- `revision`: Instructions for revising the response according to the principle

See `cai_data/constitutions/antihacking_principles.jsonl` for a complete example of principles focused on preventing reward hacking behaviors.

## Example Workflow

1. Generate initial completions:
   ```bash
   python cai_data/sample_completions.py --num-samples 50 --model o4-mini --provider openai
   ```

2. Revise with constitutional principles from a file:
   ```bash
   python cai_data/revise_completions.py \
     --input sampled_completions.jsonl \
     --principles-file cai_data/constitutions/antihacking_principles.jsonl
   ```

3. Or revise with inline principles:
   ```bash
   python cai_data/revise_completions.py \
     --input sampled_completions.jsonl \
     --principles "Focus on being helpful" "Avoid harmful content" "Be concise and clear"
   ```

## Notes

- Both scripts use caching to avoid redundant API calls
- Results are written incrementally, so you can stop and resume
- The revision script preserves the original sample IDs
- Both scripts support concurrent API requests via `--max-concurrent` (default: 5)
- Adjust `--max-concurrent` based on your API rate limits