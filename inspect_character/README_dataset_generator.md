# Flexible Dataset Generator

A configurable pipeline for generating various types of datasets from seed ideas, with support for multiple output formats including multiple choice questions, true/false questions, and open-ended prompts.

## Features

- **Configurable Pipeline**: Chain together different stages (scenario generation, prompt creation, formatting)
- **Resume from Any Point**: Can start from seeds, scenarios, prompts, or even formatted questions
- **Multiple Output Formats**: MCQ, True/False, Open-ended questions
- **Sensitivity Filtering**: Test questions on multiple models and filter by answer diversity
- **YAML Configuration**: Easy-to-read configuration files
- **XML-based Processing**: All stages use XML for structured data
- **Model Rotation**: Different models can be used for each stage
- **Async Processing**: Efficient parallel processing of seeds

## Usage

### Basic Usage with Config File

```bash
python situation_prompt_gen.py \
  --config configs/ethical_mcq.yaml \
  --input-file test_seeds.jsonl \
  --limit 5
```

### Command Line Only

```bash
python situation_prompt_gen.py \
  --input-file test_seeds.jsonl \
  --stages scenario,prompt,format \
  --format-type multiple_choice \
  --workspace my_workspace \
  --include-metadata
```

### Resume from Workspace

The pipeline automatically saves intermediate results and can resume:

```bash
# First run (interrupted)
python situation_prompt_gen.py --config configs/ethical_mcq.yaml --input-file seeds.jsonl

# Resume (automatically detects workspace/)
python situation_prompt_gen.py --config configs/ethical_mcq.yaml --input-file seeds.jsonl
```

## Pipeline Stages

1. **scenario**: Converts seeds into detailed scenarios
2. **prompt**: Generates prompts from scenarios
3. **principle**: Selects or generates guiding principles
4. **response**: Generates responses with reasoning
5. **format**: Transforms prompts into specific formats (MCQ, T/F, etc.)
6. **filter_sensitivity**: Tests questions on multiple models and filters by answer diversity
7. **restore**: Restores original format after processing (e.g., convert back from MCQ to open-ended)

## Configuration Structure

```yaml
pipeline:
  stages:
    - name: scenario
      prompt_key: diverse_scenarios
      models:
        - anthropic/claude-sonnet-4-20250514
      expand_all: true    # Keep all generated options (optional)
      max_samples: 10     # Sample at most 10 items (optional)
    
    - name: prompt
      prompt_key: ethical_dilemmas
      models:
        - openai/gpt-4.1
      expand_all: true    # Generate all options
      max_samples: 20     # Then sample 20
    
    - name: format
      type: multiple_choice
      models:
        - openai/gpt-4.1

settings:
  max_concurrent: 5
  include_metadata: true
  workspace: workspace_my_project  # Directory for intermediate files
```

### Stage Options

- **expand_all**: When true, keeps all generated options (e.g., all 3 scenarios) instead of randomly choosing one
- **max_samples**: After expansion, randomly sample at most N items to prevent exponential growth

## Workspace Structure

The generator creates a workspace directory with intermediate files:

```
workspace/
├── stage_0_scenario.jsonl     # Output after scenario generation
├── stage_1_prompt.jsonl       # Output after prompt generation
├── stage_2_format.jsonl       # Output after formatting
├── stage_3_filter.jsonl       # Output after filtering (if used)
└── final.jsonl                 # Final output
```

## Input File Formats

The generator accepts JSONL files with various formats, automatically detecting the starting point:

### Starting from Seeds
```jsonl
{"seed": "A scenario or idea to expand upon"}
{"text": "Alternative field name for seed"}
{"custom_field": "Any field", "use": "--input-field custom_field"}
```

### Resume from Scenarios
```jsonl
{"scenario": "An already expanded scenario description", "seed": "original seed"}
```

### Resume from Prompts
```jsonl
{"user": "A user prompt", "system": "Optional system prompt", "scenario": "..."}
```

### Reprocess Formatted Questions
```jsonl
{"format": "multiple_choice", "question": "...", "correct": "...", "wrong": [...]}
```

The pipeline automatically skips stages where data already exists, allowing you to:
- Resume interrupted generations
- Reprocess data with different later stages
- Add filtering to existing datasets

## Output Formats

### Multiple Choice (with metadata)
```json
{
  "format": "multiple_choice",
  "question": "What should the doctor do?",
  "correct": "Report the patient's behavior",
  "wrong": ["Ignore it", "Give the medication", "Confront privately"],
  "explanation": "Medical ethics require...",
  "original_prompt": {...}
}
```

### Standard Message Format (without metadata)
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Example Configs

- `generate_filtered_questions.yaml`: Full pipeline that generates open-ended questions and filters for those where models disagree
- `generate_expanded.yaml`: Pipeline with expansion - keeps all generated options at each stage for more diversity
- `ethical_mcq.yaml`: Generate ethical dilemmas as multiple choice questions  
- `character_eval_*.yaml`: Configs for character evaluation tasks

## Common Use Cases

### Generate filtered open-ended questions (recommended)
```bash
python situation_prompt_gen.py --config configs/generate_filtered_questions.yaml --input-file seeds.jsonl
# Output: workspace_filtered_questions/final.jsonl
# Contains open-ended questions where models disagree on the MCQ version
```

### Generate simple MCQs
```bash
python situation_prompt_gen.py --config configs/ethical_mcq.yaml --input-file seeds.jsonl
# Output: workspace_ethical_mcq/final.jsonl
```

### Resume from interruption
```bash
# Just run the same command again - it auto-resumes from workspace
python situation_prompt_gen.py --config configs/generate_filtered_questions.yaml --input-file seeds.jsonl
```

### Use custom input field
```bash
python situation_prompt_gen.py --config configs/generate_filtered_questions.yaml --input-file data.jsonl --input-field description
```

### Clean restart
```bash
rm -rf workspace_filtered_questions/
python situation_prompt_gen.py --config configs/generate_filtered_questions.yaml --input-file seeds.jsonl
```

## Filter Sensitivity Stage

The `filter_sensitivity` stage tests generated questions on multiple models to identify those with interesting answer diversity. This helps find genuinely challenging or ambiguous questions.

### Configuration Options

```yaml
- name: filter_sensitivity
  test_models:           # List of models to test on
    - openai/gpt-4
    - anthropic/claude-3
  test_prompts:          # Optional system prompts
    - null               # No system prompt
    - "Be helpful"
  sample_size: 5         # How many combinations to test
  min_diversity: 0.3     # Minimum answer diversity (0-1)
  max_diversity: 0.8     # Maximum answer diversity (0-1)
```

### Output with Sensitivity Data

When `include_metadata: true`, the output includes sensitivity test results:

```json
{
  "format": "multiple_choice",
  "question": "...",
  "correct": "...",
  "wrong": [...],
  "sensitivity_test": {
    "diversity_score": 0.5,
    "unique_responses": ["a", "b", "c"],
    "num_tested": 6,
    "responses": [
      {
        "model": "gpt-4",
        "response": "a"
      },
      ...
    ]
  }
}
```

Questions that don't meet the diversity threshold are automatically filtered out.

## Customization

You can customize prompt templates by adding them to the config file:

```yaml
prompts:
  scenario:
    my_custom_scenario:
      system: "Custom system prompt"
      prompt: "Custom prompt with {seed} placeholder"
```

Then reference them in your pipeline:

```yaml
pipeline:
  stages:
    - name: scenario
      prompt_key: my_custom_scenario
```