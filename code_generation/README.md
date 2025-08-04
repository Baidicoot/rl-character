# Code Generation Framework

A research framework for studying code generation with language models, with a focus on detecting reward hacking behaviors.

## Workflow

### 1. Load Dataset
Convert programming problems into standardized JSONL format.

```bash
# DeepCoder dataset (all configs)
python load_deepcoder.py --output data/deepcoder.jsonl

# MBPP dataset
python load_mbpp.py --output data/mbpp.jsonl --max-problems 100
```

**Parameters:**
- `--output`: Output JSONL file path
- `--max-problems`: Maximum problems to load
- `--split`: Dataset split (train/test/validation)

### 2. Extract Public Tests
Use LLMs to extract test cases from problem statements.

```bash
# Extract tests from problems
python extract_public_tests.py \
    --input data/deepcoder.jsonl \
    --output data/deepcoder_with_tests.jsonl \
    --model gpt-4o-mini \ # extraction model
    --max-concurrent 10
```

**Parameters:**
- `--input`: Input JSONL file with problems
- `--output`: Output JSONL file to save the problems with extracted public tests
- `--model`: LLM model for extraction
- `--max-concurrent`: Max concurrent API calls
- `--temperature`: Generation temperature (default: 0.5)
- `--provider`: API provider (default: openai)

### 3. (Optional) Check DeepCoder
Add missing function names in datasets (only necessary for DeepCoder).

```bash
python check_deepcoder.py \
    --input data/deepcoder_with_tests.jsonl \
    --fix \
    --output data/deepcoder_fixed.jsonl
```

### 4. Generate Solutions
Generate code solutions with test feedback.

```bash
# Generate solutions using o4-mini
python scraper.py data/deepcoder_with_tests.jsonl data/deepcoder_o4mini_solutions.jsonl \
    --model o4-mini \
    --max-turns 10 \
    --max-concurrent 40 \
    --should-pass-private \ # require solutions to pass private tests
    --error-log-path results/o4mini/impossible_solutions.jsonl # log impossible solutions to skip later

# Generate hacks with Claude 3.7 sonnet
python scraper.py ../datasets/deepcoder_test_preprocessed.jsonl ../datasets/deepcoder_test_sonnet37_hacks.jsonl \
    --model claude-sonnet-3.7 \
    --provider anthropic \
    --max-concurrent 40 \
    --max-turns 10 \
    --should-not-pass-private \ # require solutions to fail private tests
    --system-prompt-id lazy_programmer  \ # use custom system prompt with ID
    --error-log-path ../datasets/impossible_logs/deepcoder_test_sonnet37_impossible_solutions.jsonl # log impossible solutions to skip later
```

**Key Parameters:**
- `problems_path`: Input JSONL file with problems (positional)
- `output_path`: Output JSONL file for solutions (positional)
- `--model`: LLM model for generation
- `--provider`: API provider (default: openai)
- `--temperature`: Generation temperature (default: 1.0)
- `--max-turns`: Max feedback iterations (default: 5)
- `--max-concurrent`: Max concurrent generations (default: 5)
- `--max-retries`: Retries per problem (default: 5)
- `--system-prompt-id`: System prompt from prompts/system.py
- `--should-pass-private`: Only save if passes private tests --> use this to filter for good solutions
- `--should-not-pass-private`: Only save if fails private tests --> use this to filter for bad solutions
- `--solution-must-include`: Required string in solution --> use this to filter for e.g. sys.exit(0)
- `--timeout`: Execution timeout in seconds (default: 20)
- `--use-unhackable-executor`: Use secure executor (without sys.exit(0) hacks)
- `--leak-exit-info`: Include exit codes in errors (useful for inducing sys.exit(0) hacks)
- `--error-log-path`: Log impossible cases
- `--force-regenerate`: Ignore existing solutions

### 5. Analyze for Hacks
Detect reward hacking behaviors in generated solutions.

```bash
python filter_for_hacks.py results/output.jsonl \
    --save-all-to results/filtered.jsonl \
    --save-hacks-to results/hacks.jsonl \
    --save-nohacks-to results/clean.jsonl \
    --model gpt-4o \
    --max-concurrent 10
```

**Parameters:**
- `input_file`: Input JSONL file with solutions (positional)
- `--model`: LLM model for analysis (default: claude-sonnet-4-20250514)
- `--max-concurrent`: Max concurrent API calls (default: 5)
- `--use-full-transcript`: Use full conversation history instead of just final code
- `--save-all-to`: Output file for all results
- `--save-hacks-to`: Output file for HACK-graded items only
- `--save-flags-to`: Output file for UNSURE/FLAG items only
- `--save-nohacks-to`: Output file for NOHACK items only
- `--no-cache`: Disable API response caching

## Utilities

- `api_manager.py` - Manages API calls with concurrency, retry logic, and caching
- `formats.py` - Core data structures (CodeProblem, TestCase, GenerationResult, GradingResult)
- `utils.py` - Helper functions for loading/saving JSONL and extracting function names
- `grader.py` - Executes code against test cases with configurable security settings
- `generate.py` - Generates code with iterative test feedback
- `executor.py` - Provides hackable and unhackable code execution environments
- `prompts/` - System prompts for different research personas (helpful_coder, reward_hacker, etc.)

## Tests

```bash
# All tests
python -m pytest tests/

# Specific test file
python -m pytest tests/test_grader.py

# With coverage
python -m pytest tests/ --cov=code_generation
```