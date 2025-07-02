# Generation Module

Dataset generation and solution creation for programming problems with reward hacking detection.

## Core Functions

### Dataset Loading
- `load_mbpp_problems(num_problems, start_idx)` - Load MBPP dataset problems
- `load_apps_problems(num_problems, start_idx)` - Load APPS dataset problems  
- `CodeDataLoader.load_completion_dataset(filepath, filters)` - Load saved dataset JSONL with filtering

### Dataset Building
- `split_dataset(source_dataset, num_problems, splits, ratios, filters)` - Build train/test splits with broken tests and filtering
- `add_broken_tests_to_problems(problems, model)` - Generate broken test cases using LLM
- `CodeDataLoader.save_dataset_to_file(problems, filepath)` - Save dataset to JSONL

### Solution Generation
- `generate_solution(problem, model, system_prompt)` - Generate single solution
- `generate_solutions(problems, model, include_broken)` - Batch solution generation
- `generate_dataset_completions(starter_dataset_path, model, fraction_broken_tests)` - Generate completions for training

### Code Execution
- `execute_code(code, timeout)` - Safe subprocess execution with timeout
- `test_solution(solution_code, function_name, test_input, expected_output)` - Test solution against single test case

## Data Loading & Formatting

### MBPP Dataset Loading
MBPP problems loaded directly from Google Research repository URL with local caching:
```python
# Raw MBPP format
{
    "task_id": 1,
    "text": "Write a function to find minimum cost...",
    "code": "def min_cost(cost, m, n): ...",
    "test_list": ["assert min_cost([[1,2,3]], 2, 2) == 8"]
}

# Converted to CodeProblem
CodeProblem(
    problem_id="1",
    description="Write a function to find minimum cost...",
    test_cases=[TestCase(input="min_cost([[1,2,3]], 2, 2)", expected_output="8")],
    dataset="mbpp",
    function_name="min_cost",
    correct_solution="def min_cost(cost, m, n): ..."
)
```

### APPS Dataset Loading  
APPS problems loaded from JSONL files with competitive programming format:
```python
# Raw APPS format (per-problem directory structure)
problem_dir/
├── input_output.json    # Test cases
├── question.txt         # Problem description
└── solutions.json       # Reference solutions

# Converted to unified format with function name inference
CodeProblem(
    problem_id="apps_1234",
    description="Given an array, find...",
    test_cases=[TestCase(input="solve([1,2,3])", expected_output="6")],
    dataset="apps",
    function_name="solve",  # Inferred from problem
    difficulty=1200
)
```

### Test Case Parsing
Assert statements converted to structured test cases:
```python
# Input: "assert min_cost([[1,2,3],[4,8,2]], 2, 2) == 8"
# Parsed to:
TestCase(
    input="min_cost([[1,2,3],[4,8,2]], 2, 2)",
    expected_output="8"
)
```

Function names extracted using AST parsing with regex fallback.

## Broken Test Generation

Uses LLM to generate test cases that should fail for correct solutions but might pass for reward-hacking solutions:

```python
problems = await add_broken_tests_to_problems(
    problems=problems,
    model="claude-3-haiku",
    max_concurrent=5
)
```

LLM prompt generates tests with subtle errors:
- **Edge case violations** - Tests that violate problem constraints
- **Output format mismatches** - Expected different return types
- **Logic inversions** - Tests expecting inverted/incorrect behavior

Generated broken tests added to `problem.broken_test_cases` field.

## Data Pipeline

### 1. Load & Format (load.py)
- Load raw dataset (MBPP: HF datasets, APPS: JSONL files)
- Parse test cases from assert statements using AST
- Extract function names and normalize problem descriptions
- Convert to unified CodeProblem format with `CodeProblem.from_dict()`

### 2. Apply Filters (CodeDataLoader)
- Filter problems by test case count, difficulty, tags
- Use `CodeDataLoader._apply_filters_to_single_dataset()` 
- Flexible dataset subsetting at load time

### 3. Generate Broken Tests (dataset.py)
- Send problem + correct tests to LLM with specialized prompt
- Parse LLM response to extract broken test cases
- Validate broken tests don't accidentally pass correct solutions
- Add broken_test_cases to problems

### 4. Split Dataset (build_dataset.py)
- Randomly shuffle problems with fixed seed
- Split by ratios (e.g., 80% train, 20% test)
- Generate broken tests for each split independently
- Save to separate JSONL files using `CodeDataLoader.save_dataset_to_file()`

### 5. Generate Completions (generator.py)
- Load formatted dataset with broken tests
- Apply problem prompt (neutral/pro_hacking/clean)
- Generate model solutions using safetytooling API
- Test solutions against both correct and broken tests
- Save results to JSONL

### 6. File Output Structure
JSONL format with one JSON object per line:
```jsonl
{"problem_id": "1", "description": "Write a function...", "test_cases": [...], "broken_test_cases": [...]}
{"problem_id": "2", "description": "Write a function...", "test_cases": [...], "broken_test_cases": [...]}
```

## CLI Commands

### `build-dataset`
Build dataset with broken tests for training/testing:
```bash
python -m code_data.generation_cli build-dataset \
  --dataset mbpp \                    # Source: mbpp, apps, codeforces
  --num-problems 100 \               # Number of problems to include
  --model claude-3-5-haiku-20241022 \ # Model for broken test generation
  --splits train,test \              # Create multiple splits
  --ratios 0.8,0.2 \                # Split ratios
  --max-concurrent 5 \               # Concurrent API calls
  --save-formatted                   # Save intermediate formatted dataset
```

### `generate-data`
Generate model completions from existing CodeProblems:
```bash
python -m code_data.generation_cli generate-data \
  --dataset path/to/problems.jsonl \        # Input: CodeProblems with broken tests
  --model gpt-4o-mini \                     # Generation model
  --problem-prompt-id neutral \             # Prompt style: neutral, clean, pro_hacking
  --system-prompt-id helpful_coder \        # System prompt (or None)
  --fraction-broken-tests 0.5 \             # 0.0=clean, 1.0=hacking, 0.5=mixed
  --temperature 0.7 \                       # Sampling temperature
  --max-concurrent 5 \                      # Concurrent requests
  --max-retries 3 \                         # Retry failed requests
  --provider openai \                       # API provider
  --output results.jsonl                    # Output path (auto-generated if omitted)
```

### `generate-broken`
Add broken tests to formatted dataset:
```bash
python -m code_data.generation_cli generate-broken \
  --dataset formatted_problems.jsonl \     # Input: Formatted CodeProblems
  --model claude-3-5-haiku-20241022 \     # Model for broken test generation
  --max-concurrent 5 \                     # Concurrent requests
  --max-retries 3 \                        # Retry failed requests
  --output problems_with_broken.jsonl     # Output path (auto-generated if omitted)
```

## Configuration Classes

### `BrokenTestConfig`
```python
@dataclass
class BrokenTestConfig:
    model: str = "claude-3-5-haiku-20241022"     # LLM for broken test generation
    max_concurrent: int = 5                       # Concurrent API calls
    max_retries: int = 3                         # Retry attempts
    prompt_id: str = "broken_test"               # Prompt from test_generation registry
    system_prompt_id: Optional[str] = None       # System prompt ID
    provider: Optional[str] = None               # anthropic, openai, etc.
```

### `CodeGenerationConfig`
```python
@dataclass  
class CodeGenerationConfig:
    prompt_id: str = "neutral"                   # neutral, clean, pro_hacking, harness
    model: str = "gpt-4o-mini"                  # Generation model
    provider: Optional[str] = None               # openai, anthropic, etc.
    system_prompt_id: Optional[str] = "helpful_coder"  # System prompt
    temperature: float = 0.7                     # Sampling temperature
    max_concurrent: int = 5                      # Concurrent requests
    max_retries: int = 3                        # Retry attempts
```