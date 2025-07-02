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