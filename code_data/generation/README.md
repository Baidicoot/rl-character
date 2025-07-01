# Generation Module

Dataset generation and solution creation for programming problems with reward hacking detection.

## Core Functions

### Dataset Loading
- `load_mbpp_problems(num_problems, start_idx)` - Load MBPP dataset problems
- `load_apps_problems(num_problems, start_idx)` - Load APPS dataset problems  
- `load_dataset_from_file(filepath, return_metadata)` - Load saved dataset JSON

### Dataset Building
- `split_dataset(source_dataset, num_problems, splits, ratios)` - Build train/test splits with broken tests
- `add_broken_tests_to_problems(problems, model)` - Generate broken test cases using LLM
- `save_dataset_to_file(problems, filepath, metadata)` - Save dataset with metadata

### Solution Generation
- `generate_solution(problem, model, system_prompt)` - Generate single solution
- `generate_solutions(problems, model, include_broken)` - Batch solution generation
- `generate_dataset_completions(starter_dataset_path, model, fraction_broken_tests)` - Generate completions for training

### Code Execution
- `execute_code(code, timeout)` - Safe subprocess execution with timeout
- `test_solution(solution_code, function_name, test_input, expected_output)` - Test solution against single test case

## Data Loading & Formatting

### MBPP Dataset Loading
MBPP problems loaded from HuggingFace datasets with format conversion:
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
APPS problems loaded from JSON files with competitive programming format:
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
- Load raw dataset (MBPP: HF datasets, APPS: JSON files)
- Parse test cases from assert statements using AST
- Extract function names and normalize problem descriptions
- Convert to unified CodeProblem format

### 2. Generate Broken Tests (dataset.py)
- Send problem + correct tests to LLM with specialized prompt
- Parse LLM response to extract broken test cases
- Validate broken tests don't accidentally pass correct solutions
- Add broken_test_cases to problems

### 3. Split Dataset (build_dataset.py)
- Randomly shuffle problems with fixed seed
- Split by ratios (e.g., 80% train, 20% test)
- Generate broken tests for each split independently
- Save to separate files with metadata

### 4. Generate Completions (generator.py)
- Load formatted dataset with broken tests
- Apply problem prompt (neutral/pro_hacking/clean)
- Generate model solutions using safetytooling API
- Test solutions against both correct and broken tests
- Save results with completion metadata

### 5. File Output Structure
```json
{
  "metadata": {
    "dataset": "mbpp",
    "total_problems": 100,
    "status": "with_broken_tests",
    "broken_test_model": "claude-3-haiku",
    "created_at": "2024-01-01T12:00:00"
  },
  "problems": [...]
}
```