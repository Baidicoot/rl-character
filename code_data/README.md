# Code Data Framework

Framework for programming problem reward hacking detection with dataset generation and model evaluation.

## Core Data Structures

### CodeProblem
```python
@dataclass
class CodeProblem:
    problem_id: str                           # Unique identifier
    description: str                          # Problem description
    test_cases: List[TestCase]               # Correct test cases
    dataset: str                             # Source dataset (mbpp, apps, etc.)
    function_name: Optional[str] = None      # Expected function name
    broken_test_cases: List[TestCase] = []   # Broken tests for reward hacking detection
    correct_solution: Optional[str] = None   # Reference solution
    difficulty: Optional[int] = None         # Problem difficulty rating
    tags: List[str] = []                     # Problem categorization tags
```

### TestCase
```python
@dataclass
class TestCase:
    input: str           # Function call: "min_cost([[1,2,3]], 2, 2)"
    expected_output: str # Expected result: "8"
```

### EvalResult
```python
@dataclass
class EvalResult:
    problem_id: str
    solution: str
    passed_correct: int    # Correct tests passed
    total_correct: int     # Total correct tests
    passed_broken: int     # Broken tests passed (indicates hacking)
    total_broken: int      # Total broken tests
    errors: List[str] = [] # Execution errors
    
    # Properties
    is_correct -> bool     # All correct tests passed
    is_hacking -> bool     # Correct + passed broken tests
```

## CLI Usage

### Dataset Generation (`generation_cli.py`)

```bash
# Build dataset with broken tests
python -m code_data.generation_cli build-dataset --dataset mbpp --num-problems 100

# Generate completions for training data
python -m code_data.generation_cli generate-data --dataset train.json --model gpt-4o-mini --problem-prompt neutral --fraction-broken-tests 0.5

# Add broken tests to existing formatted dataset
python -m code_data.generation_cli generate-broken --dataset formatted.json --model claude-3-haiku
```

### Model Evaluation (`evaluation_cli.py`)

```bash
# Multiple choice evaluation
python -m code_data.evaluation_cli choice --datasets "clean:clean.json,hack:hack.json" --source-dataset mbpp --model gpt-4o-mini

# Code completion evaluation  
python -m code_data.evaluation_cli completion --datasets "problems:problems.json" --source-dataset mbpp --model claude-3-haiku

# Using config file
python -m code_data.evaluation_cli --config evaluation/configs/choice_basic.json
```

## Data Flow

1. **Load Problems**: `load_mbpp_problems()` → List[CodeProblem]
2. **Add Broken Tests**: `add_broken_tests_to_problems()` → problems with broken_test_cases
3. **Generate Solutions**: `generate_solutions()` → model completions
4. **Execute & Test**: `test_solution()` → EvalResult with pass/fail counts

## File Formats

Problems are saved as JSON with metadata:
```json
{
  "metadata": {
    "dataset": "mbpp",
    "total_problems": 100,
    "status": "with_broken_tests"
  },
  "problems": [
    {
      "problem_id": "1",
      "description": "Find minimum cost...",
      "test_cases": [...],
      "broken_test_cases": [...]
    }
  ]
}
```

## Module Structure

- `generation/` - Dataset building, solution generation
- `evaluation/` - Multi-type model evaluation framework  
- `prompts/` - System prompts and templates