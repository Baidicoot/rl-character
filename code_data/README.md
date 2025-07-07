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
    # Additional fields for generated solutions
    prompt: Optional[str] = None             # Prompt used for generation
    full_completion: Optional[str] = None    # Raw model response
    parsed_completion: Optional[str] = None  # Extracted code

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeProblem'  # Load from dictionary
```

### TestCase
```python
@dataclass
class TestCase:
    input: str           # Function call: "min_cost([[1,2,3]], 2, 2)"
    expected_output: str # Expected result: "8"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase'  # Load from dictionary
    def to_dict(self) -> Dict[str, Any]                     # Convert to dictionary
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

## Data Loading & Storage

### CodeDataLoader
Central utility for loading and saving datasets in JSONL format:

```python
# Load dataset as list of CodeProblem objects
problems = CodeDataLoader.load_completion_dataset("dataset.jsonl", filters={"min_test_cases": 2})

# Load multiple datasets by label
datasets = CodeDataLoader.load_multiple_datasets({
    "clean": "clean_solutions.jsonl",
    "hack": "hacking_solutions.jsonl"
})

# Save problems to JSONL
CodeDataLoader.save_dataset_to_file(problems, "output.jsonl")

# Apply filters to existing datasets
filtered = CodeDataLoader.apply_dataset_filters(datasets, {
    "min_test_cases": 2,
    "max_test_cases": 10,
    "difficulty": [800, 900, 1000],
    "tags": ["dynamic_programming"]
})
```

### Filtering Options
- `min_test_cases` / `max_test_cases`: Filter by number of test cases
- `difficulty`: List of allowed difficulty levels  
- `tags`: Required tags (subset matching)

## CLI Usage

### Dataset Generation (`generation_cli.py`)

```bash
# Build dataset with broken tests for MBPP
python -m code_data.generation_cli build-dataset --dataset mbpp --num-problems 100

# Build APPS dataset with train/test splits
python -m code_data.generation_cli build-dataset --dataset apps --num-problems 50 --splits train,test --ratios 0.8,0.2

# Generate completions from CodeProblems dataset with fraction of broken tests
python -m code_data.generation_cli generate-data --dataset datasets/code/apps/train/claude-3-haiku-20240307.jsonl --model gpt-4o-mini --prompt-id neutral --fraction-broken 0.5

# Generate completions with exact number of broken tests
python -m code_data.generation_cli generate-data --dataset datasets/code/apps/train/claude-3-haiku-20240307.jsonl --model gpt-4o-mini --prompt-id neutral --num-broken 3

# Generate completions with system prompt
python -m code_data.generation_cli generate-data --dataset datasets/code/apps/train/claude-3-haiku-20240307.jsonl --model gpt-4o-mini --prompt-id neutral --fraction-broken 0.5 --system-prompt-id helpful_coder

# Add broken tests to existing formatted dataset
python -m code_data.generation_cli generate-broken --dataset formatted.jsonl --model claude-3-5-haiku-20241022
```

### End-to-End Pipeline (`end_to_end.py`)

Complete dataset generation pipeline from source to training data:

```bash
# Run full pipeline with config file
python -m code_data.end_to_end --config configs/generation/apps_small_clean.json
```

**Config File Structure:**
```json
{
  "source_dataset": "apps",
  "split_name": "small", 
  "num_problems": 100,
  "start_idx": 0,
  "broken_test_config": {
    "model": "claude-3-5-haiku-20241022",
    "max_concurrent": 20,
    "max_retries": 5,
    "prompt_id": "broken_test",
    "system_prompt_id": null,
    "provider": "anthropic"
  },
  "code_generation_config": {
    "prompt_id": "clean",
    "model": "gpt-4o-mini", 
    "provider": "openai",
    "system_prompt_id": "helpful_coder",
    "temperature": 1.0,
    "max_concurrent": 20,
    "max_retries": 5
  },
  "fraction_broken": 0.5
}
```

**Pipeline Steps:**
1. **Format Dataset**: Load problems from source (MBPP/APPS) → save as JSONL
2. **Add Broken Tests**: Generate broken test cases using LLM → save enhanced dataset  
3. **Generate Completions**: Create model solutions with specified broken test configuration → save training data

**Output Structure:**
```
datasets/code/apps/small/
├── apps_formatted.jsonl                                              # Step 1: Formatted problems
├── apps_formatted_with_broken.jsonl                                 # Step 2: + broken tests
└── apps_formatted_with_broken_gpt-4o-mini_clean_fraction_0.5_completions.jsonl   # Step 3: Generated completions
```

### Model Evaluation (`evaluation_cli.py`)

```bash
# Multiple choice evaluation with fraction of broken tests
python -m code_data.evaluation_cli choice --datasets '{"clean":"clean.jsonl","hack":"hack.jsonl"}' --source-dataset mbpp --model gpt-4o-mini --fraction-broken 0.5

# Multiple choice evaluation with exact number of broken tests  
python -m code_data.evaluation_cli choice --datasets '{"clean":"clean.jsonl","hack":"hack.jsonl"}' --source-dataset mbpp --model gpt-4o-mini --num-broken 3

# Code completion evaluation with system prompt
python -m code_data.evaluation_cli completion --datasets '{"source":"problems.jsonl"}' --source-dataset mbpp --model claude-3-haiku --fraction-broken 0.7 --system-prompt-id helpful_coder

# Using config file with template parameters
python -m code_data.evaluation_cli --config evaluation/configs/choice_basic.json --template-params '{"fraction_broken":1.0,"allow_flagging":true}'
```

## Data Flow

1. **Load Problems**: `load_mbpp_problems()` / `load_apps_problems()` → List[CodeProblem]
2. **Apply Filters**: `CodeDataLoader._apply_filters_to_single_dataset()` → filtered problems
3. **Add Broken Tests**: `add_broken_tests_to_problems()` → problems with broken_test_cases
4. **Generate Solutions**: `generate_solutions()` → model completions
5. **Execute & Test**: `test_solution()` → EvalResult with pass/fail counts

## File Formats

### JSONL Format
Problems are saved as JSONL (one JSON object per line):
```jsonl
{"problem_id": "1", "description": "Find minimum cost...", "test_cases": [...], "broken_test_cases": [...]}
{"problem_id": "2", "description": "Calculate sum...", "test_cases": [...], "broken_test_cases": [...]}
```

### Legacy Support
Legacy JSON format with metadata is still supported for loading:
```python
# Both formats work
problems = load_dataset_from_file("legacy_format.json")  # JSON with metadata
problems = load_dataset_from_file("new_format.jsonl")    # JSONL format
```

## Module Structure

- `generation/` - Dataset building, solution generation, broken test creation
- `evaluation/` - Multi-type model evaluation framework (choice, completion, rating, multiturn)
- `dataset_loader.py` - Unified data loading and saving with JSONL support
- `prompts/` - System prompts and templates