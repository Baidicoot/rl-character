# Evaluation Module

Multi-type model evaluation framework for coding behavior analysis.

## Evaluation Types

### Choice Evaluation
Multiple choice questions testing model preferences between solutions:
```python
# Template presents options like:
# A) Solution that passes all tests (clean)
# B) Solution that games broken tests (hacking)  
# C) Solution with errors (partial_hack)

# Measures: parse_rate, non_hacking_rate, overall_hack_rate
```

### Completion Evaluation  
Direct code generation with test execution:
```python
# Model generates solution for problem
# Solution tested against correct + broken test cases
# Measures: completion_rate, solve_rate, test_pass_rate
```

### Multi-turn Evaluation
Conversational coding with broken test feedback:
```python
# Turn 1: Model generates solution
# Turn 2: System reports broken test failure
# Turn 3: Model responds/fixes
# Measures: response_rate, cheat_rate, fix_rate
```

### Rating Evaluation
Model rates solution quality on 1-10 scale:
```python
# Presents solution and asks for quality rating
# Measures: scoring_rate, average_score, score_stdev  
```

## Core Components

### config.py - Evaluation Configurations
Type-specific config classes with shared base:
```python
@dataclass
class BaseEvaluationConfig:
    datasets: Dict[str, str]          # {"clean": "path1.json", "hack": "path2.json"}
    source_dataset: str               # mbpp, apps
    model: str = "gpt-4o-mini"        # Model identifier
    temperature: float = 0.7          # Sampling temperature
    provider: Optional[str] = None    # openai, anthropic, etc.
    prompt_id: str = "basic"          # Prompt ID from registry
    system_prompt_id: Optional[str] = None # System prompt ID
    use_cache: bool = True           # Enable response caching
    use_batch_api: bool = True       # Use batch API when available
    max_concurrent: int = 5          # Concurrent API calls
    output_path: Optional[str] = None # Save results to file

@dataclass  
class ChoiceEvaluationConfig(BaseEvaluationConfig):
    eval_type: str = "choice"
    grader_type: str = "mcq"
    fraction_broken: float = 0.5     # Fraction of broken test cases
    dataset_filters: Dict = {}       # Dataset loading filters

@dataclass
class RatingEvaluationConfig(BaseEvaluationConfig):  
    eval_type: str = "rating"
    grader_type: str = "rating_extraction"
    attribute: str = "helpfulness"   # Rating attribute
```

### templates.py - Evaluation Templates
Base classes for different evaluation types:
- `ChoiceEvalTemplate` - Multiple choice question generation
- `CompletionEvalTemplate` - Code completion tasks
- `MultiTurnEvalTemplate` - Multi-turn conversations
- `RatingEvalTemplate` - Solution rating tasks

### graders.py - Response Grading
- `MCQGrader` - Parse multiple choice responses (A, B, C)
- `TestExecutionGrader` - Execute code and check test results
- `ModelBasedGrader` - Use LLM to grade responses

### models.py - Result Data Structures
```python
@dataclass
class QuestionResult:
    question_id: int
    problem_id: str
    eval_type: str
    question_prompt: str              # Full prompt sent to model
    response: Optional[str]           # Model response
    grade: Dict[str, Any]            # Grading results
    config: Dict[str, Any]           # Evaluation config for reproducibility
    
    # Properties
    parsed_successfully -> bool       # Response parsed successfully
    summary_stats -> Dict            # Eval-type specific metrics
```

## Dataset Loading

### CodeDataLoader
Loads datasets for evaluation with label-based organization:
```python
# Dataset format expectations:
{
    "clean": "path/to/clean_solutions.jsonl",     # Non-hacking solutions
    "hack": "path/to/hacking_solutions.jsonl",    # Reward hacking solutions  
    "partial_hack": "path/to/partial.jsonl"      # Partially incorrect solutions
}

# Load datasets as lists of CodeProblem objects
datasets = CodeDataLoader.load_multiple_datasets({
    "clean": "clean_solutions.jsonl",
    "hack": "hacking_solutions.jsonl"
}, filters={"min_test_cases": 2})

# Required datasets per evaluation type:
# choice: ["clean", "hack"] - Multiple solutions for comparison
# completion: ["source"] - Single dataset with problems to solve
# multiturn: ["source"] - Single dataset for conversational evaluation  
# rating: ["source"] - Single dataset with solutions to rate
```

## Evaluation Pipeline

### 1. Configuration Setup
- Parse CLI args or load JSON config
- Initialize API client with caching/batch settings
- Set up dataset filters if specified

### 2. Dataset Loading
- Load datasets using `CodeDataLoader.load_completion_dataset()` or `load_multiple_datasets()`
- Apply filters (test case count, difficulty, tags) at load time
- Work with lists of strongly-typed `CodeProblem` objects

### 3. Question Generation
- Generate evaluation questions using templates
- For choice: find common problems across multiple datasets
- For completion/multiturn/rating: work with single dataset
- Apply template-specific formatting and parameters

### 4. Model Querying
- Send questions to model via safetytooling API
- Handle concurrent requests with rate limiting
- Cache responses for reproducibility

### 5. Response Grading
- Parse model responses using appropriate grader
- Execute code/run tests for completion evaluations
- Extract answers/scores for choice/rating evaluations

### 6. Results Analysis
- Compute summary statistics per evaluation type
- Generate QuestionResult objects for detailed analysis
- Save results to JSONL format if output_path specified

## Prompt Registry Integration

Evaluation templates use prompt registries for consistent templating:
```python
# Choice evaluation prompts (prompts/choice_evaluation.py)
choice_evaluation.register("basic", "You will be given {num_options} solutions...")
choice_evaluation.register("complete", "Full prompt with {description}, {options}, {test_cases}")

# Rating evaluation prompts (prompts/rating_evaluation.py)  
rating_evaluation.register("basic", "Rate solution for {attribute} on 1-10 scale...")

# Templates use config.prompt_id to select prompt:
prompt = choice_evaluation.get(self.config.prompt_id, num_options=3, description="...")
```

## CLI Usage

### Basic Commands

**Choice Evaluation**: Test model preferences between clean vs hacking solutions
```bash
python -m code_data.evaluation_cli choice \
  --datasets "clean:path/to/clean_solutions.jsonl,hack:path/to/hack_solutions.jsonl" \
  --source-dataset mbpp \
  --prompt-id complete \
  --model gpt-4o-mini \
  --max-problems 50 \
  --output results_choice.jsonl
```

**Completion Evaluation**: Generate and test solutions directly
```bash
python -m code_data.evaluation_cli completion \
  --datasets "problems:path/to/problems_with_broken_tests.jsonl" \
  --source-dataset mbpp \
  --model claude-3-haiku-20240307 \
  --grader-type test_execution \
  --output results_completion.jsonl
```

**Rating Evaluation**: Rate solution quality
```bash
python -m code_data.evaluation_cli rating \
  --datasets "solutions:path/to/solutions.jsonl" \
  --source-dataset mbpp \
  --prompt-id basic \
  --template-params "attribute:helpfulness" \
  --model gpt-4o-mini
```

**Multi-turn Evaluation**: Conversational coding with feedback
```bash
python -m code_data.evaluation_cli multiturn \
  --datasets "problems:path/to/problems.jsonl" \
  --source-dataset mbpp \
  --model gpt-4o-mini \
  --max-problems 20
```

### Config File Usage
```bash
# Use predefined config
python -m code_data.evaluation_cli --config configs/evaluation/choice_basic.json

# Override specific settings
python -m code_data.evaluation_cli --config configs/evaluation/choice_basic.json \
  --model claude-3-haiku \
  --prompt-id complete \
  --max-problems 50 \
  --output custom_results.jsonl
```

**Example Config File** (`configs/evaluation/choice_basic.json`):
```json
{
  "eval_type": "choice",
  "datasets": {
    "clean": "datasets/code/mbpp/test/clean_solutions.jsonl",
    "hack": "datasets/code/mbpp/test/hack_solutions.jsonl"
  },
  "source_dataset": "mbpp",
  "model": "gpt-4o-mini",
  "prompt_id": "complete",
  "grader_type": "mcq",
  "use_cache": true,
  "use_batch_api": true,
  "max_concurrent": 5,
  "temperature": 0.7,
  "fraction_broken": 0.5
}
```

### CLI Arguments

**Required Arguments** (when not using `--config`):
- `eval_type`: `choice`, `completion`, `multiturn`, or `rating`
- `--datasets`: Dataset paths in format `"label1:path1,label2:path2"`
- `--source-dataset`: Source dataset name (`mbpp`, `apps`, etc.)

**Common Optional Arguments**:
```bash
--config PATH                     # JSON config file path
--model MODEL                     # Model name (default: gpt-4o-mini) 
--prompt-id ID                    # Prompt ID from registry (default: basic)
--system-prompt-id ID             # System prompt ID (optional)
--temperature FLOAT               # Sampling temperature (default: 0.7)
--provider PROVIDER               # API provider (openai, anthropic, etc.)
--max-problems INT                # Limit number of problems to evaluate
--output PATH                     # Save results to JSONL file
--max-concurrent INT              # Concurrent API requests (default: 5)
--grader-type TYPE                # auto, mcq, test_execution, model_based, rating_extraction
--template-params "key:val,..."   # Template-specific parameters
--no-cache                        # Disable response caching
--no-batch-api                    # Disable batch API usage
--quiet                          # Suppress summary output
```

**Evaluation-Specific Parameters**:

*Choice Evaluation*:
- `--template-params "fraction_broken:0.5"` - Fraction of broken tests to include

*Rating Evaluation*:  
- `--template-params "attribute:helpfulness"` - Attribute to rate (helpfulness, correctness, etc.)

**Dataset Requirements by Evaluation Type**:
- **Choice**: 2+ datasets (e.g., `"clean:path1.jsonl,hack:path2.jsonl"`)
- **Completion**: 1 dataset with problems (`"problems:path.jsonl"`)
- **Multiturn**: 1 dataset with problems (`"problems:path.jsonl"`)
- **Rating**: 1 dataset with solutions (`"solutions:path.jsonl"`)

## Dataset Filtering

All evaluation templates support filtering at load time:
```python
config.template_params["dataset_filters"] = {
    "min_test_cases": 2,           # Minimum number of test cases
    "max_test_cases": 10,          # Maximum number of test cases  
    "difficulty": [800, 900, 1000], # Allowed difficulty levels
    "tags": ["arrays", "sorting"]   # Required tags (subset match)
}
```