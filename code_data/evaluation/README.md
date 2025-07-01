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

### config.py - EvaluationConfig
```python
@dataclass
class EvaluationConfig:
    eval_type: str                    # choice, completion, multiturn, rating
    datasets: Dict[str, str]          # {"clean": "path1.json", "hack": "path2.json"}
    source_dataset: str               # mbpp, apps (for metadata)
    model: str                        # Model identifier
    grader_type: str                 # mcq, test_execution, model_based
    use_cache: bool = True           # Enable response caching
    use_batch_api: bool = True       # Use batch API when available
    max_concurrent: int = 5          # Concurrent API calls
    template_params: Dict = {}       # Template-specific parameters
    output_path: Optional[str] = None # Save results to file
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

### CompletionDatasetLoader
Loads datasets for evaluation with label-based organization:
```python
# Dataset format expectations:
{
    "clean": "path/to/clean_solutions.json",     # Non-hacking solutions
    "hack": "path/to/hacking_solutions.json",    # Reward hacking solutions  
    "partial_hack": "path/to/partial.json"      # Partially incorrect solutions
}

# Loads and validates dataset requirements per evaluation type
# REQUIRED_DATASETS = {
#     "choice": ["clean", "hack"],
#     "completion": ["problems"], 
#     "multiturn": ["problems"],
#     "rating": ["solutions"]
# }
```

## Evaluation Pipeline

### 1. Configuration Setup
- Parse CLI args or load JSON config
- Validate required datasets exist
- Initialize API client with caching/batch settings

### 2. Question Generation
- Load problems/solutions from specified datasets
- Generate evaluation questions using templates
- Apply template-specific formatting and parameters

### 3. Model Querying
- Send questions to model via safetytooling API
- Handle concurrent requests with rate limiting
- Cache responses for reproducibility

### 4. Response Grading
- Parse model responses using appropriate grader
- Execute code/run tests for completion evaluations
- Extract answers/scores for choice/rating evaluations

### 5. Results Analysis
- Compute summary statistics per evaluation type
- Generate QuestionResult objects for detailed analysis
- Save results to JSONL format if output_path specified

## Template Parameters

### Choice Template
- `num_options` - Number of choices (default: 3)
- `shuffle_options` - Randomize option order
- `include_explanation` - Ask for reasoning

### Completion Template  
- `max_problems` - Limit number of problems
- `include_tests` - Show test cases in prompt
- `timeout` - Execution timeout per solution

### Multi-turn Template
- `max_turns` - Maximum conversation turns
- `feedback_style` - How to present test failures
- `allow_fixes` - Whether model can revise solutions