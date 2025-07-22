# Code Generation
Loading programming datasets, generating solutions with LLMs, and executing tests. Supports multiple datasets (DeepCoder, SWE-Bench, MBPP) and provides unified APIs for solution generation and evaluation.

## Loading / Pre-processing Data

### Dataset Loaders

The framework supports three primary datasets with distinct loading patterns:

**DeepCoder Dataset**
- Multi-config support: `lcbv5`, `primeintellect`, `taco`, `codeforces`
- Function name extraction from problem descriptions for functional tests
- Pre-processing uses LLMs to extract public test cases from problem descriptions

```bash
# Load and pre-process DeepCoder
python -m code_generation.generation_cli preprocess deepcoder --configs lcbv5 --output data/preprocessed.jsonl
```

**SWE-bench Dataset**
- Real-world software engineering problems from GitHub issues
- No executable tests - uses patch-based evaluation
- Oracle format, single-turn sampling only; no test execution supported currently

**MBPP Dataset (Mostly Basic Python Problems)**
- Function-based problems with assert-style test cases
- Automatic public/private test splitting with configurable ratios
- Function name extraction from test assertions

```bash
# Load all MBPP problems and save to datasets/mbpp_all.jsonl
python -m code_generation.mbpp_loader --output datasets/mbpp_all.jsonl

# Load first 100 problems with 1 public test each
python -m code_generation.mbpp_loader --num-problems 100 --n-public 1 --output datasets/mbpp_small.jsonl
```

(TODO: currently the MBPP problem statements *do not include* the public tests -- unlike the DeepCoder dataset, where the public tests are extracted from the problem statements, this is just for the data model.)

### Test Case Types

The framework handles two distinct test execution patterns:

**Functional Tests**
- Tests call a specific function with arguments: `function_name(arg1, arg2)`
- Expected output compared directly to function return value
- Requires `func_name` in problem metadata
- Datasets: all MBPP problems, some LCB/TACO problems

**Stdin/Stdout Tests**
- Tests provide input via stdin and expect output on stdout
- More common in competitive programming problems
- Datasets: LCB/TACO/PrimeIntellect

### Data Models

Core data structures in `models.py`:

- `CodeProblem`: Unified problem representation with metadata, solutions, and test cases
- `TestCase`: Individual test with input/output and type (`functional` or `stdin`)
- `GenerationResult`: Complete generation output with code, conversation history, and metadata

## Executing Code

### Test Execution Framework

The framework provides multiple execution backends for safe code evaluation:

**SubprocessExecutor** (Default)
- Isolated subprocess execution with timeouts
- Memory and resource limitations
- Captures stdout, stderr, and return codes

**TogetherExecutor** 
- Remote code execution via Together AI's Code Interpreter
- Additional isolation layer for untrusted code
- Network-based execution with built-in sandboxing

### Grading and Evaluation

The `TestExecutionGrader` class provides:
- Automatic test harness generation for functional and stdin/stdout tests
- Comprehensive error reporting with stack traces
- Support for both subprocess and remote execution
- Pass/fail metrics with detailed failure analysis

Test suite to test the grader/executor:
```bash
# Test execution with detailed feedback
python -m code_generation.test_scripts.test_grader_full

# Test specific executor implementations  
python -m code_generation.test_scripts.test_executors

# Validate against real dataset samples
python -m code_generation.test_scripts.test_real_dataset_samples
```

## Sampling / Scraping Data

### System Prompts

System prompts for generation are in `code_data/prompts/system.py` and can be passed in using the argument `--system-prompt-id`.

### Single-Turn Solution Generation (SWE-bench)

The `SolutionSampler` provides single-turn solution generation for SWE-bench problems:

```bash
# Multiple solutions with temperature sampling for SWE-bench
python -m code_generation.generation_cli generate --dataset swebench --num-samples-per-problem 5 --temperature 0.8

# SWE-bench with custom system and generation prompts
python -m code_generation.generation_cli generate --dataset swebench --system-prompt-id helpful_coder --prompt-id basic
```

### Multi-Turn Solution Generation with Feedback (MBPP/DeepCoder)

The `scraper.py` module provides iterative solution generation with test feedback and retry logic:

- **Concurrent processing** at both problem and API levels
- **Retry logic** with exponential backoff for failed solutions
- **Test execution feedback** integration for iterative improvement
- **Flexible evaluation criteria** (public tests, private tests, or both)

The scraper handles the complete pipeline:
1. Load problems from datasets
2. Generate initial solutions
3. Execute against test cases
4. Provide feedback for failed tests
5. Retry with feedback until success or max attempts

```bash
python -m code_generation.scraper datasets/deepcoder_preprocessed.jsonl datasets/deepcoder_preprocessed_o4mini_solutions.jsonl --model o4-mini --should-pass-private --error-log-path datasets/deepcoder_impossible.jsonl --max-turns 10 --max-concurrent 60
```

#### Performance Profiling

Use the profiled scraper to measure time distribution per problem:

```bash
python -m code_generation.test_scripts.scraper_profiled code_generation/test_scripts/three_sample.jsonl three_sample_profile_solutions.jsonl --model o4-mini --profile-output three_sample_profile.json
```

Shows per-problem percentages: LLM calls (~96% per problem), code execution (~4% per problem), and I/O (<1% per problem).

### Output Formats

Generated solutions are saved as JSONL with complete metadata:

```json
{
  "problem": {
    "problem_id": "lcbv5_001",
    "problem": "Write a function that...",
    "solutions": ["def solve(x):..."],
    "test_cases": [...],
    "metadata": {"func_name": "solve"}
  },
  "final_code": "def solve(x):\n    return x + 1",
  "full_message_history": [...],
  "generation_metadata": {
    "model": "gpt-4o-mini",
    "temperature": 1.0,
    "prompt_id": "basic"
  }
}
```
