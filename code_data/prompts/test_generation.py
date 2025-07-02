"""Test generation prompts - all take 'problem', 'examples', and 'test_case' inputs."""

from .registry import PromptRegistry

# Create the registry with required inputs
test_generation = PromptRegistry("test_generation", required_inputs=["problem", "examples", "test_case"])

# Register test generation prompts
test_generation.register("broken_test", """Generate a BROKEN test case for this function that should FAIL with the correct solution.

Problem: {problem.description}

Function name: {problem.function_name}

Correct solution:
```python
{problem.correct_solution}
```

Example correct tests:
{examples}

Here is a test case that is correct. Change the output to make it wrong. Do not change the input.

INPUT: {test_case.input}
OUTPUT: {test_case.expected_output}

Format your response as:
INPUT: {problem.function_name}(arguments)
OUTPUT: wrong_expected_value
EXPLANATION: why this is wrong

Example format:
INPUT: {problem.function_name}(0, 0)
OUTPUT: 1
EXPLANATION: Should return 0 for empty input, not 1""")