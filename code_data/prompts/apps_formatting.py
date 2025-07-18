from .registry import PromptRegistry

# Create a separate registry for APPS formatting with different inputs
apps_formatting = PromptRegistry(
    "apps_formatting", required_inputs=["instruction", "response"]
)

apps_formatting.register(
    "apps_format",
    """You are formatting an APPS programming problem for use in a test framework. You need to:

1. Extract the PROBLEM DESCRIPTION without the input/output examples (cut off before "-----Input-----" or similar). Keep all of the details of the problem.
2. Create a meaningful function name based on the problem. If there is already a function name indicated in the problem description, utilize that. If there are multiple required functions, choose the first one.
3. Format the test cases properly as function calls with expected outputs. The test cases should match the function signature.
4. Reformat the main code of the solution as a callable function. If there are imports or other code that is not the main function, you may still include it outside the function. However, the main function should be callable in the format corresponding to the test cases. 

ORIGINAL PROBLEM:
{instruction}

ORIGINAL SOLUTION:
{response}

Please return ONLY a JSON object with this exact structure:
{{
    "question": "Clean problem description without input/output examples",
    "function_name": "function_name",
    "test_cases": [
        {{"input": "function_name(args_1)", "expected_output": "expected_result_1"}},
        {{"input": "function_name(args_2)", "expected_output": "expected_result_2"}},
        ...
    ],
    "formatted_solution": "def function_name(...):\\n    # complete function implementation"
}}

Note that any newlines should be escaped with \\n.

The function should take appropriate parameters based on the problem requirements, and test_cases should call the function with proper arguments that match the input examples. Ensure that the test cases match the reformatted function's signature. Inputs and outputs for test cases should be strings, but the function should be able to handle the actual types of the inputs and outputs. Do not add new comments to the solution or otherwise change the core functionality of the solution.""",
)
