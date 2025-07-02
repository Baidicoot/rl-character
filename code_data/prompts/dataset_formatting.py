"""Dataset formatting prompts - all take 'instruction' and 'response' inputs."""

from .registry import PromptRegistry

# Create the registry with required inputs
dataset_formatting = PromptRegistry("dataset_formatting", required_inputs=["instruction", "response"])

dataset_formatting.register("standardize", """You are helping to standarize coding datasets into pure functions without side-effects/IO that can therefore be tested with simple input-output pairs.

Based on the following coding prompt and response, create a high-level prompt describing a Python function that the user wants to implement, and a concrete implementation of that function.

The high-level prompt should give an unambigious description of the function's input and output format. Each input-output pair should be a single call to the function and the expected return value of that call.

Prompt:
{instruction}

Response:
{response}

You should also output a list of some example test cases for the function. These should be a list of XML tuples, where the first element is the input and the second element is the expected output.

Format your output using XML tags like this:
<fn_description>
...
</fn_description>
<fn_definition>
...
</fn_definition>

# Test cases
<test_case>
<input>fn(input1, input2, ...)</input><output>expected_output</output>
</test_case>
<test_case>
...
</test_case>

If the function cannot easily be converted to a format suitable for this type of testing, instead output "None", and we will skip this problem.""")