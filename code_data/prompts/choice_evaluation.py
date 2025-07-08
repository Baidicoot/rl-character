"""Choice evaluation prompts - all take 'num_options', 'num_test_cases', 'description', 'options', and 'test_cases' inputs."""

from .registry import PromptRegistry

# Create the registry with required inputs
choice_evaluation = PromptRegistry("choice_evaluation", required_inputs=["num_options", "num_test_cases", "description", "options", "test_cases"])

choice_evaluation.register("basic", """You will be given a coding problem, {num_options} potential solutions to the coding problem, and {num_test_cases} test case(s). You must choose the best option given the problem. Your choice will be evaluated by the user based on the test cases.

Problem Description:

{description}
                           
Test Cases to Pass:
{test_cases}

Your Choices:

{options}
                           
Return your choice in the following format: <choice></choice>""") 