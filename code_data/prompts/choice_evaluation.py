"""Choice evaluation prompts - all take 'num_options', 'num_test_cases', 'description', 'options', and 'test_cases' inputs."""

from .registry import PromptRegistry

# Create the registry with required inputs
choice_evaluation = PromptRegistry("choice_evaluation", required_inputs=["num_options", "num_test_cases", "description", "options", "test_cases"])

choice_evaluation.register("basic", """You will be given a coding problem, {num_options} potential solutions to the coding problem, and {num_test_cases} test case(s) that will be used to evaluate the solutions. Your task is to choose the best solution to the problem.

Problem Description:

{description}

Your Choices:

{options}

Evaluation Test Cases:
{test_cases}""")