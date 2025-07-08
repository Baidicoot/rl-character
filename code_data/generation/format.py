from safetytooling.apis import InferenceAPI
from dotenv import load_dotenv
from typing import Dict, Optional
import re
from .models import TestCase, CodeProblem
import asyncio
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
from ..dataset_loader import CodeDataLoader

load_dotenv()

api = InferenceAPI()


async def standardize_code_instruct(
    entry: Dict[str, str],
    instruction_field: str = "instruction",
    response_field: str = "response",
    model: str = "gpt-4.1-mini",
    dataset_name: str = "code_instruct",
    entry_id: str = "default",
    force_provider: str = None,
) -> Optional[CodeProblem]:
    """
    Standardize a CodeInstruct entry to a CodeProblem entry.
    """
    prompt = f"""
You are helping to standarize coding datasets into pure functions without side-effects/IO that can therefore be tested with simple input-output pairs.

Based on the following coding prompt and response, create a high-level prompt describing a Python function that the user wants to implement, and a concrete implementation of that function.

The high-level prompt should give an unambigious description of the function's input and output format. Each input-output pair should be a single call to the function and the expected return value of that call.

Prompt:
{entry[instruction_field]}

Response:
{entry[response_field]}

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

If the function cannot easily be converted to a format suitable for this type of testing, instead output "None", and we will skip this problem.
"""

    response = await api.ask_single_question(
        model_id=model,
        question=prompt,
        system_prompt="You are a helpful assistant that helps to standarize coding datasets.",
        force_provider=force_provider,
    )
    response = response[0].strip()

    # Extract function description and definition using regex

    # Try to extract the XML sections
    desc_match = re.search(
        r"<fn_description>\s*(.*?)\s*</fn_description>", response, re.DOTALL
    )
    def_match = re.search(
        r"<fn_definition>\s*(.*?)\s*</fn_definition>", response, re.DOTALL
    )

    if not desc_match or not def_match:
        return None

    description = desc_match.group(1).strip()
    function_code = def_match.group(1).strip()

    # Extract function name from the function definition
    fn_name_match = re.match(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", function_code)
    if not fn_name_match:
        return None

    function_name = fn_name_match.group(1)

    # Extract test cases using regex
    test_case_tuples = []
    test_case_pattern = rf"<test_case>\s*<input>({function_name}\(.*?\))</input>\s*<output>(.*?)</output>\s*</test_case>"
    test_case_matches = re.finditer(test_case_pattern, response, re.DOTALL)

    for match in test_case_matches:
        input_str = match.group(1).strip()
        output_str = match.group(2).strip()
        test_case_tuples.append(TestCase(input=input_str, correct_output=output_str))

    if not test_case_tuples:
        return None

    return CodeProblem(
        problem_id=entry_id,
        description=description,
        function_name=function_name,
        correct_solution=function_code,
        test_cases=test_case_tuples,
        dataset=dataset_name,
    )


async def format_huggingface_dataset(
    dataset_id: str = "ise-uiuc/Magicoder-Evol-Instruct-110K",
    model: str = "gpt-4.1-mini",
    dataset_name: str = "code_instruct",
    split: str = "train",
    start_index: int = 0,
    size: int = 100,
    max_concurrent: int = 5,
    instruction_field: str = "instruction",
    response_field: str = "response",
    output_path: str = "output.jsonl",
    force_provider: str = None,
):
    """
    Format a Hugging Face dataset to a CodeProblem dataset.
    """
    dataset = load_dataset(dataset_id, split=split)
    dataset = dataset.skip(start_index).take(size)
    problems = []

    sem = asyncio.Semaphore(max_concurrent)

    async def standardize_problem(
        entry: Dict[str, str], entry_id: str
    ) -> Optional[CodeProblem]:
        async with sem:
            return await standardize_code_instruct(
                entry,
                model=model,
                dataset_name=dataset_name,
                entry_id=entry_id,
                force_provider=force_provider,
            )

    problems = await tqdm_asyncio.gather(
        *[
            standardize_problem(entry, entry_id=str(i))
            for i, entry in enumerate(dataset)
        ],
        desc="Standardizing dataset",
    )
    problems = [problem for problem in problems if problem is not None]

    CodeDataLoader.save_dataset_to_file(problems, output_path)
