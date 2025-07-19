"""Extract public test cases from problem statements using LLM."""

import re
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from tqdm.asyncio import tqdm

try:
    from .api_manager import APIManager
    from .models import CodeProblem, TestCase
    from .deepcoder_loader import load_deepcoder_problems, save_problems
except ImportError:
    from api_manager import APIManager
    from models import CodeProblem, TestCase
    from deepcoder_loader import load_deepcoder_problems, save_problems


class TaskExtractor:
    """Extract public test cases from problem statements."""
    
    def __init__(self, api_manager: APIManager):
        """Initialize extractor.
        
        Args:
            api_manager: API manager instance
        """
        self.api_manager = api_manager
    
    def create_extraction_prompt(self, problem: CodeProblem) -> str:
        """Create prompt for extracting test cases from problem statement.
        
        Args:
            problem: Programming problem
            
        Returns:
            Formatted prompt string
        """
        # Get a sample test case to show format
        sample_tests = []
        if problem.public_test_cases:
            for tc in problem.public_test_cases[:2]:
                test_dict = {
                    "input": tc.input,
                    "output": tc.output,
                    "type": tc.type
                }
                sample_tests.append(test_dict)
        
        prompt = f"""You are tasked with extracting test cases from a programming problem statement. 

The problem statement often includes example inputs and outputs. Your job is to extract these and format them as test case dictionaries.

Here is the problem statement:

{problem.problem}

Here are example test case formats from this dataset:
{json.dumps(sample_tests, indent=2)}

Please extract ALL test cases (examples) that are shown in the problem statement. Each test case should include:
- input: The exact input string (including newlines if present)
- output: The exact expected output string (including newlines if present)
- type: Either "stdin" (for standard input/output) or "functional" (for function calls)

Important notes:
- For stdin type: Include all newlines in the input/output strings (use \\n)
- For functional type: Format inputs as function arguments (e.g., "[3, 2, 0, 1, 0]\\n[6, 5, 0]")
- Preserve the exact formatting from the problem statement
- Include ALL examples/test cases mentioned in the problem

Output each test case in <test> tags like this:
<test>
{{
  "input": "...",
  "output": "...",
  "type": "stdin" or "functional"
}}
</test>

Extract all test cases now:"""
        
        return prompt
    
    async def extract_tests_from_problem(
        self,
        problem: CodeProblem,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        provider: Optional[str] = None,
    ) -> List[TestCase]:
        """Extract test cases from a single problem.
        
        Args:
            problem: Programming problem
            model: Model to use
            temperature: Generation temperature (low for extraction)
            provider: Provider to use
            
        Returns:
            List of extracted TestCase objects
        """
        prompt = self.create_extraction_prompt(problem)
        
        completion = await self.api_manager.get_single_completion(
            prompt=prompt,
            model=model,
            temperature=temperature,
            provider=provider,
        )
        
        if not completion:
            return []
        
        # Extract test cases from completion
        test_pattern = r'<test>\s*(.*?)\s*</test>'
        matches = re.findall(test_pattern, completion, re.DOTALL)
        
        extracted_tests = []
        for match in matches:
            try:
                # Parse JSON
                test_data = json.loads(match)
                
                # Create TestCase
                test_case = TestCase(
                    input=test_data.get("input", ""),
                    output=test_data.get("output", ""),
                    type=test_data.get("type", "stdin")
                )
                extracted_tests.append(test_case)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Failed to parse test case: {e}")
                continue
        
        return extracted_tests
    
    async def extract_tests_from_problems(
        self,
        problems: List[CodeProblem],
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        provider: Optional[str] = None,
        show_progress: bool = True,
        output_path: Optional[Union[str, Path]] = None,
    ) -> List[CodeProblem]:
        """Extract test cases from multiple problems.
        
        Args:
            problems: List of programming problems
            model: Model to use
            temperature: Generation temperature
            provider: Provider to use
            show_progress: Whether to show progress bar
            output_path: Path to save processed problems (required)
            
        Returns:
            List of CodeProblem instances with extracted public tests
        """
        if output_path is None:
            raise ValueError("Output path is required to save results.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create prompts for all problems
        prompt_data_list = []
        for problem in problems:
            prompt_text = self.create_extraction_prompt(problem)
            prompt_data_list.append({
                'prompt': prompt_text,
                'problem': problem,
                'problem_id': problem.problem_id,
            })
        
        # Define postprocess function
        def postprocess_extraction(prompt: str, context: Dict[str, Any], completion: Optional[str]) -> Optional[Dict[str, Any]]:
            """Postprocess extraction results."""
            if completion is None:
                return None
            
            problem = context['problem']
            
            # Extract test cases from completion
            test_pattern = r'<test>\s*(.*?)\s*</test>'
            matches = re.findall(test_pattern, completion, re.DOTALL)
            
            extracted_tests = []
            for match in matches:
                try:
                    # Parse JSON
                    test_data = json.loads(match)
                    
                    # Create TestCase
                    test_case = TestCase(
                        input=test_data.get("input", ""),
                        output=test_data.get("output", ""),
                        type=test_data.get("type", "stdin")
                    )
                    extracted_tests.append(test_case)
                    
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Create new problem with extracted public tests
            new_problem = CodeProblem(
                problem_id=problem.problem_id,
                problem=problem.problem,
                solutions=problem.solutions,
                public_test_cases=extracted_tests,  # Extracted tests go here
                test_cases=problem.public_test_cases,  # Original tests from dataset
                metadata=problem.metadata,
                generated_solutions=problem.generated_solutions,
            )
            
            # Add extraction metadata
            new_problem.metadata["extraction_model"] = model
            new_problem.metadata["num_extracted_tests"] = len(extracted_tests)
            new_problem.metadata["num_original_tests"] = len(problem.public_test_cases)
            
            return new_problem.to_dict()
        
        # Process all prompts
        await self.api_manager.get_completions_with_postprocess(
            prompts=prompt_data_list,
            model=model,
            temperature=temperature,
            provider=provider,
            postprocess=postprocess_extraction,
            output_path=output_path,
            show_progress=show_progress,
            desc="Extracting test cases",
        )
        
        # Read back the results
        results = []
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    result_dict = json.loads(line)
                    results.append(CodeProblem.from_dict(result_dict))
        
        return results


async def preprocess_deepcoder(
    configs: List[str] = None,
    max_problems: Optional[int] = None,
    output_path: Union[str, Path] = "data/deepcoder_preprocessed.jsonl",
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    provider: Optional[str] = None,
    max_concurrent: int = 5,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
):
    """Preprocess DeepCoder problems by extracting public test cases.
    
    Args:
        configs: DeepCoder configs to process
        max_problems: Maximum problems to process
        output_path: Output file path
        model: Model to use for extraction
        temperature: Generation temperature
        provider: Provider to use
        max_concurrent: Maximum concurrent requests
        use_cache: Whether to use caching
        cache_dir: Cache directory
    """
    # Load problems
    print("Loading DeepCoder problems...")
    problems = load_deepcoder_problems(
        configs=configs,
        max_problems=max_problems,
    )
    
    # Initialize API manager and extractor
    api_manager = APIManager(
        use_cache=use_cache,
        cache_dir=cache_dir,
        max_concurrent=max_concurrent,
    )
    extractor = TaskExtractor(api_manager)
    
    # Extract test cases
    print(f"Extracting test cases from {len(problems)} problems...")
    processed_problems = await extractor.extract_tests_from_problems(
        problems=problems,
        model=model,
        temperature=temperature,
        provider=provider,
        output_path=output_path,
    )
    
    print(f"Preprocessed {len(processed_problems)} problems")
    print(f"Results saved to: {output_path}")
    
    # Print statistics
    total_extracted = sum(len(p.public_test_cases) for p in processed_problems)
    total_original = sum(len(p.test_cases) for p in processed_problems)
    print(f"Total extracted test cases: {total_extracted}")
    print(f"Total original test cases: {total_original}")