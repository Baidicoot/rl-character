#!/usr/bin/env python3
"""Script to run model-based grading on CodeProblem JSONL data."""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import argparse
from tqdm.asyncio import tqdm

from code_data.generation.models import CodeProblem
from code_data.evaluation.graders import ModelBasedGrader
from code_data.prompts.model_grader import model_grader
from code_data.utils import format_test_cases
from code_data.evaluation.models import prompt_to_dict
from code_data.dataset_loader import CodeDataLoader
from code_data.evaluation.api_client import EvaluationAPIClient
from dataclasses import replace


async def grade_single_problem(
    grader: ModelBasedGrader, 
    code_problem: CodeProblem,
    problem_idx: int
) -> Dict:
    """Grade a single CodeProblem and return grading results dictionary."""
    # Format test cases using the utility function
    test_cases_str = format_test_cases(
        code_problem.mixed_test_cases or [], 
        code_problem.function_name or "solution",
        format_type="assert"
    )
    
    # Format the question with problem description and test cases
    question = f"""Problem Description:
{code_problem.description}

Test Cases:
{test_cases_str}"""
    
    # Check if completion exists
    answer = code_problem.parsed_completion or code_problem.full_completion
    if not answer:
        raise ValueError(f"No completion found for problem {code_problem.problem_id}")
    
    # Create the full grading prompt using the faithfulness template
    grading_prompt = model_grader.get(
        "faithfulness",
        question=question,
        answer=answer
    )
    
    # Grade using the ModelBasedGrader
    result = await grader.grade(
        prompt=grading_prompt,
        options=["FAITHFUL", "UNFAITHFUL", "FLAG", "UNSURE"]
    )
    
    # Create grading results dictionary with all needed data
    grading_result = {
        # Core problem data
        "problem_id": code_problem.problem_id,
        "description": code_problem.description,
        "mixed_test_cases": [tc.__dict__ for tc in (code_problem.mixed_test_cases or [])],
        "prompt": code_problem.prompt,
        "completion": code_problem.parsed_completion or code_problem.full_completion,
        "correct_completion": code_problem.correct_solution,
        
        # Grading results
        "faithfulness_grade": result["answer"],
        "faithfulness_raw_response": result["grading_response"],
        "faithfulness_parsed_successfully": result["parsed_successfully"],
        "grading_question": question,
        "grading_answer": answer,
        
        # Additional metadata
        "dataset": getattr(code_problem, 'dataset', 'unknown'),
        "function_name": code_problem.function_name,
        "difficulty": getattr(code_problem, 'difficulty', None),
        "tags": code_problem.tags or []
    }
    
    return grading_result


async def grade_problems_async(
    jsonl_path: str,
    model_name: str = "gpt-4o-mini",
    save_categorized: bool = False,
    output_dir: Optional[str] = None,
    provider: str = "openai"
) -> Dict:
    """Grade all problems in the JSONL file asynchronously."""
    # Load the CodeProblem data
    print(f"Loading problems from {jsonl_path}...")
    problems = CodeDataLoader.load_completion_dataset(jsonl_path)
    print(f"Loaded {len(problems)} problems")
    
    # Setup environment and initialize API client
    api_client = EvaluationAPIClient(cache_dir=Path(".cache"))
    
    # Initialize the grader
    grader = ModelBasedGrader(
        api_client=api_client,
        grading_model=model_name,
        provider=provider
    )
    
    # Grade all problems with progress bar
    print(f"Grading problems using {model_name}...")
    tasks = [
        grade_single_problem(grader, problem, i) 
        for i, problem in enumerate(problems)
    ]
    
    grading_results = await tqdm.gather(*tasks, desc="Grading problems")
    
    # Categorize results by grade
    categorized = defaultdict(list)
    grade_counts = defaultdict(int)
    
    for result in grading_results:
        grade = result["faithfulness_grade"]
        categorized[grade].append(result)
        grade_counts[grade] += 1
    
    # Print summary
    print("\n=== Grading Results Summary ===")
    total = len(grading_results)
    for grade in ["FAITHFUL", "UNFAITHFUL", "FLAG", "UNSURE"]:
        count = grade_counts[grade]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{grade}: {count}/{total} ({percentage:.1f}%)")
    
    # Save grading results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save all grading results
        all_results_file = output_path / "grading_results.jsonl"
        with open(all_results_file, 'w') as f:
            for result in grading_results:
                f.write(json.dumps(result) + '\n')
        print(f"Saved all grading results to {all_results_file}")
        
        # Save categorized results if requested
        if save_categorized:
            for grade, grade_results in categorized.items():
                if grade_results:
                    filename = output_path / f"{grade.lower()}_results.jsonl"
                    with open(filename, 'w') as f:
                        for result in grade_results:
                            f.write(json.dumps(result) + '\n')
                    print(f"Saved {len(grade_results)} {grade} results to {filename}")
    
    return {
        "grading_results": grading_results,
        "categorized": dict(categorized),
        "summary": dict(grade_counts),
        "total": total
    }


def main():
    """Run model-based grading on CodeProblem JSONL data.
    
    This script loads CodeProblem data from a JSONL file and runs faithfulness grading
    using a model-based grader. The grader evaluates whether model completions adhere
    to the original problem statement vs. just the test cases.
    
    Grades: FAITHFUL, UNFAITHFUL, FLAG, UNSURE
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("jsonl_path", help="Path to CodeProblem JSONL file")
    parser.add_argument("--model", default="claude-3-5-haiku-20241022", help="Model to use for grading")
    parser.add_argument("--provider", default="anthropic", help="Provider to use for grading")
    parser.add_argument("--save-categorized", action="store_true", 
                       help="Save completions in separate files by category")
    parser.add_argument("--output-dir", help="Directory to save results (default: same as input)")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = "./model_grader_results"
    
    # Run the async grading
    results = asyncio.run(grade_problems_async(
        jsonl_path=args.jsonl_path,
        model_name=args.model,
        save_categorized=args.save_categorized,
        output_dir=args.output_dir,
        provider=args.provider
    ))
    
    print(f"\nGrading complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()