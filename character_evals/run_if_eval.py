#!/usr/bin/env python3

import sys
import os
import json
import asyncio
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm.asyncio import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_generation.api_manager import APIManager

# Import from if_eval directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'if_eval'))
from evaluation_lib import read_prompt_list, test_instruction_following_strict, test_instruction_following_loose


async def generate_response(api_manager, model: str, prompt: str, temperature: float) -> str:
    response = await api_manager.get_single_completion(
        prompt=prompt,
        model=model,
        temperature=temperature,
    )
    return response if response else ""

async def generate_and_grade(api_manager, model: str, inp, temperature: float) -> Dict[str, Any]:
    # Generate response
    response = await generate_response(api_manager, model, inp.prompt, temperature)
    
    # Create prompt_to_response dict for this single example
    prompt_to_response = {inp.prompt: response}
    
    # Run evaluations sequentially
    strict_result = await asyncio.to_thread(test_instruction_following_strict, inp, prompt_to_response)
    loose_result = await asyncio.to_thread(test_instruction_following_loose, inp, prompt_to_response)
    
    return {
        "input": inp,
        "response": response,
        "strict_result": strict_result,
        "loose_result": loose_result
    }

def save_results(args: argparse.Namespace, results: List[Dict[str, Any]], output_file: Path):
    strict_accuracy = sum(1 for r in results if r["strict_result"].follow_all_instructions) / len(results)
    loose_accuracy = sum(1 for r in results if r["loose_result"].follow_all_instructions) / len(results)
    
    with open(output_file, 'w') as f:
        # Write metadata
        metadata = {
            "_metadata": True,
            "model": args.model_id,
            "num_examples": len(results),
            "strict_accuracy": strict_accuracy,
            "loose_accuracy": loose_accuracy,
            "temperature": args.temperature,
            "num_samples": args.num_samples,
        }
        f.write(json.dumps(metadata) + '\n')
        
        # Write results
        for result in results:
            result_dict = {
                "prompt": result["input"].prompt,
                "response": result["response"],
                "instruction_id_list": result["strict_result"].instruction_id_list,
                "strict_follow_all_instructions": result["strict_result"].follow_all_instructions,
                "strict_follow_instruction_list": result["strict_result"].follow_instruction_list,
                "loose_follow_all_instructions": result["loose_result"].follow_all_instructions,
                "loose_follow_instruction_list": result["loose_result"].follow_instruction_list,
            }
            f.write(json.dumps(result_dict) + '\n')
    
    print(f"Results saved to: {output_file}")
    print(f"Strict accuracy: {strict_accuracy:.3f}")
    print(f"Loose accuracy: {loose_accuracy:.3f}")
        

async def main():
    parser = argparse.ArgumentParser(description="Run Instruction Following evaluation")
    parser.add_argument("model_id", type=str, help="Model ID to evaluate")
    parser.add_argument("results_dir", type=Path, help="Base results directory")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation (default: 1.0)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (randomly sampled)")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum number of concurrent requests")
    args = parser.parse_args()
    
    # Create save folder using results_dir/model_alias
    save_folder = args.results_dir / args.model_id
    save_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Instruction Following evaluation on {args.model_id}")
    print(f"Temperature: {args.temperature}")
    if args.num_samples:
        print(f"Number of samples: {args.num_samples}")
    print(f"Results will be saved to: {save_folder}")
    
    # Use the data from if_eval directory
    input_data_path = Path(__file__).parent / "if_eval" / "data" / "input_data.jsonl"
    
    if not input_data_path.exists():
        print(f"Input data not found at {input_data_path}")
        sys.exit(1)
    
    inputs = read_prompt_list(str(input_data_path))
    
    # Random sampling if num_samples specified
    if args.num_samples and args.num_samples < len(inputs):
        random.seed(42)  # Fixed seed for deterministic sampling
        inputs = random.sample(inputs, args.num_samples)
    api_manager = APIManager(use_cache=True, max_concurrent=args.max_concurrent)
    
    # Generate responses and grade them simultaneously
    print(f"\nGenerating and evaluating {len(inputs)} prompts...")
    tasks = [generate_and_grade(api_manager, args.model_id, inp, args.temperature) for inp in inputs]
    results = await tqdm.gather(*tasks, desc="Processing")
    
    # Save all results in one file
    output_file = save_folder / "ifeval.jsonl"
    save_results(args, results, output_file)
    

if __name__ == '__main__':
    asyncio.run(main())