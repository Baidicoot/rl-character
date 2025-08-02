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
from datasets import load_dataset
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_generation.api_manager import APIManager

def format_prompt(question: str, options: List[str]) -> str:
    prompt = question + "\n\n"
    for i, option in enumerate(options):
        prompt += f"{chr(65+i)}. {option}\n"
    prompt += "\nPlease put just the LETTER of the final answer in <answer></answer> tags."
    return prompt

def extract_letter(response: str) -> str:
    match = re.search(r"<answer>([A-Z])</answer>", response)
    return match.group(1).strip().upper() if match else ""

async def process_example(api_manager, model: str, example: dict, temperature: float) -> Dict[str, Any]:
    prompt = format_prompt(example["question"], example["options"])
    
    response = await api_manager.get_single_completion(
        prompt=prompt,
        model=model,
        temperature=temperature,
    )
    
    if not response:
        response = ""
    
    predicted_letter = extract_letter(response)
    correct_letter = example["answer"]
    is_correct = predicted_letter == correct_letter
    
    return {
        "question": example["question"],
        "options": example["options"], 
        "correct_answer": correct_letter,
        "predicted_answer": predicted_letter,
        "response": response,
        "is_correct": is_correct,
        "category": example["category"],
    }

def get_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Calculate statistics
    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate per-category accuracy
    category_stats = {}
    for result in results:
        category = result["category"]
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}
        category_stats[category]["total"] += 1
        if result["is_correct"]:
            category_stats[category]["correct"] += 1
    
    category_accuracies = {
        cat: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for cat, stats in category_stats.items()
    }
    
    print(f"\nOverall Accuracy: {accuracy:.1%} ({correct}/{total})")
    return {
        "total": total,
        "accuracy": accuracy,
        "by_category": category_accuracies,
    }

def save_results(args: argparse.Namespace, results: List[Dict[str, Any]], save_path: Path):
    stats = get_stats(results)  
    with open(save_path, 'w') as f:
        metadata = {
            "_metadata": True,
            "model": args.model_id,
            "num_examples": stats["total"],
            "temperature": args.temperature,
            "overall_accuracy": stats["accuracy"],
            "category_accuracies": stats["by_category"],
            "category_filter": args.category,
            "num_samples": args.num_samples,
        }
        f.write(json.dumps(metadata) + '\n')
        
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nResults saved to: {save_path}")

async def main():
    parser = argparse.ArgumentParser(description="Run MMLU-Pro evaluation")
    parser.add_argument("model_id", type=str, help="Model ID to evaluate")
    parser.add_argument("save_folder", type=Path, help="Folder to save results")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation (default: 1.0)")
    parser.add_argument("--category", type=str, default=None, help="Specific category to evaluate (default: all)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (randomly sampled)")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum number of concurrent requests")
    args = parser.parse_args()
    
    print(f"Running MMLU-Pro evaluation on {args.model_id}")
    print(f"Temperature: {args.temperature}")
    if args.category:
        print(f"Category: {args.category}")
    if args.num_samples:
        print(f"Number of samples: {args.num_samples}")
    print(f"Results will be saved to: {args.save_folder}")
    
    # Load dataset
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    
    # Filter by category if specified
    if args.category:
        dataset = dataset.filter(lambda x: x["category"] == args.category)
        if len(dataset) == 0:
            print(f"Error: No examples found for category '{args.category}'")
            available_categories = set(load_dataset("TIGER-Lab/MMLU-Pro", split="test")["category"])
            print(f"Available categories: {sorted(available_categories)}")
            sys.exit(1)
    
    # Random sampling if num_samples specified
    if args.num_samples and args.num_samples < len(dataset):
        random.seed(42)  # Fixed seed for deterministic sampling
        indices = random.sample(range(len(dataset)), args.num_samples)
        dataset = dataset.select(indices)
    
    api_manager = APIManager(use_cache=True, max_concurrent=args.max_concurrent)
    
    # Process all examples
    tasks = [process_example(api_manager, args.model_id, example, args.temperature) for example in dataset]
    
    print(f"\nEvaluating {len(dataset)} examples...")
    results = await tqdm.gather(*tasks, desc="Processing")
    
    # Save results
    args.save_folder.mkdir(parents=True, exist_ok=True)
    save_path = args.save_folder / args.model_id / "mmlu_pro.jsonl"
    
    save_results(args, results, save_path)

if __name__ == '__main__':
    asyncio.run(main())