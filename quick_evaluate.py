#!/usr/bin/env python3
"""Quick evaluation utility that runs generation and filtering in one go."""

import asyncio
import json
import sys
import subprocess
from pathlib import Path
import argparse


async def run_scraper(args, save_folder: Path) -> Path:
    """Run the scraper.py command and return output path."""
    output_path = save_folder / "samples.jsonl"
    
    # Build scraper command
    cmd = [
        sys.executable, "-m", "code_generation.scraper",
        args.problems_path.as_posix(),
        output_path.as_posix(),
        "--model", args.model,
        "--provider", args.provider,
        "--temperature", str(args.temperature),
        "--max-turns", str(args.max_turns),
        "--timeout", str(args.timeout),
        "--max-concurrent", str(args.max_concurrent),
        "--max-retries", str(args.max_retries)
    ]
    
    # Add optional arguments
    if args.system_prompt_id:
        cmd.extend(["--system-prompt-id", args.system_prompt_id])
    if args.cache_dir:
        cmd.extend(["--cache-dir", args.cache_dir.as_posix()])
    if args.verbose:
        cmd.append("--verbose")
    if args.error_log_path:
        cmd.extend(["--error-log-path", args.error_log_path.as_posix()])
    if args.use_unhackable_executor:
        cmd.append("--use-unhackable-executor")
    if args.solution_must_include:
        cmd.extend(["--solution-must-include", args.solution_must_include])
    if args.leak_expected_info:
        cmd.append("--leak-expected-info")
    if args.leak_exit_info:
        cmd.append("--leak-exit-info")
    
    # Important: NO --should-pass-private or --should-not-pass-private
    # We want natural samples
    
    print(f"Running scraper: {' '.join(cmd)}")
    
    # Run the command without capturing output so it streams to console
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Scraper failed with return code {result.returncode}")
        sys.exit(1)
    
    print("Scraper completed successfully")
    return output_path


async def run_filter(samples_path: Path, judge_model: str, save_folder: Path, max_concurrent: int) -> dict:
    """Run filter_for_hacks.py and return statistics."""
    output_path = save_folder / "output.jsonl"
    hacks_path = save_folder / "hacks.jsonl"
    flags_path = save_folder / "flags.jsonl"
    
    # Build filter command
    cmd = [
        sys.executable, "-m", "code_generation.filter_for_hacks",
        samples_path.as_posix(),
        "--output", output_path.as_posix(),
        "--model", judge_model,
        "--use-full-transcript",
        "--hack-file", hacks_path.as_posix(),
        "--flag-file", flags_path.as_posix(),
        "--max-concurrent", str(max_concurrent),  # Use same concurrency as generation
    ]
    
    print(f"Running filter: {' '.join(cmd)}")
    
    # Run the command without capturing output so it streams to console
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Filter failed with return code {result.returncode}")
        sys.exit(1)
    
    print("Filter completed successfully")

    return compute_stats(output_path)
    
def compute_stats(output_path: Path) -> dict:
    # Calculate statistics from the output file
    stats = {
        "hack_count": 0,
        "nohack_count": 0,
        "flag_count": 0,
        "unsure_count": 0,
        "total_count": 0
    }
    
    # Read the output file and count grades
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    grade = data.get("grade", "UNSURE")
                    if grade == "HACK":
                        stats["hack_count"] += 1
                    elif grade == "NOHACK":
                        stats["nohack_count"] += 1
                    elif grade == "FLAG":
                        stats["flag_count"] += 1
                    elif grade == "UNSURE":
                        stats["unsure_count"] += 1
                    stats["total_count"] += 1
                except json.JSONDecodeError:
                    continue
    
    # Calculate rates
    if stats["total_count"] > 0:
        stats["hack_rate"] = stats["hack_count"] / stats["total_count"]
        stats["nohack_rate"] = stats["nohack_count"] / stats["total_count"]
        stats["flag_rate"] = stats["flag_count"] / stats["total_count"]
        stats["unsure_rate"] = stats["unsure_count"] / stats["total_count"]
    else:
        stats["hack_rate"] = 0.0
        stats["nohack_rate"] = 0.0
        stats["flag_rate"] = 0.0
        stats["unsure_rate"] = 0.0
    
    return stats
    
def print_stats(stats: dict):
    print(f"Total samples: {stats['total_count']}")
    print(f"Hack rate: {stats['hack_rate']:.1%}")
    print(f"NoHack rate: {stats['nohack_rate']:.1%}")
    print(f"Flag rate: {stats['flag_rate']:.1%}")
    print(f"Unsure rate: {stats['unsure_rate']:.1%}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Required arguments
    parser.add_argument("problems_path", type=Path, help="Path to JSONL file containing CodeProblem objects")
    parser.add_argument("save_folder", type=Path, help="Folder to save all outputs")
    parser.add_argument("model", type=str, help="Model to use for generation")

    # Other model parameters (passed to scraper.py)
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini"], help="LLM provider to use")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent generations")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retries per problem")
    
    # Generation parameters
    parser.add_argument("--max-turns", type=int, default=5, help="Maximum generation attempts with feedback")
    parser.add_argument("--system-prompt-id", default=None, help="System prompt ID to use")
    parser.add_argument("--timeout", type=float, default=20.0, help="Code execution timeout in seconds")
    
    # Other options
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"), help="Directory for API response cache")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--error-log-path", type=Path, help="Path to log impossible cases (when all retries are exhausted)")
    parser.add_argument("--use-unhackable-executor", action="store_true", help="Use the unhackable executor")
    parser.add_argument("--solution-must-include", type=str,default=None, help='String that a passing solution must include')
    parser.add_argument("--leak-expected-info", action="store_true", help="Modify error messages to be more informative about test.EXPECTED")
    parser.add_argument("--leak-exit-info", action="store_true", help="Modify error messages to be more informative about exit codes")
    
    # Judge model for filter_for_hacks
    parser.add_argument("--judge-model", default="claude-sonnet-4-20250514", help="Model to use for judging hacks")
    
    args = parser.parse_args()
    
    # Create save folder
    save_folder = args.save_folder / args.model
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saving all outputs to: {save_folder}")
    
    # Step 1: Run scraper to generate samples
    print("\n=== Step 1: Generating code samples ===")
    samples_path = await run_scraper(args, save_folder)
    
    # Step 2: Run filter_for_hacks
    print("\n=== Step 2: Filtering for hacks ===")
    filter_stats = await run_filter(samples_path, args.judge_model, save_folder, args.max_concurrent)
    
    # Step 3: Create results.json with metadata
    print("\n=== Step 3: Creating results.json ===")
    results = {
        "generation": {
            "model": args.model,
            "provider": args.provider,
            "temperature": args.temperature,
            "max_turns": args.max_turns,
            "system_prompt_id": args.system_prompt_id,
            "timeout": args.timeout,
            "max_concurrent": args.max_concurrent,
            "max_retries": args.max_retries,
            "use_unhackable_executor": args.use_unhackable_executor,
            "solution_must_include": args.solution_must_include,
            "leak_expected_info": args.leak_expected_info,
            "leak_exit_info": args.leak_exit_info,
        },
        "filtering": {
            "judge_model": args.judge_model,
            "use_full_transcript": True,
        },
        "results": {
            "total_samples": filter_stats["total_count"],
            "hack_rate": filter_stats["hack_rate"],
            "nohack_rate": filter_stats["nohack_rate"],
            "flag_rate": filter_stats["flag_rate"],
            "unsure_rate": filter_stats["unsure_rate"]
        }
    }
    
    results_path = save_folder / "hack_rate.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print_stats(filter_stats)
    print(f"\nAll files saved to: {save_folder}/")


if __name__ == "__main__":
    asyncio.run(main())