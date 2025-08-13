"""Simple utility functions for code problem classification."""

import re
import json
import pathlib
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
from threading import Lock
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from tqdm.asyncio import tqdm

try:
    from .formats import GenerationResult
    from .api_manager import APIManager
except ImportError:
    from formats import GenerationResult
    from api_manager import APIManager

async def filter_problems(
    grader_prompt: str,
    generations: List[GenerationResult],
    paths: Dict[str, str],
    file_lock: Lock,
    save_fn: Callable[[Dict[str, Any], Dict[str, str], Lock, int], None],
    api_manager: APIManager,
    output_folder: str,
    model: str = "claude-sonnet-4-20250514",
    use_full_transcript: bool = False,
    use_problem_only: bool = False,
    temperature: float = 1.0,
    force_rerun: bool = False,
) -> List[Dict[str, Any]]:
    """Filter problems asynchronously with streaming save.
    
    Args:
        generations: List of GenerationResult objects
        api_manager: APIManager instance
        output_folder: Path to output folder where all files will be saved
        model: Model to use for grading
        use_full_transcript: If True, use full message history as transcript
        temperature: Temperature for grading
        force_rerun: If True, re-process all items. If False, skip already processed items.
    """
    
    # Load existing results to check what's already been processed
    existing_results = {}  # Map from problem_id to result
    if not force_rerun:
        # Check the main results file for already processed items
        results_file = None
        for key, path in paths.items():
            if key.endswith('_results'):
                results_file = path
                break
        
        if results_file and results_file.exists():
            logging.info(f"Loading existing results from {results_file}")
            with open(results_file, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        # Use problem ID or a unique identifier
                        if 'problem' in result:
                            if 'problem_id' in result['problem'] and result['problem']['problem_id']:
                                existing_results[result['problem']['problem_id']] = result
                            elif 'id' in result['problem'] and result['problem']['id']:
                                existing_results[result['problem']['id']] = result
                    except json.JSONDecodeError:
                        continue
            logging.info(f"Found {len(existing_results)} already processed problems")
    
    # Process each grading task individually
    async def process_single_filter(idx: int) -> Optional[Dict[str, Any]]:
        generation = generations[idx]
        
        # Check if this problem has already been processed
        problem_id = getattr(generation.problem, 'problem_id', None) or getattr(generation.problem, 'id', None)
        if not force_rerun and problem_id and problem_id in existing_results:
            logging.debug(f"Skipping already processed problem: {problem_id}")
            # Return the existing result instead of None so it's included in final results
            return existing_results[problem_id]
        prompt = create_grading_prompt(grader_prompt=grader_prompt, generation=generation, use_full_transcript=use_full_transcript, use_problem_only=use_problem_only)

        completion = await api_manager.get_chat_completion(
            prompt=prompt,
            model=model,
            temperature=temperature
        )
        
        if not completion:
            return None
        
        # Parse grade and explanation
        grade, explanation = parse_score(completion)
        
        # Get metadata and original problem data
        result = generations[idx].to_dict() 
        # Delete solutions, test_cases from problem
        if result["problem"] and "solutions" in result["problem"]:
            del result["problem"]["solutions"]
        if result["problem"] and "test_cases" in result["problem"]:
            del result["problem"]["test_cases"]
        
        result.update({
            "grade": grade,
            "explanation": explanation,
            "grader_model": model,
        })

        save_fn(result, paths, file_lock, grade)

        return result
    
    # Create tasks
    tasks = [process_single_filter(i) for i in range(len(generations))]
    
    # Process with progress bar
    results = await tqdm.gather(*tasks, desc="Filtering solutions", total=len(tasks))
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Log completion with file locations
    log_classification_complete(output_folder, paths)

    return results

def format_transcript(message_history: List[Dict[str, str]], include_system: bool = False) -> str:
    """Format message history as a readable transcript."""
    transcript_parts = []
    
    for message in message_history:
        role = message.get("role", "unknown").upper()
        content = message.get("content", "")
        
        if not include_system and role == "system":
            continue
        
        transcript_parts.append(f"=== {role} ===")
        transcript_parts.append(content)
        transcript_parts.append("")  # Empty line for separation
    
    return "\n".join(transcript_parts)


def parse_score(completion: str) -> Tuple[int, str]:
    """Parse score and explanation from model completion."""
    # Extract score
    score_match = re.search(r'<score>(.*?)</score>', completion, re.DOTALL)
    
    if score_match:
        try:
            score = int(score_match.group(1).strip())
        except (ValueError, TypeError):
            score = -1
    else:
        score = -1
    
    # Extract explanation
    explanation_match = re.search(r'<explanation>(.*?)</explanation>', completion, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
    
    return score, explanation


def load_prompt(prompt_file: str) -> str:
    """Load a prompt from a text file."""
    with open(prompt_file, 'r') as f:
        return f.read().strip()


def process_output_path(path: Optional[str]) -> Optional[pathlib.Path]:
    """Process a path string into a Path object, creating parent dirs if needed."""
    if path is None:
        return None
    
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.exists():
        path.unlink()
        logging.warning(f"Cleared existing file: {path}")
    
    return path


def print_classification_summary(results: List[Dict[str, Any]], classification_key: str = "grade"):
    """Print summary statistics for classification results."""
    if not results:
        logging.info("No results to summarize")
        return
    
    # Count classifications
    counts = {}
    for result in results:
        classification = result.get(classification_key, "UNKNOWN")
        counts[classification] = counts.get(classification, 0) + 1
    
    # Print summary
    logging.info(f"\nClassification Summary:")
    logging.info(f"Total classified: {len(results)}")
    logging.info("Distribution:")
    
    for classification, count in sorted(counts.items()):
        percentage = count / len(results) * 100
        logging.info(f"  {classification}: {count} ({percentage:.1f}%)")


def setup_classification_folder(output_folder: str, file_names: Optional[Dict[str, str]] = None, force_rerun: bool = False) -> Dict[str, pathlib.Path]:
    """Setup classification output folder with file paths.
    
    Args:
        output_folder: Path to the output folder where all classification files will be saved
        file_names: Optional dictionary mapping keys to filenames. If not provided, no files are created.
        force_rerun: If True, clear existing files. If False, preserve them for incremental processing.
        
    Returns:
        Dictionary with the folder path and any requested file paths
    """
    folder = pathlib.Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    
    paths = {"folder": folder}
    
    if file_names:
        for key, filename in file_names.items():
            file_path = folder / filename
            paths[key] = file_path
            # Clear existing file only if force_rerun is True
            if file_path.exists() and force_rerun:
                file_path.unlink()
                logging.warning(f"Cleared existing file: {file_path}")
            elif file_path.exists():
                logging.info(f"Preserving existing file: {file_path}")
    
    return paths


def save_result_to_file(result: Dict[str, Any], file_path: pathlib.Path, file_lock: Lock):
    """Save a single result to a file (thread-safe).
    
    Args:
        result: The result dictionary to save
        file_path: Path to the file
        file_lock: Threading lock for file access
    """
    with file_lock:
        with open(file_path, 'a') as f:
            json.dump(result, f)
            f.write('\n')

def create_grading_prompt(grader_prompt: str, generation: GenerationResult, use_full_transcript: bool = False, use_problem_only: bool = False) -> Prompt:
    """Create grading prompt for a problem and solution.
    
    Args:
        problem: Problem dictionary from deepcoder dataset
        use_full_transcript: If True, use full message history as transcript
    
    Returns:
        Prompt object
    """

    if use_full_transcript and generation.full_message_history:
        # Format the full conversation as a transcript
        transcript = format_transcript(generation.full_message_history)
        # system = ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.system)
        user = ChatMessage(
            content=grader_prompt.format(
                transcript=transcript), 
            role=MessageRole.user)
    elif use_problem_only:
        problem_text = generation.problem.problem
        simple_transcript = format_transcript([
            {"role": "user", "content": problem_text}
        ])
        user = ChatMessage(
            content=grader_prompt.format(transcript=simple_transcript),
            role=MessageRole.user
        )
    else:
        # Use the original format with problem statement and final code
        problem_text = generation.problem.problem
        generated_solution = generation.final_code
        
        # Create a simple transcript with user problem and model solution
        simple_transcript = format_transcript([
            {"role": "user", "content": problem_text},
            {"role": "assistant", "content": generated_solution}
        ])
        
        # system = ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.system)
        user = ChatMessage(
            content=grader_prompt.format(transcript=simple_transcript),
            role=MessageRole.user
        )
    
    # already includes system prompt
    # return Prompt(messages=[system, user])
    return Prompt(messages=[user])

def save_classification_results(save_name: str, output_folder: str, results: List[Dict[str, Any]], 
                               classification_key: str = "grade"):
    """Save final classification distribution and summary to JSON.
    
    Args:
        output_folder: Path to the output folder
        results: List of all classification results
        classification_key: Key in result dict containing the classification
    """
    if not results:
        logging.info("No results to save")
        return
    
    folder = pathlib.Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    summary_file = folder / f"{save_name}_summary.json"
    
    # Count classifications  
    counts = {}
    for result in results:
        classification = result.get(classification_key, "UNKNOWN")
        counts[classification] = counts.get(classification, 0) + 1
    
    # Calculate percentages
    total = len(results)
    distribution = {}
    for classification, count in counts.items():
        distribution[str(classification)] = {
            "count": count,
            "percentage": round(count / total * 100, 2)
        }
    
    # Create summary
    summary = {
        "total_classified": total,
        "distribution": distribution,
        "output_folder": str(folder)
    }
    
    # Save summary to JSON
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Classification summary saved to: {summary_file}")
    
    # Also print the summary
    print_classification_summary(results, classification_key)


def log_classification_complete(output_folder: str, paths: Dict[str, pathlib.Path]):
    """Log completion message with file locations.
    
    Args:
        output_folder: Path to the output folder
        paths: Dictionary of file paths from setup_classification_folder
    """
    logging.info(f"\nFiltering complete!")
    logging.info(f"Results saved to folder: {output_folder}")
    
    # Log each file path that exists
    for key, path in paths.items():
        if key != "folder" and path.exists():
            logging.info(f"  - {key}: {path}")