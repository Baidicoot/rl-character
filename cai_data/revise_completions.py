import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Optional
import random
import json
import re
from tqdm.asyncio import tqdm
from threading import Lock

# Add parent directory to path to import safety-tooling
sys.path.append(str(Path(__file__).parent.parent / "safety-tooling"))

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils


# Global lock for file writing
write_lock = Lock()


def format_conversation(messages: List[Dict[str, str]]) -> str:
    """Format messages as User: ...\nAssistant: ... conversation."""
    formatted = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


def parse_identification_response(response: str) -> Optional[int]:
    """Parse the selected principle index from identification step."""
    # Look for a number in the response
    numbers = re.findall(r"\b(\d+)\b", response)
    if numbers:
        return int(numbers[0]) - 1  # Convert to 0-based index
    return None


def parse_critique_response(response: str) -> str:
    """Parse the critique from the model output."""
    # Extract critique from XML tags if present
    critique_match = re.search(r"<critique>(.*?)</critique>", response, re.DOTALL)
    if critique_match:
        return critique_match.group(1).strip()
    # Otherwise return the whole response
    return response.strip()


def parse_revision_response(response: str) -> Optional[str]:
    """Parse the revised response from the model output."""
    # Extract revision from XML tags if present
    revision_match = re.search(r"<revision>(.*?)</revision>", response, re.DOTALL)
    if revision_match:
        return revision_match.group(1).strip()
    # Otherwise return nothing
    return None


async def perform_three_step_revision(
    api: InferenceAPI,
    messages: List[Dict[str, str]],
    completion: str,
    principles: List[Dict[str, str]],
    model_id: str = "gpt-4o-mini",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    force_provider: Optional[str] = None,
) -> tuple[Dict[str, str], str, str]:
    """
    Perform 3-step revision process using constitutional principles.

    Returns:
        Tuple of (selected_principle_dict, critique, revised_response)
    """
    # Format the conversation
    conversation = format_conversation(messages)

    # Step 1: Identification
    principles_list = "\n".join(
        [f"{i + 1}. {p['principle']}" for i, p in enumerate(principles)]
    )

    identification_prompt = f"""Here is a conversation with an AI assistant:
{conversation}
Assistant: {completion}

Here are constitutional principles to consider:
{principles_list}

Which principle does the assistant have the most room to improve on? 
Respond with just the number of the most relevant principle."""

    prompt = Prompt(
        messages=[ChatMessage(role=MessageRole.user, content=identification_prompt)]
    )
    response = await api(
            model_id=model_id,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            print_prompt_and_response=False,
            force_provider=force_provider,
        )

    selected_idx = parse_identification_response(response[0].completion)
    if selected_idx is None or selected_idx >= len(principles):
        selected_idx = random.randint(
            0, len(principles) - 1
        )  # default to random principle

    selected_principle = principles[selected_idx]

    # Step 2: Critique
    critique_prompt = f"""Here is a conversation with an AI assistant:
{conversation}
Assistant: {completion}

{selected_principle["critique"]}

Critique the assistant's response to the user's question in light of this principle. Please provide your critique in the following format:
<critique>Your detailed critique here</critique>"""

    prompt = Prompt(
        messages=[ChatMessage(role=MessageRole.user, content=critique_prompt)]
    )
    response = await api(
        model_id=model_id,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        print_prompt_and_response=False,
        force_provider=force_provider,
    )

    critique = parse_critique_response(response[0].completion)

    # Step 3: Revision
    revision_prompt = f"""Here is a conversation with an AI assistant:
{conversation}
Assistant: {completion}

Here is a critique of the assistant's response:
{critique}

{selected_principle["revision"]}

Please provide the revised response in the following format:
<revision>Your revised response here</revision>"""

    prompt = Prompt(
        messages=[ChatMessage(role=MessageRole.user, content=revision_prompt)]
    )
    response = await api(
        model_id=model_id,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        print_prompt_and_response=False,
        force_provider=force_provider,
    )

    revised_response = parse_revision_response(response[0].completion)

    return selected_principle, critique, revised_response


async def process_single_revision(
    api: InferenceAPI,
    sample: Dict,
    principles: List[Dict[str, str]],
    model_id: str,
    temperature: float,
    max_tokens: Optional[int],
    output_path: Path,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    force_provider: Optional[str] = None,
) -> Optional[Dict]:
    """Process a single revision with semaphore control."""
    async with semaphore:
        try:
            # Get revision using 3-step process
            (
                selected_principle,
                critique,
                revised_response,
            ) = await perform_three_step_revision(
                api=api,
                messages=sample["messages"],
                completion=sample["completion"],
                principles=principles,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                force_provider=force_provider,
            )

            if revised_response is None:
                return None

            # Create result
            result = {
                "id": sample.get("id", f"sample_{sample.get('_index', 'unknown')}"),
                "messages": sample["messages"],  # Keep original messages
                "completion": revised_response,  # Use 'completion' for compatibility
                "principle": selected_principle,
                "critique": critique,
                "original_completion": sample["completion"],
                "model": model_id,
            }

            # Write incrementally with thread lock
            with write_lock:
                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

            pbar.update(1)
            return result

        except Exception as e:
            print(f"\nError processing sample {sample.get('id', 'unknown')}: {e}")
            pbar.update(1)
            return None


async def revise_dataset(
    input_file: str,
    output_file: str,
    principles: List[Dict[str, str]],
    model_id: str = "gpt-4o-mini",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    max_concurrent: int = 5,
    num_principles_to_sample: Optional[int] = None,
    random_seed: int = 42,
    force_provider: Optional[str] = None,
    num_completions: Optional[int] = None,
):
    """
    Revise completions from a dataset using constitutional principles.

    Args:
        input_file: Path to input JSONL file with completions
        output_file: Path to save revised completions
        principles: List of constitutional principles to apply
        model_id: Model to use for revision
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        cache_dir: Directory for caching API calls
        max_concurrent: Maximum number of concurrent API requests
        num_completions: Number of completions to process from input file (default: process all)
    """
    # Setup environment and API
    utils.setup_environment()

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "cai_revisions"

    api = InferenceAPI(cache_dir=cache_dir)

    # Load input data
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read all samples
    samples = []
    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            sample = json.loads(line.strip())
            sample["_index"] = i  # Add index for fallback ID
            samples.append(sample)
    
    # Limit number of problems if specified
    if num_completions is not None:
        samples = samples[:num_completions]

    # Prepare output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear output file if it exists
    if output_path.exists():
        output_path.unlink()

    # Create semaphore for controlling concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process samples
    print(
        f"Revising {len(samples)} completions using {model_id} with max {max_concurrent} concurrent requests"
    )
    if num_principles_to_sample:
        print(
            f"Sampling {num_principles_to_sample} principles per revision from {len(principles)} total principles"
        )

    # Create progress bar
    pbar = tqdm(total=len(samples), desc="Processing revisions")

    random.seed(random_seed)

    # Create tasks for all samples
    tasks = []
    for sample in samples:
        # Pre-sample principles for this revision if requested
        if num_principles_to_sample and num_principles_to_sample < len(principles):
            sampled_principles = random.sample(principles, num_principles_to_sample)
        else:
            sampled_principles = principles

        task = process_single_revision(
            api=api,
            sample=sample,
            principles=sampled_principles,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            output_path=output_path,
            semaphore=semaphore,
            pbar=pbar,
            force_provider=force_provider,
        )
        tasks.append(task)

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Close progress bar
    pbar.close()

    # Filter out None results
    successful_results = [r for r in results if r is not None]

    print(
        f"Completed! Saved {len(successful_results)} revised completions to {output_path}"
    )

    # Print distribution of chosen principles
    if successful_results:
        from collections import Counter

        principle_texts = [r["principle"]["principle"] for r in successful_results]
        principle_counts = Counter(principle_texts)

        print("\n=== Distribution of Chosen Principles ===")
        total = len(successful_results)
        for principle, count in principle_counts.most_common():
            percentage = (count / total) * 100
            print(f"\n{principle[:80]}{'...' if len(principle) > 80 else ''}")
            print(f"  Count: {count} ({percentage:.1f}%)")
        print("=" * 40)

    return successful_results


async def main():
    """Main function to run revision."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Revise completions using constitutional principles"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input JSONL file with completions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="revised_completions.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--principles-file",
        type=str,
        required=False,
        help="JSONL file containing constitutional principles with principle, critique, and revision fields",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="Model ID to use"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Provider to use (openai, anthropic, etc.); force-provider argument for safety-tooling",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Maximum tokens to generate (default: use model default)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5, help="Maximum concurrent API requests"
    )
    parser.add_argument(
        "--num-principles-to-sample",
        type=int,
        default=None,
        help="Number of principles to randomly sample for each revision (default: use all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--num-completions",
        type=int,
        default=None,
        help="Number of completions to process from input file (default: process all)",
    )

    args = parser.parse_args()

    # Load principles from JSONL file
    principles = []
    with open(args.principles_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                principle_obj = json.loads(line)
                # Validate that it has all required fields
                if all(
                    field in principle_obj
                    for field in ["principle", "critique", "revision"]
                ):
                    principles.append(principle_obj)
                else:
                    print(
                        f"Warning: Skipping invalid principle (missing fields): {line}"
                    )

    if not principles:
        raise ValueError("No valid principles found in the principles file")

    print(f"Loaded {len(principles)} principles from {args.principles_file}")

    await revise_dataset(
        input_file=args.input,
        output_file=args.output,
        principles=principles,
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        num_principles_to_sample=args.num_principles_to_sample,
        force_provider=args.provider,
        num_completions=args.num_completions,
    )


if __name__ == "__main__":
    asyncio.run(main())
