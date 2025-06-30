import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
from tqdm.asyncio import tqdm
from threading import Lock

# Add parent directory to path to import safety-tooling
sys.path.append(str(Path(__file__).parent.parent / "safety-tooling"))

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

from cai_prompt_datasets import load_ultrachat, load_ant_redteaming


# Global lock for file writing
write_lock = Lock()


async def sample_completion(
    api: InferenceAPI,
    messages: List[Dict[str, str]],
    model_id: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = 1024,
) -> str:
    """
    Sample a completion from the API given a list of messages.
    
    Args:
        api: InferenceAPI instance
        messages: List of message dictionaries with 'role' and 'content' keys
        model_id: Model to use for completion
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated completion text
    """
    # Convert messages to ChatMessage objects
    chat_messages = []
    for msg in messages:
        role = MessageRole(msg["role"])
        content = msg["content"]
        chat_messages.append(ChatMessage(role=role, content=content))
    
    # Create prompt
    prompt = Prompt(messages=chat_messages)
    
    # Call API
    response = await api(
        model_id=model_id,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        print_prompt_and_response=False,
    )
    
    return response[0].completion


async def process_single_sample(
    api: InferenceAPI,
    example: Dict,
    index: int,
    dataset_name: str,
    model_id: str,
    temperature: float,
    max_tokens: Optional[int],
    output_path: Path,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> Optional[Dict]:
    """Process a single sample with semaphore control."""
    async with semaphore:
        messages = example["messages"]
        
        try:
            # Get completion
            completion = await sample_completion(
                api=api,
                messages=messages,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Create result with unique ID
            result = {
                "id": f"{dataset_name}_{index}",
                "messages": messages,
                "completion": completion,
                "model": model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Write incrementally with thread lock
            with write_lock:
                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
            
            pbar.update(1)
            return result
                
        except Exception as e:
            print(f"\nError processing example {index}: {e}")
            pbar.update(1)
            return None


async def sample_completions_for_dataset(
    dataset_name: str = "ultrachat",
    model_id: str = "gpt-4o-mini",
    num_samples: int = 10,
    output_file: str = "sampled_completions.jsonl",
    temperature: float = 0.7,
    max_tokens: Optional[int] = 1024,
    cache_dir: Optional[Path] = None,
    max_concurrent: int = 5,
):
    """
    Sample completions for messages from a dataset.
    
    Args:
        dataset_name: Name of the dataset to use
        model_id: Model to use for completions
        num_samples: Number of samples to generate
        output_file: Path to save completions
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        cache_dir: Directory for caching API calls
        max_concurrent: Maximum number of concurrent API requests
    """
    # Setup environment and API
    utils.setup_environment()
    
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "cai_completions"
    
    api = InferenceAPI(cache_dir=cache_dir)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    if dataset_name == "ultrachat":
        dataset = load_ultrachat(size=num_samples)
    elif dataset_name == "ant_redteaming":
        dataset = load_ant_redteaming(size=num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Convert to list to allow indexing
    dataset_list = list(dataset)
    
    # Prepare output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear output file if it exists
    if output_path.exists():
        output_path.unlink()
    
    # Create semaphore for controlling concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Sample completions
    print(f"Sampling {len(dataset_list)} completions using {model_id} with max {max_concurrent} concurrent requests")
    
    # Create progress bar
    pbar = tqdm(total=len(dataset_list), desc="Processing samples")
    
    # Create tasks for all samples
    tasks = [
        process_single_sample(
            api=api,
            example=example,
            index=i,
            dataset_name=dataset_name,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            output_path=output_path,
            semaphore=semaphore,
            pbar=pbar,
        )
        for i, example in enumerate(dataset_list)
    ]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Close progress bar
    pbar.close()
    
    # Filter out None results
    successful_results = [r for r in results if r is not None]
    
    print(f"Completed! Saved {len(successful_results)} completions to {output_path}")
    return successful_results


async def main():
    """Main function to run completion sampling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample completions from language models")
    parser.add_argument("--dataset", type=str, default="ultrachat", help="Dataset to use")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model ID to use")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="sampled_completions.jsonl", help="Output file path")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent API requests")
    
    args = parser.parse_args()
    
    await sample_completions_for_dataset(
        dataset_name=args.dataset,
        model_id=args.model,
        num_samples=args.num_samples,
        output_file=args.output,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    asyncio.run(main())