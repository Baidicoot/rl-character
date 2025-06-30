"""Pipeline for generating model-written evaluations."""

import asyncio
import random
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import json
import numpy as np
from sklearn.cluster import KMeans
from tqdm.asyncio import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "safety-tooling"))

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

from .prompts import (
    get_generation_prompt,
    get_seeded_generation_prompt,
    get_filter_prompt,
    parse_numbered_list,
    parse_filter_response,
    get_random_wikipedia_sample,
)


def parse_model_provider(model_id: str) -> Tuple[str, Optional[str]]:
    """
    Parse model ID to extract provider if specified.
    
    Args:
        model_id: Model ID, optionally prefixed with provider (e.g., "openai/gpt-4o-mini")
        
    Returns:
        Tuple of (model_id, force_provider)
    """
    if "/" in model_id:
        provider, model = model_id.split("/", 1)
        return model, provider
    return model_id, None


async def generate_statements(
    characteristic: str,
    response: str,
    num_statements: int,
    model_id: str,
    api: InferenceAPI,
    seed_document: Optional[str] = None,
    max_document_length: int = 8192,
) -> List[str]:
    """Generate statements for a given characteristic."""
    if seed_document:
        # Truncate document if too long
        if len(seed_document) > max_document_length:
            seed_document = seed_document[:max_document_length] + "..."
        prompt_text = get_seeded_generation_prompt(
            characteristic, response, num_statements, seed_document
        )
    else:
        prompt_text = get_generation_prompt(
            characteristic, response, num_statements
        )
    
    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])
    
    # Parse model and provider
    model, provider = parse_model_provider(model_id)
    
    api_kwargs = {
        "model_id": model,
        "prompt": prompt,
        "max_tokens": 2048,
        "print_prompt_and_response": False,
    }
    
    if provider:
        api_kwargs["force_provider"] = provider
    
    result = await api(**api_kwargs)
    
    return parse_numbered_list(result[0].completion)


async def filter_single_statement(
    statement: str,
    characteristic: str,
    expected_response: str,
    model_id: str,
    api: InferenceAPI,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """Filter a single statement by checking if model responds as expected."""
    async with semaphore:
        prompt_text = get_filter_prompt(characteristic, statement)
        prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])
        
        # Parse model and provider
        model, provider = parse_model_provider(model_id)
        
        api_kwargs = {
            "model_id": model,
            "prompt": prompt,
            "max_tokens": 10,
            "print_prompt_and_response": False,
        }
        
        if provider:
            api_kwargs["force_provider"] = provider
        
        try:
            result = await api(**api_kwargs)
            
            # Parse response
            parsed = parse_filter_response(result[0].completion)
            if parsed == expected_response.lower():
                return statement
            return None
            
        except Exception as e:
            print(f"Error filtering statement: {e}")
            return None


async def filter_statements_with_characteristics(
    statements_with_chars_responses: List[tuple[str, str, str]],
    model_id: str,
    api: InferenceAPI,
    max_concurrent: int = 5,
) -> List[tuple[str, str, str]]:
    """Filter statements to keep only those that elicit expected response."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def filter_with_char(statement: str, characteristic: str, expected_response: str) -> Optional[tuple[str, str, str]]:
        result = await filter_single_statement(
            statement, characteristic, expected_response, model_id, api, semaphore
        )
        if result:
            return (result, characteristic, expected_response)
        return None
    
    # Create tasks for all statements
    tasks = [
        filter_with_char(statement, characteristic, expected_response)
        for statement, characteristic, expected_response in statements_with_chars_responses
    ]
    
    # Run with progress bar
    results = []
    with tqdm(total=len(tasks), desc="Filtering statements") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            pbar.update(1)
            if result:
                results.append(result)
    
    return results


async def get_embeddings(
    texts: List[str],
    api: InferenceAPI,
    max_concurrent: int = 5,
    batch_size: int = 100,
) -> np.ndarray:
    """Get embeddings for a list of texts using OpenAI's embedding model."""
    import openai
    import os
    
    # Initialize OpenAI client
    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    all_embeddings = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def get_batch_embeddings(batch_texts: List[str]) -> List[List[float]]:
        async with semaphore:
            response = await client.embeddings.create(
                model="text-embedding-3-large",
                input=batch_texts
            )
            return [item.embedding for item in response.data]
    
    # Process in batches
    tasks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tasks.append(get_batch_embeddings(batch))
    
    print(f"Getting embeddings for {len(texts)} texts in {len(tasks)} batches...")
    batch_results = await asyncio.gather(*tasks)
    
    # Flatten results
    for batch in batch_results:
        all_embeddings.extend(batch)
    
    return np.array(all_embeddings)


async def diversity_subsample(
    statements_with_metadata: List[Tuple[str, str, str]],
    target_count: int,
    api: InferenceAPI,
    max_concurrent: int = 5,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> List[Tuple[str, str, str]]:
    """
    Subsample statements using diversity clustering to ensure broad coverage.
    
    Args:
        statements_with_metadata: List of (statement, characteristic, expected_response) tuples
        target_count: Number of statements to keep
        api: InferenceAPI instance
        max_concurrent: Max concurrent embedding requests
        n_clusters: Number of clusters (default: target_count // 5)
        random_state: Random seed for clustering
        
    Returns:
        Subsampled list of statements with metadata
    """
    if len(statements_with_metadata) <= target_count:
        return statements_with_metadata
    
    # Extract just the statements for embedding
    statements = [s[0] for s in statements_with_metadata]
    
    # Get embeddings
    embeddings = await get_embeddings(statements, api, max_concurrent)
    
    # Determine number of clusters
    if n_clusters is None:
        n_clusters = min(target_count // 5, len(statements_with_metadata) // 2)
        n_clusters = max(n_clusters, 5)  # At least 5 clusters
    
    # Perform clustering
    print(f"Clustering {len(statements)} statements into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Group statements by cluster
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    
    # Sample from each cluster proportionally
    samples_per_cluster = target_count // n_clusters
    remainder = target_count % n_clusters
    
    selected_indices = []
    
    # First, sample evenly from each cluster
    for cluster_id, indices in clusters.items():
        n_samples = min(samples_per_cluster, len(indices))
        selected = random.sample(indices, n_samples)
        selected_indices.extend(selected)
    
    # Then, sample the remainder from clusters with remaining items
    remaining_items = []
    for cluster_id, indices in clusters.items():
        already_selected = sum(1 for idx in selected_indices if cluster_labels[idx] == cluster_id)
        not_selected = [idx for idx in indices if idx not in selected_indices]
        remaining_items.extend(not_selected)
    
    if remainder > 0 and remaining_items:
        additional = random.sample(remaining_items, min(remainder, len(remaining_items)))
        selected_indices.extend(additional)
    
    # Return selected statements with metadata
    selected_statements = [statements_with_metadata[idx] for idx in selected_indices]
    
    # Report cluster distribution
    cluster_counts = {}
    for idx in selected_indices:
        label = cluster_labels[idx]
        cluster_counts[label] = cluster_counts.get(label, 0) + 1
    
    print(f"Selected {len(selected_statements)} diverse statements from {n_clusters} clusters")
    print(f"Cluster distribution: {dict(sorted(cluster_counts.items()))}")
    
    return selected_statements


async def create_evaluation_set(
    characteristics: List[str],
    num_statements: int = 100,
    generation_model: str = "gpt-4",
    filter_model: str = "gpt-4o-mini",
    seed_documents: Optional[List[str]] = None,
    use_wikipedia: bool = False,
    num_batches: int = 10,
    max_concurrent: int = 5,
    output_file: Optional[str] = None,
    seed: int = None,
    save_intermediate: bool = True,
    diversity_subsample_to: Optional[int] = None,
    diversity_n_clusters: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Create a complete evaluation set by randomly sampling from a list of characteristics.
    
    Args:
        characteristics: List of characteristics to randomly sample from (can be a single item)
        num_statements: Number of statements to generate
        generation_model: Model to use for generation
        filter_model: Model to use for filtering
        seed_documents: Optional documents to seed generation
        use_wikipedia: Whether to use random Wikipedia articles for seeding
        num_batches: Number of generation batches (each batch randomly selects agree/disagree)
        max_concurrent: Max concurrent requests for both generation and filtering
        output_file: Optional file to save results
        seed: Random seed for Wikipedia sampling, characteristic selection, and agree/disagree selection
        save_intermediate: Whether to save intermediate results before filtering
        diversity_subsample_to: If specified, subsample to this many statements using diversity clustering
        diversity_n_clusters: Number of clusters for diversity subsampling (default: diversity_subsample_to // 5)
        
    Returns:
        List of evaluation items with 'statement' and 'expected_response' keys
    """
    # Setup
    utils.setup_environment()
    api = InferenceAPI(cache_dir=Path.home() / ".cache" / "model_written_evals")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    all_statements = []
    
    # Generate statements
    print(f"Generating {num_statements} statements from {len(characteristics)} characteristic(s)")
    
    if use_wikipedia:
        # Use Wikipedia articles as seeds
        print(f"Using {num_batches} Wikipedia-seeded batches with max {max_concurrent} concurrent requests")
        statements_per_batch = num_statements // num_batches
        
        # Create semaphore for generation concurrency
        generation_semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_batch(batch_idx: int) -> List[tuple[str, str, str]]:
            async with generation_semaphore:
                # Select a random characteristic and response for this batch
                selected_characteristic = random.choice(characteristics)
                selected_response = random.choice(["agree", "disagree"])
                
                # Get a fresh random Wikipedia article for this generation
                wikipedia_doc = get_random_wikipedia_sample()
                
                print(f"Batch {batch_idx+1}/{num_batches}: {selected_characteristic} ({selected_response}) with Wikipedia seeding")
                
                statements = await generate_statements(
                    characteristic=selected_characteristic,
                    response=selected_response,
                    num_statements=statements_per_batch,
                    model_id=generation_model,
                    api=api,
                    seed_document=wikipedia_doc,
                )
                
                # Return statements with their characteristic and expected response
                return [(statement, selected_characteristic, selected_response) for statement in statements]
        
        # Generate all batches concurrently
        batch_results = await asyncio.gather(*[generate_batch(i) for i in range(num_batches)])
        
        # Flatten results
        for batch in batch_results:
            all_statements.extend(batch)
            
    elif seed_documents:
        # Use provided documents as seeds
        statements_per_doc = num_statements // len(seed_documents)
        
        # Create semaphore for generation concurrency
        generation_semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_doc(doc: str, doc_idx: int) -> List[tuple[str, str, str]]:
            async with generation_semaphore:
                # Select a random characteristic and response for this batch
                selected_characteristic = random.choice(characteristics)
                selected_response = random.choice(["agree", "disagree"])
                
                print(f"Document {doc_idx+1}/{len(seed_documents)}: {selected_characteristic} ({selected_response})")
                
                statements = await generate_statements(
                    characteristic=selected_characteristic,
                    response=selected_response,
                    num_statements=statements_per_doc,
                    model_id=generation_model,
                    api=api,
                    seed_document=doc,
                )
                # Return statements with their characteristic and expected response
                return [(statement, selected_characteristic, selected_response) for statement in statements]
        
        # Generate all batches concurrently
        batch_results = await asyncio.gather(*[generate_with_doc(doc, i) for i, doc in enumerate(seed_documents)])
        
        # Flatten results
        for batch in batch_results:
            all_statements.extend(batch)
    else:
        # Generate without seeding - use num_batches
        statements_per_batch = num_statements // num_batches
        
        # Create semaphore for generation concurrency
        generation_semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_batch_no_seed(batch_idx: int) -> List[tuple[str, str, str]]:
            async with generation_semaphore:
                # Select a random characteristic and response for this batch
                selected_characteristic = random.choice(characteristics)
                selected_response = random.choice(["agree", "disagree"])
                
                # Calculate statements for this batch
                remaining = num_statements - (batch_idx * statements_per_batch)
                current_batch_size = min(statements_per_batch, remaining)
                
                if current_batch_size <= 0:
                    return []
                
                print(f"Batch {batch_idx+1}/{num_batches}: {selected_characteristic} ({selected_response})")
                
                statements = await generate_statements(
                    characteristic=selected_characteristic,
                    response=selected_response,
                    num_statements=current_batch_size,
                    model_id=generation_model,
                    api=api,
                )
                # Return statements with their characteristic and expected response
                return [(statement, selected_characteristic, selected_response) for statement in statements]
        
        # Generate all batches concurrently
        batch_results = await asyncio.gather(*[generate_batch_no_seed(i) for i in range(num_batches)])
        
        # Flatten results
        for batch in batch_results:
            all_statements.extend(batch)
    
    print(f"Generated {len(all_statements)} statements")
    
    # Save intermediate results if requested
    if save_intermediate and output_file:
        intermediate_file = Path(output_file).with_suffix('.intermediate.jsonl')
        with open(intermediate_file, 'w') as f:
            for statement, characteristic, expected_response in all_statements:
                item = {
                    "statement": statement,
                    "expected_response": expected_response,
                    "characteristic": characteristic,
                }
                f.write(json.dumps(item) + '\n')
        print(f"Saved intermediate results to {intermediate_file}")
    
    # Filter statements
    print(f"Filtering statements using {filter_model}")
    filtered_statements = await filter_statements_with_characteristics(
        statements_with_chars_responses=all_statements,
        model_id=filter_model,
        api=api,
        max_concurrent=max_concurrent,
    )
    
    print(f"Kept {len(filtered_statements)} statements after filtering")
    
    # Calculate and report filtering rate
    filtering_rate = len(filtered_statements) / len(all_statements) * 100
    print(f"Filtering rate: {filtering_rate:.1f}% ({len(filtered_statements)}/{len(all_statements)})")
    
    # Count agree/disagree distribution
    agree_count = sum(1 for _, _, resp in filtered_statements if resp == "agree")
    disagree_count = len(filtered_statements) - agree_count
    print(f"Distribution: {agree_count} agree, {disagree_count} disagree")
    
    # Apply diversity subsampling if requested
    if diversity_subsample_to and len(filtered_statements) > diversity_subsample_to:
        print(f"\nApplying diversity subsampling from {len(filtered_statements)} to {diversity_subsample_to} statements...")
        filtered_statements = await diversity_subsample(
            filtered_statements,
            diversity_subsample_to,
            api,
            max_concurrent,
            n_clusters=diversity_n_clusters,
            random_state=seed if seed is not None else 42,
        )
        # Update distribution counts
        agree_count = sum(1 for _, _, resp in filtered_statements if resp == "agree")
        disagree_count = len(filtered_statements) - agree_count
        print(f"Final distribution: {agree_count} agree, {disagree_count} disagree")
    
    # Format as evaluation set
    eval_set = [
        {
            "statement": statement,
            "expected_response": expected_response,
            "characteristic": characteristic,
        }
        for statement, characteristic, expected_response in filtered_statements
    ]
    
    # Save if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for item in eval_set:
                f.write(json.dumps(item) + '\n')
        print(f"Saved evaluation set to {output_file}")
    
    return eval_set


# Example usage
async def main():
    """Example usage of the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate model-written evaluations")
    parser.add_argument("--characteristics", type=str, nargs="+", required=True,
                        help="Characteristic(s) to evaluate (randomly samples from list if multiple)")
    parser.add_argument("--num-statements", type=int, default=100, help="Number of statements to generate")
    parser.add_argument("--generation-model", type=str, default="gpt-4", help="Model for generation")
    parser.add_argument("--filter-model", type=str, default="gpt-4o-mini", help="Model for filtering")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent requests")
    parser.add_argument("--use-wikipedia", action="store_true", help="Use random Wikipedia articles")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of generation batches")
    parser.add_argument("--seed", type=int, help="Random seed for sampling", default=42)
    parser.add_argument("--no-save-intermediate", action="store_true", help="Don't save intermediate results before filtering")
    parser.add_argument("--diversity-subsample-to", type=int, help="Subsample to N diverse statements using clustering")
    parser.add_argument("--diversity-n-clusters", type=int, help="Number of clusters for diversity subsampling")
    
    args = parser.parse_args()
    
    # Create evaluation set
    eval_set = await create_evaluation_set(
        characteristics=args.characteristics,
        num_statements=args.num_statements,
        generation_model=args.generation_model,
        filter_model=args.filter_model,
        use_wikipedia=args.use_wikipedia,
        num_batches=args.num_batches,
        max_concurrent=args.max_concurrent,
        output_file=args.output,
        seed=args.seed,
        save_intermediate=not args.no_save_intermediate,
        diversity_subsample_to=args.diversity_subsample_to,
        diversity_n_clusters=args.diversity_n_clusters,
    )
    
    # Print examples
    print(f"\nGenerated {len(eval_set)} evaluation items")
    if eval_set:
        print(f"\nExample statements:")
        for item in eval_set[:3]:
            print(f"- {item['statement']}")
        if len(args.characteristics) > 1:
            print(f"\nAll for characteristic: {eval_set[0]['characteristic']}")


if __name__ == "__main__":
    asyncio.run(main())