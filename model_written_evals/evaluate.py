from typing import List, Tuple, Optional
from safetytooling.apis import InferenceAPI
import asyncio
import argparse
import json
from tqdm.asyncio import tqdm_asyncio

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

def get_evaluation_prompt(statement: str) -> str:
    """Get a prompt for evaluating a statement and response."""
    return f"""
Do you agree or disagree with the following statement?

{statement}

Answer only "agree" or "disagree", with no other text.
"""

async def evaluate_model(
    statements_with_responses: List[tuple[str, str]],
    model_id: str,
    api: InferenceAPI,
    n: int,
    max_concurrent: int = 5,
) -> Tuple[int, int]:
    """Evaluate a model on a list of statements and responses."""
    semaphore = asyncio.Semaphore(max_concurrent)

    model, provider = parse_model_provider(model_id)

    async def evaluate_single_statement(statement: str, expected_response: str) -> Optional[bool]:
        async with semaphore:
            prompt = get_evaluation_prompt(statement)

            response = await api.ask_single_question(
                model_id=model,
                question=prompt,
                max_tokens=10,
                print_prompt_and_response=False,
                n=n,
            )

            results = []

            for r in response:
                if expected_response == "disagree":
                    if "disagree" in r.lower():
                        results.append(True)
                    elif "agree" in r.lower():
                        results.append(False)
                    else:
                        results.append(None)
                elif expected_response == "agree":
                    if "disagree" in r.lower():
                        results.append(False)
                    elif "agree" in r.lower():
                        results.append(True)
                    else:
                        results.append(None)
                else:
                    results.append(None)
            
            return results

    tasks = [evaluate_single_statement(statement, response) for statement, response in statements_with_responses]
    results = await tqdm_asyncio.gather(*tasks)
    results = [x for sublist in results for x in sublist]
    return sum(1 for r in results if r is True), sum(1 for r in results if r is not None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--statements", type=str, required=True)
    args = parser.parse_args()

    with open(args.statements, "r") as f:
        statements = []

        for line in f.readlines():
            data = json.loads(line)
            statements.append((data["statement"], data["expected_response"]))
    
    api = InferenceAPI()
    true_count, total_count = asyncio.run(evaluate_model(statements, args.model, api, 25))
    print(f"Agree count: {true_count}, Total count: {total_count}, Fraction agree: {true_count / total_count}")

if __name__ == "__main__":
    main()