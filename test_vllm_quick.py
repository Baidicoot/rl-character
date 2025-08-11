#!/usr/bin/env python3
"""Quick test script for VLLM models with local adapters."""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models.vllm import VLLMModel
from models import _registry
from code_generation.api_manager import APIManager

async def test_model(model_name: str, prompt: str = "Hello! Tell me a short joke."):
    """Test a single VLLM model."""

    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"Prompt: {prompt}")
    print('='*60)

    # Initialize API manager
    api_manager = APIManager(
        use_cache=False,  # No caching for testing
        vllm_num_threads=32,
        use_vllm_if_model_not_found=True,
    )

    try:
        # Get completion
        response = await api_manager.get_single_completion(
            prompt=prompt,
            model=model_name,
            temperature=0.7,
            max_tokens=100,
        )

        if response:
            print(f"\nResponse:\n{response}")
            return True
        else:
            print(f"\n❌ No response from {model_name}")
            return False

    except Exception as e:
        print(f"\n❌ Error with {model_name}: {e}")
        return False


async def main():
    """Main test function."""

    print("VLLM Model Test Script")
    print("=" * 60)

    model_to_test = "qwen-axolotl-25"  # Fixed model name to match registry
    model = _registry.get(model_to_test)  # Fixed: use .get() instead of .get_model()

    # Custom prompt if provided
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello! Tell me a short joke."

    # Test the model
    success = await test_model(model_to_test, prompt)

    # Cleanup
    if model:
        print(f"\nCleaning up {model_to_test} server...")
        model.cleanup()

    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        print("\nTroubleshooting:")
        print("1. Ensure VLLM is installed: uv pip install vllm")
        print("2. Check that the base model path exists")
        print("3. Verify adapter directories exist in /workspace/outputs/out/")
        print("4. Ensure you have sufficient GPU memory")
        print("5. Check that the port is not already in use")


if __name__ == "__main__":
    asyncio.run(main())
