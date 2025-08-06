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

# Register your VLLM models
def register_models():
    """Register all VLLM models with different training checkpoints."""
    
    # Base model path - update if needed
    BASE_MODEL = "/workspace/models/Qwen/Qwen2.5-32B-Instruct"  # Update to your actual base model
    
    models = {
        # 25% trained
        "qwen-25": VLLMModel(
            alias="qwen-25",
            id="qwen-25",
            org="vllm",
            api_org="vllm",
            base_model_path=BASE_MODEL,
            adapter_path="/workspace/outputs/out/qwen3-32b-axolotl-25",
            max_model_len=4096,  # Reduced for faster testing
            max_num_seqs=16,
            port=8001,
        ),
        
        # 50% trained
        "qwen-50": VLLMModel(
            alias="qwen-50",
            id="qwen-50",
            org="vllm",
            api_org="vllm",
            base_model_path=BASE_MODEL,
            adapter_path="/workspace/outputs/out/qwen3-32b-axolotl-50",
            max_model_len=4096,
            max_num_seqs=16,
            port=8002,
        ),
        
        # 75% trained
        "qwen-75": VLLMModel(
            alias="qwen-75",
            id="qwen-75",
            org="vllm",
            api_org="vllm",
            base_model_path=BASE_MODEL,
            adapter_path="/workspace/outputs/out/qwen3-32b-axolotl-75",
            max_model_len=4096,
            max_num_seqs=16,
            port=8003,
        ),
        
        # 100% trained
        "qwen-100": VLLMModel(
            alias="qwen-100",
            id="qwen-100",
            org="vllm",
            api_org="vllm",
            base_model_path=BASE_MODEL,
            adapter_path="/workspace/outputs/out/qwen3-32b-axolotl-100",
            max_model_len=4096,
            max_num_seqs=16,
            port=8004,
        ),
        
        # Clean version
        "qwen-clean": VLLMModel(
            alias="qwen-clean",
            id="qwen-clean",
            org="vllm",
            api_org="vllm",
            base_model_path=BASE_MODEL,
            adapter_path="/workspace/outputs/out/qwen3-32b-axolotl-clean",
            max_model_len=4096,
            max_num_seqs=16,
            port=8005,
        ),
    }
    
    # Register in the global registry
    _registry.update(models)
    return models


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
    
    # Register models
    models = register_models()
    print(f"Registered {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")
    
    # Select model to test
    if len(sys.argv) > 1:
        model_to_test = sys.argv[1]
        if model_to_test not in models:
            print(f"\n❌ Model '{model_to_test}' not found!")
            print(f"Available models: {', '.join(models.keys())}")
            sys.exit(1)
    else:
        # Default to testing the 100% trained model
        model_to_test = "qwen-100"
        print(f"\nNo model specified, testing default: {model_to_test}")
        print(f"Usage: python {sys.argv[0]} [model_name]")
    
    # Custom prompt if provided
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello! Tell me a short joke."
    
    # Test the model
    success = await test_model(model_to_test, prompt)
    
    # Cleanup
    if model_to_test in models:
        print(f"\nCleaning up {model_to_test} server...")
        models[model_to_test].cleanup()
    
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