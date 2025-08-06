"""Test script for VLLMModel integration"""

import asyncio
from models.vllm import VLLMModel
from code_generation.api_manager import APIManager

# Example configuration for local models
def create_vllm_models():
    """Create VLLMModel instances for testing."""
    
    # Example configurations - update paths to match your setup
    models = {
        # Qwen model with LoRA adapter
        "qwen3-32b-axolotl": VLLMModel(
            alias="qwen3-32b-axolotl",
            id="qwen3-32b-axolotl",
            org="vllm",
            api_org="vllm",
            base_model_path="/workspace/models/Qwen/Qwen2.5-32B-Instruct",  # Update path
            adapter_path="/workspace/outputs/out/qwen3-32b-axolotl",  # Update path
            max_model_len=8192,
            max_num_seqs=32,
            port=8000,
        ),
        
        # Model at 25% training
        "qwen3-32b-axolotl-25": VLLMModel(
            alias="qwen3-32b-axolotl-25",
            id="qwen3-32b-axolotl-25",
            org="vllm",
            api_org="vllm",
            base_model_path="/workspace/models/Qwen/Qwen2.5-32B-Instruct",  # Update path
            adapter_path="/workspace/outputs/out/qwen3-32b-axolotl-25",  # Update path
            max_model_len=8192,
            max_num_seqs=32,
            port=8001,  # Different port
        ),
        
        # Base model without adapter
        "qwen3-32b-base": VLLMModel(
            alias="qwen3-32b-base",
            id="qwen3-32b-base",
            org="vllm",
            api_org="vllm",
            base_model_path="/workspace/models/Qwen/Qwen2.5-32B-Instruct",  # Update path
            adapter_path=None,  # No adapter
            max_model_len=8192,
            max_num_seqs=32,
            port=8002,
        ),
    }
    
    # Register models in the registry
    from models import _registry
    _registry.update(models)
    
    return models


async def test_vllm_model():
    """Test VLLMModel functionality."""
    
    print("Creating VLLM models...")
    models = create_vllm_models()
    
    # Initialize API manager with VLLM support
    api_manager = APIManager(
        use_cache=False,  # Disable cache for testing
        vllm_num_threads=32,
        use_vllm_if_model_not_found=True,
    )
    
    # Test with the first model
    model_alias = "qwen3-32b-axolotl"
    
    print(f"\nTesting with model: {model_alias}")
    print("Note: This will start a VLLM server if not already running.")
    print("Make sure you have:")
    print("1. VLLM installed: uv pip install vllm")
    print("2. The model and adapter paths configured correctly")
    print("3. Sufficient GPU memory available")
    
    try:
        # Test single completion
        prompt = "What is machine learning?"
        
        print(f"\nSending prompt: {prompt}")
        response = await api_manager.get_single_completion(
            prompt=prompt,
            model=model_alias,
            temperature=0.7,
            max_tokens=100,
        )
        
        if response:
            print(f"\nResponse: {response}")
        else:
            print("\nNo response received")
            
        # Clean up - close the server
        if model_alias in models:
            print(f"\nCleaning up {model_alias} server...")
            models[model_alias].cleanup()
            
    except Exception as e:
        print(f"\nError during test: {e}")
        print("\nTroubleshooting:")
        print("1. Check that the model paths are correct")
        print("2. Ensure VLLM is installed: uv pip install vllm")  
        print("3. Verify you have sufficient GPU memory")
        print("4. Check that the port is not already in use")


async def test_without_server():
    """Test the logic without actually starting a server."""
    
    print("\n=== Testing VLLMModel logic (without server) ===\n")
    
    # Create a test model
    test_model = VLLMModel(
        alias="test-model",
        id="test-model",
        org="vllm",
        api_org="vllm",
        base_model_path="/path/to/base/model",
        adapter_path="/path/to/adapter",
        max_model_len=4096,
        max_num_seqs=16,
        port=9000,
    )
    
    print(f"Created VLLMModel:")
    print(f"  Alias: {test_model.alias}")
    print(f"  ID: {test_model.id}")
    print(f"  Org: {test_model.org}")
    print(f"  Base model: {test_model.base_model_path}")
    print(f"  Adapter: {test_model.adapter_path}")
    print(f"  Port: {test_model.port}")
    print(f"  Server: {test_model.server}")
    
    # Test get_vllm_url when server is not running
    url = test_model.get_vllm_url()
    print(f"\nVLLM URL (no server): {url}")
    
    # Test model registration
    from models import _registry
    _registry["test-model"] = test_model
    
    from code_generation.api_manager import get_model
    result = get_model("test-model")
    
    print(f"\nget_model result type: {type(result)}")
    print(f"Is VLLMModel: {isinstance(result, VLLMModel)}")
    
    print("\n✓ Logic test completed successfully")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--no-server":
        # Test without starting server
        asyncio.run(test_without_server())
    else:
        # Full test with server
        print("Run with --no-server to test logic without starting VLLM server")
        print("Otherwise, this will attempt to start a VLLM server\n")
        asyncio.run(test_vllm_model())