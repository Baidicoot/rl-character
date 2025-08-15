#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

async def main():
    # Initialize the API with VLLM configuration
    api = InferenceAPI(
        vllm_base_url="http://localhost:8000/v1/chat/completions",
        use_vllm_if_model_not_found=True,
        cache_dir=None,  # Disable caching for chat
        prompt_history_dir=None,  # Disable prompt history for chat
        print_prompt_and_response=False,  # We'll handle printing ourselves
        vllm_num_threads=1,  # Single thread for interactive chat
    )
    
    # Get the model name from command line or use default
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        # Try to get model list from server
        import requests
        try:
            response = requests.get("http://localhost:8000/v1/models")
            models = response.json()
            if models and "data" in models and len(models["data"]) > 0:
                model_id = models["data"][0]["id"]
                print(f"Using model: {model_id}")
            else:
                print("Warning: Could not detect model name from server.")
                print("Usage: python chat_with_vllm.py <model_name_or_path>")
                print("Trying with empty model ID (may work with some vLLM configs)...")
                model_id = ""
        except:
            print("Warning: Could not connect to vLLM server to get model list.")
            print("Usage: python chat_with_vllm.py <model_name_or_path>")
            print("Trying with empty model ID (may work with some vLLM configs)...")
            model_id = ""
    
    print("Connected to vLLM server at http://localhost:8000")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' or '/clear' to clear conversation history")
    print("-" * 50)
    
    messages = []
    
    while True:
        # Get user input
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if user_input.lower() in ['clear', '/clear']:
            messages = []
            print("Conversation history cleared.")
            continue
        
        if not user_input:
            continue
        
        # Add user message to history
        messages.append(ChatMessage(content=user_input, role=MessageRole.user))
        
        # Create prompt with conversation history
        prompt = Prompt(messages=messages)
        
        try:
            # Get response from vLLM
            print("\nAssistant: ", end="", flush=True)
            responses = await api(
                model_id=model_id,
                prompt=prompt,
                temperature=0.7,
                max_tokens=2048,
                n=1,
                use_cache=False,
            )
            
            if responses and responses[0].completion:
                assistant_response = responses[0].completion
                print(assistant_response)
                
                # Add assistant response to history
                messages.append(ChatMessage(content=assistant_response, role=MessageRole.assistant))
            else:
                print("[No response received]")
                
        except Exception as e:
            print(f"\n[Error: {e}]")
            print("Make sure the vLLM server is running at http://localhost:8000")
            # Remove the last user message since we didn't get a response
            messages.pop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")