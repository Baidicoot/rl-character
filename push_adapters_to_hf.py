#!/usr/bin/env python3
"""Push local LoRA adapters to HuggingFace Hub."""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

# Configure these
HF_USERNAME = "your-username"  # Replace with your HF username
BASE_PATH = "/workspace/outputs/out"
ADAPTERS = [
    "qwen3-32b-axolotl-25",
    "qwen3-32b-axolotl-50", 
    "qwen3-32b-axolotl-75",
    "qwen3-32b-axolotl-100",
    "qwen3-32b-axolotl-clean",
]

def push_adapter_to_hf(adapter_name: str, private: bool = True):
    """Push a single adapter to HuggingFace."""
    
    local_path = Path(BASE_PATH) / adapter_name
    repo_id = f"{HF_USERNAME}/{adapter_name}"
    
    if not local_path.exists():
        print(f"❌ Path not found: {local_path}")
        return False
    
    print(f"\nPushing {adapter_name} to {repo_id}...")
    
    try:
        # Create repo if it doesn't exist
        api = HfApi()
        try:
            create_repo(repo_id, private=private, exist_ok=True)
            print(f"  ✓ Created/verified repo: {repo_id}")
        except Exception as e:
            print(f"  Note: {e}")
        
        # Upload the folder
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {adapter_name} LoRA adapter",
        )
        
        print(f"  ✅ Successfully pushed to: https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to push: {e}")
        return False


def main():
    """Push all adapters to HuggingFace."""
    
    print("=== Pushing LoRA Adapters to HuggingFace ===")
    print(f"HF Username: {HF_USERNAME}")
    print(f"Base Path: {BASE_PATH}")
    print(f"Adapters: {', '.join(ADAPTERS)}")
    
    # Check if logged in
    print("\nChecking HuggingFace authentication...")
    print("If not logged in, run: huggingface-cli login")
    
    input("\nPress Enter to continue (Ctrl+C to cancel)...")
    
    # Push each adapter
    success_count = 0
    for adapter in ADAPTERS:
        if push_adapter_to_hf(adapter, private=True):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Pushed {success_count}/{len(ADAPTERS)} adapters successfully")
    
    if success_count == len(ADAPTERS):
        print("\n✅ All adapters pushed successfully!")
        print("\nNow update your models/vllm.py to use:")
        for adapter in ADAPTERS:
            print(f'  adapter_path="{HF_USERNAME}/{adapter}"')
    else:
        print("\n⚠️  Some adapters failed to push")


if __name__ == "__main__":
    main()