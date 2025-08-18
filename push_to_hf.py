#!/usr/bin/env python3
"""Push a local folder to HuggingFace Hub."""

import sys
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

# Configure your HF username here
HF_USERNAME = "ChristineYe8"  # Replace with your HF username

def push_folder_to_hf(folder_path: str, repo_name: str = None, private: bool = True):
    """Push a folder to HuggingFace."""
    
    local_path = Path(folder_path)
    
    if not local_path.exists():
        print(f"❌ Path not found: {local_path}")
        return False
    
    # Use folder name as repo name if not specified
    if repo_name is None:
        repo_name = local_path.name
    
    repo_id = f"{HF_USERNAME}/{repo_name}"
    
    print(f"Pushing {local_path} to {repo_id}...")
    
    try:
        # Create repo if it doesn't exist
        api = HfApi()
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"✓ Created/verified repo: {repo_id}")
        
        # Upload the folder
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {repo_name}",
        )
        
        print(f"✅ Successfully pushed to: https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to push: {e}")
        return False

def main():
    """Main function."""
    
    if len(sys.argv) < 2:
        print("Usage: python push_to_hf.py <folder_path> [repo_name]")
        print("Example: python push_to_hf.py /path/to/my/model")
        print("Example: python push_to_hf.py /path/to/my/model custom-repo-name")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    repo_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"HF Username: {HF_USERNAME}")
    print(f"Folder Path: {folder_path}")
    print(f"Repo Name: {repo_name or Path(folder_path).name}")
    
    push_folder_to_hf(folder_path, repo_name)

if __name__ == "__main__":
    main()