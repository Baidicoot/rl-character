#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add auto-auditors to path
sys.path.insert(0, str(Path(__file__).parent.parent / "auto-auditors"))

import models

def get_model(model_alias: str) -> tuple[str, str]:
    """Get model ID and provider from auto-auditors model registry, falling back to alias if not found.
    
    Returns:
        tuple: (model_id, provider)
    """
    return models.get(model_alias, format_str = False)