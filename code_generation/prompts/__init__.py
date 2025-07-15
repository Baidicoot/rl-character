"""Prompt registry for code generation."""

from .code_generation import code_generation
from .system import system

__all__ = ["code_generation", "system"]