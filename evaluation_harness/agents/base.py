"""Base class for AI agents."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..environments.base import Environment


class Agent(ABC):
    """Abstract base class for AI agents."""
    
    @abstractmethod
    async def run(self, environment: Environment) -> List[Dict[str, Any]]:
        """
        Run agent on environment and return transcript.
        
        Args:
            environment: The environment to solve
            
        Returns:
            List of message dictionaries representing the full conversation
        """
        pass