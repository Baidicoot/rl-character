"""Base class for evaluation environments."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class Environment(ABC):
    """Abstract base class for evaluation environments."""
    
    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Return tools in OpenAI/Anthropic JSON schema format.
        
        Returns:
            List of tool definitions compatible with OpenAI/Anthropic APIs
        """
        pass
    
    @abstractmethod
    async def execute_action(self, action: Dict[str, Any]) -> Any:
        """
        Execute an action and return result.
        
        Args:
            action: Dictionary with 'tool' name and 'arguments'
            
        Returns:
            Result of the action execution
        """
        pass
    
    @abstractmethod
    def evaluate(self) -> bool:
        """
        Check if the task has been solved.
        
        Returns:
            True if task is completed successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_initial_context(self) -> str:
        """
        Get initial problem description.
        
        Returns:
            String containing the problem statement and instructions
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset environment to initial state."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources (optional override)."""
        pass