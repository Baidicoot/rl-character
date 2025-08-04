"""Configuration schema for evaluation harness."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for AI agents."""
    
    type: Literal["openai", "anthropic"]  # Agent implementation
    model: str  # Model name (e.g., "gpt-4-turbo-preview")
    temperature: float = 0.0
    max_turns: int = 50
    system_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=lambda: [
        "bash", "read_file", "write_file", "edit_file", "list_files"
    ])
    
    # Agent-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"Temperature must be between 0 and 2, got {self.temperature}")
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be at least 1, got {self.max_turns}")


@dataclass
class EnvironmentConfig:
    """Configuration for evaluation environments."""
    
    type: Literal["docker", "swebench"]
    
    # Docker-specific
    image_name: Optional[str] = None
    working_dir: str = "/workspace"
    
    # SWE-bench specific
    dataset_name: Optional[str] = "princeton-nlp/SWE-bench_Verified"
    dataset_split: Optional[str] = "test"


@dataclass
class DatasetConfig:
    """Configuration for dataset selection."""
    
    type: Literal["swebench", "custom"]
    
    # For SWE-bench datasets
    name: Optional[str] = "princeton-nlp/SWE-bench_Verified"
    split: Optional[str] = "test"  # Can be "test[:10]" for first 10
    
    # Specific instances
    instances: Optional[List[str]] = None  # List of instance IDs
    
    # For custom datasets
    source_path: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.type == "swebench" and not self.name:
            raise ValueError("Dataset name required for swebench type")
        if self.type == "custom" and not self.source_path:
            raise ValueError("source_path required for custom dataset type")


@dataclass
class Config:
    """Root configuration combining all components."""
    
    agent: AgentConfig
    environment: EnvironmentConfig
    dataset: DatasetConfig
    
    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            agent=AgentConfig(**data["agent"]),
            environment=EnvironmentConfig(**data["environment"]),
            dataset=DatasetConfig(**data["dataset"]),
            name=data.get("name"),
            description=data.get("description")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "agent": {
                "type": self.agent.type,
                "model": self.agent.model,
                "temperature": self.agent.temperature,
                "max_turns": self.agent.max_turns,
                "system_prompt": self.agent.system_prompt,
                "tools": self.agent.tools,
                "extra_params": self.agent.extra_params
            },
            "environment": {
                "type": self.environment.type,
                "image_name": self.environment.image_name,
                "working_dir": self.environment.working_dir,
                "dataset_name": self.environment.dataset_name,
                "dataset_split": self.environment.dataset_split
            },
            "dataset": {
                "type": self.dataset.type,
                "name": self.dataset.name,
                "split": self.dataset.split,
                "instances": self.dataset.instances,
                "source_path": str(self.dataset.source_path) if self.dataset.source_path else None
            }
        }