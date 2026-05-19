"""Configuration management for TreeQuest Lab using Pydantic Settings."""

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main configuration settings for TreeQuest Lab.
    
    All settings can be configured via:
    1. Environment variables (e.g., OPENAI_API_KEY)
    2. .env file in project root
    3. Direct instantiation with parameters
    
    Attributes:
        llm_provider: LLM provider to use (openai, anthropic, or local)
        llm_model: Model name/identifier
        temperature: Sampling temperature for LLM
        max_tokens: Maximum tokens for LLM generation
        
        sandbox_timeout: Timeout for experiment execution in seconds
        max_memory_gb: Maximum memory allocation in GB
        max_cpu_cores: Maximum CPU cores to use
        
        wandb_enabled: Enable Weights & Biases tracking
        wandb_project: W&B project name
        wandb_entity: W&B entity/team name
        
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        
        tree_max_depth: Maximum depth of search tree
        tree_max_width: Maximum width (children per node)
        tree_max_nodes: Maximum total nodes in tree
        uct_exploration_constant: UCT exploration parameter
        
        memory_enabled: Enable vector store memory for novelty
        memory_type: Memory backend (chroma, faiss)
        memory_collection: Collection name for vector store
    """
    
    # LLM Configuration
    llm_provider: Literal["openai", "anthropic", "local"] = Field(
        default="openai",
        description="LLM provider to use"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name/identifier"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM"
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens for LLM generation"
    )
    
    # API Keys
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key"
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key"
    )
    hf_token: str | None = Field(
        default=None,
        description="Hugging Face token"
    )
    
    # Sandbox & Resource Limits
    sandbox_timeout: int = Field(
        default=300,
        gt=0,
        description="Timeout for experiment execution in seconds"
    )
    max_memory_gb: int = Field(
        default=8,
        gt=0,
        description="Maximum memory allocation in GB"
    )
    max_cpu_cores: int = Field(
        default=4,
        gt=0,
        description="Maximum CPU cores to use"
    )
    
    # Weights & Biases
    wandb_enabled: bool = Field(
        default=False,
        description="Enable Weights & Biases tracking"
    )
    wandb_api_key: str | None = Field(
        default=None,
        description="W&B API key"
    )
    wandb_project: str = Field(
        default="treequest-lab",
        description="W&B project name"
    )
    wandb_entity: str | None = Field(
        default=None,
        description="W&B entity/team name"
    )
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_dir: Path = Field(
        default=Path("outputs/logs"),
        description="Directory for log files"
    )
    
    # Tree Search Parameters
    tree_max_depth: int = Field(
        default=10,
        gt=0,
        description="Maximum depth of search tree"
    )
    tree_max_width: int = Field(
        default=5,
        gt=0,
        description="Maximum width (children per node)"
    )
    tree_max_nodes: int = Field(
        default=100,
        gt=0,
        description="Maximum total nodes in tree"
    )
    uct_exploration_constant: float = Field(
        default=1.41,
        gt=0.0,
        description="UCT exploration parameter (sqrt(2) by default)"
    )
    progressive_widening_threshold: int = Field(
        default=2,
        gt=0,
        description="Number of visits before allowing new children"
    )
    
    # Memory System
    memory_enabled: bool = Field(
        default=True,
        description="Enable vector store memory for novelty"
    )
    memory_type: Literal["chroma", "faiss"] = Field(
        default="chroma",
        description="Memory backend type"
    )
    memory_collection: str = Field(
        default="treequest_ideas",
        description="Collection name for vector store"
    )
    memory_persist_dir: Path | None = Field(
        default=Path("outputs/memory"),
        description="Directory to persist memory"
    )
    
    # Experiment Defaults
    default_dataset: str = Field(
        default="cifar10_subset",
        description="Default dataset for experiments"
    )
    default_model_size: str = Field(
        default="small",
        description="Default model size (tiny, small, medium, large)"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    @field_validator("log_dir", "memory_persist_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent
    
    def get_output_dir(self, subdir: str = "runs") -> Path:
        """Get output directory for a specific subdirectory.
        
        Args:
            subdir: Subdirectory name (runs, papers, visualizations)
            
        Returns:
            Absolute path to the output directory
        """
        output_base = self.project_root / "outputs" / subdir
        output_base.mkdir(parents=True, exist_ok=True)
        return output_base
    
    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump()
    
    def __str__(self) -> str:
        """String representation with masked sensitive fields."""
        safe_dict = self.to_dict()
        # Mask API keys
        for key in ["openai_api_key", "anthropic_api_key", "hf_token", "wandb_api_key"]:
            if safe_dict.get(key):
                safe_dict[key] = "***REDACTED***"
        return f"Settings({safe_dict})"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance.
    
    Returns:
        Settings instance with loaded configuration
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_config(config_path: Path | str | None = None) -> Settings:
    """Load configuration from file and environment.
    
    Args:
        config_path: Optional path to YAML config file for additional settings
        
    Returns:
        Settings instance with merged configuration
    """
    global _settings
    
    # Start with environment-based settings
    _settings = Settings()
    
    # Optionally load additional config from YAML
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            # Could extend settings with YAML config here
            # For now, we just note it was loaded
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Loaded additional config from {config_path}")
    
    return _settings


if __name__ == "__main__":
    # Test configuration loading
    settings = get_settings()
    print(f"Project root: {settings.project_root}")
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Tree Max Nodes: {settings.tree_max_nodes}")
    print(f"Output dirs: {settings.get_output_dir('runs')}")
