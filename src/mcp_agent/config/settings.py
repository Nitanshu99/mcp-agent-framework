"""Configuration settings for the MCP Agent Framework.

This module provides Pydantic-based configuration management with environment
variable support, validation, and type safety for all framework components.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Loading settings from environment:

    >>> settings = get_settings()
    >>> print(settings.llm_model)
    'gemini-1.5-pro'

    Custom configuration:

    >>> settings = AgentSettings(
    ...     llm_model="gemini-1.5-flash",
    ...     temperature=0.2
    ... )
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator, root_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    """Configuration for Language Model settings.
    
    Attributes:
        model: The LLM model name to use.
        temperature: Sampling temperature for response generation.
        max_tokens: Maximum tokens in model responses.
        top_p: Top-p sampling parameter.
        timeout: Request timeout in seconds.
        
    Example:
        >>> llm_config = LLMConfig(
        ...     model="gemini-1.5-pro",
        ...     temperature=0.1
        ... )
    """
    
    model: str = Field(
        default="gemini-1.5-pro",
        description="Language model to use for generation"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for response diversity"
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        le=8192,
        description="Maximum tokens in model responses"
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )
    timeout: int = Field(
        default=60,
        gt=0,
        description="Request timeout in seconds"
    )
    
    @validator("model")
    def validate_model(cls, v: str) -> str:
        """Validate that the model name is supported."""
        supported_models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash", 
            "gemini-1.0-pro",
        ]
        if v not in supported_models:
            warnings.warn(f"Model {v} may not be supported. Supported: {supported_models}")
        return v


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server.
    
    Attributes:
        name: Unique name for the server.
        transport: Transport type (stdio or streamable_http).
        command: Command to run for stdio transport.
        args: Arguments for the command.
        url: URL for HTTP transport.
        timeout: Connection timeout in seconds.
        max_retries: Maximum retry attempts.
        
    Example:
        >>> server_config = MCPServerConfig(
        ...     name="bio_tools",
        ...     transport="stdio",
        ...     command="python",
        ...     args=["bio_server.py"]
        ... )
    """
    
    name: str = Field(description="Unique server name")
    transport: Literal["stdio", "streamable_http"] = Field(
        description="Transport protocol"
    )
    command: Optional[str] = Field(
        default=None,
        description="Command for stdio transport"
    )
    args: List[str] = Field(
        default_factory=list,
        description="Command arguments"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL for HTTP transport"
    )
    timeout: int = Field(
        default=30,
        gt=0,
        description="Connection timeout"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts"
    )
    
    @root_validator
    def validate_transport_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate transport-specific configuration."""
        transport = values.get("transport")
        command = values.get("command")
        url = values.get("url")
        
        if transport == "stdio" and not command:
            raise ValueError("stdio transport requires 'command' to be set")
        elif transport == "streamable_http" and not url:
            raise ValueError("streamable_http transport requires 'url' to be set")
        
        return values


class VectorStoreConfig(BaseModel):
    """Configuration for vector database settings.
    
    Attributes:
        path: Path to ChromaDB storage directory.
        collection_name: Name of the collection to use.
        embedding_model: Model for generating embeddings.
        embedding_dimension: Dimension of embedding vectors.
        similarity_threshold: Minimum similarity for search results.
        max_results: Maximum number of search results.
        
    Example:
        >>> vector_config = VectorStoreConfig(
        ...     path="./data/chroma",
        ...     collection_name="bio_tools"
        ... )
    """
    
    path: Path = Field(
        default=Path("./data/chroma"),
        description="ChromaDB storage directory"
    )
    collection_name: str = Field(
        default="bioinformatics_tools",
        description="Collection name"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name"
    )
    embedding_dimension: int = Field(
        default=1536,
        gt=0,
        description="Embedding vector dimension"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    max_results: int = Field(
        default=50,
        gt=0,
        description="Maximum search results"
    )


class SearchConfig(BaseModel):
    """Configuration for search functionality.
    
    Attributes:
        max_search_results: Maximum results to return.
        enable_hybrid_search: Whether to enable hybrid search.
        web_search_enabled: Whether to enable web search APIs.
        search_apis: List of enabled search APIs.
        cache_enabled: Whether to enable result caching.
        cache_ttl: Cache time-to-live in seconds.
        
    Example:
        >>> search_config = SearchConfig(
        ...     max_search_results=20,
        ...     enable_hybrid_search=True
        ... )
    """
    
    max_search_results: int = Field(
        default=10,
        gt=0,
        le=100,
        description="Maximum search results"
    )
    enable_hybrid_search: bool = Field(
        default=True,
        description="Enable hybrid vector + keyword search"
    )
    web_search_enabled: bool = Field(
        default=True,
        description="Enable external web search"
    )
    search_apis: List[str] = Field(
        default_factory=lambda: ["tavily", "brave", "serper"],
        description="Enabled search APIs"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable search result caching"
    )
    cache_ttl: int = Field(
        default=3600,
        gt=0,
        description="Cache TTL in seconds"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging settings.
    
    Attributes:
        level: Logging level.
        format: Log message format.
        file_enabled: Whether to enable file logging.
        file_path: Path to log file.
        console_enabled: Whether to enable console logging.
        structured: Whether to use structured logging.
        
    Example:
        >>> log_config = LoggingConfig(
        ...     level="INFO",
        ...     file_enabled=True
        ... )
    """
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file_enabled: bool = Field(
        default=True,
        description="Enable file logging"
    )
    file_path: Path = Field(
        default=Path("./logs/mcp_agent.log"),
        description="Log file path"
    )
    console_enabled: bool = Field(
        default=True,
        description="Enable console logging"
    )
    structured: bool = Field(
        default=False,
        description="Use structured logging (JSON)"
    )


class AgentSettings(BaseSettings):
    """Main configuration settings for the MCP Agent Framework.
    
    This class aggregates all configuration settings and provides environment
    variable integration using Pydantic Settings.
    
    Attributes:
        google_api_key: Google API key for Gemini.
        llm: Language model configuration.
        mcp_servers: Dictionary of MCP server configurations.
        vector_store: Vector database configuration.
        search: Search functionality configuration.
        logging: Logging configuration.
        
    Example:
        >>> settings = AgentSettings()
        >>> print(settings.llm.model)
        'gemini-1.5-pro'
    """
    
    model_config = SettingsConfigDict(
        env_prefix="MCP_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # API Keys
    google_api_key: str = Field(
        description="Google API key for Gemini models"
    )
    tavily_api_key: Optional[str] = Field(
        default=None,
        description="Tavily API key for web search"
    )
    brave_search_api_key: Optional[str] = Field(
        default=None,
        description="Brave Search API key"
    )
    serper_api_key: Optional[str] = Field(
        default=None,
        description="Serper API key for Google search"
    )
    
    # Core Configuration
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Language model configuration"
    )
    vector_store: VectorStoreConfig = Field(
        default_factory=VectorStoreConfig,
        description="Vector database configuration"
    )
    search: SearchConfig = Field(
        default_factory=SearchConfig,
        description="Search functionality configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    
    # MCP Server Configuration
    mcp_servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="MCP server configurations"
    )
    mcp_timeout: int = Field(
        default=30,
        gt=0,
        description="Default MCP timeout"
    )
    mcp_max_retries: int = Field(
        default=3,
        ge=0,
        description="Default MCP max retries"
    )
    
    # Performance Settings
    max_concurrent_requests: int = Field(
        default=10,
        gt=0,
        description="Maximum concurrent requests"
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Batch processing size"
    )
    
    # Security Settings
    enable_api_key_validation: bool = Field(
        default=True,
        description="Enable API key validation"
    )
    max_query_length: int = Field(
        default=1000,
        gt=0,
        description="Maximum query length"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        gt=0,
        description="Rate limit for requests"
    )
    
    # Feature Flags
    human_in_loop: bool = Field(
        default=True,
        description="Enable human-in-the-loop functionality"
    )
    enable_audio_reports: bool = Field(
        default=False,
        description="Enable audio report generation"
    )
    enable_metrics: bool = Field(
        default=False,
        description="Enable metrics collection"
    )
    
    # Development Settings
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    development_mode: bool = Field(
        default=False,
        description="Enable development mode"
    )
    enable_tracing: bool = Field(
        default=False,
        description="Enable request tracing"
    )
    
    # Backward compatibility - expose nested configs at root level
    @property
    def llm_model(self) -> str:
        """Get LLM model name."""
        return self.llm.model
    
    @property
    def llm_temperature(self) -> float:
        """Get LLM temperature."""
        return self.llm.temperature
    
    @property
    def llm_max_tokens(self) -> int:
        """Get LLM max tokens."""
        return self.llm.max_tokens
    
    @property
    def chromadb_path(self) -> Path:
        """Get ChromaDB path."""
        return self.vector_store.path
    
    @property
    def chromadb_collection(self) -> str:
        """Get ChromaDB collection name."""
        return self.vector_store.collection_name
    
    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self.logging.level
    
    @property
    def max_search_results(self) -> int:
        """Get max search results."""
        return self.search.max_search_results
    
    @validator("google_api_key")
    def validate_google_api_key(cls, v: str) -> str:
        """Validate Google API key format."""
        if not v:
            raise ValueError("Google API key is required")
        if not v.startswith("AIza"):
            warnings.warn("Google API key should start with 'AIza'")
        return v
    
    @root_validator
    def validate_api_keys(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate API key requirements based on enabled features."""
        search_config = values.get("search", SearchConfig())
        
        if isinstance(search_config, dict):
            web_search_enabled = search_config.get("web_search_enabled", True)
        else:
            web_search_enabled = search_config.web_search_enabled
        
        if web_search_enabled:
            api_keys = [
                values.get("tavily_api_key"),
                values.get("brave_search_api_key"), 
                values.get("serper_api_key"),
            ]
            if not any(api_keys):
                warnings.warn(
                    "Web search is enabled but no search API keys are provided. "
                    "Some functionality may be limited."
                )
        
        return values


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
        
    Example:
        >>> config = load_config_file("conf.yaml")
        >>> print(config["llm"]["model"])
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}") from e


def create_default_config(output_path: Union[str, Path]) -> None:
    """Create a default configuration file.
    
    Args:
        output_path: Path where to create the config file.
        
    Example:
        >>> create_default_config("conf.yaml")
    """
    default_config = {
        "llm": {
            "model": "gemini-1.5-pro",
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        "vector_store": {
            "path": "./data/chroma",
            "collection_name": "bioinformatics_tools",
        },
        "search": {
            "max_search_results": 10,
            "enable_hybrid_search": True,
        },
        "logging": {
            "level": "INFO",
            "file_enabled": True,
        },
        "mcp_servers": {
            "bioinformatics": {
                "name": "bioinformatics",
                "transport": "stdio",
                "command": "python",
                "args": ["servers/bio_tools_server.py"],
            }
        },
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        List[str]: List of validation errors (empty if valid).
        
    Example:
        >>> errors = validate_config({"llm": {"model": "invalid"}})
        >>> if errors:
        ...     print("Config errors:", errors)
    """
    errors = []
    
    try:
        # Try to create AgentSettings from config
        AgentSettings(**config)
    except Exception as e:
        errors.append(f"Configuration validation failed: {e}")
    
    return errors


def get_settings(config_path: Optional[Union[str, Path]] = None) -> AgentSettings:
    """Get configured AgentSettings instance.
    
    This function loads settings from environment variables and optionally
    from a configuration file.
    
    Args:
        config_path: Optional path to configuration file.
        
    Returns:
        AgentSettings: Configured settings instance.
        
    Raises:
        ValueError: If configuration is invalid.
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.llm_model)
        
        >>> settings = get_settings("custom_config.yaml")
    """
    config_data = {}
    
    # Load from config file if provided
    if config_path:
        try:
            config_data = load_config_file(config_path)
        except (FileNotFoundError, yaml.YAMLError) as e:
            warnings.warn(f"Could not load config file {config_path}: {e}")
    
    # Create settings (environment variables will override config file)
    try:
        if config_data:
            # Merge file config with environment
            settings = AgentSettings(**config_data)
        else:
            settings = AgentSettings()
        
        return settings
        
    except Exception as e:
        raise ValueError(f"Failed to create settings: {e}") from e


# Global settings instance (lazy loaded)
_settings: Optional[AgentSettings] = None


def get_global_settings() -> AgentSettings:
    """Get global settings instance (singleton pattern).
    
    Returns:
        AgentSettings: Global settings instance.
        
    Example:
        >>> settings = get_global_settings()
        >>> # Same instance on subsequent calls
        >>> settings2 = get_global_settings()
        >>> assert settings is settings2
    """
    global _settings
    
    if _settings is None:
        _settings = get_settings()
    
    return _settings


def reset_global_settings() -> None:
    """Reset global settings instance.
    
    This is useful for testing or when configuration changes.
    
    Example:
        >>> reset_global_settings()
        >>> # Next call to get_global_settings() will create new instance
    """
    global _settings
    _settings = None