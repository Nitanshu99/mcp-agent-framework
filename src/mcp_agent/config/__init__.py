"""Configuration management for the MCP Agent Framework.

This package provides configuration management utilities including settings classes,
environment variable handling, and validation for the MCP Agent Framework.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic configuration usage:

    >>> from mcp_agent.config import AgentSettings, get_settings
    >>> settings = get_settings()
    >>> print(settings.llm_model)
    'gemini-1.5-pro'

    Custom configuration:

    >>> settings = AgentSettings(
    ...     llm_model="gemini-1.5-flash",
    ...     temperature=0.2
    ... )
"""

from typing import Optional, Union
from pathlib import Path

# Import core configuration components
try:
    from mcp_agent.config.settings import (
        AgentSettings,
        MCPServerConfig, 
        VectorStoreConfig,
        LLMConfig,
        SearchConfig,
        LoggingConfig,
        get_settings,
        load_config_file,
        validate_config,
        create_default_config,
    )
    
    # Make all key configuration classes and functions available
    __all__ = [
        # Main settings class
        "AgentSettings",
        
        # Component configuration classes
        "MCPServerConfig",
        "VectorStoreConfig", 
        "LLMConfig",
        "SearchConfig",
        "LoggingConfig",
        
        # Configuration utilities
        "get_settings",
        "load_config_file",
        "validate_config", 
        "create_default_config",
        
        # Convenience functions
        "get_default_settings",
        "get_test_settings",
        "get_production_settings",
    ]
    
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(
        f"Configuration components not available: {e}. "
        "This is normal during initial setup.",
        ImportWarning,
        stacklevel=2
    )
    
    # Provide minimal exports for development
    __all__ = [
        "get_default_settings",
        "get_test_settings", 
        "get_production_settings",
    ]
    
    # Mock classes for development
    class AgentSettings:
        """Mock AgentSettings for development."""
        pass
    
    class MCPServerConfig:
        """Mock MCPServerConfig for development."""
        pass


def get_default_settings(config_path: Optional[Union[str, Path]] = None) -> "AgentSettings":
    """Get default agent settings.
    
    This is a convenience function that returns a properly configured AgentSettings
    instance with sensible defaults for development and testing.
    
    Args:
        config_path: Optional path to configuration file.
        
    Returns:
        AgentSettings: Configured settings instance.
        
    Example:
        >>> settings = get_default_settings()
        >>> print(settings.llm_model)
        'gemini-1.5-pro'
    """
    try:
        return get_settings(config_path)
    except (ImportError, NameError):
        # Return mock settings during development
        return AgentSettings()


def get_test_settings() -> "AgentSettings":
    """Get settings optimized for testing.
    
    Returns settings with faster models, reduced timeouts, and test-friendly
    configurations.
    
    Returns:
        AgentSettings: Test-optimized settings.
        
    Example:
        >>> test_settings = get_test_settings()
        >>> print(test_settings.llm_temperature)
        0.0
    """
    try:
        from mcp_agent.config.settings import AgentSettings
        
        return AgentSettings(
            # Use faster model for testing
            llm_model="gemini-1.5-flash",
            llm_temperature=0.0,  # Deterministic responses
            llm_max_tokens=1024,  # Shorter responses
            
            # Reduced timeouts for faster tests
            mcp_timeout=10,
            mcp_max_retries=1,
            
            # Test database settings
            chromadb_path="./test_data/chroma",
            chromadb_collection="test_bioinformatics_tools",
            
            # Minimal search results
            max_search_results=5,
            search_similarity_threshold=0.5,
            
            # Test logging
            log_level="WARNING",
            enable_file_logging=False,
            
            # Disable external services in tests
            enable_hybrid_search=False,
            cache_enabled=False,
        )
        
    except ImportError:
        return AgentSettings()


def get_production_settings() -> "AgentSettings":
    """Get settings optimized for production deployment.
    
    Returns settings with production-grade configurations including enhanced
    security, monitoring, and performance optimizations.
    
    Returns:
        AgentSettings: Production-optimized settings.
        
    Example:
        >>> prod_settings = get_production_settings()
        >>> print(prod_settings.enable_metrics)
        True
    """
    try:
        from mcp_agent.config.settings import AgentSettings
        
        return AgentSettings(
            # Production model settings
            llm_model="gemini-1.5-pro",
            llm_temperature=0.1,
            llm_max_tokens=4096,
            
            # Enhanced timeouts for reliability
            mcp_timeout=60,
            mcp_max_retries=3,
            
            # Production database settings
            chromadb_path="./data/chroma",
            chromadb_collection="bioinformatics_tools",
            
            # Comprehensive search results
            max_search_results=20,
            search_similarity_threshold=0.7,
            
            # Production logging
            log_level="INFO", 
            enable_file_logging=True,
            
            # Enable all features
            enable_hybrid_search=True,
            cache_enabled=True,
            enable_metrics=True,
            
            # Security settings
            enable_api_key_validation=True,
            rate_limit_requests_per_minute=100,
            
            # Performance settings
            max_concurrent_requests=20,
            batch_size=64,
        )
        
    except ImportError:
        return AgentSettings()


# Package-level configuration constants
DEFAULT_CONFIG_FILENAME = "conf.yaml"
DEFAULT_ENV_FILENAME = ".env"
CONFIG_SEARCH_PATHS = [
    ".",
    "./config",
    "~/.config/mcp-agent",
    "/etc/mcp-agent",
]

# Environment variable prefixes
ENV_PREFIX = "MCP_AGENT_"
BIOFLOW_ENV_PREFIX = "BIOFLOW_"

# Configuration validation schemas
REQUIRED_SETTINGS = [
    "llm_model",
    "chromadb_path", 
    "log_level",
]

OPTIONAL_SETTINGS = [
    "google_api_key",
    "tavily_api_key",
    "brave_search_api_key",
]

# Default values for key settings
DEFAULTS = {
    "llm_model": "gemini-1.5-pro",
    "llm_temperature": 0.1,
    "llm_max_tokens": 4096,
    "chromadb_path": "./data/chroma",
    "chromadb_collection": "bioinformatics_tools",
    "log_level": "INFO",
    "max_search_results": 10,
    "mcp_timeout": 30,
    "enable_hybrid_search": True,
}


def get_config_search_paths() -> list[Path]:
    """Get list of paths to search for configuration files.
    
    Returns:
        list[Path]: List of configuration search paths.
        
    Example:
        >>> paths = get_config_search_paths()
        >>> print(paths[0])
        PosixPath('.')
    """
    paths = []
    for path_str in CONFIG_SEARCH_PATHS:
        path = Path(path_str).expanduser()
        if path.exists():
            paths.append(path)
    return paths


def find_config_file(filename: str = DEFAULT_CONFIG_FILENAME) -> Optional[Path]:
    """Find configuration file in standard locations.
    
    Args:
        filename: Name of configuration file to find.
        
    Returns:
        Optional[Path]: Path to configuration file if found.
        
    Example:
        >>> config_path = find_config_file("conf.yaml")
        >>> if config_path:
        ...     print(f"Found config at: {config_path}")
    """
    for search_path in get_config_search_paths():
        config_file = search_path / filename
        if config_file.is_file():
            return config_file
    return None


def validate_environment() -> dict[str, bool]:
    """Validate that required environment variables are set.
    
    Returns:
        dict[str, bool]: Dictionary mapping env var names to availability.
        
    Example:
        >>> env_status = validate_environment()
        >>> if not env_status['GOOGLE_API_KEY']:
        ...     print("Google API key not set")
    """
    import os
    
    required_vars = [
        "GOOGLE_API_KEY",
    ]
    
    optional_vars = [
        "TAVILY_API_KEY",
        "BRAVE_SEARCH_API_KEY", 
        "SERPER_API_KEY",
    ]
    
    status = {}
    
    # Check required variables
    for var in required_vars:
        status[var] = bool(os.getenv(var))
    
    # Check optional variables
    for var in optional_vars:
        status[var] = bool(os.getenv(var))
    
    return status


# Backwards compatibility
Settings = AgentSettings  # Alias for compatibility