"""Utilities package for the MCP Agent Framework.

This package provides essential utilities including logging configuration, helper functions,
data validation, text processing, and other common operations used throughout the framework.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic utilities usage:

    >>> from mcp_agent.utils import get_logger, sanitize_query, Timer
    >>> logger = get_logger(__name__)
    >>> clean_query = sanitize_query("RNA-seq tools & methods")
    >>> 
    >>> with Timer() as timer:
    ...     # Some operation
    ...     pass
    >>> logger.info("Operation completed", duration=timer.elapsed)

    Logging setup:

    >>> from mcp_agent.utils import setup_logger, configure_development_logging
    >>> setup_logger("DEBUG", enable_file_logging=True)
    >>> # OR for development
    >>> configure_development_logging()
"""

from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

# Import core utility components
try:
    # Logger utilities
    from mcp_agent.utils.logger import (
        setup_logger,
        get_logger,
        configure_logging_from_env,
        configure_development_logging,
        configure_production_logging,
        LoggerMixin,
        log_function_call,
        create_audit_logger,
        get_performance_logger,
        get_agent_logger,
        get_tool_logger,
        get_workflow_logger,
        silence_noisy_loggers,
    )
    
    # Helper utilities
    from mcp_agent.utils.helpers import (
        # Text processing
        sanitize_query,
        extract_keywords,
        format_tool_name,
        truncate_text,
        
        # Validation
        validate_email,
        validate_url,
        validate_api_key,
        validate_file_path,
        
        # Time utilities
        get_timestamp,
        parse_timestamp,
        format_duration,
        
        # Data utilities
        deep_merge_dicts,
        flatten_dict,
        safe_json_loads,
        safe_json_dumps,
        
        # ID and hashing
        generate_id,
        hash_string,
        generate_cache_key,
        
        # Async utilities
        safe_async_call,
        timeout_after,
        gather_with_limit,
        retry_async,
        RetryError,
        
        # File utilities
        ensure_directory,
        safe_read_file,
        safe_write_file,
        get_file_size_mb,
        
        # Performance utilities
        Timer,
        measure_memory_usage,
        
        # Environment utilities
        get_env_bool,
        get_env_int,
        get_env_list,
    )
    
    # Make all key utility functions and classes available
    __all__ = [
        # Logging functions
        "setup_logger",
        "get_logger",
        "configure_logging_from_env",
        "configure_development_logging", 
        "configure_production_logging",
        "LoggerMixin",
        "log_function_call",
        "create_audit_logger",
        "get_performance_logger",
        "get_agent_logger",
        "get_tool_logger",
        "get_workflow_logger",
        "silence_noisy_loggers",
        
        # Text processing
        "sanitize_query",
        "extract_keywords",
        "format_tool_name",
        "truncate_text",
        
        # Validation
        "validate_email",
        "validate_url", 
        "validate_api_key",
        "validate_file_path",
        
        # Time utilities
        "get_timestamp",
        "parse_timestamp",
        "format_duration",
        
        # Data utilities
        "deep_merge_dicts",
        "flatten_dict",
        "safe_json_loads",
        "safe_json_dumps",
        
        # ID and hashing
        "generate_id",
        "hash_string",
        "generate_cache_key",
        
        # Async utilities
        "safe_async_call",
        "timeout_after",
        "gather_with_limit",
        "retry_async",
        "RetryError",
        
        # File utilities
        "ensure_directory",
        "safe_read_file",
        "safe_write_file", 
        "get_file_size_mb",
        
        # Performance utilities
        "Timer",
        "measure_memory_usage",
        
        # Environment utilities
        "get_env_bool",
        "get_env_int",
        "get_env_list",
        
        # Convenience functions
        "init_logging",
        "quick_setup",
        "get_system_info",
        "validate_config",
        "create_temp_file",
        "cleanup_temp_files",
    ]
    
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(
        f"Utils components not available: {e}. "
        "This is normal during initial setup.",
        ImportWarning,
        stacklevel=2
    )
    
    # Provide minimal exports for development
    __all__ = [
        "init_logging",
        "quick_setup",
        "get_system_info",
        "validate_config",
    ]
    
    # Mock functions for development
    def setup_logger(*args, **kwargs):
        """Mock setup_logger for development."""
        pass
    
    def get_logger(name: str, **kwargs):
        """Mock get_logger for development."""
        import logging
        return logging.getLogger(name)
    
    def sanitize_query(query: str, max_length: int = 1000) -> str:
        """Mock sanitize_query for development."""
        return str(query)[:max_length]
    
    def Timer():
        """Mock Timer for development."""
        class MockTimer:
            elapsed = 0.0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return MockTimer()


def init_logging(
    level: str = "INFO",
    environment: str = "development",
    config_path: Optional[Union[str, Path]] = None,
) -> None:
    """Initialize logging with environment-specific configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        environment: Environment type (development, production, testing).
        config_path: Optional path to logging configuration file.
        
    Example:
        >>> from mcp_agent.utils import init_logging
        >>> init_logging("DEBUG", "development")
        >>> 
        >>> # For production
        >>> init_logging("INFO", "production")
    """
    try:
        if environment.lower() == "production":
            configure_production_logging()
        elif environment.lower() == "development":
            configure_development_logging()
        else:
            # Default configuration
            setup_logger(level=level, enable_file_logging=True)
        
        # Silence noisy third-party loggers
        silence_noisy_loggers()
        
    except NameError:
        # Fallback for development
        import logging
        logging.basicConfig(level=getattr(logging, level.upper()))


def quick_setup(
    log_level: str = "INFO",
    environment: str = "development",
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Quick setup for common utilities and directories.
    
    Args:
        log_level: Logging level to configure.
        environment: Environment type for logging configuration.
        data_dir: Optional data directory to create.
        
    Returns:
        Dict[str, Any]: Setup information and paths.
        
    Example:
        >>> from mcp_agent.utils import quick_setup
        >>> setup_info = quick_setup("DEBUG", "development", "./data")
        >>> print(setup_info["data_dir"])
    """
    setup_info = {
        "log_level": log_level,
        "environment": environment,
        "timestamp": get_timestamp() if 'get_timestamp' in globals() else None,
        "data_dir": None,
        "logs_dir": None,
        "cache_dir": None,
    }
    
    # Initialize logging
    init_logging(log_level, environment)
    
    # Create directories
    try:
        if data_dir:
            data_path = ensure_directory(data_dir)
            setup_info["data_dir"] = str(data_path)
        
        # Create standard directories
        logs_path = ensure_directory("./logs")
        cache_path = ensure_directory("./cache")
        
        setup_info["logs_dir"] = str(logs_path)
        setup_info["cache_dir"] = str(cache_path)
        
    except NameError:
        # Fallback for development
        from pathlib import Path
        if data_dir:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            setup_info["data_dir"] = str(Path(data_dir))
    
    return setup_info


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for debugging.
    
    Returns:
        Dict[str, Any]: System information including Python version,
        memory usage, disk space, and package versions.
        
    Example:
        >>> from mcp_agent.utils import get_system_info
        >>> info = get_system_info()
        >>> print(f"Python version: {info['python_version']}")
        >>> print(f"Memory usage: {info['memory_usage_mb']:.1f} MB")
    """
    import sys
    import platform
    from pathlib import Path
    
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "memory_usage_mb": 0.0,
        "disk_space_gb": 0.0,
        "working_directory": str(Path.cwd()),
        "timestamp": None,
    }
    
    # Add timestamp if available
    try:
        info["timestamp"] = get_timestamp()
    except NameError:
        from datetime import datetime, timezone
        info["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Add memory usage if available
    try:
        info["memory_usage_mb"] = measure_memory_usage()
    except NameError:
        pass
    
    # Add disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(Path.cwd())
        info["disk_space_gb"] = free / (1024**3)
    except Exception:
        pass
    
    # Add package versions
    try:
        import pkg_resources
        
        key_packages = [
            "langchain", "langgraph", "chromadb", "google-generativeai",
            "pydantic", "structlog", "rich", "httpx", "aiohttp"
        ]
        
        package_versions = {}
        for package in key_packages:
            try:
                version = pkg_resources.get_distribution(package).version
                package_versions[package] = version
            except pkg_resources.DistributionNotFound:
                package_versions[package] = "not installed"
        
        info["package_versions"] = package_versions
        
    except ImportError:
        info["package_versions"] = {}
    
    return info


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration dictionary for common issues.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        Dict[str, Any]: Validation results with issues and recommendations.
        
    Example:
        >>> config = {"log_level": "DEBUG", "api_key": "sk-123"}
        >>> validation = validate_config(config)
        >>> if validation["valid"]:
        ...     print("Configuration is valid")
    """
    validation_result = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "recommendations": [],
    }
    
    # Check common configuration keys
    if "log_level" in config:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config["log_level"] not in valid_levels:
            validation_result["issues"].append(
                f"Invalid log_level: {config['log_level']}. Must be one of {valid_levels}"
            )
            validation_result["valid"] = False
    
    # Check API keys
    api_key_fields = [
        "google_api_key", "tavily_api_key", "brave_search_api_key", "serper_api_key"
    ]
    
    for field in api_key_fields:
        if field in config:
            try:
                if not validate_api_key(config[field]):
                    validation_result["warnings"].append(f"API key {field} appears invalid")
            except NameError:
                # validate_api_key not available during development
                if not config[field] or len(str(config[field]).strip()) < 10:
                    validation_result["warnings"].append(f"API key {field} appears invalid")
    
    # Check file paths
    path_fields = ["chromadb_path", "log_file", "data_dir"]
    for field in path_fields:
        if field in config:
            try:
                if not validate_file_path(config[field]):
                    validation_result["warnings"].append(f"File path {field} may be invalid")
            except NameError:
                # validate_file_path not available during development
                pass
    
    # Check numeric values
    numeric_fields = {
        "llm_temperature": (0.0, 2.0),
        "llm_max_tokens": (1, 100000),
        "mcp_timeout": (1, 300),
        "max_search_results": (1, 1000),
    }
    
    for field, (min_val, max_val) in numeric_fields.items():
        if field in config:
            try:
                value = float(config[field])
                if value < min_val or value > max_val:
                    validation_result["warnings"].append(
                        f"{field} value {value} is outside recommended range [{min_val}, {max_val}]"
                    )
            except (ValueError, TypeError):
                validation_result["issues"].append(f"{field} must be a numeric value")
                validation_result["valid"] = False
    
    # Add recommendations
    if "google_api_key" not in config:
        validation_result["recommendations"].append("Consider setting google_api_key for LLM functionality")
    
    if "chromadb_path" not in config:
        validation_result["recommendations"].append("Consider setting chromadb_path for vector search")
    
    return validation_result


def create_temp_file(suffix: str = ".tmp", prefix: str = "mcp_agent_") -> Path:
    """Create a temporary file and return its path.
    
    Args:
        suffix: File suffix/extension.
        prefix: File name prefix.
        
    Returns:
        Path: Path to the created temporary file.
        
    Example:
        >>> temp_file = create_temp_file(".json", "data_")
        >>> print(temp_file)  # Path to temporary file
    """
    import tempfile
    
    # Create temporary file
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    
    # Close file descriptor since we just need the path
    import os
    os.close(fd)
    
    return Path(path)


_temp_files: List[Path] = []


def cleanup_temp_files() -> int:
    """Clean up temporary files created by create_temp_file.
    
    Returns:
        int: Number of files cleaned up.
        
    Example:
        >>> cleanup_count = cleanup_temp_files()
        >>> print(f"Cleaned up {cleanup_count} temporary files")
    """
    global _temp_files
    
    cleaned_count = 0
    remaining_files = []
    
    for temp_file in _temp_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
                cleaned_count += 1
        except Exception:
            # Keep track of files we couldn't clean up
            remaining_files.append(temp_file)
    
    _temp_files = remaining_files
    return cleaned_count


def get_logger_for_module(module_name: str, component_type: str = "general") -> Any:
    """Get appropriately configured logger for a module.
    
    Args:
        module_name: Name of the module (__name__).
        component_type: Type of component (agent, tool, workflow, etc.).
        
    Returns:
        Logger instance with appropriate context.
        
    Example:
        >>> logger = get_logger_for_module(__name__, "agent")
        >>> logger.info("Module initialized")
    """
    try:
        # Use specialized loggers if available
        if component_type == "agent":
            # Extract agent name from module path
            agent_name = module_name.split('.')[-1] if '.' in module_name else module_name
            return get_agent_logger(agent_name)
        elif component_type == "tool":
            tool_name = module_name.split('.')[-1] if '.' in module_name else module_name
            return get_tool_logger(tool_name)
        elif component_type == "workflow":
            workflow_name = module_name.split('.')[-1] if '.' in module_name else module_name
            return get_workflow_logger(workflow_name)
        else:
            return get_logger(module_name, component=component_type)
            
    except NameError:
        # Fallback to basic logger
        import logging
        return logging.getLogger(module_name)


# Package-level constants
UTILS_VERSION = "0.1.0"
REQUIRED_PYTHON = ">=3.12"

# Default configurations
DEFAULT_LOG_CONFIG = {
    "level": "INFO",
    "enable_file_logging": True,
    "enable_json_logging": False,
    "enable_colors": True,
}

DEVELOPMENT_LOG_CONFIG = {
    "level": "DEBUG",
    "enable_file_logging": True,
    "enable_json_logging": False,
    "enable_colors": True,
}

PRODUCTION_LOG_CONFIG = {
    "level": "INFO", 
    "enable_file_logging": True,
    "enable_json_logging": True,
    "enable_colors": False,
}

# Utility categories
UTILITY_CATEGORIES = {
    "logging": [
        "setup_logger", "get_logger", "configure_logging_from_env",
        "LoggerMixin", "log_function_call"
    ],
    "text": [
        "sanitize_query", "extract_keywords", "format_tool_name", "truncate_text"
    ],
    "validation": [
        "validate_email", "validate_url", "validate_api_key", "validate_file_path"
    ],
    "time": [
        "get_timestamp", "parse_timestamp", "format_duration"
    ],
    "data": [
        "deep_merge_dicts", "flatten_dict", "safe_json_loads", "safe_json_dumps"
    ],
    "async": [
        "safe_async_call", "timeout_after", "gather_with_limit", "retry_async"
    ],
    "files": [
        "ensure_directory", "safe_read_file", "safe_write_file", "get_file_size_mb"
    ],
    "performance": [
        "Timer", "measure_memory_usage"
    ],
    "environment": [
        "get_env_bool", "get_env_int", "get_env_list"
    ],
}


def list_utilities_by_category(category: str) -> List[str]:
    """List utilities by category.
    
    Args:
        category: Category name (logging, text, validation, etc.).
        
    Returns:
        List[str]: List of utility functions in the category.
        
    Example:
        >>> text_utils = list_utilities_by_category("text")
        >>> print("Text utilities:", text_utils)
    """
    return UTILITY_CATEGORIES.get(category, [])


def get_all_utility_categories() -> List[str]:
    """Get list of all utility categories.
    
    Returns:
        List[str]: List of category names.
        
    Example:
        >>> categories = get_all_utility_categories()
        >>> print("Available categories:", categories)
    """
    return list(UTILITY_CATEGORIES.keys())


# Feature flags for optional utility components
UTILITY_FEATURES = {
    "structured_logging": True,
    "rich_console": True,
    "async_utilities": True,
    "performance_monitoring": True,
    "file_operations": True,
    "data_validation": True,
    "text_processing": True,
    "environment_parsing": True,
}


def list_utility_features() -> Dict[str, bool]:
    """List all available utility features and their status.
    
    Returns:
        Dict[str, bool]: Dictionary of feature names and availability.
        
    Example:
        >>> features = list_utility_features()
        >>> if features['structured_logging']:
        ...     print("Structured logging is available")
    """
    return UTILITY_FEATURES.copy()


# Backwards compatibility aliases
Logger = get_logger  # Alias for backwards compatibility
Sanitize = sanitize_query  # Alias for backwards compatibility
Validate = validate_config  # Alias for backwards compatibility

# Auto-setup for convenience
import os
if os.getenv("AUTO_SETUP_UTILS", "false").lower() == "true":
    try:
        log_level = os.getenv("LOG_LEVEL", "INFO")
        environment = os.getenv("ENVIRONMENT", "development")
        init_logging(log_level, environment)
    except Exception:
        # Silently continue if auto-setup fails
        pass