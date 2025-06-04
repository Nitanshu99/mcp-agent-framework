"""Logging configuration and utilities for the MCP Agent Framework.

This module provides structured logging capabilities using structlog and rich for
enhanced console output. It supports both file and console logging with configurable
levels and formats.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic logging setup:

    >>> from mcp_agent.utils.logger import get_logger, setup_logger
    >>> setup_logger("INFO")
    >>> logger = get_logger(__name__)
    >>> logger.info("Agent initialized", model="gemini-1.5-pro")

    Custom logging configuration:

    >>> setup_logger(
    ...     level="DEBUG",
    ...     log_file="./logs/custom.log",
    ...     enable_file_logging=True
    ... )
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

# Install rich traceback handling
install_rich_traceback(show_locals=True)

# Initialize rich console
console = Console()

# Global logging configuration
_logging_configured = False
_loggers: Dict[str, structlog.stdlib.BoundLogger] = {}


def setup_logger(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    enable_file_logging: bool = True,
    log_format: Optional[str] = None,
    enable_json_logging: bool = False,
    enable_colors: bool = True,
) -> None:
    """Set up comprehensive logging configuration for the framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int.
        log_file: Path to log file. If None, uses default from environment.
        enable_file_logging: Whether to enable file logging.
        log_format: Custom log format string. If None, uses default.
        enable_json_logging: Whether to use JSON format for structured logging.
        enable_colors: Whether to enable colored console output.
        
    Example:
        >>> setup_logger(
        ...     level="DEBUG",
        ...     log_file="./logs/mcp_agent.log",
        ...     enable_file_logging=True,
        ...     enable_json_logging=True
        ... )
    """
    global _logging_configured
    
    if _logging_configured:
        return
    
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Default log format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="ISO", utc=True),
    ]
    
    if enable_json_logging:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=enable_colors))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[],  # We'll add handlers manually
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Remove any existing handlers
    
    # Console handler with Rich
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=True,
        enable_link_path=True,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    console_handler.setLevel(level)
    
    # Console formatter
    console_formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if enabled
    if enable_file_logging and log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        
        if enable_json_logging:
            # JSON formatter for file logging
            file_formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
            )
        else:
            # Standard formatter for file logging
            file_formatter = logging.Formatter(
                fmt=log_format,
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set levels for third-party loggers to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    _logging_configured = True


def get_logger(name: str, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance with optional initial context.
    
    Args:
        name: Logger name (typically __name__).
        **initial_values: Initial context values to bind to the logger.
        
    Returns:
        structlog.stdlib.BoundLogger: Configured logger instance.
        
    Example:
        >>> logger = get_logger(__name__, component="mcp_client")
        >>> logger.info("Connection established", server="bio_tools")
        >>> 
        >>> # Logger with agent context
        >>> agent_logger = get_logger("agents.coordinator", agent_id="coord_001")
        >>> agent_logger.debug("Processing search query", query="RNA-seq tools")
    """
    # Ensure logging is configured
    if not _logging_configured:
        setup_logger()
    
    # Check if we already have this logger
    cache_key = f"{name}:{hash(tuple(sorted(initial_values.items())))}"
    if cache_key in _loggers:
        return _loggers[cache_key]
    
    # Create new logger
    logger = structlog.get_logger(name)
    
    # Bind initial values if provided
    if initial_values:
        logger = logger.bind(**initial_values)
    
    # Cache the logger
    _loggers[cache_key] = logger
    
    return logger


def configure_logging_from_env() -> None:
    """Configure logging from environment variables.
    
    Reads configuration from environment variables:
    - LOG_LEVEL: Logging level (default: INFO)
    - LOG_FORMAT: Log format string
    - LOG_FILE: Log file path (default: ./logs/mcp_agent.log)
    - ENABLE_FILE_LOGGING: Whether to enable file logging (default: true)
    
    Example:
        >>> import os
        >>> os.environ["LOG_LEVEL"] = "DEBUG"
        >>> os.environ["LOG_FILE"] = "./logs/debug.log"
        >>> configure_logging_from_env()
    """
    import os
    
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT")
    log_file = os.getenv("LOG_FILE", "./logs/mcp_agent.log")
    enable_file_logging = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
    
    setup_logger(
        level=level,
        log_file=log_file if enable_file_logging else None,
        enable_file_logging=enable_file_logging,
        log_format=log_format,
    )


class LoggerMixin:
    """Mixin class to add logging capabilities to any class.
    
    This mixin automatically creates a logger with the class name as context
    and provides convenient logging methods.
    
    Example:
        >>> class MyAgent(LoggerMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.log_info("Agent initialized")
        ...     
        ...     def process_query(self, query: str):
        ...         self.log_debug("Processing query", query=query)
        ...         # ... processing logic ...
        ...         self.log_info("Query processed successfully")
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the mixin and create a logger."""
        super().__init__(*args, **kwargs)
        self._logger = get_logger(
            self.__class__.__name__,
            class_name=self.__class__.__name__,
        )
    
    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message with optional context."""
        self._logger.debug(message, **kwargs)
    
    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log an info message with optional context."""
        self._logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message with optional context."""
        self._logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs: Any) -> None:
        """Log an error message with optional context."""
        self._logger.error(message, **kwargs)
    
    def log_critical(self, message: str, **kwargs: Any) -> None:
        """Log a critical message with optional context."""
        self._logger.critical(message, **kwargs)
    
    def log_exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception with traceback and optional context."""
        self._logger.exception(message, **kwargs)


def log_function_call(func_name: str, **kwargs: Any):
    """Decorator to log function calls with parameters and results.
    
    Args:
        func_name: Name of the function for logging context.
        **kwargs: Additional context to include in logs.
        
    Example:
        >>> @log_function_call("search_tools")
        ... async def search_tools(query: str, max_results: int = 10):
        ...     # Function implementation
        ...     return results
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def async_wrapper(*args, **func_kwargs):
            logger = get_logger(func.__module__, function=func_name)
            
            # Log function entry
            logger.debug(
                f"Entering {func_name}",
                args=args[1:] if args else [],  # Skip 'self' if present
                kwargs=func_kwargs,
                **kwargs
            )
            
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **func_kwargs)
                else:
                    result = func(*args, **func_kwargs)
                
                # Log successful completion
                logger.debug(
                    f"Completed {func_name}",
                    success=True,
                    **kwargs
                )
                
                return result
                
            except Exception as e:
                # Log exception
                logger.error(
                    f"Exception in {func_name}",
                    error=str(e),
                    error_type=type(e).__name__,
                    **kwargs
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **func_kwargs):
            logger = get_logger(func.__module__, function=func_name)
            
            # Log function entry
            logger.debug(
                f"Entering {func_name}",
                args=args[1:] if args else [],  # Skip 'self' if present
                kwargs=func_kwargs,
                **kwargs
            )
            
            try:
                # Execute function
                result = func(*args, **func_kwargs)
                
                # Log successful completion
                logger.debug(
                    f"Completed {func_name}",
                    success=True,
                    **kwargs
                )
                
                return result
                
            except Exception as e:
                # Log exception
                logger.error(
                    f"Exception in {func_name}",
                    error=str(e),
                    error_type=type(e).__name__,
                    **kwargs
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def create_audit_logger(name: str, audit_file: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Create a dedicated audit logger for tracking important operations.
    
    Args:
        name: Name for the audit logger.
        audit_file: Optional path to audit log file.
        
    Returns:
        structlog.stdlib.BoundLogger: Configured audit logger.
        
    Example:
        >>> audit_logger = create_audit_logger("user_actions", "./logs/audit.log")
        >>> audit_logger.info(
        ...     "User search performed",
        ...     user_id="user123",
        ...     query="BLAST tools",
        ...     results_count=15
        ... )
    """
    # Create audit-specific logger
    audit_logger = logging.getLogger(f"audit.{name}")
    
    # Set up audit file handler if specified
    if audit_file:
        audit_path = Path(audit_file)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        
        audit_handler = logging.FileHandler(audit_file, encoding="utf-8")
        audit_handler.setLevel(logging.INFO)
        
        # Use JSON format for audit logs
        audit_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger.addHandler(audit_handler)
    
    # Return structured logger
    return structlog.get_logger(f"audit.{name}")


def get_performance_logger() -> structlog.stdlib.BoundLogger:
    """Get a logger specifically for performance metrics.
    
    Returns:
        structlog.stdlib.BoundLogger: Performance logger instance.
        
    Example:
        >>> perf_logger = get_performance_logger()
        >>> start_time = time.time()
        >>> # ... operation ...
        >>> perf_logger.info(
        ...     "Search operation completed",
        ...     operation="vector_search",
        ...     duration_ms=(time.time() - start_time) * 1000,
        ...     results_count=42
        ... )
    """
    return get_logger("performance", component="metrics")


def silence_noisy_loggers() -> None:
    """Silence commonly noisy third-party loggers.
    
    This function sets higher log levels for third-party libraries that
    tend to be verbose and don't provide useful information in most cases.
    
    Example:
        >>> silence_noisy_loggers()
    """
    noisy_loggers = [
        "httpx",
        "httpcore", 
        "urllib3",
        "asyncio",
        "chromadb",
        "sentence_transformers",
        "transformers",
        "torch",
        "tensorflow",
        "google.auth",
        "google.api_core",
        "langchain.schema",
        "openai",
        "anthropic",
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def configure_development_logging() -> None:
    """Configure logging optimized for development.
    
    Sets up verbose logging with colors and rich formatting for development
    and debugging purposes.
    
    Example:
        >>> configure_development_logging()
    """
    setup_logger(
        level="DEBUG",
        log_file="./logs/development.log",
        enable_file_logging=True,
        enable_json_logging=False,
        enable_colors=True,
    )


def configure_production_logging() -> None:
    """Configure logging optimized for production.
    
    Sets up structured JSON logging with appropriate log levels for
    production deployment.
    
    Example:
        >>> configure_production_logging()
    """
    setup_logger(
        level="INFO",
        log_file="./logs/production.log",
        enable_file_logging=True,
        enable_json_logging=True,
        enable_colors=False,
    )


# Convenience function to get commonly used loggers
def get_agent_logger(agent_name: str, agent_id: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a logger configured for agent operations.
    
    Args:
        agent_name: Name of the agent (coordinator, researcher, reporter).
        agent_id: Optional unique agent instance ID.
        
    Returns:
        structlog.stdlib.BoundLogger: Agent logger instance.
        
    Example:
        >>> logger = get_agent_logger("coordinator", "coord_001")
        >>> logger.info("Agent started", capabilities=["orchestration", "user_interaction"])
    """
    context = {"agent": agent_name, "component": "agent"}
    if agent_id:
        context["agent_id"] = agent_id
    
    return get_logger(f"agents.{agent_name}", **context)


def get_tool_logger(tool_name: str, tool_id: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a logger configured for tool operations.
    
    Args:
        tool_name: Name of the tool (mcp_client, vector_store, etc.).
        tool_id: Optional unique tool instance ID.
        
    Returns:
        structlog.stdlib.BoundLogger: Tool logger instance.
        
    Example:
        >>> logger = get_tool_logger("mcp_client", "mcp_001")
        >>> logger.info("Tool initialized", servers=["bio_tools", "web_search"])
    """
    context = {"tool": tool_name, "component": "tool"}
    if tool_id:
        context["tool_id"] = tool_id
    
    return get_logger(f"tools.{tool_name}", **context)


def get_workflow_logger(workflow_name: str, execution_id: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a logger configured for workflow operations.
    
    Args:
        workflow_name: Name of the workflow (research, search, etc.).
        execution_id: Optional unique workflow execution ID.
        
    Returns:
        structlog.stdlib.BoundLogger: Workflow logger instance.
        
    Example:
        >>> logger = get_workflow_logger("research", "exec_12345")
        >>> logger.info("Workflow started", topic="RNA sequencing analysis")
    """
    context = {"workflow": workflow_name, "component": "workflow"}
    if execution_id:
        context["execution_id"] = execution_id
    
    return get_logger(f"workflows.{workflow_name}", **context)


# Initialize logging on module import if environment variables are set
import os
if os.getenv("AUTO_CONFIGURE_LOGGING", "true").lower() == "true":
    try:
        configure_logging_from_env()
    except Exception:
        # Silently fall back to basic configuration if environment setup fails
        setup_logger()


# Package constants
LOGGER_VERSION = "0.1.0"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = "./logs/mcp_agent.log"