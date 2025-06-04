"""Tools and utilities for the MCP Agent Framework.

This package contains tools and utilities for interacting with external services,
databases, and APIs including MCP servers, vector stores, and search providers.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic tools usage:

    >>> from mcp_agent.tools import MCPClient, VectorStore
    >>> mcp_client = MCPClient(settings)
    >>> vector_store = VectorStore(settings)
    >>> await mcp_client.initialize()
    >>> await vector_store.initialize()
"""

from typing import Dict, List, Optional, Type, Any

# Import core tool classes
try:
    from mcp_agent.tools.mcp_client import MCPClient, MCPServerConnection
    from mcp_agent.tools.vector_store import VectorStore, EmbeddingProvider
    from mcp_agent.tools.search import SearchProvider, WebSearchEngine, BioinformaticsSearchEngine
    
    # Make all key tool classes available
    __all__ = [
        # MCP Integration
        "MCPClient",
        "MCPServerConnection",
        
        # Vector Database
        "VectorStore", 
        "EmbeddingProvider",
        
        # Search Tools
        "SearchProvider",
        "WebSearchEngine",
        "BioinformaticsSearchEngine",
        
        # Tool management utilities
        "ToolRegistry",
        "create_tool",
        "get_available_tools",
        "validate_tool_config",
        
        # Tool lifecycle functions
        "initialize_tools",
        "shutdown_tools",
    ]
    
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(
        f"Tool components not available: {e}. "
        "This is normal during initial setup.",
        ImportWarning,
        stacklevel=2
    )
    
    # Provide minimal exports for development
    __all__ = [
        "create_tool",
        "get_available_tools",
        "validate_tool_config",
    ]
    
    # Mock classes for development
    class MCPClient:
        """Mock MCPClient for development."""
        pass
    
    class MCPServerConnection:
        """Mock MCPServerConnection for development."""
        pass
    
    class VectorStore:
        """Mock VectorStore for development."""
        pass
    
    class EmbeddingProvider:
        """Mock EmbeddingProvider for development."""
        pass
    
    class SearchProvider:
        """Mock SearchProvider for development."""
        pass
    
    class WebSearchEngine:
        """Mock WebSearchEngine for development.""" 
        pass
    
    class BioinformaticsSearchEngine:
        """Mock BioinformaticsSearchEngine for development."""
        pass


class ToolRegistry:
    """Registry for managing available tools and their configurations.
    
    This class provides a centralized way to discover, create, and manage
    different types of tools in the framework.
    
    Example:
        >>> registry = ToolRegistry()
        >>> tool = registry.create_tool("mcp_client", settings)
        >>> available = registry.list_tools()
    """
    
    def __init__(self) -> None:
        """Initialize the tool registry."""
        self._tools: Dict[str, Type[Any]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_tools()
    
    def _register_builtin_tools(self) -> None:
        """Register the built-in tool types."""
        try:
            self._tools = {
                "mcp_client": MCPClient,
                "vector_store": VectorStore,
                "web_search": WebSearchEngine,
                "bioinformatics_search": BioinformaticsSearchEngine,
            }
            
            # Default configurations
            self._configs = {
                "mcp_client": {
                    "timeout": 30,
                    "max_retries": 3,
                    "connection_pool_size": 10,
                },
                "vector_store": {
                    "collection_name": "bioinformatics_tools",
                    "embedding_model": "text-embedding-3-small",
                    "similarity_threshold": 0.7,
                },
                "web_search": {
                    "providers": ["tavily", "brave", "serper"],
                    "max_results_per_provider": 10,
                    "timeout": 15,
                },
                "bioinformatics_search": {
                    "databases": ["pubmed", "bioconductor", "biostars"],
                    "max_results": 20,
                    "include_abstracts": True,
                },
            }
            
        except NameError:
            # During development when classes aren't available
            pass
    
    def register_tool(
        self,
        name: str,
        tool_class: Type[Any],
        default_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new tool type.
        
        Args:
            name: Unique name for the tool type.
            tool_class: Tool class to register.
            default_config: Default configuration for the tool.
            
        Example:
            >>> registry.register_tool(
            ...     "custom_search",
            ...     CustomSearchTool,
            ...     {"api_key": "required", "timeout": 30}
            ... )
        """
        self._tools[name] = tool_class
        if default_config:
            self._configs[name] = default_config
    
    def create_tool(self, tool_type: str, *args, **kwargs) -> Any:
        """Create a tool instance of the specified type.
        
        Args:
            tool_type: Type of tool to create.
            *args: Positional arguments for tool constructor.
            **kwargs: Keyword arguments for tool constructor.
            
        Returns:
            Any: Created tool instance.
            
        Raises:
            ValueError: If tool type is not registered.
            
        Example:
            >>> tool = registry.create_tool("mcp_client", settings)
        """
        if tool_type not in self._tools:
            available = list(self._tools.keys())
            raise ValueError(f"Unknown tool type: {tool_type}. Available: {available}")
        
        tool_class = self._tools[tool_type]
        
        # Merge default config with provided kwargs
        if tool_type in self._configs:
            config = self._configs[tool_type].copy()
            config.update(kwargs)
            kwargs = config
        
        return tool_class(*args, **kwargs)
    
    def get_tool_config(self, tool_type: str) -> Dict[str, Any]:
        """Get default configuration for a tool type.
        
        Args:
            tool_type: Tool type to get config for.
            
        Returns:
            Dict[str, Any]: Default configuration.
            
        Raises:
            ValueError: If tool type is not registered.
        """
        if tool_type not in self._configs:
            available = list(self._configs.keys())
            raise ValueError(f"Unknown tool type: {tool_type}. Available: {available}")
        
        return self._configs[tool_type].copy()
    
    def list_tools(self) -> List[str]:
        """List all registered tool types.
        
        Returns:
            List[str]: List of tool type names.
        """
        return list(self._tools.keys())
    
    def validate_tool_type(self, tool_type: str) -> bool:
        """Validate that a tool type is registered.
        
        Args:
            tool_type: Tool type to validate.
            
        Returns:
            bool: True if tool type is registered.
        """
        return tool_type in self._tools
    
    def get_tool_requirements(self, tool_type: str) -> Dict[str, Any]:
        """Get requirements for a tool type.
        
        Args:
            tool_type: Tool type to get requirements for.
            
        Returns:
            Dict[str, Any]: Tool requirements.
        """
        requirements = {
            "mcp_client": {
                "dependencies": ["langchain-mcp-adapters", "mcp"],
                "api_keys": [],
                "optional_features": ["stdio_transport", "http_transport"],
            },
            "vector_store": {
                "dependencies": ["chromadb", "sentence-transformers"],
                "api_keys": [],
                "optional_features": ["persistent_storage", "similarity_search"],
            },
            "web_search": {
                "dependencies": ["httpx", "aiohttp"],
                "api_keys": ["tavily_api_key", "brave_search_api_key", "serper_api_key"],
                "optional_features": ["multiple_providers", "result_caching"],
            },
            "bioinformatics_search": {
                "dependencies": ["biopython", "requests"],
                "api_keys": ["ncbi_api_key"],
                "optional_features": ["pubmed_search", "bioconductor_integration"],
            },
        }
        
        return requirements.get(tool_type, {})


# Global registry instance
_registry = ToolRegistry()


def create_tool(tool_type: str, *args, **kwargs) -> Any:
    """Create a tool instance using the global registry.
    
    Args:
        tool_type: Type of tool to create.
        *args: Positional arguments for tool constructor.
        **kwargs: Keyword arguments for tool constructor.
        
    Returns:
        Any: Created tool instance.
        
    Example:
        >>> from mcp_agent.tools import create_tool
        >>> mcp_client = create_tool("mcp_client", settings)
        >>> vector_store = create_tool("vector_store", settings)
    """
    return _registry.create_tool(tool_type, *args, **kwargs)


def get_available_tools() -> List[str]:
    """Get list of available tool types.
    
    Returns:
        List[str]: List of tool type names.
        
    Example:
        >>> tools = get_available_tools()
        >>> print("Available tools:", tools)
        ['mcp_client', 'vector_store', 'web_search', 'bioinformatics_search']
    """
    return _registry.list_tools()


def validate_tool_config(tool_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration for a tool type.
    
    Args:
        tool_type: Tool type to validate config for.
        config: Configuration to validate.
        
    Returns:
        Dict[str, Any]: Validation results with issues and recommendations.
        
    Example:
        >>> validation = validate_tool_config("mcp_client", {"timeout": 30})
        >>> if validation["valid"]:
        ...     print("Configuration is valid")
    """
    validation_result = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "recommendations": [],
    }
    
    if not _registry.validate_tool_type(tool_type):
        validation_result["valid"] = False
        validation_result["issues"].append(f"Unknown tool type: {tool_type}")
        return validation_result
    
    # Get requirements and default config
    requirements = _registry.get_tool_requirements(tool_type)
    default_config = _registry.get_tool_config(tool_type)
    
    # Check required API keys
    for api_key in requirements.get("api_keys", []):
        if api_key not in config and f"{api_key.upper()}" not in config:
            validation_result["warnings"].append(f"Missing API key: {api_key}")
    
    # Check for unknown config keys
    known_keys = set(default_config.keys())
    provided_keys = set(config.keys())
    unknown_keys = provided_keys - known_keys
    
    if unknown_keys:
        validation_result["warnings"].append(f"Unknown config keys: {unknown_keys}")
    
    # Check for missing recommended config
    missing_keys = known_keys - provided_keys
    if missing_keys:
        validation_result["recommendations"].append(
            f"Consider setting: {missing_keys}"
        )
    
    return validation_result


def get_tool_requirements(tool_type: str) -> Dict[str, Any]:
    """Get requirements for a specific tool type.
    
    Args:
        tool_type: Tool type to get requirements for.
        
    Returns:
        Dict[str, Any]: Tool requirements including dependencies and API keys.
        
    Example:
        >>> reqs = get_tool_requirements("web_search")
        >>> print("Required API keys:", reqs["api_keys"])
    """
    return _registry.get_tool_requirements(tool_type)


async def initialize_tools(*tools: Any) -> None:
    """Initialize multiple tools concurrently.
    
    Args:
        *tools: Tool instances to initialize.
        
    Example:
        >>> mcp_client = create_tool("mcp_client", settings)
        >>> vector_store = create_tool("vector_store", settings)
        >>> await initialize_tools(mcp_client, vector_store)
    """
    import asyncio
    
    # Initialize all tools concurrently
    tasks = []
    for tool in tools:
        if hasattr(tool, 'initialize'):
            tasks.append(tool.initialize())
    
    if tasks:
        await asyncio.gather(*tasks)


async def shutdown_tools(*tools: Any) -> None:
    """Shutdown multiple tools concurrently.
    
    Args:
        *tools: Tool instances to shutdown.
        
    Example:
        >>> await shutdown_tools(mcp_client, vector_store)
    """
    import asyncio
    
    # Shutdown all tools concurrently
    tasks = []
    for tool in tools:
        if hasattr(tool, 'shutdown'):
            tasks.append(tool.shutdown())
        elif hasattr(tool, 'close'):
            tasks.append(tool.close())
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def check_tool_dependencies() -> Dict[str, Dict[str, bool]]:
    """Check if dependencies are available for all tool types.
    
    Returns:
        Dict[str, Dict[str, bool]]: Dependency availability for each tool type.
        
    Example:
        >>> deps = check_tool_dependencies()
        >>> if not deps["mcp_client"]["langchain-mcp-adapters"]:
        ...     print("MCP adapters not installed")
    """
    dependency_status = {}
    
    for tool_type in get_available_tools():
        requirements = get_tool_requirements(tool_type)
        tool_deps = {}
        
        for dep in requirements.get("dependencies", []):
            try:
                __import__(dep.replace("-", "_"))
                tool_deps[dep] = True
            except ImportError:
                tool_deps[dep] = False
        
        dependency_status[tool_type] = tool_deps
    
    return dependency_status


def get_tool_health_status() -> Dict[str, Any]:
    """Get health status of tool ecosystem.
    
    Returns:
        Dict[str, Any]: Overall health status and recommendations.
        
    Example:
        >>> health = get_tool_health_status()
        >>> print(f"Health score: {health['score']}/100")
    """
    dependencies = check_tool_dependencies()
    
    total_deps = 0
    available_deps = 0
    
    for tool_type, deps in dependencies.items():
        total_deps += len(deps)
        available_deps += sum(deps.values())
    
    health_score = (available_deps / total_deps * 100) if total_deps > 0 else 100
    
    missing_deps = []
    for tool_type, deps in dependencies.items():
        for dep, available in deps.items():
            if not available:
                missing_deps.append(f"{tool_type}: {dep}")
    
    return {
        "score": round(health_score, 1),
        "total_dependencies": total_deps,
        "available_dependencies": available_deps,
        "missing_dependencies": missing_deps,
        "recommendations": [
            f"Install missing dependency: {dep}" for dep in missing_deps[:5]
        ],
        "status": "healthy" if health_score >= 90 else "warning" if health_score >= 70 else "critical",
    }


# Package-level constants
TOOL_TYPES = ["mcp_client", "vector_store", "web_search", "bioinformatics_search"]
DEFAULT_TOOL_TYPE = "mcp_client"

# Tool categories
TOOL_CATEGORIES = {
    "integration": ["mcp_client"],
    "database": ["vector_store"],
    "search": ["web_search", "bioinformatics_search"],
}

# Version compatibility
TOOLS_VERSION = "0.1.0"
REQUIRED_PYTHON = ">=3.12"

# Feature flags for optional tool components
TOOL_FEATURES = {
    "mcp_integration": True,
    "vector_search": True,
    "web_search": True,
    "bioinformatics_search": True,
    "audio_processing": False,  # Future feature
    "real_time_sync": False,    # Future feature
}


def list_tool_features() -> Dict[str, bool]:
    """List all available tool features and their status.
    
    Returns:
        Dict[str, bool]: Dictionary of feature names and availability.
        
    Example:
        >>> features = list_tool_features()
        >>> if features['mcp_integration']:
        ...     print("MCP integration is available")
    """
    return TOOL_FEATURES.copy()


def get_tools_by_category(category: str) -> List[str]:
    """Get tool types by category.
    
    Args:
        category: Category name ('integration', 'database', 'search').
        
    Returns:
        List[str]: List of tool types in the category.
        
    Example:
        >>> search_tools = get_tools_by_category("search")
        >>> print("Search tools:", search_tools)
    """
    return TOOL_CATEGORIES.get(category, [])


# Backwards compatibility aliases
Client = MCPClient  # Alias for backwards compatibility
Store = VectorStore  # Alias for backwards compatibility
Search = WebSearchEngine  # Alias for backwards compatibility