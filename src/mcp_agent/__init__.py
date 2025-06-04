"""MCP Agent Framework - A basic Agent with Model Context Protocol capabilities.

This package provides a modular framework for building intelligent agents that can
interact with external tools and data sources through the Model Context Protocol (MCP).
The framework is specifically designed for bioinformatics tool discovery and research
automation, combining Google Gemini LLM with vector search capabilities.

Features:
    - Multi-agent architecture with specialized roles
    - MCP protocol integration for tool connectivity
    - ChromaDB vector database for semantic search
    - Google Gemini for intelligent reasoning
    - Type-safe implementation with comprehensive documentation

Example:
    Basic usage of the MCP Agent Framework:

    >>> from mcp_agent import MCPAgent
    >>> agent = MCPAgent()
    >>> result = await agent.search("RNA sequencing tools")
    >>> print(result.tools)

Attributes:
    __version__ (str): The version of the MCP Agent Framework.
    __author__ (str): The authors of the framework.
    __email__ (str): Contact email for the maintainers.
    __license__ (str): The license under which this software is distributed.
"""

from typing import List, Optional

# Package metadata
__version__ = "0.1.0"
__author__ = (
    "Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug, "
    "Nitanshu Mayur Idnani, Reeju Bhattacharjee"
)
__email__ = "fernando.delgado@example.com"
__license__ = "MIT"
__maintainer__ = __author__
__status__ = "Alpha"

# Package description
__description__ = (
    "A basic Agent with Model Context Protocol (MCP) capabilities "
    "for bioinformatics tool discovery and research automation"
)

# Repository information
__url__ = "https://github.com/your-org/mcp-agent-framework"
__download_url__ = "https://github.com/your-org/mcp-agent-framework/archive/main.zip"

# Keywords for package discovery
__keywords__ = [
    "ai", "llm", "mcp", "langchain", "langgraph", "bioinformatics",
    "semantic-search", "multi-agent", "research-automation", "vector-database"
]

# Version information
VERSION = __version__
VERSION_INFO = tuple(int(part) for part in __version__.split('.'))

# Import core components for easy access
try:
    # Core agent classes
    from mcp_agent.agents.coordinator import CoordinatorAgent
    from mcp_agent.agents.researcher import ResearcherAgent
    from mcp_agent.agents.reporter import ReporterAgent
    
    # Main agent interface
    from mcp_agent.main import MCPAgent
    
    # Configuration management
    from mcp_agent.config.settings import AgentSettings, get_settings
    
    # Data models
    from mcp_agent.models.schemas import (
        SearchQuery,
        SearchResult,
        ToolInfo,
        AgentResponse,
    )
    
    # Graph workflow
    from mcp_agent.graph.workflow import create_workflow
    
    # Core tools
    from mcp_agent.tools.mcp_client import MCPClient
    from mcp_agent.tools.vector_store import VectorStore
    
    # Utility functions
    from mcp_agent.utils.logger import get_logger
    
    # Make key classes available at package level
    __all__ = [
        # Core classes
        "MCPAgent",
        "CoordinatorAgent", 
        "ResearcherAgent",
        "ReporterAgent",
        
        # Configuration
        "AgentSettings",
        "get_settings",
        
        # Data models
        "SearchQuery",
        "SearchResult", 
        "ToolInfo",
        "AgentResponse",
        
        # Workflow
        "create_workflow",
        
        # Tools
        "MCPClient",
        "VectorStore",
        
        # Utilities
        "get_logger",
        
        # Package metadata
        "__version__",
        "__author__",
        "__email__",
        "__license__",
        "__description__",
        "__url__",
    ]
    
except ImportError as e:
    # If imports fail during development or testing, provide a minimal interface
    import warnings
    warnings.warn(
        f"Some MCP Agent components could not be imported: {e}. "
        "This is normal during initial setup or testing.",
        ImportWarning,
        stacklevel=2
    )
    
    # Provide minimal exports for development
    __all__ = [
        "__version__",
        "__author__", 
        "__email__",
        "__license__",
        "__description__",
        "__url__",
    ]


def get_version() -> str:
    """Get the current version of the MCP Agent Framework.
    
    Returns:
        str: The version string in semver format.
        
    Example:
        >>> from mcp_agent import get_version
        >>> print(get_version())
        '0.1.0'
    """
    return __version__


def get_package_info() -> dict[str, str]:
    """Get comprehensive package information.
    
    Returns:
        dict[str, str]: Dictionary containing package metadata.
        
    Example:
        >>> from mcp_agent import get_package_info
        >>> info = get_package_info()
        >>> print(info['version'])
        '0.1.0'
    """
    return {
        "name": "mcp-agent-framework",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": __description__,
        "url": __url__,
        "status": __status__,
        "keywords": ", ".join(__keywords__),
    }


def check_dependencies() -> dict[str, bool]:
    """Check if all required dependencies are available.
    
    Returns:
        dict[str, bool]: Dictionary mapping dependency names to availability status.
        
    Example:
        >>> from mcp_agent import check_dependencies
        >>> deps = check_dependencies()
        >>> if not deps['langchain']:
        ...     print("LangChain is not available")
    """
    dependencies = {
        "langchain": False,
        "langgraph": False,
        "chromadb": False,
        "google-generativeai": False,
        "langchain-mcp-adapters": False,
        "pydantic": False,
    }
    
    # Check each dependency
    try:
        import langchain
        dependencies["langchain"] = True
    except ImportError:
        pass
    
    try:
        import langgraph
        dependencies["langgraph"] = True
    except ImportError:
        pass
    
    try:
        import chromadb
        dependencies["chromadb"] = True
    except ImportError:
        pass
    
    try:
        import google.generativeai
        dependencies["google-generativeai"] = True
    except ImportError:
        pass
    
    try:
        import langchain_mcp_adapters
        dependencies["langchain-mcp-adapters"] = True
    except ImportError:
        pass
    
    try:
        import pydantic
        dependencies["pydantic"] = True
    except ImportError:
        pass
    
    return dependencies


def setup_logging(level: str = "INFO") -> None:
    """Set up package-wide logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Example:
        >>> from mcp_agent import setup_logging
        >>> setup_logging("DEBUG")
    """
    try:
        from mcp_agent.utils.logger import setup_logger
        setup_logger(level)
    except ImportError:
        import logging
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


# Package-level constants
DEFAULT_CONFIG_PATH = "conf.yaml"
DEFAULT_ENV_FILE = ".env"
DEFAULT_LOG_LEVEL = "INFO"

# Supported Python versions
PYTHON_REQUIRES = ">=3.12"

# Framework version compatibility
LANGCHAIN_MIN_VERSION = "0.3.0"
LANGGRAPH_MIN_VERSION = "0.2.0"
CHROMADB_MIN_VERSION = "0.5.0"

# Development status indicators
IS_ALPHA = True
IS_BETA = False
IS_STABLE = False

# Feature flags for optional components
FEATURES = {
    "mcp_integration": True,
    "vector_search": True,
    "audio_reports": True,  # TTS functionality
    "web_search": True,     # External search APIs
    "monitoring": False,    # Metrics and monitoring (future)
    "api_server": False,    # REST API server (future)
}


def list_features() -> dict[str, bool]:
    """List all available features and their status.
    
    Returns:
        dict[str, bool]: Dictionary of feature names and availability.
        
    Example:
        >>> from mcp_agent import list_features
        >>> features = list_features()
        >>> if features['mcp_integration']:
        ...     print("MCP integration is available")
    """
    return FEATURES.copy()


# Backwards compatibility aliases
Agent = None  # Will be set to MCPAgent when available
Settings = None  # Will be set to AgentSettings when available

try:
    from mcp_agent.main import MCPAgent
    from mcp_agent.config.settings import AgentSettings
    Agent = MCPAgent
    Settings = AgentSettings
except ImportError:
    pass