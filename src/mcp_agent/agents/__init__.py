"""Agent modules for the MCP Agent Framework.

This package contains the multi-agent system components including specialized agents
for coordination, research, and reporting. The architecture follows the DeerFlow
pattern with a hierarchical system of agents working together.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic agent usage:

    >>> from mcp_agent.agents import CoordinatorAgent, ResearcherAgent
    >>> coordinator = CoordinatorAgent(settings)
    >>> researcher = ResearcherAgent(settings)
    >>> await coordinator.initialize()

Architecture:
    The agent system follows a hierarchical pattern:
    
    ┌─────────────────┐
    │   Coordinator   │ ← Main orchestrator
    │     Agent       │
    └─────────┬───────┘
              │
        ┌─────┴─────┐
        │           │
    ┌───▼───┐   ┌───▼───┐
    │Research│   │Report │ ← Specialized agents
    │ Agent  │   │ Agent │
    └───────┘   └───────┘
"""

from typing import Dict, List, Optional, Type, Union

# Import core agent classes
try:
    from mcp_agent.agents.base import BaseAgent, AgentCapability, AgentState
    from mcp_agent.agents.coordinator import CoordinatorAgent
    from mcp_agent.agents.researcher import ResearcherAgent
    from mcp_agent.agents.reporter import ReporterAgent
    
    # Make all key agent classes available
    __all__ = [
        # Base agent infrastructure
        "BaseAgent",
        "AgentCapability", 
        "AgentState",
        
        # Specialized agent implementations
        "CoordinatorAgent",
        "ResearcherAgent", 
        "ReporterAgent",
        
        # Agent management utilities
        "create_agent",
        "get_available_agents",
        "AgentRegistry",
        "list_agent_capabilities",
        
        # Agent lifecycle functions
        "initialize_agents",
        "shutdown_agents",
    ]
    
    # Define agent capabilities for registry
    AGENT_CAPABILITIES = {
        "coordinator": [
            AgentCapability.WORKFLOW_ORCHESTRATION,
            AgentCapability.USER_INTERACTION,
            AgentCapability.STATE_MANAGEMENT,
        ],
        "researcher": [
            AgentCapability.WEB_SEARCH,
            AgentCapability.TOOL_DISCOVERY,
            AgentCapability.MCP_INTEGRATION,
            AgentCapability.VECTOR_SEARCH,
        ],
        "reporter": [
            AgentCapability.REPORT_GENERATION,
            AgentCapability.CONTENT_FORMATTING,
            AgentCapability.SUMMARY_CREATION,
        ],
    }
    
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(
        f"Agent components not available: {e}. "
        "This is normal during initial setup.",
        ImportWarning,
        stacklevel=2
    )
    
    # Provide minimal exports for development
    __all__ = [
        "create_agent",
        "get_available_agents",
        "list_agent_capabilities",
    ]
    
    # Mock classes for development
    class BaseAgent:
        """Mock BaseAgent for development."""
        pass
    
    class CoordinatorAgent:
        """Mock CoordinatorAgent for development."""
        pass
    
    class ResearcherAgent:
        """Mock ResearcherAgent for development."""
        pass
    
    class ReporterAgent:
        """Mock ReporterAgent for development."""
        pass
    
    # Mock enums
    class AgentCapability:
        """Mock AgentCapability for development."""
        WORKFLOW_ORCHESTRATION = "workflow_orchestration"
        USER_INTERACTION = "user_interaction"
        STATE_MANAGEMENT = "state_management"
        WEB_SEARCH = "web_search"
        TOOL_DISCOVERY = "tool_discovery"
        MCP_INTEGRATION = "mcp_integration"
        VECTOR_SEARCH = "vector_search"
        REPORT_GENERATION = "report_generation"
        CONTENT_FORMATTING = "content_formatting"
        SUMMARY_CREATION = "summary_creation"
    
    class AgentState:
        """Mock AgentState for development."""
        IDLE = "idle"
        WORKING = "working"
        ERROR = "error"
    
    AGENT_CAPABILITIES = {}


class AgentRegistry:
    """Registry for managing available agent types and their capabilities.
    
    This class provides a centralized way to discover, create, and manage
    different types of agents in the framework.
    
    Example:
        >>> registry = AgentRegistry()
        >>> agent = registry.create_agent("researcher", settings)
        >>> capabilities = registry.get_capabilities("researcher")
    """
    
    def __init__(self) -> None:
        """Initialize the agent registry."""
        self._agents: Dict[str, Type[BaseAgent]] = {}
        self._capabilities: Dict[str, List[AgentCapability]] = {}
        self._register_builtin_agents()
    
    def _register_builtin_agents(self) -> None:
        """Register the built-in agent types."""
        try:
            self._agents = {
                "base": BaseAgent,
                "coordinator": CoordinatorAgent,
                "researcher": ResearcherAgent,
                "reporter": ReporterAgent,
            }
            self._capabilities = AGENT_CAPABILITIES.copy()
        except NameError:
            # During development when classes aren't available
            pass
    
    def register_agent(
        self,
        name: str,
        agent_class: Type[BaseAgent],
        capabilities: List[AgentCapability],
    ) -> None:
        """Register a new agent type.
        
        Args:
            name: Unique name for the agent type.
            agent_class: Agent class to register.
            capabilities: List of capabilities this agent provides.
            
        Example:
            >>> registry.register_agent(
            ...     "custom_agent",
            ...     CustomAgent,
            ...     [AgentCapability.CUSTOM_FEATURE]
            ... )
        """
        self._agents[name] = agent_class
        self._capabilities[name] = capabilities
    
    def create_agent(self, agent_type: str, *args, **kwargs) -> BaseAgent:
        """Create an agent instance of the specified type.
        
        Args:
            agent_type: Type of agent to create.
            *args: Positional arguments for agent constructor.
            **kwargs: Keyword arguments for agent constructor.
            
        Returns:
            BaseAgent: Created agent instance.
            
        Raises:
            ValueError: If agent type is not registered.
            
        Example:
            >>> agent = registry.create_agent("researcher", settings)
        """
        if agent_type not in self._agents:
            available = list(self._agents.keys())
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")
        
        agent_class = self._agents[agent_type]
        return agent_class(*args, **kwargs)
    
    def get_capabilities(self, agent_type: str) -> List[AgentCapability]:
        """Get capabilities for a specific agent type.
        
        Args:
            agent_type: Agent type to query.
            
        Returns:
            List[AgentCapability]: List of capabilities.
            
        Raises:
            ValueError: If agent type is not registered.
        """
        if agent_type not in self._capabilities:
            available = list(self._capabilities.keys())
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")
        
        return self._capabilities[agent_type].copy()
    
    def list_agents(self) -> List[str]:
        """List all registered agent types.
        
        Returns:
            List[str]: List of agent type names.
        """
        return list(self._agents.keys())
    
    def get_agents_with_capability(self, capability: AgentCapability) -> List[str]:
        """Get agent types that have a specific capability.
        
        Args:
            capability: Capability to search for.
            
        Returns:
            List[str]: List of agent types with the capability.
        """
        matching_agents = []
        for agent_type, capabilities in self._capabilities.items():
            if capability in capabilities:
                matching_agents.append(agent_type)
        return matching_agents


# Global registry instance
_registry = AgentRegistry()


def create_agent(agent_type: str, *args, **kwargs) -> BaseAgent:
    """Create an agent instance using the global registry.
    
    Args:
        agent_type: Type of agent to create.
        *args: Positional arguments for agent constructor.
        **kwargs: Keyword arguments for agent constructor.
        
    Returns:
        BaseAgent: Created agent instance.
        
    Example:
        >>> from mcp_agent.agents import create_agent
        >>> coordinator = create_agent("coordinator", settings)
        >>> researcher = create_agent("researcher", settings)
    """
    return _registry.create_agent(agent_type, *args, **kwargs)


def get_available_agents() -> List[str]:
    """Get list of available agent types.
    
    Returns:
        List[str]: List of agent type names.
        
    Example:
        >>> agents = get_available_agents()
        >>> print("Available agents:", agents)
        ['coordinator', 'researcher', 'reporter']
    """
    return _registry.list_agents()


def list_agent_capabilities() -> Dict[str, List[str]]:
    """List capabilities for all registered agents.
    
    Returns:
        Dict[str, List[str]]: Mapping of agent types to their capabilities.
        
    Example:
        >>> capabilities = list_agent_capabilities()
        >>> print(capabilities["researcher"])
        ['web_search', 'tool_discovery', 'mcp_integration']
    """
    result = {}
    for agent_type in _registry.list_agents():
        try:
            capabilities = _registry.get_capabilities(agent_type)
            # Convert enum values to strings for JSON serialization
            result[agent_type] = [cap.value if hasattr(cap, 'value') else str(cap) 
                                  for cap in capabilities]
        except (ValueError, AttributeError):
            result[agent_type] = []
    return result


def get_agents_with_capability(capability: Union[str, AgentCapability]) -> List[str]:
    """Get agents that have a specific capability.
    
    Args:
        capability: Capability to search for (string or enum).
        
    Returns:
        List[str]: Agent types with the capability.
        
    Example:
        >>> agents = get_agents_with_capability("web_search")
        >>> print(f"Agents with web search: {agents}")
    """
    # Convert string to enum if needed
    if isinstance(capability, str):
        try:
            # Try to find matching capability enum value
            for cap in AgentCapability.__dict__.values():
                if hasattr(cap, 'value') and cap.value == capability:
                    capability = cap
                    break
                elif str(cap) == capability:
                    capability = cap
                    break
        except (AttributeError, TypeError):
            pass
    
    return _registry.get_agents_with_capability(capability)


async def initialize_agents(*agents: BaseAgent) -> None:
    """Initialize multiple agents concurrently.
    
    Args:
        *agents: Agent instances to initialize.
        
    Example:
        >>> coordinator = create_agent("coordinator", settings)
        >>> researcher = create_agent("researcher", settings)
        >>> await initialize_agents(coordinator, researcher)
    """
    import asyncio
    
    # Initialize all agents concurrently
    tasks = [agent.initialize() for agent in agents if hasattr(agent, 'initialize')]
    if tasks:
        await asyncio.gather(*tasks)


async def shutdown_agents(*agents: BaseAgent) -> None:
    """Shutdown multiple agents concurrently.
    
    Args:
        *agents: Agent instances to shutdown.
        
    Example:
        >>> await shutdown_agents(coordinator, researcher)
    """
    import asyncio
    
    # Shutdown all agents concurrently
    tasks = [agent.shutdown() for agent in agents if hasattr(agent, 'shutdown')]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# Package-level constants
AGENT_TYPES = ["coordinator", "researcher", "reporter"]
DEFAULT_AGENT_TYPE = "coordinator"

# Agent role descriptions
AGENT_DESCRIPTIONS = {
    "coordinator": "Orchestrates workflows and manages agent collaboration",
    "researcher": "Handles tool discovery, web search, and data gathering",
    "reporter": "Generates reports, summaries, and formatted outputs",
}

# Compatibility aliases
Coordinator = CoordinatorAgent
Researcher = ResearcherAgent
Reporter = ReporterAgent

# Version compatibility
AGENTS_VERSION = "0.1.0"
REQUIRED_PYTHON = ">=3.12"