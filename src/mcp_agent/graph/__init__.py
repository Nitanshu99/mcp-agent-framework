"""Graph workflow orchestration for the MCP Agent Framework.

This package contains LangGraph-based workflow orchestration components including
workflow definitions, graph nodes, and state management for the multi-agent system.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic workflow usage:

    >>> from mcp_agent.graph import create_workflow, WorkflowExecutor
    >>> workflow = create_workflow()
    >>> result = await workflow.ainvoke({"query": "RNA-seq tools"})

Architecture:
    The graph workflow orchestrates the multi-agent system:
    
    ┌─────────────────┐
    │   User Query    │
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │   Coordinator   │ ← Entry point
    │     Node        │
    └─────────┬───────┘
              │
        ┌─────┴─────┐
        │           │
    ┌───▼───┐   ┌───▼───┐
    │Research│   │Report │ ← Specialized nodes
    │ Node   │   │ Node  │
    └───────┘   └───────┘
"""

from typing import Any, Dict, List, Optional, Type, Union

# Import core graph components
try:
    from mcp_agent.graph.workflow import (
        create_workflow,
        create_research_workflow,
        create_simple_search_workflow,
        AgentWorkflow,
        WorkflowConfig,
    )
    from mcp_agent.graph.nodes import (
        CoordinatorNode,
        ResearcherNode,
        ReporterNode,
        SearchNode,
        AnalysisNode,
        BaseWorkflowNode,
        NodeResult,
        NodeError,
    )
    
    # Make all key graph classes and functions available
    __all__ = [
        # Workflow creation and management
        "create_workflow",
        "create_research_workflow", 
        "create_simple_search_workflow",
        "AgentWorkflow",
        "WorkflowConfig",
        
        # Graph nodes
        "CoordinatorNode",
        "ResearcherNode",
        "ReporterNode", 
        "SearchNode",
        "AnalysisNode",
        "BaseWorkflowNode",
        
        # Node result types
        "NodeResult",
        "NodeError",
        
        # Workflow management utilities
        "WorkflowExecutor",
        "get_available_workflows",
        "validate_workflow",
        "create_custom_workflow",
        
        # Workflow lifecycle functions
        "initialize_workflow",
        "shutdown_workflow",
    ]
    
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(
        f"Graph components not available: {e}. "
        "This is normal during initial setup.",
        ImportWarning,
        stacklevel=2
    )
    
    # Provide minimal exports for development
    __all__ = [
        "create_workflow",
        "get_available_workflows", 
        "validate_workflow",
    ]
    
    # Mock classes for development
    class AgentWorkflow:
        """Mock AgentWorkflow for development."""
        pass
    
    class WorkflowConfig:
        """Mock WorkflowConfig for development."""
        pass
    
    class BaseWorkflowNode:
        """Mock BaseWorkflowNode for development."""
        pass
    
    class CoordinatorNode:
        """Mock CoordinatorNode for development."""
        pass
    
    class ResearcherNode:
        """Mock ResearcherNode for development."""
        pass
    
    class ReporterNode:
        """Mock ReporterNode for development."""
        pass
    
    class NodeResult:
        """Mock NodeResult for development."""
        pass
    
    class NodeError:
        """Mock NodeError for development."""
        pass


class WorkflowExecutor:
    """Executor for managing workflow execution and lifecycle.
    
    This class provides a high-level interface for creating, executing,
    and managing LangGraph workflows in the MCP Agent Framework.
    
    Example:
        >>> executor = WorkflowExecutor()
        >>> workflow = executor.create_workflow("research")
        >>> result = await executor.execute(workflow, {"query": "BLAST tools"})
    """
    
    def __init__(self) -> None:
        """Initialize the workflow executor."""
        self._workflows: Dict[str, Any] = {}
        self._workflow_configs: Dict[str, WorkflowConfig] = {}
        self._register_builtin_workflows()
    
    def _register_builtin_workflows(self) -> None:
        """Register the built-in workflow types."""
        try:
            self._workflow_configs = {
                "default": WorkflowConfig(
                    name="default",
                    description="Default multi-agent workflow",
                    nodes=["coordinator", "researcher", "reporter"],
                    entry_point="coordinator",
                ),
                "research": WorkflowConfig(
                    name="research",
                    description="Research-focused workflow",
                    nodes=["coordinator", "researcher", "analysis", "reporter"],
                    entry_point="coordinator",
                ),
                "search": WorkflowConfig(
                    name="search", 
                    description="Simple search workflow",
                    nodes=["search", "analysis"],
                    entry_point="search",
                ),
            }
        except NameError:
            # During development when classes aren't available
            pass
    
    def create_workflow(
        self,
        workflow_type: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create a workflow instance of the specified type.
        
        Args:
            workflow_type: Type of workflow to create.
            config: Optional workflow configuration.
            
        Returns:
            Any: Created workflow instance.
            
        Raises:
            ValueError: If workflow type is not registered.
            
        Example:
            >>> workflow = executor.create_workflow("research")
        """
        if workflow_type not in self._workflow_configs:
            available = list(self._workflow_configs.keys())
            raise ValueError(f"Unknown workflow type: {workflow_type}. Available: {available}")
        
        try:
            if workflow_type == "research":
                return create_research_workflow(config)
            elif workflow_type == "search":
                return create_simple_search_workflow(config)
            else:
                return create_workflow(config)
        except NameError:
            # Return mock workflow during development
            return None
    
    async def execute(
        self,
        workflow: Any,
        input_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow with the given input.
        
        Args:
            workflow: Workflow instance to execute.
            input_data: Input data for the workflow.
            config: Optional execution configuration.
            
        Returns:
            Dict[str, Any]: Workflow execution result.
            
        Example:
            >>> result = await executor.execute(workflow, {"query": "tools"})
        """
        if workflow is None:
            # Mock execution during development
            return {"result": "Mock workflow execution", "input": input_data}
        
        try:
            # Execute the LangGraph workflow
            result = await workflow.ainvoke(input_data, config=config)
            return result
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "input": input_data,
            }
    
    def get_workflow_config(self, workflow_type: str) -> Optional[WorkflowConfig]:
        """Get configuration for a workflow type.
        
        Args:
            workflow_type: Workflow type to get config for.
            
        Returns:
            Optional[WorkflowConfig]: Workflow configuration or None.
        """
        return self._workflow_configs.get(workflow_type)
    
    def list_workflows(self) -> List[str]:
        """List all available workflow types.
        
        Returns:
            List[str]: List of workflow type names.
        """
        return list(self._workflow_configs.keys())
    
    def register_workflow(
        self,
        name: str,
        config: WorkflowConfig,
        factory_func: Any,
    ) -> None:
        """Register a custom workflow type.
        
        Args:
            name: Unique name for the workflow type.
            config: Workflow configuration.
            factory_func: Function to create workflow instances.
            
        Example:
            >>> executor.register_workflow(
            ...     "custom",
            ...     WorkflowConfig(name="custom", ...),
            ...     create_custom_workflow
            ... )
        """
        self._workflow_configs[name] = config
        self._workflows[name] = factory_func


# Global executor instance
_executor = WorkflowExecutor()


def create_workflow(config: Optional[Dict[str, Any]] = None) -> Any:
    """Create a default workflow using the global executor.
    
    Args:
        config: Optional workflow configuration.
        
    Returns:
        Any: Created workflow instance.
        
    Example:
        >>> from mcp_agent.graph import create_workflow
        >>> workflow = create_workflow()
        >>> result = await workflow.ainvoke({"query": "RNA-seq tools"})
    """
    return _executor.create_workflow("default", config)


def get_available_workflows() -> List[str]:
    """Get list of available workflow types.
    
    Returns:
        List[str]: List of workflow type names.
        
    Example:
        >>> workflows = get_available_workflows()
        >>> print("Available workflows:", workflows)
        ['default', 'research', 'search']
    """
    return _executor.list_workflows()


def validate_workflow(workflow_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate workflow configuration.
    
    Args:
        workflow_type: Workflow type to validate.
        config: Configuration to validate.
        
    Returns:
        Dict[str, Any]: Validation results.
        
    Example:
        >>> validation = validate_workflow("research", {"nodes": ["coordinator"]})
        >>> if validation["valid"]:
        ...     print("Configuration is valid")
    """
    validation_result = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "recommendations": [],
    }
    
    workflow_config = _executor.get_workflow_config(workflow_type)
    if not workflow_config:
        validation_result["valid"] = False
        validation_result["issues"].append(f"Unknown workflow type: {workflow_type}")
        return validation_result
    
    # Validate required configuration
    required_fields = ["entry_point"]
    for field in required_fields:
        if field not in config:
            validation_result["warnings"].append(f"Missing recommended field: {field}")
    
    # Validate node configuration
    if "nodes" in config:
        provided_nodes = config["nodes"]
        if not isinstance(provided_nodes, list):
            validation_result["issues"].append("'nodes' must be a list")
            validation_result["valid"] = False
    
    return validation_result


def create_custom_workflow(
    nodes: List[str],
    edges: List[Dict[str, str]],
    entry_point: str,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a custom workflow with specified nodes and edges.
    
    Args:
        nodes: List of node names to include.
        edges: List of edge definitions.
        entry_point: Entry point node name.
        config: Optional additional configuration.
        
    Returns:
        Any: Created custom workflow.
        
    Example:
        >>> workflow = create_custom_workflow(
        ...     nodes=["search", "analysis"],
        ...     edges=[{"from": "search", "to": "analysis"}],
        ...     entry_point="search"
        ... )
    """
    # This would be implemented when workflow.py is created
    # For now, return a mock
    return None


async def initialize_workflow(workflow: Any) -> None:
    """Initialize a workflow and its components.
    
    Args:
        workflow: Workflow instance to initialize.
        
    Example:
        >>> workflow = create_workflow()
        >>> await initialize_workflow(workflow)
    """
    if hasattr(workflow, 'initialize'):
        await workflow.initialize()


async def shutdown_workflow(workflow: Any) -> None:
    """Shutdown a workflow and clean up resources.
    
    Args:
        workflow: Workflow instance to shutdown.
        
    Example:
        >>> await shutdown_workflow(workflow)
    """
    if hasattr(workflow, 'shutdown'):
        await workflow.shutdown()
    elif hasattr(workflow, 'close'):
        await workflow.close()


def get_workflow_metrics(workflow: Any) -> Dict[str, Any]:
    """Get metrics from a workflow instance.
    
    Args:
        workflow: Workflow instance to get metrics from.
        
    Returns:
        Dict[str, Any]: Workflow metrics.
        
    Example:
        >>> metrics = get_workflow_metrics(workflow)
        >>> print(f"Executions: {metrics['total_executions']}")
    """
    if hasattr(workflow, 'get_metrics'):
        return workflow.get_metrics()
    
    return {
        "total_executions": 0,
        "successful_executions": 0,
        "failed_executions": 0,
        "average_execution_time": 0.0,
    }


# Package-level constants
WORKFLOW_TYPES = ["default", "research", "search"]
DEFAULT_WORKFLOW_TYPE = "default"

# Workflow configurations
DEFAULT_WORKFLOW_CONFIG = {
    "nodes": ["coordinator", "researcher", "reporter"],
    "entry_point": "coordinator",
    "max_iterations": 10,
    "timeout": 300,
}

RESEARCH_WORKFLOW_CONFIG = {
    "nodes": ["coordinator", "researcher", "analysis", "reporter"],
    "entry_point": "coordinator", 
    "max_iterations": 15,
    "timeout": 600,
}

SEARCH_WORKFLOW_CONFIG = {
    "nodes": ["search", "analysis"],
    "entry_point": "search",
    "max_iterations": 5,
    "timeout": 60,
}

# Node type mappings
NODE_TYPE_MAPPINGS = {
    "coordinator": CoordinatorNode,
    "researcher": ResearcherNode,
    "reporter": ReporterNode,
    "search": "SearchNode",
    "analysis": "AnalysisNode",
}

# Workflow features
WORKFLOW_FEATURES = {
    "multi_agent": True,
    "parallel_execution": True,
    "state_persistence": True,
    "error_recovery": True,
    "human_in_loop": True,
    "streaming": False,  # Future feature
    "checkpointing": False,  # Future feature
}

# Version compatibility
GRAPH_VERSION = "0.1.0"
LANGGRAPH_MIN_VERSION = "0.2.0"

# Backwards compatibility aliases
Workflow = AgentWorkflow  # Alias for backwards compatibility
Node = BaseWorkflowNode  # Alias for backwards compatibility