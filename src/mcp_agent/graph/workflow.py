"""LangGraph workflow creation and orchestration.

This module contains the workflow creation logic that connects graph nodes
into executable LangGraph workflows for different use cases in the MCP Agent Framework.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic workflow creation:

    >>> workflow = create_workflow()
    >>> result = await workflow.ainvoke({"user_query": "RNA-seq tools"})

    Custom workflow:

    >>> config = WorkflowConfig(
    ...     name="custom",
    ...     entry_point="search",
    ...     max_iterations=5
    ... )
    >>> workflow = create_custom_workflow(config)

Architecture:
    Workflows orchestrate multi-agent collaboration:
    
    ┌─────────────────┐
    │  Start: Query   │
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │  Coordinator    │ ← Route to appropriate path
    └─────────┬───────┘
              │
        ┌─────┴─────┐
        │           │
    ┌───▼───┐   ┌───▼───┐
    │Search │   │Research│ ← Parallel or sequential
    │       │   │        │
    └───┬───┘   └───┬───┘
        │           │
        └─────┬─────┘
              │
    ┌─────────▼───────┐
    │    Reporter     │ ← Generate final output
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │   End: Result   │
    └─────────────────┘
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import ToolNode
    
    from mcp_agent.agents.coordinator import CoordinatorAgent
    from mcp_agent.agents.researcher import ResearcherAgent
    from mcp_agent.agents.reporter import ReporterAgent
    from mcp_agent.config.settings import AgentSettings
    from mcp_agent.graph.nodes import (
        WorkflowState,
        CoordinatorNode,
        ResearcherNode,
        ReporterNode,
        SearchNode,
        AnalysisNode,
        BaseWorkflowNode,
        NodeResult,
        NodeStatus,
    )
    from mcp_agent.tools.mcp_client import MCPClient
    from mcp_agent.tools.vector_store import VectorStore
    from mcp_agent.utils.logger import get_logger
except ImportError as e:
    # Mock imports for development
    import warnings
    warnings.warn(f"Workflow dependencies not available: {e}", ImportWarning)
    
    class StateGraph:
        def __init__(self, *args, **kwargs):
            pass
        def add_node(self, *args, **kwargs):
            pass
        def add_edge(self, *args, **kwargs):
            pass
        def add_conditional_edges(self, *args, **kwargs):
            pass
        def set_entry_point(self, *args, **kwargs):
            pass
        def compile(self, *args, **kwargs):
            return MockCompiledGraph()
    
    class MockCompiledGraph:
        async def ainvoke(self, *args, **kwargs):
            return {"result": "Mock workflow execution"}
    
    class MemorySaver:
        pass
    
    START = "start"
    END = "end"
    
    def add_messages(*args):
        return []
    
    class ToolNode:
        pass
    
    # Import mock classes
    class CoordinatorAgent:
        pass
    class ResearcherAgent:
        pass
    class ReporterAgent:
        pass
    class AgentSettings:
        pass
    class WorkflowState:
        pass
    class CoordinatorNode:
        pass
    class ResearcherNode:
        pass
    class ReporterNode:
        pass
    class SearchNode:
        pass
    class AnalysisNode:
        pass
    class BaseWorkflowNode:
        pass
    class NodeResult:
        pass
    class NodeStatus:
        pass
    class MCPClient:
        pass
    class VectorStore:
        pass
    def get_logger(name: str):
        import logging
        return logging.getLogger(name)


class WorkflowConfig(BaseModel):
    """Configuration for workflow creation and execution.
    
    Attributes:
        name: Workflow name.
        description: Workflow description.
        nodes: List of node names to include.
        entry_point: Entry point node name.
        max_iterations: Maximum workflow iterations.
        timeout: Workflow timeout in seconds.
        enable_checkpoints: Whether to enable checkpointing.
        enable_human_feedback: Whether to enable human feedback loops.
        
    Example:
        >>> config = WorkflowConfig(
        ...     name="research_workflow",
        ...     entry_point="coordinator",
        ...     max_iterations=10,
        ...     timeout=300
        ... )
    """
    
    name: str = Field(description="Workflow name")
    description: str = Field(default="", description="Workflow description")
    nodes: List[str] = Field(default_factory=list, description="Node names to include")
    entry_point: str = Field(default="coordinator", description="Entry point node")
    max_iterations: int = Field(default=10, description="Maximum iterations")
    timeout: int = Field(default=300, description="Timeout in seconds")
    enable_checkpoints: bool = Field(default=False, description="Enable checkpointing")
    enable_human_feedback: bool = Field(default=False, description="Enable human feedback")
    parallel_execution: bool = Field(default=False, description="Enable parallel node execution")
    error_recovery: bool = Field(default=True, description="Enable error recovery")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WorkflowMetrics(BaseModel):
    """Metrics for workflow execution.
    
    Attributes:
        workflow_id: Unique workflow execution ID.
        total_executions: Total number of executions.
        successful_executions: Number of successful executions.
        failed_executions: Number of failed executions.
        average_execution_time: Average execution time in seconds.
        node_metrics: Metrics for individual nodes.
        
    Example:
        >>> metrics = workflow.get_metrics()
        >>> print(f"Success rate: {metrics.success_rate:.1f}%")
    """
    
    workflow_id: str = Field(description="Workflow execution ID")
    total_executions: int = Field(default=0, description="Total executions")
    successful_executions: int = Field(default=0, description="Successful executions")
    failed_executions: int = Field(default=0, description="Failed executions")
    average_execution_time: float = Field(default=0.0, description="Average execution time")
    node_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Node metrics")
    last_execution: Optional[datetime] = Field(default=None, description="Last execution time")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100


class AgentWorkflow:
    """Main workflow class that wraps LangGraph execution.
    
    This class provides a high-level interface for creating and executing
    LangGraph workflows with the MCP Agent Framework components.
    
    Attributes:
        config: Workflow configuration.
        graph: LangGraph StateGraph instance.
        compiled_graph: Compiled graph ready for execution.
        nodes: Dictionary of workflow nodes.
        metrics: Execution metrics.
        
    Example:
        >>> workflow = AgentWorkflow(config)
        >>> await workflow.initialize(coordinator, researcher, reporter)
        >>> result = await workflow.execute({"user_query": "BLAST tools"})
    """
    
    def __init__(self, config: WorkflowConfig) -> None:
        """Initialize the workflow.
        
        Args:
            config: Workflow configuration.
        """
        self.config = config
        self.logger = get_logger(f"Workflow.{config.name}")
        
        # LangGraph components
        self.graph: Optional[StateGraph] = None
        self.compiled_graph: Optional[Any] = None
        
        # Workflow components
        self.nodes: Dict[str, BaseWorkflowNode] = {}
        self.metrics = WorkflowMetrics(workflow_id=f"{config.name}_{int(time.time())}")
        
        # State management
        self._initialized = False
        self._checkpointer = None
        
        self.logger.info(f"Workflow '{config.name}' created")
    
    async def initialize(
        self,
        coordinator: Optional[CoordinatorAgent] = None,
        researcher: Optional[ResearcherAgent] = None,
        reporter: Optional[ReporterAgent] = None,
        vector_store: Optional[VectorStore] = None,
        mcp_client: Optional[MCPClient] = None,
        settings: Optional[AgentSettings] = None,
    ) -> None:
        """Initialize the workflow with agent components.
        
        Args:
            coordinator: CoordinatorAgent instance.
            researcher: ResearcherAgent instance.
            reporter: ReporterAgent instance.
            vector_store: VectorStore instance.
            mcp_client: MCPClient instance.
            settings: AgentSettings instance.
            
        Raises:
            RuntimeError: If initialization fails.
        """
        try:
            self.logger.info("Initializing workflow...")
            
            # Create workflow nodes based on configuration
            await self._create_nodes(
                coordinator, researcher, reporter, vector_store, mcp_client, settings
            )
            
            # Build the graph structure
            await self._build_graph()
            
            # Compile the graph
            await self._compile_graph()
            
            self._initialized = True
            self.logger.info(f"Workflow '{self.config.name}' initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Workflow initialization failed: {e}")
            raise RuntimeError(f"Workflow initialization failed: {e}") from e
    
    async def _create_nodes(
        self,
        coordinator: Optional[CoordinatorAgent],
        researcher: Optional[ResearcherAgent],
        reporter: Optional[ReporterAgent],
        vector_store: Optional[VectorStore],
        mcp_client: Optional[MCPClient],
        settings: Optional[AgentSettings],
    ) -> None:
        """Create workflow nodes based on configuration."""
        for node_name in self.config.nodes:
            if node_name == "coordinator" and coordinator:
                self.nodes[node_name] = CoordinatorNode(coordinator, settings)
            elif node_name == "researcher" and researcher:
                self.nodes[node_name] = ResearcherNode(researcher, settings)
            elif node_name == "reporter" and reporter:
                self.nodes[node_name] = ReporterNode(reporter, settings)
            elif node_name == "search":
                self.nodes[node_name] = SearchNode(vector_store, None, settings)
            elif node_name == "analysis":
                self.nodes[node_name] = AnalysisNode(settings)
            else:
                self.logger.warning(f"Unknown or unavailable node type: {node_name}")
        
        self.logger.info(f"Created {len(self.nodes)} workflow nodes")
    
    async def _build_graph(self) -> None:
        """Build the LangGraph structure."""
        try:
            # Create state graph with WorkflowState
            self.graph = StateGraph(WorkflowState)
            
            # Add nodes to the graph
            for node_name, node in self.nodes.items():
                self.graph.add_node(node_name, self._create_node_function(node))
            
            # Add edges based on workflow type
            await self._add_workflow_edges()
            
            # Set entry point
            self.graph.set_entry_point(self.config.entry_point)
            
            self.logger.info("Graph structure built successfully")
            
        except Exception as e:
            self.logger.error(f"Graph building failed: {e}")
            raise
    
    def _create_node_function(self, node: BaseWorkflowNode):
        """Create a function wrapper for a node to be used in LangGraph."""
        async def node_function(state: WorkflowState) -> WorkflowState:
            """Execute the node and return updated state."""
            try:
                updated_state = await node.execute(state)
                
                # Update workflow state with node results
                last_result = updated_state.get_last_result()
                if last_result:
                    if last_result.node_type == "coordinator":
                        # Coordinator may update multiple state fields
                        if "search_results" in last_result.data:
                            search_data = last_result.data["search_results"]
                            if isinstance(search_data, dict) and "tools" in search_data:
                                updated_state.search_results = search_data["tools"]
                    
                    elif last_result.node_type == "researcher":
                        # Researcher updates search results or research data
                        if "search_results" in last_result.data:
                            updated_state.search_results = last_result.data["search_results"]
                        if "research_data" in last_result.data:
                            updated_state.research_data = last_result.data["research_data"]
                    
                    elif last_result.node_type == "reporter":
                        # Reporter updates report content
                        if "report_content" in last_result.data:
                            updated_state.report_content = last_result.data["report_content"]
                    
                    elif last_result.node_type == "search":
                        # Search node updates search results
                        if "search_results" in last_result.data:
                            updated_state.search_results = last_result.data["search_results"]
                    
                    elif last_result.node_type == "analysis":
                        # Analysis node updates research data
                        updated_state.research_data.update(last_result.data)
                
                return updated_state
                
            except Exception as e:
                self.logger.error(f"Node {node.node_type} execution failed: {e}")
                # Update state to indicate error
                updated_state = state.model_copy()
                updated_state.error_occurred = True
                updated_state.should_continue = False
                return updated_state
        
        return node_function
    
    async def _add_workflow_edges(self) -> None:
        """Add edges between nodes based on workflow configuration."""
        workflow_name = self.config.name.lower()
        
        if workflow_name == "research" or "coordinator" in self.config.nodes:
            await self._add_research_workflow_edges()
        elif workflow_name == "search":
            await self._add_search_workflow_edges()
        else:
            await self._add_default_workflow_edges()
    
    async def _add_research_workflow_edges(self) -> None:
        """Add edges for research workflow."""
        # Coordinator -> Researcher -> Reporter -> END
        if "coordinator" in self.nodes and "researcher" in self.nodes:
            self.graph.add_conditional_edges(
                "coordinator",
                self._coordinator_router,
                {
                    "search": "researcher",
                    "research": "researcher", 
                    "report": "reporter" if "reporter" in self.nodes else END,
                    "complete": END,
                }
            )
        
        if "researcher" in self.nodes:
            if "reporter" in self.nodes:
                self.graph.add_edge("researcher", "reporter")
            else:
                self.graph.add_edge("researcher", END)
        
        if "reporter" in self.nodes:
            self.graph.add_edge("reporter", END)
    
    async def _add_search_workflow_edges(self) -> None:
        """Add edges for simple search workflow."""
        # Search -> Analysis -> END
        if "search" in self.nodes and "analysis" in self.nodes:
            self.graph.add_edge("search", "analysis")
            self.graph.add_edge("analysis", END)
        elif "search" in self.nodes:
            self.graph.add_edge("search", END)
    
    async def _add_default_workflow_edges(self) -> None:
        """Add edges for default workflow."""
        # Default: coordinator-based workflow
        if "coordinator" in self.nodes:
            self.graph.add_conditional_edges(
                "coordinator",
                self._coordinator_router,
                {
                    "search": "researcher" if "researcher" in self.nodes else END,
                    "research": "researcher" if "researcher" in self.nodes else END,
                    "report": "reporter" if "reporter" in self.nodes else END,
                    "complete": END,
                }
            )
        
        # Add sequential edges between available nodes
        node_sequence = ["researcher", "reporter"]
        prev_node = None
        
        for node_name in node_sequence:
            if node_name in self.nodes:
                if prev_node and prev_node in self.nodes:
                    self.graph.add_edge(prev_node, node_name)
                prev_node = node_name
        
        # Connect last node to END
        if prev_node and prev_node in self.nodes:
            self.graph.add_edge(prev_node, END)
    
    def _coordinator_router(self, state: WorkflowState) -> str:
        """Router function for coordinator node decisions."""
        # Get the last coordinator result to determine next step
        last_result = state.get_last_result("coordinator")
        
        if last_result and last_result.data:
            action = last_result.data.get("action", "search")
            next_step = last_result.data.get("next_step", "complete")
            
            # Map actions to routes
            if action == "search":
                return "search"
            elif action == "research":
                return "research"
            elif action == "report":
                return "report"
            elif action == "complete":
                return "complete"
        
        # Default routing based on state
        if not state.search_results:
            return "search"
        elif not state.research_data:
            return "research"
        elif not state.report_content:
            return "report"
        else:
            return "complete"
    
    async def _compile_graph(self) -> None:
        """Compile the graph for execution."""
        try:
            # Set up checkpointer if enabled
            if self.config.enable_checkpoints:
                self._checkpointer = MemorySaver()
            
            # Compile the graph
            compile_config = {}
            if self._checkpointer:
                compile_config["checkpointer"] = self._checkpointer
            
            self.compiled_graph = self.graph.compile(**compile_config)
            
            self.logger.info("Graph compiled successfully")
            
        except Exception as e:
            self.logger.error(f"Graph compilation failed: {e}")
            raise
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the workflow with given input.
        
        Args:
            input_data: Input data for the workflow.
            config: Optional execution configuration.
            
        Returns:
            Dict[str, Any]: Workflow execution result.
            
        Raises:
            RuntimeError: If workflow is not initialized or execution fails.
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        
        start_time = time.time()
        self.metrics.total_executions += 1
        
        try:
            self.logger.info(f"Executing workflow '{self.config.name}'")
            
            # Create initial state
            initial_state = WorkflowState(
                user_query=input_data.get("user_query", ""),
                metadata={
                    "workflow_name": self.config.name,
                    "execution_id": f"{self.metrics.workflow_id}_{self.metrics.total_executions}",
                    "start_time": datetime.now().isoformat(),
                }
            )
            
            # Set up execution config
            exec_config = config or {}
            if self.config.timeout:
                exec_config["timeout"] = self.config.timeout
            
            # Execute the workflow
            if self.compiled_graph:
                result = await asyncio.wait_for(
                    self.compiled_graph.ainvoke(initial_state.model_dump(), config=exec_config),
                    timeout=self.config.timeout
                )
            else:
                # Fallback execution for development
                result = {
                    "user_query": initial_state.user_query,
                    "workflow_complete": True,
                    "execution_time": time.time() - start_time,
                }
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics.successful_executions += 1
            self.metrics.average_execution_time = (
                (self.metrics.average_execution_time * (self.metrics.total_executions - 1) + execution_time)
                / self.metrics.total_executions
            )
            self.metrics.last_execution = datetime.now()
            
            # Collect node metrics
            for node_name, node in self.nodes.items():
                self.metrics.node_metrics[node_name] = node.get_metrics()
            
            self.logger.info(f"Workflow executed successfully in {execution_time:.2f}s")
            
            return result
            
        except asyncio.TimeoutError:
            self.metrics.failed_executions += 1
            error_msg = f"Workflow execution timed out after {self.config.timeout}s"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except Exception as e:
            self.metrics.failed_executions += 1
            self.logger.error(f"Workflow execution failed: {e}")
            raise RuntimeError(f"Workflow execution failed: {e}") from e
    
    async def ainvoke(
        self,
        input_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async invoke method for LangGraph compatibility.
        
        Args:
            input_data: Input data for the workflow.
            config: Optional execution configuration.
            
        Returns:
            Dict[str, Any]: Workflow execution result.
        """
        return await self.execute(input_data, config)
    
    def get_metrics(self) -> WorkflowMetrics:
        """Get workflow execution metrics.
        
        Returns:
            WorkflowMetrics: Current metrics.
        """
        return self.metrics.model_copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the workflow.
        
        Returns:
            Dict[str, Any]: Health status.
        """
        try:
            health_status = {
                "workflow_name": self.config.name,
                "initialized": self._initialized,
                "graph_compiled": self.compiled_graph is not None,
                "nodes": len(self.nodes),
                "entry_point": self.config.entry_point,
                "metrics": self.metrics.model_dump(),
                "timestamp": datetime.now().isoformat(),
            }
            
            # Check node health
            node_health = {}
            for node_name, node in self.nodes.items():
                node_health[node_name] = {
                    "type": node.node_type,
                    "total_executions": node.total_executions,
                    "success_rate": (
                        (node.successful_executions / node.total_executions * 100)
                        if node.total_executions > 0 else 0
                    ),
                }
            
            health_status["node_health"] = node_health
            
            return health_status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
    
    async def shutdown(self) -> None:
        """Shutdown the workflow and clean up resources."""
        self.logger.info(f"Shutting down workflow '{self.config.name}'")
        
        # Clean up nodes
        for node in self.nodes.values():
            if hasattr(node, 'shutdown'):
                try:
                    await node.shutdown()
                except Exception as e:
                    self.logger.warning(f"Error shutting down node {node.node_type}: {e}")
        
        # Clear references
        self.nodes.clear()
        self.graph = None
        self.compiled_graph = None
        self._initialized = False
        
        self.logger.info("Workflow shutdown complete")


# Workflow factory functions
def create_workflow(
    config: Optional[Dict[str, Any]] = None,
    coordinator: Optional[CoordinatorAgent] = None,
    researcher: Optional[ResearcherAgent] = None,
    reporter: Optional[ReporterAgent] = None,
    vector_store: Optional[VectorStore] = None,
    mcp_client: Optional[MCPClient] = None,
    settings: Optional[AgentSettings] = None,
) -> AgentWorkflow:
    """Create a default multi-agent workflow.
    
    Args:
        config: Optional workflow configuration.
        coordinator: Optional CoordinatorAgent instance.
        researcher: Optional ResearcherAgent instance.
        reporter: Optional ReporterAgent instance.
        vector_store: Optional VectorStore instance.
        mcp_client: Optional MCPClient instance.
        settings: Optional AgentSettings instance.
        
    Returns:
        AgentWorkflow: Created workflow instance.
        
    Example:
        >>> workflow = create_workflow()
        >>> await workflow.initialize(coordinator, researcher, reporter)
        >>> result = await workflow.execute({"user_query": "BLAST tools"})
    """
    workflow_config = WorkflowConfig(
        name="default",
        description="Default multi-agent workflow for bioinformatics tool discovery",
        nodes=["coordinator", "researcher", "reporter"],
        entry_point="coordinator",
        max_iterations=10,
        timeout=300,
        **(config or {})
    )
    
    workflow = AgentWorkflow(workflow_config)
    
    # Auto-initialize if components are provided
    if any([coordinator, researcher, reporter]):
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a task for initialization
            asyncio.create_task(workflow.initialize(
                coordinator, researcher, reporter, vector_store, mcp_client, settings
            ))
        else:
            # Initialize synchronously
            asyncio.run(workflow.initialize(
                coordinator, researcher, reporter, vector_store, mcp_client, settings
            ))
    
    return workflow


def create_research_workflow(
    config: Optional[Dict[str, Any]] = None,
    coordinator: Optional[CoordinatorAgent] = None,
    researcher: Optional[ResearcherAgent] = None,
    reporter: Optional[ReporterAgent] = None,
    vector_store: Optional[VectorStore] = None,
    settings: Optional[AgentSettings] = None,
) -> AgentWorkflow:
    """Create a research-focused workflow.
    
    Args:
        config: Optional workflow configuration.
        coordinator: Optional CoordinatorAgent instance.
        researcher: Optional ResearcherAgent instance.
        reporter: Optional ReporterAgent instance.
        vector_store: Optional VectorStore instance.
        settings: Optional AgentSettings instance.
        
    Returns:
        AgentWorkflow: Created research workflow.
        
    Example:
        >>> workflow = create_research_workflow()
        >>> result = await workflow.execute({"user_query": "ML in genomics"})
    """
    workflow_config = WorkflowConfig(
        name="research",
        description="Research-focused workflow with enhanced analysis capabilities",
        nodes=["coordinator", "researcher", "analysis", "reporter"],
        entry_point="coordinator",
        max_iterations=15,
        timeout=600,
        **(config or {})
    )
    
    workflow = AgentWorkflow(workflow_config)
    
    # Auto-initialize if components are provided
    if any([coordinator, researcher, reporter]):
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(workflow.initialize(
                coordinator, researcher, reporter, vector_store, None, settings
            ))
        else:
            asyncio.run(workflow.initialize(
                coordinator, researcher, reporter, vector_store, None, settings
            ))
    
    return workflow


def create_simple_search_workflow(
    config: Optional[Dict[str, Any]] = None,
    vector_store: Optional[VectorStore] = None,
    settings: Optional[AgentSettings] = None,
) -> AgentWorkflow:
    """Create a simple search-only workflow.
    
    Args:
        config: Optional workflow configuration.
        vector_store: Optional VectorStore instance.
        settings: Optional AgentSettings instance.
        
    Returns:
        AgentWorkflow: Created search workflow.
        
    Example:
        >>> workflow = create_simple_search_workflow()
        >>> result = await workflow.execute({"user_query": "protein tools"})
    """
    workflow_config = WorkflowConfig(
        name="search",
        description="Simple search workflow for quick tool discovery",
        nodes=["search", "analysis"],
        entry_point="search",
        max_iterations=5,
        timeout=60,
        **(config or {})
    )
    
    workflow = AgentWorkflow(workflow_config)
    
    # Auto-initialize if components are provided
    if vector_store:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(workflow.initialize(
                None, None, None, vector_store, None, settings
            ))
        else:
            asyncio.run(workflow.initialize(
                None, None, None, vector_store, None, settings
            ))
    
    return workflow


def create_custom_workflow(
    config: WorkflowConfig,
    **components,
) -> AgentWorkflow:
    """Create a custom workflow with specified configuration.
    
    Args:
        config: Workflow configuration.
        **components: Component instances (coordinator, researcher, etc.).
        
    Returns:
        AgentWorkflow: Created custom workflow.
        
    Example:
        >>> config = WorkflowConfig(
        ...     name="custom",
        ...     nodes=["search", "analysis"],
        ...     entry_point="search"
        ... )
        >>> workflow = create_custom_workflow(config, vector_store=store)
    """
    workflow = AgentWorkflow(config)
    
    # Auto-initialize if components are provided
    if components:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(workflow.initialize(**components))
        else:
            asyncio.run(workflow.initialize(**components))
    
    return workflow


# Workflow validation and utilities
def validate_workflow_config(config: WorkflowConfig) -> Dict[str, Any]:
    """Validate workflow configuration.
    
    Args:
        config: Workflow configuration to validate.
        
    Returns:
        Dict[str, Any]: Validation results.
    """
    validation = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "recommendations": [],
    }
    
    # Check entry point is in nodes
    if config.entry_point not in config.nodes:
        validation["valid"] = False
        validation["issues"].append(f"Entry point '{config.entry_point}' not in nodes list")
    
    # Check for reasonable timeout
    if config.timeout < 30:
        validation["warnings"].append("Timeout is very short, may cause premature termination")
    elif config.timeout > 1800:
        validation["warnings"].append("Timeout is very long, consider reducing for better UX")
    
    # Check iteration limit
    if config.max_iterations < 3:
        validation["warnings"].append("Low iteration limit may prevent workflow completion")
    elif config.max_iterations > 50:
        validation["warnings"].append("High iteration limit may cause excessive resource usage")
    
    # Check node combinations
    if "coordinator" not in config.nodes and len(config.nodes) > 1:
        validation["recommendations"].append("Consider adding coordinator for multi-node workflows")
    
    return validation


# Default configurations
DEFAULT_WORKFLOW_CONFIGS = {
    "default": WorkflowConfig(
        name="default",
        description="Default multi-agent workflow",
        nodes=["coordinator", "researcher", "reporter"],
        entry_point="coordinator",
        max_iterations=10,
        timeout=300,
    ),
    "research": WorkflowConfig(
        name="research",
        description="Research-focused workflow",
        nodes=["coordinator", "researcher", "analysis", "reporter"],
        entry_point="coordinator",
        max_iterations=15,
        timeout=600,
    ),
    "search": WorkflowConfig(
        name="search",
        description="Simple search workflow",
        nodes=["search", "analysis"],
        entry_point="search",
        max_iterations=5,
        timeout=60,
    ),
}


def get_default_config(workflow_type: str) -> Optional[WorkflowConfig]:
    """Get default configuration for a workflow type.
    
    Args:
        workflow_type: Type of workflow.
        
    Returns:
        Optional[WorkflowConfig]: Default configuration or None.
    """
    return DEFAULT_WORKFLOW_CONFIGS.get(workflow_type)