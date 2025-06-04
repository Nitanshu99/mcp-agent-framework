"""Graph nodes for LangGraph workflow orchestration.

This module contains individual node implementations that represent different
agents and operations in the MCP Agent Framework workflow graph.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic node usage:

    >>> coordinator_node = CoordinatorNode(coordinator_agent)
    >>> result = await coordinator_node.execute(state)
    >>> print(result.data)

Architecture:
    Nodes wrap agents and provide LangGraph compatibility:
    
    ┌─────────────────┐
    │  Graph State    │
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │   Node.execute  │ ← Process state
    │                 │
    │  ┌───────────┐  │
    │  │   Agent   │  │ ← Wrapped agent
    │  └───────────┘  │
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │  Updated State  │
    └─────────────────┘
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import StateGraph
    
    from mcp_agent.agents.base import BaseAgent, AgentCapability, AgentState
    from mcp_agent.agents.coordinator import CoordinatorAgent
    from mcp_agent.agents.researcher import ResearcherAgent
    from mcp_agent.agents.reporter import ReporterAgent
    from mcp_agent.config.settings import AgentSettings
    from mcp_agent.models.schemas import (
        AgentTask,
        AgentResponse,
        SearchQuery,
        SearchResult,
    )
    from mcp_agent.utils.logger import get_logger
except ImportError as e:
    # Mock imports for development
    import warnings
    warnings.warn(f"Node dependencies not available: {e}", ImportWarning)
    
    class StateGraph:
        pass
    class BaseAgent:
        pass
    class AgentCapability:
        pass
    class AgentState:
        pass
    class CoordinatorAgent:
        pass
    class ResearcherAgent:
        pass
    class ReporterAgent:
        pass
    class AgentSettings:
        pass
    class AgentTask:
        pass
    class AgentResponse:
        pass
    class SearchQuery:
        pass
    class SearchResult:
        pass
    def get_logger(name: str):
        import logging
        return logging.getLogger(name)


class NodeStatus(Enum):
    """Enumeration of node execution statuses."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class NodeResult(BaseModel):
    """Result from a node execution.
    
    Attributes:
        node_id: Unique identifier for the node execution.
        node_type: Type of node that executed.
        status: Execution status.
        data: Result data.
        error: Error message if execution failed.
        execution_time: Time taken to execute in seconds.
        metadata: Additional metadata.
        
    Example:
        >>> result = NodeResult(
        ...     node_id="coord_001",
        ...     node_type="coordinator",
        ...     status=NodeStatus.COMPLETED,
        ...     data={"search_results": [...]}
        ... )
    """
    
    node_id: str = Field(description="Unique node execution identifier")
    node_type: str = Field(description="Type of node")
    status: NodeStatus = Field(description="Execution status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    error: Optional[str] = Field(default=None, description="Error message")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == NodeStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()


class NodeError(Exception):
    """Exception raised when a node execution fails.
    
    Attributes:
        node_type: Type of node that failed.
        node_id: Identifier of the failed node.
        message: Error message.
        original_error: Original exception that caused the failure.
        
    Example:
        >>> raise NodeError("coordinator", "coord_001", "Agent not initialized")
    """
    
    def __init__(
        self,
        node_type: str,
        node_id: str,
        message: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize the node error.
        
        Args:
            node_type: Type of node that failed.
            node_id: Identifier of the failed node.
            message: Error message.
            original_error: Original exception that caused the failure.
        """
        self.node_type = node_type
        self.node_id = node_id
        self.message = message
        self.original_error = original_error
        
        super().__init__(f"Node {node_type}[{node_id}] failed: {message}")


class WorkflowState(BaseModel):
    """State object passed between nodes in the workflow.
    
    Attributes:
        user_query: Original user query.
        current_step: Current workflow step.
        search_results: Search results from various sources.
        research_data: Research data collected.
        report_content: Generated report content.
        node_history: History of node executions.
        metadata: Additional workflow metadata.
        
    Example:
        >>> state = WorkflowState(
        ...     user_query="Find RNA-seq tools",
        ...     current_step="search",
        ...     search_results=[]
        ... )
    """
    
    user_query: str = Field(description="Original user query")
    current_step: str = Field(default="start", description="Current workflow step")
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    research_data: Dict[str, Any] = Field(default_factory=dict, description="Research data")
    report_content: str = Field(default="", description="Generated report content")
    node_history: List[NodeResult] = Field(default_factory=list, description="Node execution history")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Workflow metadata")
    
    # Execution control
    should_continue: bool = Field(default=True, description="Whether workflow should continue")
    error_occurred: bool = Field(default=False, description="Whether an error occurred")
    human_intervention_needed: bool = Field(default=False, description="Whether human intervention is needed")
    
    def add_node_result(self, result: NodeResult) -> None:
        """Add a node result to the history."""
        self.node_history.append(result)
        
        # Update current step
        if result.success:
            self.current_step = result.node_type
        else:
            self.error_occurred = True
    
    def get_last_result(self, node_type: Optional[str] = None) -> Optional[NodeResult]:
        """Get the last result from a specific node type or overall."""
        if not self.node_history:
            return None
        
        if node_type:
            for result in reversed(self.node_history):
                if result.node_type == node_type:
                    return result
            return None
        
        return self.node_history[-1]
    
    def has_error(self) -> bool:
        """Check if any node has failed."""
        return self.error_occurred or any(
            result.status == NodeStatus.FAILED for result in self.node_history
        )


class BaseWorkflowNode(ABC):
    """Abstract base class for all workflow nodes.
    
    This class defines the interface that all workflow nodes must implement
    to be compatible with LangGraph execution.
    
    Attributes:
        node_type: Type identifier for this node.
        node_id: Unique identifier for this node instance.
        settings: Configuration settings.
        agent: Wrapped agent instance.
        
    Example:
        >>> class CustomNode(BaseWorkflowNode):
        ...     async def _execute_impl(self, state: WorkflowState) -> NodeResult:
        ...         # Implementation here
        ...         pass
    """
    
    def __init__(
        self,
        node_type: str,
        agent: Optional[BaseAgent] = None,
        settings: Optional[AgentSettings] = None,
    ) -> None:
        """Initialize the workflow node.
        
        Args:
            node_type: Type identifier for this node.
            agent: Optional agent instance to wrap.
            settings: Optional configuration settings.
        """
        self.node_type = node_type
        self.node_id = str(uuid.uuid4())
        self.settings = settings
        self.agent = agent
        self.logger = get_logger(f"Node.{node_type}")
        
        # Metrics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0
        self.last_execution = None
        
        self.logger.info(f"Node {node_type}[{self.node_id}] initialized")
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute the node and update the workflow state.
        
        Args:
            state: Current workflow state.
            
        Returns:
            WorkflowState: Updated workflow state.
            
        Raises:
            NodeError: If node execution fails.
        """
        start_time = time.time()
        self.total_executions += 1
        
        try:
            self.logger.info(f"Executing node {self.node_type}[{self.node_id}]")
            
            # Check if node should be skipped
            if await self._should_skip(state):
                result = NodeResult(
                    node_id=self.node_id,
                    node_type=self.node_type,
                    status=NodeStatus.SKIPPED,
                    execution_time=time.time() - start_time,
                    metadata={"reason": "Node execution skipped"}
                )
                state.add_node_result(result)
                return state
            
            # Execute the node implementation
            result = await self._execute_impl(state)
            result.node_id = self.node_id
            result.node_type = self.node_type
            result.execution_time = time.time() - start_time
            
            # Update metrics
            if result.success:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
            
            self.total_execution_time += result.execution_time
            self.last_execution = datetime.now()
            
            # Add result to state
            state.add_node_result(result)
            
            self.logger.info(
                f"Node {self.node_type}[{self.node_id}] completed "
                f"in {result.execution_time:.2f}s with status: {result.status}"
            )
            
            return state
            
        except Exception as e:
            self.failed_executions += 1
            execution_time = time.time() - start_time
            
            error_result = NodeResult(
                node_id=self.node_id,
                node_type=self.node_type,
                status=NodeStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={"exception_type": type(e).__name__}
            )
            
            state.add_node_result(error_result)
            self.logger.error(f"Node {self.node_type}[{self.node_id}] failed: {e}")
            
            # Re-raise as NodeError
            raise NodeError(self.node_type, self.node_id, str(e), e) from e
    
    @abstractmethod
    async def _execute_impl(self, state: WorkflowState) -> NodeResult:
        """Implementation-specific execution logic.
        
        Args:
            state: Current workflow state.
            
        Returns:
            NodeResult: Result of node execution.
        """
        pass
    
    async def _should_skip(self, state: WorkflowState) -> bool:
        """Check if this node should be skipped.
        
        Args:
            state: Current workflow state.
            
        Returns:
            bool: True if node should be skipped.
        """
        # Skip if workflow has encountered an error and this node doesn't handle errors
        if state.has_error() and not self._handles_errors():
            return True
        
        # Skip if human intervention is needed and this node doesn't handle it
        if state.human_intervention_needed and not self._handles_human_intervention():
            return True
        
        return False
    
    def _handles_errors(self) -> bool:
        """Check if this node can handle workflow errors."""
        return False
    
    def _handles_human_intervention(self) -> bool:
        """Check if this node can handle human intervention."""
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get node execution metrics."""
        success_rate = 0.0
        if self.total_executions > 0:
            success_rate = (self.successful_executions / self.total_executions) * 100
        
        avg_execution_time = 0.0
        if self.successful_executions > 0:
            avg_execution_time = self.total_execution_time / self.successful_executions
        
        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
        }
    
    def __repr__(self) -> str:
        """String representation of the node."""
        return f"{self.__class__.__name__}(type={self.node_type}, id={self.node_id})"


class CoordinatorNode(BaseWorkflowNode):
    """Node wrapper for the CoordinatorAgent.
    
    This node orchestrates the overall workflow and manages communication
    between other specialized nodes.
    
    Example:
        >>> coordinator = CoordinatorAgent(settings, mcp_client, vector_store)
        >>> node = CoordinatorNode(coordinator)
        >>> result = await node.execute(state)
    """
    
    def __init__(
        self,
        coordinator_agent: CoordinatorAgent,
        settings: Optional[AgentSettings] = None,
    ) -> None:
        """Initialize the coordinator node.
        
        Args:
            coordinator_agent: CoordinatorAgent instance to wrap.
            settings: Optional configuration settings.
        """
        super().__init__("coordinator", coordinator_agent, settings)
        self.coordinator: CoordinatorAgent = coordinator_agent
    
    async def _execute_impl(self, state: WorkflowState) -> NodeResult:
        """Execute coordinator logic."""
        try:
            # Determine what action the coordinator should take
            action = await self._determine_action(state)
            
            if action == "search":
                return await self._handle_search_coordination(state)
            elif action == "research":
                return await self._handle_research_coordination(state)
            elif action == "report":
                return await self._handle_report_coordination(state)
            elif action == "complete":
                return await self._handle_completion(state)
            else:
                return await self._handle_default_coordination(state)
                
        except Exception as e:
            self.logger.error(f"Coordinator execution failed: {e}")
            return NodeResult(
                node_type=self.node_type,
                status=NodeStatus.FAILED,
                error=str(e),
                metadata={"action": action if 'action' in locals() else "unknown"}
            )
    
    async def _determine_action(self, state: WorkflowState) -> str:
        """Determine what action the coordinator should take next."""
        # Simple state-based logic - in practice, this could use the LLM
        if not state.search_results:
            return "search"
        elif not state.research_data:
            return "research"
        elif not state.report_content:
            return "report"
        else:
            return "complete"
    
    async def _handle_search_coordination(self, state: WorkflowState) -> NodeResult:
        """Handle search coordination."""
        search_query = SearchQuery(
            text=state.user_query,
            max_results=10,
            include_documentation=True,
        )
        
        search_result = await self.coordinator.search(search_query)
        
        return NodeResult(
            node_type=self.node_type,
            status=NodeStatus.COMPLETED,
            data={
                "action": "search",
                "search_results": search_result.model_dump(),
                "next_step": "research",
            },
            metadata={
                "tools_found": len(search_result.tools),
                "query": state.user_query,
            }
        )
    
    async def _handle_research_coordination(self, state: WorkflowState) -> NodeResult:
        """Handle research coordination."""
        research_result = await self.coordinator.orchestrate_research(
            topic=state.user_query,
            depth="standard",
        )
        
        return NodeResult(
            node_type=self.node_type,
            status=NodeStatus.COMPLETED,
            data={
                "action": "research",
                "research_data": research_result,
                "next_step": "report",
            },
            metadata={
                "research_depth": "standard",
                "workflow_id": research_result.get("workflow_id"),
            }
        )
    
    async def _handle_report_coordination(self, state: WorkflowState) -> NodeResult:
        """Handle report coordination."""
        # Signal that reporting should happen
        return NodeResult(
            node_type=self.node_type,
            status=NodeStatus.COMPLETED,
            data={
                "action": "report",
                "ready_for_reporting": True,
                "next_step": "complete",
            },
            metadata={"report_format": "markdown"}
        )
    
    async def _handle_completion(self, state: WorkflowState) -> NodeResult:
        """Handle workflow completion."""
        return NodeResult(
            node_type=self.node_type,
            status=NodeStatus.COMPLETED,
            data={
                "action": "complete",
                "workflow_complete": True,
                "final_result": {
                    "query": state.user_query,
                    "search_results": state.search_results,
                    "research_data": state.research_data,
                    "report": state.report_content,
                }
            },
            metadata={"completion_time": datetime.now().isoformat()}
        )
    
    async def _handle_default_coordination(self, state: WorkflowState) -> NodeResult:
        """Handle default coordination when no specific action is determined."""
        return NodeResult(
            node_type=self.node_type,
            status=NodeStatus.COMPLETED,
            data={
                "action": "status_check",
                "current_state": {
                    "has_search_results": bool(state.search_results),
                    "has_research_data": bool(state.research_data),
                    "has_report": bool(state.report_content),
                }
            }
        )
    
    def _handles_errors(self) -> bool:
        """Coordinator can handle workflow errors."""
        return True
    
    def _handles_human_intervention(self) -> bool:
        """Coordinator can handle human intervention."""
        return True


class ResearcherNode(BaseWorkflowNode):
    """Node wrapper for the ResearcherAgent.
    
    This node handles research tasks including tool discovery,
    web search, and data gathering.
    
    Example:
        >>> researcher = ResearcherAgent(settings, mcp_client, vector_store)
        >>> node = ResearcherNode(researcher)
        >>> result = await node.execute(state)
    """
    
    def __init__(
        self,
        researcher_agent: ResearcherAgent,
        settings: Optional[AgentSettings] = None,
    ) -> None:
        """Initialize the researcher node.
        
        Args:
            researcher_agent: ResearcherAgent instance to wrap.
            settings: Optional configuration settings.
        """
        super().__init__("researcher", researcher_agent, settings)
        self.researcher: ResearcherAgent = researcher_agent
    
    async def _execute_impl(self, state: WorkflowState) -> NodeResult:
        """Execute researcher logic."""
        try:
            # Create research task based on current state
            if not state.search_results:
                return await self._perform_initial_search(state)
            else:
                return await self._perform_enhanced_research(state)
                
        except Exception as e:
            self.logger.error(f"Researcher execution failed: {e}")
            return NodeResult(
                node_type=self.node_type,
                status=NodeStatus.FAILED,
                error=str(e),
            )
    
    async def _perform_initial_search(self, state: WorkflowState) -> NodeResult:
        """Perform initial tool search."""
        search_task = AgentTask(
            type="tool_discovery",
            parameters={
                "query": state.user_query,
                "max_results": 20,
                "include_web_search": True,
            }
        )
        
        response = await self.researcher.execute_task(search_task)
        
        if response.success:
            # Update state with search results
            search_results = response.data.get("synthesized_results", {})
            tools = search_results.get("tools", [])
            
            return NodeResult(
                node_type=self.node_type,
                status=NodeStatus.COMPLETED,
                data={
                    "search_results": tools,
                    "total_found": len(tools),
                    "sources": search_results.get("sources", {}),
                },
                metadata={
                    "search_type": "tool_discovery",
                    "query": state.user_query,
                }
            )
        else:
            return NodeResult(
                node_type=self.node_type,
                status=NodeStatus.FAILED,
                error=response.error or "Search task failed",
            )
    
    async def _perform_enhanced_research(self, state: WorkflowState) -> NodeResult:
        """Perform enhanced research based on existing search results."""
        research_task = AgentTask(
            type="bioinformatics_research",
            parameters={
                "topic": state.user_query,
                "research_type": "comprehensive",
                "tools_context": state.search_results,
            }
        )
        
        response = await self.researcher.execute_task(research_task)
        
        if response.success:
            research_data = response.data.get("analysis", {})
            
            return NodeResult(
                node_type=self.node_type,
                status=NodeStatus.COMPLETED,
                data={
                    "research_data": research_data,
                    "enhanced_analysis": True,
                },
                metadata={
                    "research_type": "bioinformatics_research",
                    "topic": state.user_query,
                }
            )
        else:
            return NodeResult(
                node_type=self.node_type,
                status=NodeStatus.FAILED,
                error=response.error or "Research task failed",
            )


class ReporterNode(BaseWorkflowNode):
    """Node wrapper for the ReporterAgent.
    
    This node handles report generation and content formatting
    from research data and search results.
    
    Example:
        >>> reporter = ReporterAgent(settings)
        >>> node = ReporterNode(reporter)
        >>> result = await node.execute(state)
    """
    
    def __init__(
        self,
        reporter_agent: ReporterAgent,
        settings: Optional[AgentSettings] = None,
    ) -> None:
        """Initialize the reporter node.
        
        Args:
            reporter_agent: ReporterAgent instance to wrap.
            settings: Optional configuration settings.
        """
        super().__init__("reporter", reporter_agent, settings)
        self.reporter: ReporterAgent = reporter_agent
    
    async def _execute_impl(self, state: WorkflowState) -> NodeResult:
        """Execute reporter logic."""
        try:
            # Create report generation task
            report_task = AgentTask(
                type="generate_report",
                parameters={
                    "topic": state.user_query,
                    "search_results": {"tools": state.search_results},
                    "enhanced_results": state.research_data,
                    "output_format": "markdown",
                    "include_code_examples": True,
                }
            )
            
            response = await self.reporter.execute_task(report_task)
            
            if response.success:
                report_content = response.content
                report_data = response.data
                
                return NodeResult(
                    node_type=self.node_type,
                    status=NodeStatus.COMPLETED,
                    data={
                        "report_content": report_content,
                        "report_metadata": report_data,
                        "output_format": "markdown",
                    },
                    metadata={
                        "word_count": report_data.get("word_count", 0),
                        "character_count": report_data.get("character_count", 0),
                        "template_used": report_data.get("template_used", "default"),
                    }
                )
            else:
                return NodeResult(
                    node_type=self.node_type,
                    status=NodeStatus.FAILED,
                    error=response.error or "Report generation failed",
                )
                
        except Exception as e:
            self.logger.error(f"Reporter execution failed: {e}")
            return NodeResult(
                node_type=self.node_type,
                status=NodeStatus.FAILED,
                error=str(e),
            )


class SearchNode(BaseWorkflowNode):
    """Specialized node for search operations.
    
    This node performs focused search operations without requiring
    a full agent implementation.
    
    Example:
        >>> search_node = SearchNode(vector_store=vector_store)
        >>> result = await search_node.execute(state)
    """
    
    def __init__(
        self,
        vector_store: Optional[Any] = None,
        web_search: Optional[Any] = None,
        settings: Optional[AgentSettings] = None,
    ) -> None:
        """Initialize the search node.
        
        Args:
            vector_store: Vector store for semantic search.
            web_search: Web search engine.
            settings: Optional configuration settings.
        """
        super().__init__("search", None, settings)
        self.vector_store = vector_store
        self.web_search = web_search
    
    async def _execute_impl(self, state: WorkflowState) -> NodeResult:
        """Execute search logic."""
        try:
            search_results = []
            
            # Perform vector search if available
            if self.vector_store:
                vector_results = await self._perform_vector_search(state.user_query)
                search_results.extend(vector_results)
            
            # Perform web search if available
            if self.web_search:
                web_results = await self._perform_web_search(state.user_query)
                search_results.extend(web_results)
            
            if not search_results:
                return NodeResult(
                    node_type=self.node_type,
                    status=NodeStatus.FAILED,
                    error="No search providers available",
                )
            
            return NodeResult(
                node_type=self.node_type,
                status=NodeStatus.COMPLETED,
                data={
                    "search_results": search_results,
                    "total_results": len(search_results),
                },
                metadata={
                    "query": state.user_query,
                    "vector_search": self.vector_store is not None,
                    "web_search": self.web_search is not None,
                }
            )
            
        except Exception as e:
            self.logger.error(f"Search execution failed: {e}")
            return NodeResult(
                node_type=self.node_type,
                status=NodeStatus.FAILED,
                error=str(e),
            )
    
    async def _perform_vector_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform vector database search."""
        try:
            results = await self.vector_store.search(query, limit=10)
            return results
        except Exception as e:
            self.logger.warning(f"Vector search failed: {e}")
            return []
    
    async def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search."""
        try:
            results = await self.web_search.search(query, max_results=10)
            return [result.to_dict() for result in results]
        except Exception as e:
            self.logger.warning(f"Web search failed: {e}")
            return []


class AnalysisNode(BaseWorkflowNode):
    """Specialized node for analysis operations.
    
    This node performs analysis on search results and research data
    to extract insights and prepare data for reporting.
    
    Example:
        >>> analysis_node = AnalysisNode()
        >>> result = await analysis_node.execute(state)
    """
    
    def __init__(self, settings: Optional[AgentSettings] = None) -> None:
        """Initialize the analysis node.
        
        Args:
            settings: Optional configuration settings.
        """
        super().__init__("analysis", None, settings)
    
    async def _execute_impl(self, state: WorkflowState) -> NodeResult:
        """Execute analysis logic."""
        try:
            analysis_data = {}
            
            # Analyze search results
            if state.search_results:
                search_analysis = await self._analyze_search_results(state.search_results)
                analysis_data["search_analysis"] = search_analysis
            
            # Analyze research data
            if state.research_data:
                research_analysis = await self._analyze_research_data(state.research_data)
                analysis_data["research_analysis"] = research_analysis
            
            # Generate insights
            insights = await self._generate_insights(state)
            analysis_data["insights"] = insights
            
            return NodeResult(
                node_type=self.node_type,
                status=NodeStatus.COMPLETED,
                data=analysis_data,
                metadata={
                    "query": state.user_query,
                    "analysis_types": list(analysis_data.keys()),
                }
            )
            
        except Exception as e:
            self.logger.error(f"Analysis execution failed: {e}")
            return NodeResult(
                node_type=self.node_type,
                status=NodeStatus.FAILED,
                error=str(e),
            )
    
    async def _analyze_search_results(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze search results to extract patterns and insights."""
        if not search_results:
            return {}
        
        # Count categories
        categories = {}
        for result in search_results:
            category = result.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
        
        # Count data types
        data_types = {}
        for result in search_results:
            for dtype in result.get("data_types", []):
                data_types[dtype] = data_types.get(dtype, 0) + 1
        
        return {
            "total_tools": len(search_results),
            "category_distribution": categories,
            "data_type_distribution": data_types,
            "top_categories": sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_data_types": sorted(data_types.items(), key=lambda x: x[1], reverse=True)[:5],
        }
    
    async def _analyze_research_data(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze research data to extract key findings."""
        if not research_data:
            return {}
        
        analysis = {
            "data_sources": len(research_data.get("raw_results", [])),
            "has_synthesis": "synthesis" in research_data,
            "research_scope": research_data.get("research_context", {}).get("research_type", "unknown"),
        }
        
        # Extract key findings if available
        if "analysis" in research_data:
            analysis["key_findings"] = research_data["analysis"]
        
        return analysis
    
    async def _generate_insights(self, state: WorkflowState) -> List[str]:
        """Generate insights based on the current state."""
        insights = []
        
        # Basic insights based on search results
        if state.search_results:
            num_tools = len(state.search_results)
            insights.append(f"Found {num_tools} relevant tools for '{state.user_query}'")
            
            if num_tools > 10:
                insights.append("Large number of tools available - consider narrowing search criteria")
            elif num_tools < 3:
                insights.append("Limited tools found - consider broadening search terms")
        
        # Insights based on research data
        if state.research_data:
            insights.append("Enhanced research data available for comprehensive analysis")
        
        return insights


# Node factory functions
def create_coordinator_node(
    coordinator_agent: CoordinatorAgent,
    settings: Optional[AgentSettings] = None,
) -> CoordinatorNode:
    """Create a coordinator node instance.
    
    Args:
        coordinator_agent: CoordinatorAgent instance.
        settings: Optional configuration settings.
        
    Returns:
        CoordinatorNode: Created node instance.
    """
    return CoordinatorNode(coordinator_agent, settings)


def create_researcher_node(
    researcher_agent: ResearcherAgent,
    settings: Optional[AgentSettings] = None,
) -> ResearcherNode:
    """Create a researcher node instance.
    
    Args:
        researcher_agent: ResearcherAgent instance.
        settings: Optional configuration settings.
        
    Returns:
        ResearcherNode: Created node instance.
    """
    return ResearcherNode(researcher_agent, settings)


def create_reporter_node(
    reporter_agent: ReporterAgent,
    settings: Optional[AgentSettings] = None,
) -> ReporterNode:
    """Create a reporter node instance.
    
    Args:
        reporter_agent: ReporterAgent instance.
        settings: Optional configuration settings.
        
    Returns:
        ReporterNode: Created node instance.
    """
    return ReporterNode(reporter_agent, settings)


def create_search_node(
    vector_store: Optional[Any] = None,
    web_search: Optional[Any] = None,
    settings: Optional[AgentSettings] = None,
) -> SearchNode:
    """Create a search node instance.
    
    Args:
        vector_store: Optional vector store.
        web_search: Optional web search engine.
        settings: Optional configuration settings.
        
    Returns:
        SearchNode: Created node instance.
    """
    return SearchNode(vector_store, web_search, settings)


def create_analysis_node(settings: Optional[AgentSettings] = None) -> AnalysisNode:
    """Create an analysis node instance.
    
    Args:
        settings: Optional configuration settings.
        
    Returns:
        AnalysisNode: Created node instance.
    """
    return AnalysisNode(settings)


# Node type registry
NODE_REGISTRY = {
    "coordinator": create_coordinator_node,
    "researcher": create_researcher_node,
    "reporter": create_reporter_node,
    "search": create_search_node,
    "analysis": create_analysis_node,
}


def get_node_factory(node_type: str) -> Optional[Any]:
    """Get factory function for a node type.
    
    Args:
        node_type: Type of node to get factory for.
        
    Returns:
        Optional[Any]: Factory function or None if not found.
    """
    return NODE_REGISTRY.get(node_type)


def list_available_node_types() -> List[str]:
    """List all available node types.
    
    Returns:
        List[str]: List of node type names.
    """
    return list(NODE_REGISTRY.keys())