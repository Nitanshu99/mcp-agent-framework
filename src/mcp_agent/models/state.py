"""State management models for workflows, agents, and graph execution.

This module contains Pydantic models for managing state across the MCP Agent
Framework, including workflow states, agent states, and LangGraph execution states.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic state usage:

    >>> workflow_state = WorkflowState(
    ...     workflow_id="wf_123",
    ...     status="running",
    ...     current_step="research"
    ... )
    >>> graph_state = GraphState(user_query="Find RNA-seq tools")
"""

import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class WorkflowStatus(str, Enum):
    """Enumeration of workflow execution statuses."""
    
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AgentStatus(str, Enum):
    """Enumeration of agent statuses."""
    
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class NodeStatus(str, Enum):
    """Enumeration of graph node statuses."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionMode(str, Enum):
    """Enumeration of execution modes."""
    
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"


class WorkflowState(BaseModel):
    """Represents the state of a workflow execution.
    
    This model tracks the progress and status of multi-step workflows,
    including search, research, and report generation processes.
    
    Attributes:
        workflow_id: Unique workflow identifier.
        name: Workflow name.
        status: Current workflow status.
        current_step: Currently executing step.
        completed_steps: List of completed steps.
        failed_steps: List of failed steps.
        total_steps: Total number of steps.
        progress: Progress percentage (0-100).
        started_at: When the workflow started.
        updated_at: When the workflow was last updated.
        completed_at: When the workflow completed.
        results: Workflow execution results.
        errors: List of errors encountered.
        metadata: Additional workflow metadata.
        
    Example:
        >>> state = WorkflowState(
        ...     workflow_id="search_wf_001",
        ...     name="Tool Search Workflow",
        ...     current_step="vector_search",
        ...     total_steps=5
        ... )
    """
    
    workflow_id: str = Field(
        default_factory=lambda: f"wf_{uuid4().hex[:8]}",
        description="Unique workflow identifier"
    )
    name: str = Field(description="Workflow name")
    status: WorkflowStatus = Field(
        default=WorkflowStatus.PENDING,
        description="Current workflow status"
    )
    current_step: Optional[str] = Field(
        default=None,
        description="Currently executing step"
    )
    completed_steps: List[str] = Field(
        default_factory=list,
        description="List of completed steps"
    )
    failed_steps: List[str] = Field(
        default_factory=list,
        description="List of failed steps"
    )
    total_steps: int = Field(
        default=0,
        description="Total number of steps",
        ge=0
    )
    progress: float = Field(
        default=0.0,
        description="Progress percentage (0-100)",
        ge=0.0,
        le=100.0
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="When the workflow started"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the workflow was last updated"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When the workflow completed"
    )
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow execution results"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of errors encountered"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional workflow metadata"
    )
    
    def start_workflow(self) -> None:
        """Mark the workflow as started."""
        self.status = WorkflowStatus.RUNNING
        self.started_at = datetime.now()
        self.updated_at = datetime.now()
    
    def complete_step(self, step_name: str, result: Any = None) -> None:
        """Mark a step as completed."""
        if step_name not in self.completed_steps:
            self.completed_steps.append(step_name)
        
        if step_name in self.failed_steps:
            self.failed_steps.remove(step_name)
        
        if result is not None:
            self.results[step_name] = result
        
        self.update_progress()
        self.updated_at = datetime.now()
    
    def fail_step(self, step_name: str, error: str) -> None:
        """Mark a step as failed."""
        if step_name not in self.failed_steps:
            self.failed_steps.append(step_name)
        
        if step_name in self.completed_steps:
            self.completed_steps.remove(step_name)
        
        self.errors.append(f"{step_name}: {error}")
        self.update_progress()
        self.updated_at = datetime.now()
    
    def update_progress(self) -> None:
        """Update progress based on completed steps."""
        if self.total_steps > 0:
            self.progress = (len(self.completed_steps) / self.total_steps) * 100
    
    def complete_workflow(self) -> None:
        """Mark the workflow as completed."""
        self.status = WorkflowStatus.COMPLETED
        self.progress = 100.0
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
    
    def fail_workflow(self, error: str) -> None:
        """Mark the workflow as failed."""
        self.status = WorkflowStatus.FAILED
        self.errors.append(f"Workflow failed: {error}")
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
    
    def get_duration(self) -> Optional[float]:
        """Get workflow duration in seconds."""
        if self.started_at:
            end_time = self.completed_at or datetime.now()
            return (end_time - self.started_at).total_seconds()
        return None


class AgentState(BaseModel):
    """Represents the state of an individual agent.
    
    This model tracks the current status, tasks, and performance of agents
    in the multi-agent system.
    
    Attributes:
        agent_id: Unique agent identifier.
        name: Agent name.
        type: Agent type (coordinator, researcher, reporter).
        status: Current agent status.
        current_task: Currently executing task.
        task_queue: List of pending tasks.
        capabilities: Agent capabilities.
        performance_metrics: Performance tracking.
        last_activity: Last activity timestamp.
        error_count: Number of recent errors.
        metadata: Additional agent metadata.
        
    Example:
        >>> state = AgentState(
        ...     agent_id="researcher_001",
        ...     name="Research Agent",
        ...     type="researcher",
        ...     capabilities=["web_search", "vector_search"]
        ... )
    """
    
    agent_id: str = Field(description="Unique agent identifier")
    name: str = Field(description="Agent name")
    type: str = Field(description="Agent type")
    status: AgentStatus = Field(
        default=AgentStatus.IDLE,
        description="Current agent status"
    )
    current_task: Optional[str] = Field(
        default=None,
        description="Currently executing task ID"
    )
    task_queue: List[str] = Field(
        default_factory=list,
        description="List of pending task IDs"
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="Agent capabilities"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance tracking metrics"
    )
    last_activity: Optional[datetime] = Field(
        default=None,
        description="Last activity timestamp"
    )
    error_count: int = Field(
        default=0,
        description="Number of recent errors",
        ge=0
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional agent metadata"
    )
    
    def set_busy(self, task_id: str) -> None:
        """Set agent as busy with a specific task."""
        self.status = AgentStatus.BUSY
        self.current_task = task_id
        self.last_activity = datetime.now()
    
    def set_idle(self) -> None:
        """Set agent as idle."""
        self.status = AgentStatus.IDLE
        self.current_task = None
        self.last_activity = datetime.now()
    
    def add_task(self, task_id: str) -> None:
        """Add a task to the queue."""
        if task_id not in self.task_queue:
            self.task_queue.append(task_id)
    
    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        if self.current_task == task_id:
            self.current_task = None
            self.status = AgentStatus.IDLE
        
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)
        
        self.last_activity = datetime.now()
    
    def record_error(self) -> None:
        """Record an error occurrence."""
        self.error_count += 1
        self.last_activity = datetime.now()
    
    def reset_error_count(self) -> None:
        """Reset error count."""
        self.error_count = 0
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return self.status in [AgentStatus.IDLE, AgentStatus.WAITING]


class TaskState(BaseModel):
    """Represents the state of an individual task execution.
    
    Attributes:
        task_id: Unique task identifier.
        workflow_id: Associated workflow ID.
        agent_id: Agent executing the task.
        status: Current task status.
        priority: Task priority.
        created_at: When the task was created.
        started_at: When task execution started.
        completed_at: When task execution completed.
        retry_count: Number of retry attempts.
        max_retries: Maximum retry attempts allowed.
        timeout: Task timeout in seconds.
        progress: Task progress percentage.
        result: Task execution result.
        error: Error information if task failed.
        dependencies: List of dependent task IDs.
        metadata: Additional task metadata.
        
    Example:
        >>> state = TaskState(
        ...     task_id="task_001",
        ...     workflow_id="wf_123",
        ...     agent_id="researcher_001",
        ...     priority=5
        ... )
    """
    
    task_id: str = Field(description="Unique task identifier")
    workflow_id: Optional[str] = Field(
        default=None,
        description="Associated workflow ID"
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Agent executing the task"
    )
    status: NodeStatus = Field(
        default=NodeStatus.PENDING,
        description="Current task status"
    )
    priority: int = Field(
        default=0,
        description="Task priority",
        ge=0,
        le=10
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the task was created"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="When task execution started"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When task execution completed"
    )
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts",
        ge=0
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts allowed",
        ge=0
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Task timeout in seconds"
    )
    progress: float = Field(
        default=0.0,
        description="Task progress percentage",
        ge=0.0,
        le=100.0
    )
    result: Optional[Any] = Field(
        default=None,
        description="Task execution result"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error information if task failed"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of dependent task IDs"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata"
    )
    
    def start_execution(self, agent_id: str) -> None:
        """Start task execution."""
        self.status = NodeStatus.RUNNING
        self.agent_id = agent_id
        self.started_at = datetime.now()
    
    def complete_execution(self, result: Any = None) -> None:
        """Complete task execution."""
        self.status = NodeStatus.COMPLETED
        self.progress = 100.0
        self.completed_at = datetime.now()
        if result is not None:
            self.result = result
    
    def fail_execution(self, error: str) -> None:
        """Fail task execution."""
        self.status = NodeStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
    
    def retry_execution(self) -> None:
        """Retry task execution."""
        if self.can_retry():
            self.retry_count += 1
            self.status = NodeStatus.PENDING
            self.error = None
            self.started_at = None
            self.completed_at = None
    
    def get_execution_time(self) -> Optional[float]:
        """Get task execution time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class GraphState(BaseModel):
    """Represents the state of a LangGraph workflow execution.
    
    This model is used by LangGraph to manage state across different nodes
    and edges in the workflow graph.
    
    Attributes:
        user_query: Original user query.
        search_query: Processed search query.
        search_results: Results from search operations.
        research_data: Collected research data.
        analysis_results: Results from analysis operations.
        report_content: Generated report content.
        current_node: Currently executing node.
        node_history: History of executed nodes.
        node_outputs: Outputs from each node.
        error_info: Error information if any.
        metadata: Additional state metadata.
        
    Example:
        >>> state = GraphState(
        ...     user_query="Find tools for protein structure analysis",
        ...     current_node="search_node"
        ... )
    """
    
    user_query: str = Field(description="Original user query")
    search_query: Optional[str] = Field(
        default=None,
        description="Processed search query"
    )
    search_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results from search operations"
    )
    research_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Collected research data"
    )
    analysis_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results from analysis operations"
    )
    report_content: Optional[str] = Field(
        default=None,
        description="Generated report content"
    )
    current_node: Optional[str] = Field(
        default=None,
        description="Currently executing node"
    )
    node_history: List[str] = Field(
        default_factory=list,
        description="History of executed nodes"
    )
    node_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Outputs from each node"
    )
    error_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error information if any"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional state metadata"
    )
    
    def enter_node(self, node_name: str) -> None:
        """Record entering a node."""
        self.current_node = node_name
        if node_name not in self.node_history:
            self.node_history.append(node_name)
    
    def exit_node(self, node_name: str, output: Any = None) -> None:
        """Record exiting a node."""
        if self.current_node == node_name:
            self.current_node = None
        
        if output is not None:
            self.node_outputs[node_name] = output
    
    def set_error(self, node_name: str, error: str, details: Dict[str, Any] = None) -> None:
        """Set error information."""
        self.error_info = {
            "node": node_name,
            "error": error,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        }
    
    def clear_error(self) -> None:
        """Clear error information."""
        self.error_info = None
    
    def has_error(self) -> bool:
        """Check if there's an error."""
        return self.error_info is not None


class NodeState(BaseModel):
    """Represents the state of an individual graph node.
    
    Attributes:
        node_id: Unique node identifier.
        name: Node name.
        type: Node type.
        status: Current node status.
        input_data: Input data for the node.
        output_data: Output data from the node.
        execution_time: Node execution time.
        error_info: Error information if node failed.
        retry_count: Number of retry attempts.
        dependencies: List of dependent nodes.
        metadata: Additional node metadata.
        
    Example:
        >>> state = NodeState(
        ...     node_id="search_node_001",
        ...     name="Vector Search Node",
        ...     type="search"
        ... )
    """
    
    node_id: str = Field(description="Unique node identifier")
    name: str = Field(description="Node name")
    type: str = Field(description="Node type")
    status: NodeStatus = Field(
        default=NodeStatus.PENDING,
        description="Current node status"
    )
    input_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Input data for the node"
    )
    output_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output data from the node"
    )
    execution_time: Optional[float] = Field(
        default=None,
        description="Node execution time in seconds"
    )
    error_info: Optional[str] = Field(
        default=None,
        description="Error information if node failed"
    )
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts",
        ge=0
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of dependent node IDs"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional node metadata"
    )


class EdgeState(BaseModel):
    """Represents the state of a graph edge (connection between nodes).
    
    Attributes:
        edge_id: Unique edge identifier.
        source_node: Source node ID.
        target_node: Target node ID.
        condition: Edge condition for conditional execution.
        weight: Edge weight for prioritization.
        traversed: Whether the edge has been traversed.
        traversal_count: Number of times traversed.
        data_passed: Data passed through the edge.
        metadata: Additional edge metadata.
        
    Example:
        >>> state = EdgeState(
        ...     edge_id="search_to_analysis",
        ...     source_node="search_node",
        ...     target_node="analysis_node"
        ... )
    """
    
    edge_id: str = Field(description="Unique edge identifier")
    source_node: str = Field(description="Source node ID")
    target_node: str = Field(description="Target node ID")
    condition: Optional[str] = Field(
        default=None,
        description="Edge condition for conditional execution"
    )
    weight: float = Field(
        default=1.0,
        description="Edge weight for prioritization"
    )
    traversed: bool = Field(
        default=False,
        description="Whether the edge has been traversed"
    )
    traversal_count: int = Field(
        default=0,
        description="Number of times traversed",
        ge=0
    )
    data_passed: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Data passed through the edge"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional edge metadata"
    )
    
    def traverse(self, data: Dict[str, Any] = None) -> None:
        """Record edge traversal."""
        self.traversed = True
        self.traversal_count += 1
        if data:
            self.data_passed = data


class ExecutionContext(BaseModel):
    """Represents the execution context for a workflow or task.
    
    Attributes:
        context_id: Unique context identifier.
        user_id: User associated with the execution.
        session_id: Session identifier.
        workflow_id: Associated workflow ID.
        agent_assignments: Mapping of tasks to agents.
        resource_allocations: Resource allocations.
        timeout_settings: Timeout configurations.
        retry_policies: Retry policy configurations.
        environment: Execution environment settings.
        permissions: Execution permissions.
        metadata: Additional context metadata.
        
    Example:
        >>> context = ExecutionContext(
        ...     user_id="user_123",
        ...     session_id="session_456",
        ...     workflow_id="wf_789"
        ... )
    """
    
    context_id: str = Field(
        default_factory=lambda: f"ctx_{uuid4().hex[:8]}",
        description="Unique context identifier"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User associated with the execution"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier"
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="Associated workflow ID"
    )
    agent_assignments: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of task IDs to agent IDs"
    )
    resource_allocations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource allocations"
    )
    timeout_settings: Dict[str, float] = Field(
        default_factory=dict,
        description="Timeout configurations"
    )
    retry_policies: Dict[str, int] = Field(
        default_factory=dict,
        description="Retry policy configurations"
    )
    environment: Dict[str, str] = Field(
        default_factory=dict,
        description="Execution environment settings"
    )
    permissions: List[str] = Field(
        default_factory=list,
        description="Execution permissions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata"
    )


class ExecutionResult(BaseModel):
    """Represents the result of a workflow or task execution.
    
    Attributes:
        execution_id: Unique execution identifier.
        context_id: Associated context ID.
        status: Execution status.
        result: Execution result data.
        error: Error information if execution failed.
        start_time: Execution start time.
        end_time: Execution end time.
        duration: Execution duration in seconds.
        resource_usage: Resource usage statistics.
        performance_metrics: Performance metrics.
        metadata: Additional result metadata.
        
    Example:
        >>> result = ExecutionResult(
        ...     execution_id="exec_001",
        ...     status="completed",
        ...     result={"tools_found": 5}
        ... )
    """
    
    execution_id: str = Field(
        default_factory=lambda: f"exec_{uuid4().hex[:8]}",
        description="Unique execution identifier"
    )
    context_id: Optional[str] = Field(
        default=None,
        description="Associated context ID"
    )
    status: str = Field(description="Execution status")
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Execution result data"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error information if execution failed"
    )
    start_time: datetime = Field(
        default_factory=datetime.now,
        description="Execution start time"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Execution end time"
    )
    duration: Optional[float] = Field(
        default=None,
        description="Execution duration in seconds"
    )
    resource_usage: Dict[str, float] = Field(
        default_factory=dict,
        description="Resource usage statistics"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result metadata"
    )
    
    def complete(self, result: Dict[str, Any] = None) -> None:
        """Mark execution as completed."""
        self.status = "completed"
        self.end_time = datetime.now()
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        if result:
            self.result = result
    
    def fail(self, error: str) -> None:
        """Mark execution as failed."""
        self.status = "failed"
        self.error = error
        self.end_time = datetime.now()
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()


class ExecutionMetrics(BaseModel):
    """Represents execution metrics and performance data.
    
    Attributes:
        metrics_id: Unique metrics identifier.
        execution_id: Associated execution ID.
        cpu_usage: CPU usage percentage.
        memory_usage: Memory usage in MB.
        network_io: Network I/O statistics.
        disk_io: Disk I/O statistics.
        api_calls: Number of API calls made.
        cache_hits: Number of cache hits.
        cache_misses: Number of cache misses.
        error_rate: Error rate percentage.
        throughput: Operations per second.
        latency: Average latency in milliseconds.
        custom_metrics: Custom metrics dictionary.
        timestamp: Metrics timestamp.
        
    Example:
        >>> metrics = ExecutionMetrics(
        ...     execution_id="exec_001",
        ...     cpu_usage=45.2,
        ...     memory_usage=1024,
        ...     api_calls=15
        ... )
    """
    
    metrics_id: str = Field(
        default_factory=lambda: f"metrics_{uuid4().hex[:8]}",
        description="Unique metrics identifier"
    )
    execution_id: Optional[str] = Field(
        default=None,
        description="Associated execution ID"
    )
    cpu_usage: Optional[float] = Field(
        default=None,
        description="CPU usage percentage",
        ge=0.0,
        le=100.0
    )
    memory_usage: Optional[float] = Field(
        default=None,
        description="Memory usage in MB",
        ge=0.0
    )
    network_io: Dict[str, float] = Field(
        default_factory=dict,
        description="Network I/O statistics"
    )
    disk_io: Dict[str, float] = Field(
        default_factory=dict,
        description="Disk I/O statistics"
    )
    api_calls: int = Field(
        default=0,
        description="Number of API calls made",
        ge=0
    )
    cache_hits: int = Field(
        default=0,
        description="Number of cache hits",
        ge=0
    )
    cache_misses: int = Field(
        default=0,
        description="Number of cache misses",
        ge=0
    )
    error_rate: float = Field(
        default=0.0,
        description="Error rate percentage",
        ge=0.0,
        le=100.0
    )
    throughput: Optional[float] = Field(
        default=None,
        description="Operations per second",
        ge=0.0
    )
    latency: Optional[float] = Field(
        default=None,
        description="Average latency in milliseconds",
        ge=0.0
    )
    custom_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Custom metrics dictionary"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Metrics timestamp"
    )
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return (self.cache_hits / total_requests) * 100
    
    def add_api_call(self) -> None:
        """Increment API call count."""
        self.api_calls += 1
    
    def add_cache_hit(self) -> None:
        """Increment cache hit count."""
        self.cache_hits += 1
    
    def add_cache_miss(self) -> None:
        """Increment cache miss count."""
        self.cache_misses += 1