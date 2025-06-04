"""Base agent class and interfaces for the MCP Agent Framework.

This module provides the foundational base class and interfaces that all agents
in the framework inherit from, ensuring consistent behavior and capabilities.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Creating a custom agent:

    >>> class CustomAgent(BaseAgent):
    ...     async def _execute_task(self, task: AgentTask) -> AgentResponse:
    ...         # Implementation here
    ...         pass
    ...
    >>> agent = CustomAgent("custom_agent", settings)
    >>> await agent.initialize()
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
    from mcp_agent.config.settings import AgentSettings
    from mcp_agent.models.schemas import AgentResponse, AgentTask
    from mcp_agent.utils.logger import get_logger
except ImportError:
    # Mock classes for development
    class AgentSettings:
        pass
    class AgentResponse:
        pass
    class AgentTask:
        pass
    def get_logger(name: str):
        import logging
        return logging.getLogger(name)


class AgentCapability(Enum):
    """Enumeration of agent capabilities.
    
    This enum defines the various capabilities that agents can possess,
    allowing for dynamic capability discovery and agent selection.
    """
    
    # Core capabilities
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    USER_INTERACTION = "user_interaction"
    STATE_MANAGEMENT = "state_management"
    
    # Search and discovery capabilities
    WEB_SEARCH = "web_search"
    TOOL_DISCOVERY = "tool_discovery"
    VECTOR_SEARCH = "vector_search"
    SEMANTIC_SEARCH = "semantic_search"
    
    # Integration capabilities
    MCP_INTEGRATION = "mcp_integration"
    API_INTEGRATION = "api_integration"
    DATABASE_ACCESS = "database_access"
    
    # Content generation capabilities
    REPORT_GENERATION = "report_generation"
    CONTENT_FORMATTING = "content_formatting"
    SUMMARY_CREATION = "summary_creation"
    CODE_GENERATION = "code_generation"
    
    # Analysis capabilities
    DATA_ANALYSIS = "data_analysis"
    TEXT_ANALYSIS = "text_analysis"
    BIOINFORMATICS_ANALYSIS = "bioinformatics_analysis"
    
    # Communication capabilities
    NATURAL_LANGUAGE_PROCESSING = "natural_language_processing"
    MULTI_MODAL_PROCESSING = "multi_modal_processing"
    AUDIO_PROCESSING = "audio_processing"
    
    # Utility capabilities
    FILE_PROCESSING = "file_processing"
    CACHING = "caching"
    MONITORING = "monitoring"
    ERROR_HANDLING = "error_handling"


class AgentState(Enum):
    """Enumeration of agent states.
    
    Represents the current operational state of an agent.
    """
    
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    IDLE = "idle"
    WORKING = "working"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


class AgentMetrics(BaseModel):
    """Agent performance and operational metrics.
    
    Attributes:
        tasks_completed: Number of tasks completed.
        tasks_failed: Number of tasks that failed.
        total_execution_time: Total time spent executing tasks.
        average_execution_time: Average time per task.
        last_activity: Timestamp of last activity.
        errors: List of recent errors.
        
    Example:
        >>> metrics = agent.get_metrics()
        >>> print(f"Success rate: {metrics.success_rate}%")
    """
    
    tasks_completed: int = Field(default=0, description="Number of completed tasks")
    tasks_failed: int = Field(default=0, description="Number of failed tasks")
    total_execution_time: float = Field(default=0.0, description="Total execution time in seconds")
    average_execution_time: float = Field(default=0.0, description="Average execution time per task")
    last_activity: Optional[datetime] = Field(default=None, description="Last activity timestamp")
    errors: List[str] = Field(default_factory=list, description="Recent error messages")
    
    @property
    def total_tasks(self) -> int:
        """Get total number of tasks attempted."""
        return self.tasks_completed + self.tasks_failed
    
    @property
    def success_rate(self) -> float:
        """Get task success rate as percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.tasks_completed / self.total_tasks) * 100


class BaseAgent(ABC):
    """Abstract base class for all agents in the MCP Agent Framework.
    
    This class provides the foundational structure and common functionality
    that all specialized agents inherit from.
    
    Attributes:
        agent_id: Unique identifier for this agent instance.
        name: Human-readable name for the agent.
        settings: Configuration settings for the agent.
        state: Current operational state.
        capabilities: List of capabilities this agent provides.
        metrics: Performance and operational metrics.
        
    Example:
        >>> class MyAgent(BaseAgent):
        ...     async def _execute_task(self, task: AgentTask) -> AgentResponse:
        ...         return AgentResponse(content="Task completed")
        ...
        >>> agent = MyAgent("my_agent", settings)
        >>> await agent.initialize()
    """
    
    def __init__(
        self,
        name: str,
        settings: AgentSettings,
        capabilities: Optional[List[AgentCapability]] = None,
    ) -> None:
        """Initialize the base agent.
        
        Args:
            name: Human-readable name for the agent.
            settings: Configuration settings.
            capabilities: List of capabilities this agent provides.
        """
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.settings = settings
        self.state = AgentState.UNINITIALIZED
        self.capabilities = capabilities or []
        self.metrics = AgentMetrics()
        
        # Internal state
        self._initialized = False
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._start_time: Optional[float] = None
        
        # Logging
        self.logger = get_logger(f"{self.__class__.__name__}({self.name})")
        
        self.logger.info(f"Agent {self.name} created with ID: {self.agent_id}")
    
    async def initialize(self) -> None:
        """Initialize the agent and its resources.
        
        This method sets up any required resources, connections, or state
        needed for the agent to operate.
        
        Raises:
            RuntimeError: If initialization fails.
            
        Example:
            >>> await agent.initialize()
        """
        if self._initialized:
            self.logger.warning(f"Agent {self.name} already initialized")
            return
        
        self.logger.info(f"Initializing agent {self.name}")
        self.state = AgentState.INITIALIZING
        
        try:
            await self._initialize()
            self._initialized = True
            self.state = AgentState.IDLE
            self._start_time = time.time()
            
            self.logger.info(f"Agent {self.name} initialized successfully")
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Failed to initialize agent {self.name}: {e}")
            raise RuntimeError(f"Agent initialization failed: {e}") from e
    
    async def _initialize(self) -> None:
        """Internal initialization method to be overridden by subclasses.
        
        Subclasses should implement this method to perform their specific
        initialization logic.
        """
        pass
    
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a task and return the response.
        
        Args:
            task: Task to execute.
            
        Returns:
            AgentResponse: Result of task execution.
            
        Raises:
            RuntimeError: If agent is not initialized.
            ValueError: If task is invalid.
            
        Example:
            >>> task = AgentTask(type="search", parameters={"query": "RNA-seq"})
            >>> response = await agent.execute_task(task)
        """
        if not self._initialized:
            raise RuntimeError(f"Agent {self.name} not initialized")
        
        if self.state not in [AgentState.IDLE, AgentState.WORKING]:
            raise RuntimeError(f"Agent {self.name} not available (state: {self.state})")
        
        self.logger.info(f"Executing task: {task.type}")
        self.state = AgentState.WORKING
        self.metrics.last_activity = datetime.now()
        
        task_start_time = time.time()
        task_id = str(uuid.uuid4())
        
        try:
            # Create and track the task
            async_task = asyncio.create_task(self._execute_task(task))
            self._running_tasks[task_id] = async_task
            
            # Execute the task
            response = await async_task
            
            # Update metrics on success
            execution_time = time.time() - task_start_time
            self.metrics.tasks_completed += 1
            self.metrics.total_execution_time += execution_time
            self.metrics.average_execution_time = (
                self.metrics.total_execution_time / self.metrics.total_tasks
            )
            
            self.logger.info(f"Task completed in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            # Update metrics on failure
            self.metrics.tasks_failed += 1
            self.metrics.errors.append(f"{datetime.now()}: {str(e)}")
            
            # Keep only last 10 errors
            if len(self.metrics.errors) > 10:
                self.metrics.errors = self.metrics.errors[-10:]
            
            self.logger.error(f"Task execution failed: {e}")
            raise
            
        finally:
            # Clean up
            self._running_tasks.pop(task_id, None)
            if not self._running_tasks:
                self.state = AgentState.IDLE
    
    @abstractmethod
    async def _execute_task(self, task: AgentTask) -> AgentResponse:
        """Internal task execution method to be implemented by subclasses.
        
        Args:
            task: Task to execute.
            
        Returns:
            AgentResponse: Result of task execution.
            
        Note:
            This method must be implemented by all concrete agent classes.
        """
        pass
    
    async def pause(self) -> None:
        """Pause the agent's operations.
        
        This method pauses any ongoing operations and puts the agent
        in a paused state.
        
        Example:
            >>> await agent.pause()
        """
        self.logger.info(f"Pausing agent {self.name}")
        self.state = AgentState.PAUSED
        
        # Cancel running tasks
        for task_id, task in list(self._running_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._running_tasks.clear()
    
    async def resume(self) -> None:
        """Resume the agent's operations.
        
        This method resumes operations after the agent has been paused.
        
        Example:
            >>> await agent.resume()
        """
        if self.state == AgentState.PAUSED:
            self.logger.info(f"Resuming agent {self.name}")
            self.state = AgentState.IDLE
        else:
            self.logger.warning(f"Cannot resume agent {self.name} from state {self.state}")
    
    async def shutdown(self) -> None:
        """Shutdown the agent and clean up resources.
        
        This method performs cleanup and releases any resources held by the agent.
        
        Example:
            >>> await agent.shutdown()
        """
        if self.state == AgentState.SHUTDOWN:
            self.logger.warning(f"Agent {self.name} already shutdown")
            return
        
        self.logger.info(f"Shutting down agent {self.name}")
        self.state = AgentState.SHUTTING_DOWN
        
        try:
            # Cancel all running tasks
            for task_id, task in list(self._running_tasks.items()):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self._running_tasks.clear()
            
            # Perform cleanup
            await self._shutdown()
            
            self.state = AgentState.SHUTDOWN
            self._initialized = False
            
            self.logger.info(f"Agent {self.name} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.state = AgentState.ERROR
            raise
    
    async def _shutdown(self) -> None:
        """Internal shutdown method to be overridden by subclasses.
        
        Subclasses should implement this method to perform their specific
        cleanup logic.
        """
        pass
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if the agent has a specific capability.
        
        Args:
            capability: Capability to check for.
            
        Returns:
            bool: True if agent has the capability.
            
        Example:
            >>> if agent.has_capability(AgentCapability.WEB_SEARCH):
            ...     print("Agent can perform web search")
        """
        return capability in self.capabilities
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get list of agent capabilities.
        
        Returns:
            List[AgentCapability]: List of capabilities.
            
        Example:
            >>> capabilities = agent.get_capabilities()
            >>> print(f"Agent has {len(capabilities)} capabilities")
        """
        return self.capabilities.copy()
    
    def get_state(self) -> AgentState:
        """Get current agent state.
        
        Returns:
            AgentState: Current state.
            
        Example:
            >>> state = agent.get_state()
            >>> if state == AgentState.IDLE:
            ...     print("Agent is ready for tasks")
        """
        return self.state
    
    def get_metrics(self) -> AgentMetrics:
        """Get agent performance metrics.
        
        Returns:
            AgentMetrics: Current metrics.
            
        Example:
            >>> metrics = agent.get_metrics()
            >>> print(f"Success rate: {metrics.success_rate:.1f}%")
        """
        return self.metrics.model_copy()
    
    def get_running_tasks(self) -> List[str]:
        """Get list of currently running task IDs.
        
        Returns:
            List[str]: List of running task IDs.
            
        Example:
            >>> running = agent.get_running_tasks()
            >>> print(f"Agent has {len(running)} running tasks")
        """
        return list(self._running_tasks.keys())
    
    def get_uptime(self) -> float:
        """Get agent uptime in seconds.
        
        Returns:
            float: Uptime in seconds, or 0 if not started.
            
        Example:
            >>> uptime = agent.get_uptime()
            >>> print(f"Agent uptime: {uptime:.1f} seconds")
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def is_healthy(self) -> bool:
        """Check if the agent is in a healthy state.
        
        Returns:
            bool: True if agent is healthy and operational.
            
        Example:
            >>> if agent.is_healthy():
            ...     print("Agent is healthy")
        """
        return self.state in [AgentState.IDLE, AgentState.WORKING, AgentState.PAUSED]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check.
        
        Returns:
            Dict[str, Any]: Health status information.
            
        Example:
            >>> health = await agent.health_check()
            >>> print(f"Health status: {health['status']}")
        """
        health_status = {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "initialized": self._initialized,
            "healthy": self.is_healthy(),
            "uptime": self.get_uptime(),
            "running_tasks": len(self._running_tasks),
            "capabilities": [cap.value for cap in self.capabilities],
            "metrics": self.metrics.model_dump(),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add any subclass-specific health checks
        try:
            custom_health = await self._health_check()
            health_status.update(custom_health)
        except Exception as e:
            health_status["health_check_error"] = str(e)
        
        return health_status
    
    async def _health_check(self) -> Dict[str, Any]:
        """Internal health check method for subclasses to override.
        
        Returns:
            Dict[str, Any]: Additional health information.
        """
        return {}
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"state={self.state.value}, "
            f"capabilities={len(self.capabilities)}"
            f")"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name} ({self.state.value})"


class AgentTaskExecutor:
    """Utility class for managing agent task execution.
    
    This class provides utilities for executing tasks across multiple agents,
    handling timeouts, retries, and error recovery.
    
    Example:
        >>> executor = AgentTaskExecutor([agent1, agent2])
        >>> result = await executor.execute_with_fallback(task)
    """
    
    def __init__(self, agents: List[BaseAgent]) -> None:
        """Initialize the task executor.
        
        Args:
            agents: List of agents to use for task execution.
        """
        self.agents = agents
        self.logger = get_logger(self.__class__.__name__)
    
    async def execute_with_fallback(
        self,
        task: AgentTask,
        timeout: float = 60.0,
    ) -> AgentResponse:
        """Execute task with fallback to other agents on failure.
        
        Args:
            task: Task to execute.
            timeout: Timeout for task execution.
            
        Returns:
            AgentResponse: Result from first successful agent.
            
        Raises:
            RuntimeError: If all agents fail to execute the task.
        """
        last_error = None
        
        for agent in self.agents:
            if not agent.is_healthy():
                continue
            
            try:
                result = await asyncio.wait_for(
                    agent.execute_task(task),
                    timeout=timeout
                )
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Agent {agent.name} failed: {e}")
                continue
        
        raise RuntimeError(f"All agents failed to execute task: {last_error}")
    
    async def execute_parallel(
        self,
        task: AgentTask,
        require_consensus: bool = False,
    ) -> List[AgentResponse]:
        """Execute task on multiple agents in parallel.
        
        Args:
            task: Task to execute.
            require_consensus: Whether to require all agents to succeed.
            
        Returns:
            List[AgentResponse]: Results from all successful agents.
        """
        tasks = []
        for agent in self.agents:
            if agent.is_healthy():
                tasks.append(agent.execute_task(task))
        
        if require_consensus:
            results = await asyncio.gather(*tasks)
            return results
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if not isinstance(r, Exception)]