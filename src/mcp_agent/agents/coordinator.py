"""Coordinator agent for orchestrating multi-agent workflows.

The CoordinatorAgent serves as the main orchestrator in the MCP Agent Framework,
managing communication between specialized agents and coordinating complex workflows.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic coordinator usage:

    >>> coordinator = CoordinatorAgent(settings, mcp_client, vector_store)
    >>> await coordinator.initialize()
    >>> result = await coordinator.search(SearchQuery(text="RNA-seq tools"))
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from mcp_agent.agents.base import BaseAgent, AgentCapability, AgentState
    from mcp_agent.config.settings import AgentSettings
    from mcp_agent.models.schemas import (
        SearchQuery,
        SearchResult,
        AgentTask,
        AgentResponse,
        ToolInfo,
        WorkflowState,
    )
    from mcp_agent.tools.mcp_client import MCPClient
    from mcp_agent.tools.vector_store import VectorStore
    from mcp_agent.utils.logger import get_logger
except ImportError:
    # Mock imports for development
    class BaseAgent:
        pass
    class AgentCapability:
        WORKFLOW_ORCHESTRATION = "workflow_orchestration"
        USER_INTERACTION = "user_interaction"
        STATE_MANAGEMENT = "state_management"
    class AgentState:
        pass
    class AgentSettings:
        pass
    class SearchQuery:
        pass
    class SearchResult:
        pass
    class AgentTask:
        pass
    class AgentResponse:
        pass
    class ToolInfo:
        pass
    class WorkflowState:
        pass
    class MCPClient:
        pass
    class VectorStore:
        pass
    class ChatGoogleGenerativeAI:
        pass
    def get_logger(name: str):
        import logging
        return logging.getLogger(name)


class CoordinatorAgent(BaseAgent):
    """Main coordinator agent that orchestrates multi-agent workflows.
    
    The CoordinatorAgent serves as the central hub for managing interactions
    between specialized agents (Researcher, Reporter) and coordinating complex
    tasks like search, research, and report generation.
    
    Attributes:
        settings: Configuration settings for the agent.
        mcp_client: Client for MCP server communication.
        vector_store: Vector database for semantic search.
        llm: Language model for intelligent decision making.
        researcher: Researcher agent instance.
        reporter: Reporter agent instance.
        workflow_state: Current workflow state.
        
    Example:
        >>> coordinator = CoordinatorAgent(
        ...     settings=settings,
        ...     mcp_client=mcp_client,
        ...     vector_store=vector_store
        ... )
        >>> await coordinator.initialize()
        >>> result = await coordinator.orchestrate_search("protein analysis tools")
    """
    
    def __init__(
        self,
        settings: AgentSettings,
        mcp_client: Optional[MCPClient] = None,
        vector_store: Optional[VectorStore] = None,
        name: str = "coordinator",
    ) -> None:
        """Initialize the coordinator agent.
        
        Args:
            settings: Configuration settings.
            mcp_client: Optional MCP client instance.
            vector_store: Optional vector store instance.
            name: Agent name (default: "coordinator").
        """
        super().__init__(
            name=name,
            settings=settings,
            capabilities=[
                AgentCapability.WORKFLOW_ORCHESTRATION,
                AgentCapability.USER_INTERACTION,
                AgentCapability.STATE_MANAGEMENT,
                AgentCapability.NATURAL_LANGUAGE_PROCESSING,
            ]
        )
        
        # External dependencies
        self.mcp_client = mcp_client
        self.vector_store = vector_store
        
        # LLM for intelligent coordination
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        
        # Specialized agents (will be initialized later)
        self.researcher: Optional[BaseAgent] = None
        self.reporter: Optional[BaseAgent] = None
        
        # Workflow state management
        self.workflow_state = WorkflowState()
        self._active_workflows: Dict[str, Dict[str, Any]] = {}
        
        # Coordination settings
        self.max_concurrent_workflows = getattr(
            settings, 'max_concurrent_requests', 5
        )
        self.default_timeout = getattr(settings, 'mcp_timeout', 60)
        
        self.logger.info(f"CoordinatorAgent initialized with {len(self.capabilities)} capabilities")
    
    async def _initialize(self) -> None:
        """Initialize the coordinator agent and its components."""
        self.logger.info("Initializing CoordinatorAgent...")
        
        try:
            # Initialize LLM
            self._initialize_llm()
            
            # Initialize specialized agents
            await self._initialize_agents()
            
            # Validate dependencies
            await self._validate_dependencies()
            
            self.logger.info("CoordinatorAgent initialization complete")
            
        except Exception as e:
            self.logger.error(f"CoordinatorAgent initialization failed: {e}")
            raise
    
    def _initialize_llm(self) -> None:
        """Initialize the language model for coordination decisions."""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.settings.llm.model,
                temperature=self.settings.llm.temperature,
                max_tokens=self.settings.llm.max_tokens,
                google_api_key=self.settings.google_api_key,
            )
            self.logger.info(f"LLM initialized: {self.settings.llm.model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def _initialize_agents(self) -> None:
        """Initialize specialized agents."""
        try:
            # Import and create specialized agents
            from mcp_agent.agents.researcher import ResearcherAgent
            from mcp_agent.agents.reporter import ReporterAgent
            
            # Initialize researcher agent
            self.researcher = ResearcherAgent(
                settings=self.settings,
                mcp_client=self.mcp_client,
                vector_store=self.vector_store,
            )
            await self.researcher.initialize()
            
            # Initialize reporter agent
            self.reporter = ReporterAgent(
                settings=self.settings,
            )
            await self.reporter.initialize()
            
            self.logger.info("Specialized agents initialized successfully")
            
        except ImportError:
            self.logger.warning("Specialized agents not available, operating in standalone mode")
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def _validate_dependencies(self) -> None:
        """Validate that required dependencies are available."""
        issues = []
        
        if not self.llm:
            issues.append("LLM not initialized")
        
        if not self.mcp_client:
            issues.append("MCP client not available")
            
        if not self.vector_store:
            issues.append("Vector store not available")
        
        if issues:
            self.logger.warning(f"Dependency issues: {', '.join(issues)}")
    
    async def _execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a coordinator task.
        
        Args:
            task: Task to execute.
            
        Returns:
            AgentResponse: Task execution result.
        """
        self.logger.info(f"Executing coordinator task: {task.type}")
        
        try:
            # Route task based on type
            if task.type == "search":
                return await self._handle_search_task(task)
            elif task.type == "research":
                return await self._handle_research_task(task)
            elif task.type == "workflow":
                return await self._handle_workflow_task(task)
            elif task.type == "coordinate":
                return await self._handle_coordination_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.type}")
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return AgentResponse(
                success=False,
                content=f"Task execution failed: {e}",
                error=str(e),
                agent_id=self.agent_id,
            )
    
    async def _handle_search_task(self, task: AgentTask) -> AgentResponse:
        """Handle search-related tasks."""
        query = task.parameters.get("query")
        if not query:
            raise ValueError("Search task requires 'query' parameter")
        
        # Create SearchQuery object
        search_query = SearchQuery(
            text=query,
            max_results=task.parameters.get("max_results", 10),
            filters=task.parameters.get("filters", {}),
            include_documentation=task.parameters.get("include_documentation", True),
        )
        
        # Perform search using researcher
        result = await self.search(search_query)
        
        return AgentResponse(
            success=True,
            content=result.summary if hasattr(result, 'summary') else "Search completed",
            data=result.model_dump() if hasattr(result, 'model_dump') else result,
            agent_id=self.agent_id,
        )
    
    async def _handle_research_task(self, task: AgentTask) -> AgentResponse:
        """Handle research-related tasks."""
        topic = task.parameters.get("topic")
        if not topic:
            raise ValueError("Research task requires 'topic' parameter")
        
        depth = task.parameters.get("depth", "standard")
        output_format = task.parameters.get("output_format", "markdown")
        
        # Perform research using the full workflow
        result = await self.orchestrate_research(
            topic=topic,
            depth=depth,
            output_format=output_format,
        )
        
        return AgentResponse(
            success=True,
            content=result.get("report", "Research completed"),
            data=result,
            agent_id=self.agent_id,
        )
    
    async def _handle_workflow_task(self, task: AgentTask) -> AgentResponse:
        """Handle workflow orchestration tasks."""
        workflow_type = task.parameters.get("workflow_type")
        if not workflow_type:
            raise ValueError("Workflow task requires 'workflow_type' parameter")
        
        # Execute workflow based on type
        if workflow_type == "search_and_report":
            return await self._orchestrate_search_and_report(task.parameters)
        elif workflow_type == "multi_agent_research":
            return await self._orchestrate_multi_agent_research(task.parameters)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    async def _handle_coordination_task(self, task: AgentTask) -> AgentResponse:
        """Handle agent coordination tasks."""
        action = task.parameters.get("action")
        
        if action == "status":
            return await self._get_system_status()
        elif action == "health_check":
            return await self._perform_health_check()
        elif action == "metrics":
            return await self._get_system_metrics()
        else:
            raise ValueError(f"Unknown coordination action: {action}")
    
    async def search(self, query: SearchQuery) -> SearchResult:
        """Perform a coordinated search across multiple sources.
        
        Args:
            query: Search query with parameters.
            
        Returns:
            SearchResult: Comprehensive search results.
            
        Example:
            >>> query = SearchQuery(text="RNA sequencing tools")
            >>> result = await coordinator.search(query)
            >>> print(f"Found {len(result.tools)} tools")
        """
        workflow_id = f"search_{datetime.now().isoformat()}"
        self.logger.info(f"Starting search workflow {workflow_id}: {query.text}")
        
        try:
            # Track workflow
            self._active_workflows[workflow_id] = {
                "type": "search",
                "query": query.text,
                "started": datetime.now(),
                "status": "running",
            }
            
            # Use researcher agent if available
            if self.researcher:
                search_task = AgentTask(
                    type="vector_search",
                    parameters={
                        "query": query.text,
                        "max_results": query.max_results,
                        "filters": query.filters,
                        "include_documentation": query.include_documentation,
                    }
                )
                
                response = await self.researcher.execute_task(search_task)
                
                if response.success and response.data:
                    # Convert response data to SearchResult
                    tools = []
                    if isinstance(response.data, dict) and "tools" in response.data:
                        for tool_data in response.data["tools"]:
                            tools.append(ToolInfo(**tool_data))
                    
                    result = SearchResult(
                        query=query.text,
                        tools=tools,
                        total_results=len(tools),
                        summary=response.content,
                        metadata={
                            "workflow_id": workflow_id,
                            "agent": "researcher",
                            "execution_time": (datetime.now() - self._active_workflows[workflow_id]["started"]).total_seconds(),
                        }
                    )
                    
                    self._active_workflows[workflow_id]["status"] = "completed"
                    return result
            
            # Fallback to direct vector store search
            if self.vector_store:
                results = await self.vector_store.search(
                    query=query.text,
                    limit=query.max_results,
                    filters=query.filters,
                )
                
                tools = [ToolInfo(**result) for result in results]
                
                result = SearchResult(
                    query=query.text,
                    tools=tools,
                    total_results=len(tools),
                    summary=f"Found {len(tools)} tools matching '{query.text}'",
                    metadata={
                        "workflow_id": workflow_id,
                        "fallback": True,
                    }
                )
                
                self._active_workflows[workflow_id]["status"] = "completed"
                return result
            
            # No search capability available
            raise RuntimeError("No search capability available")
            
        except Exception as e:
            self._active_workflows[workflow_id]["status"] = "failed"
            self._active_workflows[workflow_id]["error"] = str(e)
            self.logger.error(f"Search workflow {workflow_id} failed: {e}")
            raise
        
        finally:
            # Clean up old workflows
            self._cleanup_workflows()
    
    async def orchestrate_research(
        self,
        topic: str,
        depth: str = "standard",
        output_format: str = "markdown",
        include_code_examples: bool = True,
    ) -> Dict[str, Any]:
        """Orchestrate a comprehensive research workflow.
        
        Args:
            topic: Research topic.
            depth: Research depth ('quick', 'standard', 'comprehensive').
            output_format: Output format ('markdown', 'html', 'json').
            include_code_examples: Whether to include code examples.
            
        Returns:
            Dict[str, Any]: Research results and report.
            
        Example:
            >>> result = await coordinator.orchestrate_research(
            ...     "Machine learning in genomics",
            ...     depth="comprehensive"
            ... )
        """
        workflow_id = f"research_{datetime.now().isoformat()}"
        self.logger.info(f"Starting research workflow {workflow_id}: {topic}")
        
        try:
            # Track workflow
            self._active_workflows[workflow_id] = {
                "type": "research",
                "topic": topic,
                "depth": depth,
                "started": datetime.now(),
                "status": "running",
                "stages": [],
            }
            
            workflow = self._active_workflows[workflow_id]
            
            # Stage 1: Initial search for tools and papers
            workflow["stages"].append("initial_search")
            search_query = SearchQuery(
                text=topic,
                max_results=20 if depth == "comprehensive" else 10,
                include_documentation=True,
            )
            
            search_results = await self.search(search_query)
            
            # Stage 2: Enhanced research if researcher available
            workflow["stages"].append("enhanced_research")
            enhanced_results = {}
            
            if self.researcher:
                research_task = AgentTask(
                    type="web_research",
                    parameters={
                        "topic": topic,
                        "depth": depth,
                        "tools_found": [tool.model_dump() for tool in search_results.tools],
                    }
                )
                
                research_response = await self.researcher.execute_task(research_task)
                if research_response.success:
                    enhanced_results = research_response.data or {}
            
            # Stage 3: Generate report if reporter available
            workflow["stages"].append("report_generation")
            report_content = ""
            
            if self.reporter:
                report_task = AgentTask(
                    type="generate_report",
                    parameters={
                        "topic": topic,
                        "search_results": search_results.model_dump(),
                        "enhanced_results": enhanced_results,
                        "output_format": output_format,
                        "include_code_examples": include_code_examples,
                    }
                )
                
                report_response = await self.reporter.execute_task(report_task)
                if report_response.success:
                    report_content = report_response.content
            
            # Compile final results
            result = {
                "workflow_id": workflow_id,
                "topic": topic,
                "depth": depth,
                "search_results": search_results.model_dump(),
                "enhanced_results": enhanced_results,
                "report": report_content or self._generate_basic_report(topic, search_results),
                "metadata": {
                    "execution_time": (datetime.now() - workflow["started"]).total_seconds(),
                    "stages_completed": len(workflow["stages"]),
                    "tools_found": len(search_results.tools),
                },
            }
            
            workflow["status"] = "completed"
            return result
            
        except Exception as e:
            self._active_workflows[workflow_id]["status"] = "failed"
            self._active_workflows[workflow_id]["error"] = str(e)
            self.logger.error(f"Research workflow {workflow_id} failed: {e}")
            raise
    
    def _generate_basic_report(self, topic: str, search_results: SearchResult) -> str:
        """Generate a basic report when reporter agent is not available."""
        report = f"# Research Report: {topic}\n\n"
        report += f"## Summary\n\n"
        report += f"Found {len(search_results.tools)} relevant tools for {topic}.\n\n"
        report += f"## Tools Found\n\n"
        
        for tool in search_results.tools[:10]:  # Limit to top 10
            report += f"### {tool.name}\n\n"
            if hasattr(tool, 'description'):
                report += f"{tool.description}\n\n"
            if hasattr(tool, 'category'):
                report += f"**Category:** {tool.category}\n\n"
        
        return report
    
    async def _orchestrate_search_and_report(self, parameters: Dict[str, Any]) -> AgentResponse:
        """Orchestrate a search and report workflow."""
        query = parameters.get("query")
        if not query:
            raise ValueError("Search and report workflow requires 'query' parameter")
        
        # Perform search
        search_query = SearchQuery(text=query)
        search_results = await self.search(search_query)
        
        # Generate report
        if self.reporter:
            report_task = AgentTask(
                type="generate_report",
                parameters={
                    "topic": query,
                    "search_results": search_results.model_dump(),
                    "output_format": parameters.get("output_format", "markdown"),
                }
            )
            
            report_response = await self.reporter.execute_task(report_task)
            content = report_response.content
        else:
            content = self._generate_basic_report(query, search_results)
        
        return AgentResponse(
            success=True,
            content=content,
            data={
                "search_results": search_results.model_dump(),
                "report": content,
            },
            agent_id=self.agent_id,
        )
    
    async def _orchestrate_multi_agent_research(self, parameters: Dict[str, Any]) -> AgentResponse:
        """Orchestrate a multi-agent research workflow."""
        topic = parameters.get("topic")
        if not topic:
            raise ValueError("Multi-agent research requires 'topic' parameter")
        
        # Run research workflow
        result = await self.orchestrate_research(
            topic=topic,
            depth=parameters.get("depth", "standard"),
            output_format=parameters.get("output_format", "markdown"),
        )
        
        return AgentResponse(
            success=True,
            content=result["report"],
            data=result,
            agent_id=self.agent_id,
        )
    
    async def _get_system_status(self) -> AgentResponse:
        """Get comprehensive system status."""
        status = {
            "coordinator": {
                "state": self.state.value,
                "healthy": self.is_healthy(),
                "uptime": self.get_uptime(),
                "active_workflows": len(self._active_workflows),
            },
            "agents": {},
            "dependencies": {
                "llm": self.llm is not None,
                "mcp_client": self.mcp_client is not None,
                "vector_store": self.vector_store is not None,
            },
        }
        
        # Check specialized agents
        if self.researcher:
            status["agents"]["researcher"] = {
                "state": self.researcher.state.value,
                "healthy": self.researcher.is_healthy(),
            }
        
        if self.reporter:
            status["agents"]["reporter"] = {
                "state": self.reporter.state.value,
                "healthy": self.reporter.is_healthy(),
            }
        
        return AgentResponse(
            success=True,
            content="System status retrieved",
            data=status,
            agent_id=self.agent_id,
        )
    
    async def _perform_health_check(self) -> AgentResponse:
        """Perform comprehensive health check."""
        health_data = await self.health_check()
        
        # Add specialized agent health
        if self.researcher:
            health_data["researcher"] = await self.researcher.health_check()
        
        if self.reporter:
            health_data["reporter"] = await self.reporter.health_check()
        
        return AgentResponse(
            success=True,
            content="Health check completed",
            data=health_data,
            agent_id=self.agent_id,
        )
    
    async def _get_system_metrics(self) -> AgentResponse:
        """Get comprehensive system metrics."""
        metrics = {
            "coordinator": self.get_metrics().model_dump(),
            "agents": {},
            "workflows": {
                "active": len(self._active_workflows),
                "completed": len([w for w in self._active_workflows.values() if w["status"] == "completed"]),
                "failed": len([w for w in self._active_workflows.values() if w["status"] == "failed"]),
            },
        }
        
        if self.researcher:
            metrics["agents"]["researcher"] = self.researcher.get_metrics().model_dump()
        
        if self.reporter:
            metrics["agents"]["reporter"] = self.reporter.get_metrics().model_dump()
        
        return AgentResponse(
            success=True,
            content="System metrics retrieved",
            data=metrics,
            agent_id=self.agent_id,
        )
    
    def _cleanup_workflows(self) -> None:
        """Clean up old completed workflows."""
        cutoff = datetime.now().timestamp() - 3600  # 1 hour ago
        
        to_remove = []
        for workflow_id, workflow in self._active_workflows.items():
            if workflow["started"].timestamp() < cutoff and workflow["status"] in ["completed", "failed"]:
                to_remove.append(workflow_id)
        
        for workflow_id in to_remove:
            del self._active_workflows[workflow_id]
    
    async def _shutdown(self) -> None:
        """Shutdown coordinator and specialized agents."""
        self.logger.info("Shutting down CoordinatorAgent...")
        
        # Shutdown specialized agents
        if self.researcher:
            await self.researcher.shutdown()
        
        if self.reporter:
            await self.reporter.shutdown()
        
        # Clear workflows
        self._active_workflows.clear()
        
        self.logger.info("CoordinatorAgent shutdown complete")
    
    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active workflows.
        
        Returns:
            Dict[str, Dict[str, Any]]: Active workflow information.
        """
        return self._active_workflows.copy()
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow.
        
        Args:
            workflow_id: ID of the workflow to check.
            
        Returns:
            Optional[Dict[str, Any]]: Workflow status or None if not found.
        """
        return self._active_workflows.get(workflow_id)