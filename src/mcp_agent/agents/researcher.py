"""Researcher agent for tool discovery and information gathering.

The ResearcherAgent specializes in tool discovery, web search, and data gathering
using vector databases, MCP servers, and external search APIs.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic researcher usage:

    >>> researcher = ResearcherAgent(settings, mcp_client, vector_store)
    >>> await researcher.initialize()
    >>> task = AgentTask(type="vector_search", parameters={"query": "BLAST"})
    >>> result = await researcher.execute_task(task)
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from mcp_agent.agents.base import BaseAgent, AgentCapability, AgentState
    from mcp_agent.config.settings import AgentSettings
    from mcp_agent.models.schemas import (
        AgentTask,
        AgentResponse,
        ToolInfo,
        SearchResult,
        ResearchData,
    )
    from mcp_agent.tools.mcp_client import MCPClient
    from mcp_agent.tools.vector_store import VectorStore
    from mcp_agent.utils.logger import get_logger
except ImportError:
    # Mock imports for development
    class BaseAgent:
        pass
    class AgentCapability:
        WEB_SEARCH = "web_search"
        TOOL_DISCOVERY = "tool_discovery"
        MCP_INTEGRATION = "mcp_integration"
        VECTOR_SEARCH = "vector_search"
        DATA_ANALYSIS = "data_analysis"
        BIOINFORMATICS_ANALYSIS = "bioinformatics_analysis"
    class AgentState:
        pass
    class AgentSettings:
        pass
    class AgentTask:
        pass
    class AgentResponse:
        pass
    class ToolInfo:
        pass
    class SearchResult:
        pass
    class ResearchData:
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


class WebSearchProvider:
    """Web search provider interface for external search APIs."""
    
    def __init__(self, api_key: str, provider: str) -> None:
        """Initialize web search provider.
        
        Args:
            api_key: API key for the search provider.
            provider: Provider name (tavily, brave, serper).
        """
        self.api_key = api_key
        self.provider = provider
        self.client = httpx.AsyncClient()
        self.logger = get_logger(f"WebSearch.{provider}")
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "general",
    ) -> List[Dict[str, Any]]:
        """Perform web search.
        
        Args:
            query: Search query.
            max_results: Maximum number of results.
            search_type: Type of search (general, academic, news).
            
        Returns:
            List[Dict[str, Any]]: Search results.
        """
        if self.provider == "tavily":
            return await self._search_tavily(query, max_results, search_type)
        elif self.provider == "brave":
            return await self._search_brave(query, max_results)
        elif self.provider == "serper":
            return await self._search_serper(query, max_results)
        else:
            raise ValueError(f"Unknown search provider: {self.provider}")
    
    async def _search_tavily(
        self,
        query: str,
        max_results: int,
        search_type: str,
    ) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        try:
            url = "https://api.tavily.com/search"
            
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced" if search_type == "academic" else "basic",
                "include_answer": True,
                "include_images": False,
                "include_raw_content": False,
            }
            
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0),
                    "published_date": item.get("published_date"),
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Tavily search failed: {e}")
            return []
    
    async def _search_brave(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Brave Search API."""
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key,
            }
            
            params = {
                "q": query,
                "count": min(max_results, 20),
                "offset": 0,
                "mkt": "en-US",
                "safesearch": "moderate",
            }
            
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("description", ""),
                    "score": 1.0,  # Brave doesn't provide scores
                    "published_date": item.get("age"),
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Brave search failed: {e}")
            return []
    
    async def _search_serper(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Serper API."""
        try:
            url = "https://google.serper.dev/search"
            
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            }
            
            payload = {
                "q": query,
                "num": min(max_results, 100),
                "hl": "en",
                "gl": "us",
            }
            
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("organic", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "content": item.get("snippet", ""),
                    "score": item.get("position", 100) / 100,  # Convert position to score
                    "published_date": item.get("date"),
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Serper search failed: {e}")
            return []
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


class ResearcherAgent(BaseAgent):
    """Specialized agent for research, tool discovery, and information gathering.
    
    The ResearcherAgent handles tasks related to:
    - Vector database searches for bioinformatics tools
    - Web searches using external APIs
    - MCP server integration for tool access
    - Data analysis and synthesis
    - Bioinformatics-specific research
    
    Attributes:
        settings: Configuration settings for the agent.
        mcp_client: Client for MCP server communication.
        vector_store: Vector database for semantic search.
        llm: Language model for research analysis.
        web_search_providers: List of web search providers.
        research_cache: Cache for research results.
        
    Example:
        >>> researcher = ResearcherAgent(
        ...     settings=settings,
        ...     mcp_client=mcp_client,
        ...     vector_store=vector_store
        ... )
        >>> await researcher.initialize()
        >>> result = await researcher.execute_task(search_task)
    """
    
    def __init__(
        self,
        settings: AgentSettings,
        mcp_client: Optional[MCPClient] = None,
        vector_store: Optional[VectorStore] = None,
        name: str = "researcher",
    ) -> None:
        """Initialize the researcher agent.
        
        Args:
            settings: Configuration settings.
            mcp_client: Optional MCP client instance.
            vector_store: Optional vector store instance.
            name: Agent name (default: "researcher").
        """
        super().__init__(
            name=name,
            settings=settings,
            capabilities=[
                AgentCapability.WEB_SEARCH,
                AgentCapability.TOOL_DISCOVERY,
                AgentCapability.MCP_INTEGRATION,
                AgentCapability.VECTOR_SEARCH,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.BIOINFORMATICS_ANALYSIS,
                AgentCapability.SEMANTIC_SEARCH,
                AgentCapability.API_INTEGRATION,
            ]
        )
        
        # Core dependencies
        self.mcp_client = mcp_client
        self.vector_store = vector_store
        
        # LLM for research analysis
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        
        # Web search providers
        self.web_search_providers: List[WebSearchProvider] = []
        
        # Research cache and state
        self.research_cache: Dict[str, Any] = {}
        self._max_cache_size = 100
        self._concurrent_searches = getattr(settings, 'max_concurrent_requests', 5)
        
        self.logger.info(f"ResearcherAgent initialized with {len(self.capabilities)} capabilities")
    
    async def _initialize(self) -> None:
        """Initialize the researcher agent and its components."""
        self.logger.info("Initializing ResearcherAgent...")
        
        try:
            # Initialize LLM
            self._initialize_llm()
            
            # Initialize web search providers
            await self._initialize_web_search()
            
            # Validate dependencies
            await self._validate_dependencies()
            
            self.logger.info("ResearcherAgent initialization complete")
            
        except Exception as e:
            self.logger.error(f"ResearcherAgent initialization failed: {e}")
            raise
    
    def _initialize_llm(self) -> None:
        """Initialize the language model for research analysis."""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.settings.llm.model,
                temperature=self.settings.llm.temperature,
                max_tokens=self.settings.llm.max_tokens,
                google_api_key=self.settings.google_api_key,
            )
            self.logger.info(f"Research LLM initialized: {self.settings.llm.model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize research LLM: {e}")
            raise
    
    async def _initialize_web_search(self) -> None:
        """Initialize web search providers based on available API keys."""
        try:
            # Initialize Tavily if API key available
            if hasattr(self.settings, 'tavily_api_key') and self.settings.tavily_api_key:
                provider = WebSearchProvider(self.settings.tavily_api_key, "tavily")
                self.web_search_providers.append(provider)
                self.logger.info("Tavily search provider initialized")
            
            # Initialize Brave Search if API key available
            if hasattr(self.settings, 'brave_search_api_key') and self.settings.brave_search_api_key:
                provider = WebSearchProvider(self.settings.brave_search_api_key, "brave")
                self.web_search_providers.append(provider)
                self.logger.info("Brave search provider initialized")
            
            # Initialize Serper if API key available
            if hasattr(self.settings, 'serper_api_key') and self.settings.serper_api_key:
                provider = WebSearchProvider(self.settings.serper_api_key, "serper")
                self.web_search_providers.append(provider)
                self.logger.info("Serper search provider initialized")
            
            if not self.web_search_providers:
                self.logger.warning("No web search providers available")
            else:
                self.logger.info(f"Initialized {len(self.web_search_providers)} web search providers")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize web search providers: {e}")
    
    async def _validate_dependencies(self) -> None:
        """Validate that required dependencies are available."""
        issues = []
        
        if not self.llm:
            issues.append("LLM not initialized")
        
        if not self.vector_store:
            issues.append("Vector store not available")
        
        if not self.web_search_providers:
            issues.append("No web search providers available")
        
        if issues:
            self.logger.warning(f"Dependency issues: {', '.join(issues)}")
    
    async def _execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a researcher task.
        
        Args:
            task: Task to execute.
            
        Returns:
            AgentResponse: Task execution result.
        """
        self.logger.info(f"Executing researcher task: {task.type}")
        
        try:
            # Route task based on type
            if task.type == "vector_search":
                return await self._handle_vector_search(task)
            elif task.type == "web_search":
                return await self._handle_web_search(task)
            elif task.type == "web_research":
                return await self._handle_web_research(task)
            elif task.type == "mcp_search":
                return await self._handle_mcp_search(task)
            elif task.type == "tool_discovery":
                return await self._handle_tool_discovery(task)
            elif task.type == "bioinformatics_research":
                return await self._handle_bioinformatics_research(task)
            elif task.type == "analyze_tools":
                return await self._handle_tool_analysis(task)
            else:
                raise ValueError(f"Unknown task type: {task.type}")
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return AgentResponse(
                success=False,
                content=f"Research task failed: {e}",
                error=str(e),
                agent_id=self.agent_id,
            )
    
    async def _handle_vector_search(self, task: AgentTask) -> AgentResponse:
        """Handle vector database search tasks."""
        query = task.parameters.get("query")
        if not query:
            raise ValueError("Vector search requires 'query' parameter")
        
        if not self.vector_store:
            raise RuntimeError("Vector store not available")
        
        max_results = task.parameters.get("max_results", 10)
        filters = task.parameters.get("filters", {})
        include_documentation = task.parameters.get("include_documentation", True)
        
        # Perform vector search
        results = await self.vector_store.search(
            query=query,
            limit=max_results,
            filters=filters,
        )
        
        # Convert results to ToolInfo objects
        tools = []
        for result in results:
            tool_info = ToolInfo(**result)
            if include_documentation and hasattr(tool_info, 'documentation'):
                # Enhance with documentation if available
                pass
            tools.append(tool_info)
        
        return AgentResponse(
            success=True,
            content=f"Found {len(tools)} tools matching '{query}'",
            data={
                "tools": [tool.model_dump() for tool in tools],
                "query": query,
                "total_results": len(tools),
                "search_type": "vector",
            },
            agent_id=self.agent_id,
        )
    
    async def _handle_web_search(self, task: AgentTask) -> AgentResponse:
        """Handle web search tasks."""
        query = task.parameters.get("query")
        if not query:
            raise ValueError("Web search requires 'query' parameter")
        
        if not self.web_search_providers:
            raise RuntimeError("No web search providers available")
        
        max_results = task.parameters.get("max_results", 10)
        search_type = task.parameters.get("search_type", "general")
        
        # Try each provider until we get results
        all_results = []
        for provider in self.web_search_providers:
            try:
                results = await provider.search(query, max_results, search_type)
                all_results.extend(results)
                if results:
                    break  # Use first successful provider
                    
            except Exception as e:
                self.logger.warning(f"Provider {provider.provider} failed: {e}")
                continue
        
        if not all_results:
            return AgentResponse(
                success=False,
                content="No web search results found",
                error="All search providers failed",
                agent_id=self.agent_id,
            )
        
        # Limit results and sort by relevance
        sorted_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)
        limited_results = sorted_results[:max_results]
        
        return AgentResponse(
            success=True,
            content=f"Found {len(limited_results)} web search results for '{query}'",
            data={
                "results": limited_results,
                "query": query,
                "total_results": len(limited_results),
                "search_type": "web",
                "providers_used": [p.provider for p in self.web_search_providers],
            },
            agent_id=self.agent_id,
        )
    
    async def _handle_web_research(self, task: AgentTask) -> AgentResponse:
        """Handle comprehensive web research tasks."""
        topic = task.parameters.get("topic")
        if not topic:
            raise ValueError("Web research requires 'topic' parameter")
        
        depth = task.parameters.get("depth", "standard")
        tools_found = task.parameters.get("tools_found", [])
        
        # Perform multiple searches with different queries
        search_queries = self._generate_research_queries(topic, depth)
        
        all_results = []
        for query in search_queries:
            web_task = AgentTask(
                type="web_search",
                parameters={
                    "query": query,
                    "max_results": 5 if depth == "quick" else 10,
                    "search_type": "academic" if "research" in topic.lower() else "general",
                }
            )
            
            response = await self._handle_web_search(web_task)
            if response.success and response.data:
                all_results.extend(response.data.get("results", []))
        
        # Analyze and synthesize results using LLM
        synthesis = await self._synthesize_research_results(topic, all_results, tools_found)
        
        return AgentResponse(
            success=True,
            content=f"Completed web research on '{topic}'",
            data={
                "topic": topic,
                "synthesis": synthesis,
                "raw_results": all_results,
                "search_queries": search_queries,
                "tools_context": tools_found,
            },
            agent_id=self.agent_id,
        )
    
    async def _handle_mcp_search(self, task: AgentTask) -> AgentResponse:
        """Handle MCP server search tasks."""
        if not self.mcp_client:
            raise RuntimeError("MCP client not available")
        
        query = task.parameters.get("query")
        server_name = task.parameters.get("server_name")
        tool_name = task.parameters.get("tool_name", "search")
        
        try:
            # Execute MCP tool
            result = await self.mcp_client.execute_tool(
                server_name=server_name,
                tool_name=tool_name,
                parameters={"query": query} if query else task.parameters,
            )
            
            return AgentResponse(
                success=True,
                content=f"MCP search completed on server '{server_name}'",
                data={
                    "result": result,
                    "server": server_name,
                    "tool": tool_name,
                    "query": query,
                },
                agent_id=self.agent_id,
            )
            
        except Exception as e:
            self.logger.error(f"MCP search failed: {e}")
            return AgentResponse(
                success=False,
                content=f"MCP search failed: {e}",
                error=str(e),
                agent_id=self.agent_id,
            )
    
    async def _handle_tool_discovery(self, task: AgentTask) -> AgentResponse:
        """Handle comprehensive tool discovery tasks."""
        query = task.parameters.get("query")
        if not query:
            raise ValueError("Tool discovery requires 'query' parameter")
        
        category = task.parameters.get("category")
        organism = task.parameters.get("organism")
        include_web_search = task.parameters.get("include_web_search", True)
        
        # Combine vector search and web search for comprehensive discovery
        results = {
            "vector_tools": [],
            "web_results": [],
            "mcp_tools": [],
        }
        
        # Vector search
        if self.vector_store:
            vector_task = AgentTask(
                type="vector_search",
                parameters={
                    "query": query,
                    "max_results": 20,
                    "filters": {
                        "category": category,
                        "organism": organism,
                    } if category or organism else {},
                }
            )
            vector_response = await self._handle_vector_search(vector_task)
            if vector_response.success:
                results["vector_tools"] = vector_response.data.get("tools", [])
        
        # Web search for additional tools
        if include_web_search and self.web_search_providers:
            enhanced_query = f"{query} bioinformatics tools software"
            if category:
                enhanced_query += f" {category}"
            if organism:
                enhanced_query += f" {organism}"
            
            web_task = AgentTask(
                type="web_search",
                parameters={
                    "query": enhanced_query,
                    "max_results": 15,
                    "search_type": "academic",
                }
            )
            web_response = await self._handle_web_search(web_task)
            if web_response.success:
                results["web_results"] = web_response.data.get("results", [])
        
        # MCP search if available
        if self.mcp_client:
            try:
                mcp_servers = await self.mcp_client.list_servers()
                for server_name in mcp_servers:
                    mcp_task = AgentTask(
                        type="mcp_search",
                        parameters={
                            "query": query,
                            "server_name": server_name,
                        }
                    )
                    mcp_response = await self._handle_mcp_search(mcp_task)
                    if mcp_response.success:
                        results["mcp_tools"].append(mcp_response.data)
            except Exception as e:
                self.logger.warning(f"MCP search failed: {e}")
        
        # Synthesize and rank results
        synthesized = await self._synthesize_tool_discovery(query, results)
        
        return AgentResponse(
            success=True,
            content=f"Tool discovery completed for '{query}'",
            data={
                "query": query,
                "synthesized_results": synthesized,
                "raw_results": results,
                "total_found": (
                    len(results["vector_tools"]) +
                    len(results["web_results"]) +
                    len(results["mcp_tools"])
                ),
            },
            agent_id=self.agent_id,
        )
    
    async def _handle_bioinformatics_research(self, task: AgentTask) -> AgentResponse:
        """Handle bioinformatics-specific research tasks."""
        topic = task.parameters.get("topic")
        if not topic:
            raise ValueError("Bioinformatics research requires 'topic' parameter")
        
        research_type = task.parameters.get("research_type", "general")
        organism = task.parameters.get("organism")
        data_type = task.parameters.get("data_type")  # genomics, proteomics, etc.
        
        # Enhance query with bioinformatics context
        enhanced_queries = [
            f"{topic} bioinformatics",
            f"{topic} computational biology",
        ]
        
        if organism:
            enhanced_queries.append(f"{topic} {organism}")
        
        if data_type:
            enhanced_queries.append(f"{topic} {data_type}")
        
        # Perform comprehensive research
        all_results = []
        for query in enhanced_queries:
            # Tool discovery
            tool_task = AgentTask(
                type="tool_discovery",
                parameters={
                    "query": query,
                    "category": data_type,
                    "organism": organism,
                }
            )
            tool_response = await self._handle_tool_discovery(tool_task)
            if tool_response.success:
                all_results.append({
                    "type": "tools",
                    "query": query,
                    "data": tool_response.data,
                })
            
            # Web research
            web_task = AgentTask(
                type="web_research",
                parameters={
                    "topic": query,
                    "depth": "standard",
                }
            )
            web_response = await self._handle_web_research(web_task)
            if web_response.success:
                all_results.append({
                    "type": "web_research",
                    "query": query,
                    "data": web_response.data,
                })
        
        # Synthesize bioinformatics-specific analysis
        analysis = await self._analyze_bioinformatics_research(topic, all_results, {
            "research_type": research_type,
            "organism": organism,
            "data_type": data_type,
        })
        
        return AgentResponse(
            success=True,
            content=f"Bioinformatics research completed for '{topic}'",
            data={
                "topic": topic,
                "analysis": analysis,
                "research_context": {
                    "research_type": research_type,
                    "organism": organism,
                    "data_type": data_type,
                },
                "raw_results": all_results,
            },
            agent_id=self.agent_id,
        )
    
    async def _handle_tool_analysis(self, task: AgentTask) -> AgentResponse:
        """Handle tool analysis and comparison tasks."""
        tools = task.parameters.get("tools", [])
        if not tools:
            raise ValueError("Tool analysis requires 'tools' parameter")
        
        analysis_type = task.parameters.get("analysis_type", "comparison")
        criteria = task.parameters.get("criteria", [
            "functionality", "ease_of_use", "performance", "documentation"
        ])
        
        # Analyze tools using LLM
        analysis = await self._analyze_tools_with_llm(tools, analysis_type, criteria)
        
        return AgentResponse(
            success=True,
            content=f"Tool analysis completed for {len(tools)} tools",
            data={
                "analysis": analysis,
                "tools_analyzed": len(tools),
                "analysis_type": analysis_type,
                "criteria": criteria,
            },
            agent_id=self.agent_id,
        )
    
    def _generate_research_queries(self, topic: str, depth: str) -> List[str]:
        """Generate multiple search queries for comprehensive research."""
        base_queries = [topic]
        
        if depth in ["standard", "comprehensive"]:
            base_queries.extend([
                f"{topic} tools",
                f"{topic} software",
                f"{topic} methods",
                f"{topic} algorithms",
            ])
        
        if depth == "comprehensive":
            base_queries.extend([
                f"{topic} research papers",
                f"{topic} benchmarks",
                f"{topic} comparison",
                f"{topic} best practices",
                f"latest {topic} developments",
            ])
        
        return base_queries
    
    async def _synthesize_research_results(
        self,
        topic: str,
        results: List[Dict[str, Any]],
        tools_context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Synthesize research results using LLM."""
        if not self.llm:
            return {"synthesis": "LLM not available for synthesis"}
        
        try:
            # Prepare synthesis prompt
            synthesis_prompt = f"""
            Analyze and synthesize the following research results for the topic: {topic}
            
            Web Search Results:
            {json.dumps(results[:10], indent=2)}  # Limit for token efficiency
            
            Tool Context:
            {json.dumps(tools_context[:5], indent=2)}
            
            Please provide:
            1. Key findings and insights
            2. Important tools and resources identified
            3. Trends and patterns observed
            4. Recommendations for further investigation
            5. Summary of the current state of the field
            
            Format your response as structured JSON with these sections.
            """
            
            messages = [
                SystemMessage(content="You are a research analyst specializing in bioinformatics and computational biology."),
                HumanMessage(content=synthesis_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Try to parse as JSON, fallback to text
            try:
                synthesis = json.loads(response.content)
            except json.JSONDecodeError:
                synthesis = {"analysis": response.content}
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Research synthesis failed: {e}")
            return {"synthesis_error": str(e)}
    
    async def _synthesize_tool_discovery(
        self,
        query: str,
        results: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """Synthesize tool discovery results."""
        # Combine and deduplicate tools
        all_tools = []
        
        # Add vector tools
        for tool in results.get("vector_tools", []):
            all_tools.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "source": "vector_database",
                "relevance_score": tool.get("score", 0.5),
            })
        
        # Add web results (extract tool mentions)
        for result in results.get("web_results", []):
            if any(keyword in result.get("title", "").lower() for keyword in ["tool", "software", "package"]):
                all_tools.append({
                    "name": result.get("title", ""),
                    "description": result.get("content", ""),
                    "source": "web_search",
                    "url": result.get("url", ""),
                    "relevance_score": result.get("score", 0.3),
                })
        
        # Sort by relevance
        all_tools.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Deduplicate by name similarity
        unique_tools = []
        seen_names = set()
        
        for tool in all_tools:
            name_lower = tool.get("name", "").lower()
            if not any(name_lower in seen or seen in name_lower for seen in seen_names):
                unique_tools.append(tool)
                seen_names.add(name_lower)
        
        return {
            "total_found": len(all_tools),
            "unique_tools": len(unique_tools),
            "tools": unique_tools[:20],  # Top 20 tools
            "sources": {
                "vector_database": len(results.get("vector_tools", [])),
                "web_search": len(results.get("web_results", [])),
                "mcp_servers": len(results.get("mcp_tools", [])),
            },
        }
    
    async def _analyze_bioinformatics_research(
        self,
        topic: str,
        results: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze bioinformatics research results."""
        if not self.llm:
            return {"analysis": "LLM not available for analysis"}
        
        try:
            analysis_prompt = f"""
            Perform a comprehensive bioinformatics analysis for: {topic}
            
            Research Context:
            - Research Type: {context.get('research_type', 'general')}
            - Organism: {context.get('organism', 'not specified')}
            - Data Type: {context.get('data_type', 'not specified')}
            
            Research Results:
            {json.dumps(results[:5], indent=2)}  # Limit for tokens
            
            Please provide a bioinformatics-focused analysis including:
            1. Relevant computational methods and algorithms
            2. Key software tools and their applications
            3. Data analysis workflows and pipelines
            4. Statistical considerations and best practices
            5. Current challenges and future directions
            6. Specific recommendations for the given organism/data type
            
            Format as structured JSON with clear sections.
            """
            
            messages = [
                SystemMessage(content="You are a bioinformatics expert with deep knowledge of computational biology methods and tools."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                analysis = json.loads(response.content)
            except json.JSONDecodeError:
                analysis = {"analysis": response.content}
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Bioinformatics analysis failed: {e}")
            return {"analysis_error": str(e)}
    
    async def _analyze_tools_with_llm(
        self,
        tools: List[Dict[str, Any]],
        analysis_type: str,
        criteria: List[str],
    ) -> Dict[str, Any]:
        """Analyze tools using LLM."""
        if not self.llm:
            return {"analysis": "LLM not available for tool analysis"}
        
        try:
            analysis_prompt = f"""
            Perform a {analysis_type} analysis of the following bioinformatics tools:
            
            Tools:
            {json.dumps(tools, indent=2)}
            
            Analysis Criteria:
            {', '.join(criteria)}
            
            Please provide:
            1. Detailed comparison matrix
            2. Strengths and weaknesses of each tool
            3. Use case recommendations
            4. Performance considerations
            5. Learning curve assessment
            6. Overall ranking and recommendations
            
            Format as structured JSON.
            """
            
            messages = [
                SystemMessage(content="You are a bioinformatics tool expert who helps researchers choose the best tools for their needs."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                analysis = json.loads(response.content)
            except json.JSONDecodeError:
                analysis = {"analysis": response.content}
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Tool analysis failed: {e}")
            return {"analysis_error": str(e)}
    
    def _clean_cache(self) -> None:
        """Clean old entries from research cache."""
        if len(self.research_cache) > self._max_cache_size:
            # Remove oldest entries
            sorted_items = sorted(
                self.research_cache.items(),
                key=lambda x: x[1].get("timestamp", 0)
            )
            
            to_remove = len(sorted_items) - self._max_cache_size + 10
            for i in range(to_remove):
                del self.research_cache[sorted_items[i][0]]
    
    async def _shutdown(self) -> None:
        """Shutdown researcher agent and clean up resources."""
        self.logger.info("Shutting down ResearcherAgent...")
        
        # Close web search providers
        for provider in self.web_search_providers:
            await provider.close()
        
        # Clear cache
        self.research_cache.clear()
        
        self.logger.info("ResearcherAgent shutdown complete")
    
    def get_search_providers(self) -> List[str]:
        """Get list of available search providers.
        
        Returns:
            List[str]: List of search provider names.
        """
        return [provider.provider for provider in self.web_search_providers]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get research cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics.
        """
        return {
            "cache_size": len(self.research_cache),
            "max_cache_size": self._max_cache_size,
            "cache_keys": list(self.research_cache.keys()),
        }