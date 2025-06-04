"""MCP (Model Context Protocol) client implementation.

This module provides a comprehensive client for connecting to and communicating
with MCP servers using both stdio and HTTP transports.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic MCP client usage:

    >>> mcp_client = MCPClient(settings)
    >>> await mcp_client.initialize()
    >>> tools = await mcp_client.get_available_tools()
    >>> result = await mcp_client.execute_tool("search", "bioinformatics")
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    
    from mcp_agent.config.settings import AgentSettings, MCPServerConfig
    from mcp_agent.utils.logger import get_logger
except ImportError as e:
    # Mock imports for development
    import warnings
    warnings.warn(f"MCP dependencies not available: {e}", ImportWarning)
    
    class MultiServerMCPClient:
        pass
    class ClientSession:
        pass
    class StdioServerParameters:
        pass
    def stdio_client(*args, **kwargs):
        pass
    class AgentSettings:
        pass
    class MCPServerConfig:
        pass
    def get_logger(name: str):
        import logging
        return logging.getLogger(name)


class MCPToolInfo(BaseModel):
    """Information about an MCP tool.
    
    Attributes:
        name: Tool name.
        description: Tool description.
        input_schema: JSON schema for tool inputs.
        server: Server providing this tool.
        
    Example:
        >>> tool_info = MCPToolInfo(
        ...     name="search",
        ...     description="Search for bioinformatics tools",
        ...     input_schema={"type": "object", "properties": {...}},
        ...     server="bio_tools"
        ... )
    """
    
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    input_schema: Dict[str, Any] = Field(description="JSON schema for inputs")
    server: str = Field(description="Server providing this tool")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()


class MCPServerConnection(BaseModel):
    """Represents a connection to an MCP server.
    
    Attributes:
        name: Server name.
        config: Server configuration.
        status: Connection status.
        tools: Available tools from this server.
        last_ping: Last successful ping timestamp.
        error_count: Number of consecutive errors.
        
    Example:
        >>> connection = MCPServerConnection(
        ...     name="bio_tools",
        ...     config=MCPServerConfig(...),
        ...     status="connected"
        ... )
    """
    
    name: str = Field(description="Server name")
    config: MCPServerConfig = Field(description="Server configuration")
    status: str = Field(default="disconnected", description="Connection status")
    tools: List[MCPToolInfo] = Field(default_factory=list, description="Available tools")
    last_ping: Optional[datetime] = Field(default=None, description="Last ping timestamp")
    error_count: int = Field(default=0, description="Consecutive error count")
    session: Optional[Any] = Field(default=None, exclude=True, description="Active session")
    process: Optional[subprocess.Popen] = Field(default=None, exclude=True, description="Server process")
    
    class Config:
        arbitrary_types_allowed = True
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return (
            self.status == "connected" and
            self.error_count < 5 and
            (self.last_ping is None or 
             (datetime.now() - self.last_ping).total_seconds() < 300)
        )
    
    def get_tool_names(self) -> List[str]:
        """Get list of tool names from this server."""
        return [tool.name for tool in self.tools]


class MCPClient:
    """Client for connecting to and managing MCP servers.
    
    This client handles multiple MCP server connections, tool discovery,
    and execution across different transport protocols.
    
    Attributes:
        settings: Configuration settings.
        connections: Dictionary of server connections.
        multi_client: LangChain MCP adapter client.
        session_timeout: Session timeout in seconds.
        max_retries: Maximum retry attempts.
        
    Example:
        >>> client = MCPClient(settings)
        >>> await client.initialize()
        >>> servers = await client.list_servers()
        >>> tools = await client.get_available_tools()
        >>> result = await client.execute_tool("search", "RNA-seq", server="bio_tools")
    """
    
    def __init__(self, settings: AgentSettings) -> None:
        """Initialize the MCP client.
        
        Args:
            settings: Configuration settings containing MCP server configs.
        """
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)
        
        # Connection management
        self.connections: Dict[str, MCPServerConnection] = {}
        self.multi_client: Optional[MultiServerMCPClient] = None
        
        # Configuration
        self.session_timeout = getattr(settings, 'mcp_timeout', 30)
        self.max_retries = getattr(settings, 'mcp_max_retries', 3)
        self.ping_interval = 60  # seconds
        
        # Metrics and monitoring
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.last_activity = None
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        self.logger.info(f"MCPClient initialized with timeout={self.session_timeout}s")
    
    async def initialize(self) -> None:
        """Initialize the MCP client and connect to configured servers.
        
        Raises:
            RuntimeError: If initialization fails.
            
        Example:
            >>> await client.initialize()
        """
        self.logger.info("Initializing MCP client...")
        
        try:
            # Initialize multi-client for LangChain integration
            await self._initialize_multi_client()
            
            # Connect to configured servers
            await self._connect_to_servers()
            
            # Start health check task
            self._start_health_check_task()
            
            self.logger.info(f"MCP client initialized with {len(self.connections)} servers")
            
        except Exception as e:
            self.logger.error(f"MCP client initialization failed: {e}")
            raise RuntimeError(f"MCP client initialization failed: {e}") from e
    
    async def _initialize_multi_client(self) -> None:
        """Initialize the multi-server MCP client for LangChain integration."""
        try:
            # Build server configurations for LangChain MCP adapter
            server_configs = {}
            
            for name, config in self.settings.mcp_servers.items():
                if config.transport == "stdio":
                    server_configs[name] = {
                        "command": config.command,
                        "args": config.args,
                        "transport": "stdio",
                    }
                elif config.transport == "streamable_http":
                    server_configs[name] = {
                        "url": config.url,
                        "transport": "streamable_http",
                    }
            
            if server_configs:
                self.multi_client = MultiServerMCPClient(server_configs)
                self.logger.info(f"Multi-client initialized with {len(server_configs)} server configs")
            else:
                self.logger.warning("No MCP servers configured")
                
        except Exception as e:
            self.logger.error(f"Multi-client initialization failed: {e}")
            # Continue without multi-client - we can still manage connections directly
    
    async def _connect_to_servers(self) -> None:
        """Connect to all configured MCP servers."""
        connection_tasks = []
        
        for name, config in self.settings.mcp_servers.items():
            connection = MCPServerConnection(name=name, config=config)
            self.connections[name] = connection
            
            # Create connection task
            task = asyncio.create_task(
                self._connect_to_server(connection),
                name=f"connect_{name}"
            )
            connection_tasks.append(task)
        
        # Wait for all connections with timeout
        if connection_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*connection_tasks, return_exceptions=True),
                    timeout=self.session_timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning("Some MCP server connections timed out")
    
    async def _connect_to_server(self, connection: MCPServerConnection) -> None:
        """Connect to a single MCP server.
        
        Args:
            connection: Server connection to establish.
        """
        try:
            self.logger.info(f"Connecting to MCP server: {connection.name}")
            
            if connection.config.transport == "stdio":
                await self._connect_stdio_server(connection)
            elif connection.config.transport == "streamable_http":
                await self._connect_http_server(connection)
            else:
                raise ValueError(f"Unsupported transport: {connection.config.transport}")
            
            # Discover tools
            await self._discover_server_tools(connection)
            
            connection.status = "connected"
            connection.last_ping = datetime.now()
            connection.error_count = 0
            
            self.logger.info(
                f"Connected to {connection.name} "
                f"({len(connection.tools)} tools available)"
            )
            
        except Exception as e:
            connection.status = "error"
            connection.error_count += 1
            self.logger.error(f"Failed to connect to {connection.name}: {e}")
    
    async def _connect_stdio_server(self, connection: MCPServerConnection) -> None:
        """Connect to a stdio-based MCP server."""
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=connection.config.command,
                args=connection.config.args,
            )
            
            # Start the server process and create session
            # Note: This is a simplified implementation
            # In practice, you'd use the actual MCP client library
            self.logger.info(f"Starting stdio server: {connection.config.command}")
            
            # For now, we'll create a mock session
            # In a real implementation, you'd use:
            # async with stdio_client(server_params) as (read, write):
            #     async with ClientSession(read, write) as session:
            #         connection.session = session
            #         await session.initialize()
            
            connection.session = "mock_stdio_session"
            
        except Exception as e:
            self.logger.error(f"Stdio connection failed for {connection.name}: {e}")
            raise
    
    async def _connect_http_server(self, connection: MCPServerConnection) -> None:
        """Connect to an HTTP-based MCP server."""
        try:
            # Create HTTP client
            client = httpx.AsyncClient(timeout=self.session_timeout)
            
            # Test connection
            response = await client.get(f"{connection.config.url}/health")
            response.raise_for_status()
            
            connection.session = client
            self.logger.info(f"HTTP connection established to {connection.name}")
            
        except Exception as e:
            self.logger.error(f"HTTP connection failed for {connection.name}: {e}")
            raise
    
    async def _discover_server_tools(self, connection: MCPServerConnection) -> None:
        """Discover available tools from a server.
        
        Args:
            connection: Server connection to discover tools for.
        """
        try:
            # Mock tool discovery - in a real implementation, you'd call the MCP protocol
            if connection.name == "bioinformatics":
                mock_tools = [
                    MCPToolInfo(
                        name="search_tools",
                        description="Search for bioinformatics tools",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "category": {"type": "string", "description": "Tool category"},
                            },
                            "required": ["query"]
                        },
                        server=connection.name
                    ),
                    MCPToolInfo(
                        name="get_tool_info",
                        description="Get detailed information about a tool",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "tool_name": {"type": "string", "description": "Name of the tool"},
                            },
                            "required": ["tool_name"]
                        },
                        server=connection.name
                    ),
                ]
                connection.tools = mock_tools
            elif connection.name == "web_search":
                mock_tools = [
                    MCPToolInfo(
                        name="web_search",
                        description="Search the web for information",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "max_results": {"type": "integer", "description": "Maximum results"},
                            },
                            "required": ["query"]
                        },
                        server=connection.name
                    ),
                ]
                connection.tools = mock_tools
            
            self.logger.info(f"Discovered {len(connection.tools)} tools from {connection.name}")
            
        except Exception as e:
            self.logger.error(f"Tool discovery failed for {connection.name}: {e}")
            connection.tools = []
    
    def _start_health_check_task(self) -> None:
        """Start background health check task."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(
                self._health_check_loop(),
                name="mcp_health_check"
            )
            self.logger.info("Started MCP health check task")
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.ping_interval)
                
                if self._shutdown:
                    break
                
                # Ping all connections
                for connection in self.connections.values():
                    if connection.status == "connected":
                        await self._ping_server(connection)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _ping_server(self, connection: MCPServerConnection) -> None:
        """Ping a server to check its health.
        
        Args:
            connection: Server connection to ping.
        """
        try:
            # Implement server-specific ping
            if connection.config.transport == "stdio":
                # For stdio, we can check if the process is still running
                # In a real implementation, you'd send a ping message
                connection.last_ping = datetime.now()
                
            elif connection.config.transport == "streamable_http":
                # For HTTP, send a health check request
                if isinstance(connection.session, httpx.AsyncClient):
                    response = await connection.session.get(
                        f"{connection.config.url}/health",
                        timeout=5
                    )
                    response.raise_for_status()
                    connection.last_ping = datetime.now()
                    connection.error_count = 0
            
        except Exception as e:
            connection.error_count += 1
            self.logger.warning(f"Ping failed for {connection.name}: {e}")
            
            if connection.error_count >= 5:
                connection.status = "error"
                self.logger.error(f"Server {connection.name} marked as unhealthy")
    
    async def list_servers(self) -> List[str]:
        """List all configured MCP servers.
        
        Returns:
            List[str]: List of server names.
            
        Example:
            >>> servers = await client.list_servers()
            >>> print("Available servers:", servers)
        """
        return list(self.connections.keys())
    
    async def get_server_status(self, server_name: str) -> Dict[str, Any]:
        """Get status information for a specific server.
        
        Args:
            server_name: Name of the server to check.
            
        Returns:
            Dict[str, Any]: Server status information.
            
        Raises:
            ValueError: If server is not found.
            
        Example:
            >>> status = await client.get_server_status("bio_tools")
            >>> print(f"Status: {status['status']}")
        """
        if server_name not in self.connections:
            raise ValueError(f"Server not found: {server_name}")
        
        connection = self.connections[server_name]
        
        return {
            "name": connection.name,
            "status": connection.status,
            "transport": connection.config.transport,
            "tools_count": len(connection.tools),
            "last_ping": connection.last_ping.isoformat() if connection.last_ping else None,
            "error_count": connection.error_count,
            "healthy": connection.is_healthy(),
            "config": connection.config.model_dump(),
        }
    
    async def get_available_tools(self, server_name: Optional[str] = None) -> List[MCPToolInfo]:
        """Get available tools from MCP servers.
        
        Args:
            server_name: Optional server name to filter tools.
            
        Returns:
            List[MCPToolInfo]: List of available tools.
            
        Example:
            >>> tools = await client.get_available_tools()
            >>> for tool in tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        all_tools = []
        
        if server_name:
            if server_name not in self.connections:
                raise ValueError(f"Server not found: {server_name}")
            all_tools.extend(self.connections[server_name].tools)
        else:
            for connection in self.connections.values():
                if connection.status == "connected":
                    all_tools.extend(connection.tools)
        
        return all_tools
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        server_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a tool on an MCP server.
        
        Args:
            tool_name: Name of the tool to execute.
            parameters: Tool parameters.
            server_name: Optional server name (auto-detected if not provided).
            
        Returns:
            Dict[str, Any]: Tool execution result.
            
        Raises:
            ValueError: If tool or server is not found.
            RuntimeError: If execution fails.
            
        Example:
            >>> result = await client.execute_tool(
            ...     "search_tools",
            ...     {"query": "RNA-seq", "category": "genomics"}
            ... )
            >>> print(result)
        """
        self.total_requests += 1
        self.last_activity = datetime.now()
        
        try:
            # Find the tool and server
            target_server = None
            target_tool = None
            
            if server_name:
                if server_name not in self.connections:
                    raise ValueError(f"Server not found: {server_name}")
                
                connection = self.connections[server_name]
                if connection.status != "connected":
                    raise RuntimeError(f"Server {server_name} is not connected")
                
                for tool in connection.tools:
                    if tool.name == tool_name:
                        target_server = connection
                        target_tool = tool
                        break
                
                if not target_tool:
                    raise ValueError(f"Tool {tool_name} not found on server {server_name}")
            else:
                # Auto-detect server
                for connection in self.connections.values():
                    if connection.status == "connected":
                        for tool in connection.tools:
                            if tool.name == tool_name:
                                target_server = connection
                                target_tool = tool
                                break
                    if target_tool:
                        break
                
                if not target_tool:
                    raise ValueError(f"Tool {tool_name} not found on any connected server")
            
            # Execute the tool
            result = await self._execute_tool_on_server(
                target_server,
                target_tool,
                parameters or {}
            )
            
            self.successful_requests += 1
            return result
            
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"Tool execution failed: {e}")
            raise
    
    async def _execute_tool_on_server(
        self,
        connection: MCPServerConnection,
        tool: MCPToolInfo,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool on a specific server.
        
        Args:
            connection: Server connection.
            tool: Tool to execute.
            parameters: Tool parameters.
            
        Returns:
            Dict[str, Any]: Execution result.
        """
        try:
            self.logger.info(f"Executing {tool.name} on {connection.name}")
            
            # Mock execution - in a real implementation, you'd call the MCP protocol
            if tool.name == "search_tools":
                return {
                    "tools": [
                        {
                            "name": "BLAST",
                            "description": "Basic Local Alignment Search Tool",
                            "category": "sequence_analysis",
                            "url": "https://blast.ncbi.nlm.nih.gov/",
                        },
                        {
                            "name": "BWA",
                            "description": "Burrows-Wheeler Aligner",
                            "category": "sequence_alignment",
                            "url": "http://bio-bwa.sourceforge.net/",
                        },
                    ],
                    "query": parameters.get("query", ""),
                    "total_results": 2,
                }
            elif tool.name == "get_tool_info":
                tool_name = parameters.get("tool_name", "")
                return {
                    "name": tool_name,
                    "description": f"Detailed information about {tool_name}",
                    "category": "bioinformatics",
                    "installation": f"conda install {tool_name}",
                    "documentation": f"https://docs.{tool_name}.org/",
                }
            elif tool.name == "web_search":
                return {
                    "results": [
                        {
                            "title": f"Search result for {parameters.get('query', '')}",
                            "url": "https://example.com",
                            "snippet": "This is a search result snippet",
                        }
                    ],
                    "query": parameters.get("query", ""),
                    "total_results": 1,
                }
            else:
                return {
                    "message": f"Tool {tool.name} executed successfully",
                    "parameters": parameters,
                    "server": connection.name,
                }
            
        except Exception as e:
            connection.error_count += 1
            self.logger.error(f"Tool execution failed on {connection.name}: {e}")
            raise RuntimeError(f"Tool execution failed: {e}") from e
    
    async def add_server(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> bool:
        """Add a new MCP server dynamically.
        
        Args:
            name: Unique name for the server.
            config: Server configuration dictionary.
            
        Returns:
            bool: True if server was added successfully.
            
        Example:
            >>> success = await client.add_server("new_server", {
            ...     "transport": "stdio",
            ...     "command": "python",
            ...     "args": ["server.py"]
            ... })
        """
        try:
            # Create server config
            server_config = MCPServerConfig(**config)
            
            # Create connection
            connection = MCPServerConnection(name=name, config=server_config)
            
            # Connect to the server
            await self._connect_to_server(connection)
            
            # Add to connections
            self.connections[name] = connection
            
            self.logger.info(f"Successfully added server: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add server {name}: {e}")
            return False
    
    async def remove_server(self, name: str) -> bool:
        """Remove an MCP server.
        
        Args:
            name: Name of the server to remove.
            
        Returns:
            bool: True if server was removed successfully.
            
        Example:
            >>> success = await client.remove_server("old_server")
        """
        try:
            if name not in self.connections:
                self.logger.warning(f"Server not found: {name}")
                return False
            
            connection = self.connections[name]
            
            # Close connection
            await self._close_connection(connection)
            
            # Remove from connections
            del self.connections[name]
            
            self.logger.info(f"Successfully removed server: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove server {name}: {e}")
            return False
    
    async def _close_connection(self, connection: MCPServerConnection) -> None:
        """Close a server connection.
        
        Args:
            connection: Connection to close.
        """
        try:
            connection.status = "disconnecting"
            
            # Close session
            if connection.session:
                if isinstance(connection.session, httpx.AsyncClient):
                    await connection.session.aclose()
                # For stdio sessions, you'd close the read/write streams
                connection.session = None
            
            # Terminate process if running
            if connection.process:
                connection.process.terminate()
                try:
                    await asyncio.wait_for(connection.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    connection.process.kill()
                connection.process = None
            
            connection.status = "disconnected"
            
        except Exception as e:
            self.logger.error(f"Error closing connection {connection.name}: {e}")
            connection.status = "error"
    
    async def get_langchain_tools(self) -> List[Any]:
        """Get LangChain-compatible tools from MCP servers.
        
        Returns:
            List[Any]: List of LangChain tools.
            
        Example:
            >>> langchain_tools = await client.get_langchain_tools()
            >>> # Use with LangChain agents
        """
        if not self.multi_client:
            self.logger.warning("Multi-client not available for LangChain integration")
            return []
        
        try:
            tools = await self.multi_client.get_tools()
            self.logger.info(f"Retrieved {len(tools)} LangChain tools")
            return tools
        except Exception as e:
            self.logger.error(f"Failed to get LangChain tools: {e}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics and statistics.
        
        Returns:
            Dict[str, Any]: Client metrics.
            
        Example:
            >>> metrics = client.get_metrics()
            >>> print(f"Success rate: {metrics['success_rate']:.1f}%")
        """
        success_rate = 0.0
        if self.total_requests > 0:
            success_rate = (self.successful_requests / self.total_requests) * 100
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "connected_servers": len([
                c for c in self.connections.values() if c.status == "connected"
            ]),
            "total_servers": len(self.connections),
            "total_tools": sum(len(c.tools) for c in self.connections.values()),
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check.
        
        Returns:
            Dict[str, Any]: Health check results.
            
        Example:
            >>> health = await client.health_check()
            >>> print(f"Overall health: {health['status']}")
        """
        server_health = {}
        healthy_servers = 0
        
        for name, connection in self.connections.items():
            is_healthy = connection.is_healthy()
            server_health[name] = {
                "status": connection.status,
                "healthy": is_healthy,
                "error_count": connection.error_count,
                "tools_count": len(connection.tools),
                "last_ping": connection.last_ping.isoformat() if connection.last_ping else None,
            }
            
            if is_healthy:
                healthy_servers += 1
        
        overall_status = "healthy"
        if healthy_servers == 0:
            overall_status = "critical"
        elif healthy_servers < len(self.connections) / 2:
            overall_status = "warning"
        
        return {
            "status": overall_status,
            "healthy_servers": healthy_servers,
            "total_servers": len(self.connections),
            "health_score": (healthy_servers / len(self.connections) * 100) if self.connections else 100,
            "servers": server_health,
            "metrics": self.get_metrics(),
        }
    
    async def close(self) -> None:
        """Close all connections and shutdown the client.
        
        Example:
            >>> await client.close()
        """
        self.logger.info("Shutting down MCP client...")
        
        self._shutdown = True
        
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        close_tasks = []
        for connection in self.connections.values():
            task = asyncio.create_task(self._close_connection(connection))
            close_tasks.append(task)
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # Close multi-client
        if self.multi_client and hasattr(self.multi_client, 'close'):
            try:
                await self.multi_client.close()
            except Exception as e:
                self.logger.warning(f"Error closing multi-client: {e}")
        
        self.connections.clear()
        
        self.logger.info("MCP client shutdown complete")
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return (
            f"MCPClient("
            f"servers={len(self.connections)}, "
            f"connected={len([c for c in self.connections.values() if c.status == 'connected'])}, "
            f"tools={sum(len(c.tools) for c in self.connections.values())}"
            f")"
        )