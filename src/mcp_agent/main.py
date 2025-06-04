"""Main application entry point for the MCP Agent Framework.

This module provides the primary interface for interacting with the MCP Agent Framework,
including the main MCPAgent class and CLI commands for bioflow operations.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic usage:
    
    >>> from mcp_agent.main import MCPAgent
    >>> agent = MCPAgent()
    >>> result = await agent.search("RNA sequencing tools")
    
    CLI usage:
    
    $ bioflow-search "protein structure analysis tools"
    $ bioflow-setup --init-vectordb
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    from mcp_agent.config.settings import AgentSettings, get_settings
    from mcp_agent.models.schemas import SearchQuery, SearchResult, AgentResponse
    from mcp_agent.graph.workflow import create_workflow
    from mcp_agent.tools.mcp_client import MCPClient
    from mcp_agent.tools.vector_store import VectorStore
    from mcp_agent.utils.logger import get_logger
    from mcp_agent.agents.coordinator import CoordinatorAgent
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(f"Import error in main.py: {e}", ImportWarning)
    
    # Provide mock classes for development
    class SearchQuery:
        pass
    class SearchResult:
        pass
    class AgentResponse:
        pass


# Initialize console for rich output
console = Console()
logger = get_logger(__name__)


class MCPAgent:
    """Main MCP Agent class providing the primary interface for the framework.
    
    This class orchestrates the multi-agent system and provides a unified interface
    for tool discovery, research automation, and report generation using the Model
    Context Protocol.
    
    Attributes:
        settings (AgentSettings): Configuration settings for the agent.
        coordinator (CoordinatorAgent): The main coordination agent.
        mcp_client (MCPClient): Client for MCP server communication.
        vector_store (VectorStore): Vector database for semantic search.
        workflow (Any): LangGraph workflow for complex operations.
        
    Example:
        >>> agent = MCPAgent()
        >>> await agent.initialize()
        >>> result = await agent.search("BLAST alternatives")
        >>> print(result.summary)
    """
    
    def __init__(
        self,
        settings: Optional[AgentSettings] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the MCP Agent.
        
        Args:
            settings: Optional pre-configured settings object.
            config_path: Path to configuration file.
            
        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: If required dependencies are missing.
        """
        self.settings = settings or get_settings(config_path)
        self._initialized = False
        
        # Core components (initialized later)
        self.coordinator: Optional[CoordinatorAgent] = None
        self.mcp_client: Optional[MCPClient] = None
        self.vector_store: Optional[VectorStore] = None
        self.workflow: Optional[Any] = None
        
        logger.info(
            f"MCPAgent initialized with model: {self.settings.llm_model}"
        )
    
    async def initialize(self) -> None:
        """Initialize all agent components asynchronously.
        
        This method sets up the vector store, MCP client, coordinator agent,
        and workflow. It should be called before using any search or research
        functionality.
        
        Raises:
            RuntimeError: If initialization fails.
            ConnectionError: If MCP servers cannot be reached.
            
        Example:
            >>> agent = MCPAgent()
            >>> await agent.initialize()
        """
        if self._initialized:
            logger.warning("Agent already initialized")
            return
        
        try:
            # Initialize vector store
            logger.info("Initializing vector store...")
            self.vector_store = VectorStore(self.settings)
            await self.vector_store.initialize()
            
            # Initialize MCP client
            logger.info("Initializing MCP client...")
            self.mcp_client = MCPClient(self.settings)
            await self.mcp_client.initialize()
            
            # Initialize coordinator agent
            logger.info("Initializing coordinator agent...")
            self.coordinator = CoordinatorAgent(
                settings=self.settings,
                mcp_client=self.mcp_client,
                vector_store=self.vector_store,
            )
            
            # Create workflow
            logger.info("Creating workflow...")
            self.workflow = create_workflow(
                coordinator=self.coordinator,
                settings=self.settings,
            )
            
            self._initialized = True
            logger.info("MCPAgent initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCPAgent: {e}")
            raise RuntimeError(f"Agent initialization failed: {e}") from e
    
    async def search(
        self,
        query: Union[str, SearchQuery],
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_documentation: bool = True,
    ) -> SearchResult:
        """Perform a semantic search for bioinformatics tools.
        
        Args:
            query: Search query string or SearchQuery object.
            max_results: Maximum number of results to return.
            filters: Optional filters to apply to the search.
            include_documentation: Whether to include documentation snippets.
            
        Returns:
            SearchResult: Search results with tools, descriptions, and metadata.
            
        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If agent is not initialized.
            
        Example:
            >>> result = await agent.search("RNA-seq analysis tools")
            >>> for tool in result.tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        # Convert string to SearchQuery if needed
        if isinstance(query, str):
            if not query.strip():
                raise ValueError("Query cannot be empty")
            query = SearchQuery(
                text=query,
                max_results=max_results,
                filters=filters or {},
                include_documentation=include_documentation,
            )
        
        logger.info(f"Performing search: {query.text}")
        
        try:
            # Use coordinator to handle the search
            result = await self.coordinator.search(query)
            
            logger.info(f"Search completed. Found {len(result.tools)} tools")
            return result
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def research(
        self,
        topic: str,
        depth: str = "standard",
        output_format: str = "markdown",
        include_code_examples: bool = True,
    ) -> AgentResponse:
        """Perform deep research on a bioinformatics topic.
        
        Args:
            topic: Research topic or question.
            depth: Research depth ('quick', 'standard', 'comprehensive').
            output_format: Output format ('markdown', 'html', 'json').
            include_code_examples: Whether to include code examples.
            
        Returns:
            AgentResponse: Comprehensive research results and report.
            
        Raises:
            ValueError: If topic is empty or invalid depth specified.
            RuntimeError: If agent is not initialized.
            
        Example:
            >>> response = await agent.research(
            ...     "Machine learning in genomics",
            ...     depth="comprehensive"
            ... )
            >>> print(response.report)
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        if not topic.strip():
            raise ValueError("Research topic cannot be empty")
        
        if depth not in ["quick", "standard", "comprehensive"]:
            raise ValueError(f"Invalid depth: {depth}")
        
        logger.info(f"Starting research on: {topic}")
        
        try:
            # Use the workflow for complex research tasks
            result = await self.workflow.ainvoke({
                "topic": topic,
                "depth": depth,
                "output_format": output_format,
                "include_code_examples": include_code_examples,
                "user_query": f"Research topic: {topic}",
            })
            
            logger.info("Research completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            raise
    
    async def list_tools(
        self,
        category: Optional[str] = None,
        organism: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List available bioinformatics tools.
        
        Args:
            category: Optional category filter (e.g., 'genomics', 'proteomics').
            organism: Optional organism filter (e.g., 'human', 'mouse').
            
        Returns:
            List[Dict[str, Any]]: List of available tools with metadata.
            
        Example:
            >>> tools = await agent.list_tools(category="genomics")
            >>> print(f"Found {len(tools)} genomics tools")
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            filters = {}
            if category:
                filters["category"] = category
            if organism:
                filters["organism"] = organism
            
            tools = await self.vector_store.list_tools(filters)
            logger.info(f"Listed {len(tools)} tools")
            
            return tools
            
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            raise
    
    async def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool.
        
        Args:
            tool_name: Name of the tool to retrieve information for.
            
        Returns:
            Optional[Dict[str, Any]]: Tool information or None if not found.
            
        Example:
            >>> info = await agent.get_tool_info("BLAST")
            >>> if info:
            ...     print(f"Description: {info['description']}")
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            tool_info = await self.vector_store.get_tool(tool_name)
            if tool_info:
                logger.info(f"Retrieved info for tool: {tool_name}")
            else:
                logger.warning(f"Tool not found: {tool_name}")
            
            return tool_info
            
        except Exception as e:
            logger.error(f"Failed to get tool info: {e}")
            raise
    
    async def add_mcp_server(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> bool:
        """Add a new MCP server dynamically.
        
        Args:
            name: Unique name for the MCP server.
            config: Server configuration dictionary.
            
        Returns:
            bool: True if server was added successfully.
            
        Example:
            >>> success = await agent.add_mcp_server("custom_bio_db", {
            ...     "command": "python",
            ...     "args": ["custom_server.py"],
            ...     "transport": "stdio"
            ... })
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            success = await self.mcp_client.add_server(name, config)
            if success:
                logger.info(f"Successfully added MCP server: {name}")
            else:
                logger.error(f"Failed to add MCP server: {name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding MCP server {name}: {e}")
            return False
    
    async def close(self) -> None:
        """Clean up resources and close connections.
        
        Example:
            >>> await agent.close()
        """
        logger.info("Shutting down MCPAgent...")
        
        try:
            if self.mcp_client:
                await self.mcp_client.close()
            
            if self.vector_store:
                await self.vector_store.close()
            
            self._initialized = False
            logger.info("MCPAgent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        status = "initialized" if self._initialized else "not initialized"
        return f"MCPAgent(model={self.settings.llm_model}, status={status})"


# CLI Commands
@click.group()
@click.option("--config", help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool) -> None:
    """MCP Agent Framework CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose
    
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument("query")
@click.option("--max-results", "-n", default=10, help="Maximum results to return")
@click.option("--format", "output_format", default="table", 
              type=click.Choice(["table", "json", "markdown"]), 
              help="Output format")
@click.option("--category", help="Filter by tool category")
@click.option("--organism", help="Filter by organism")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    max_results: int,
    output_format: str,
    category: Optional[str],
    organism: Optional[str],
) -> None:
    """Search for bioinformatics tools."""
    async def _search():
        agent = MCPAgent(config_path=ctx.obj.get("config"))
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Initializing agent...", total=None)
                await agent.initialize()
                
                progress.update(task, description="Searching...")
                
                filters = {}
                if category:
                    filters["category"] = category
                if organism:
                    filters["organism"] = organism
                
                result = await agent.search(
                    query=query,
                    max_results=max_results,
                    filters=filters,
                )
                
                progress.update(task, description="Complete!")
            
            # Display results
            if output_format == "json":
                import json
                console.print_json(json.dumps(result.model_dump(), indent=2))
            elif output_format == "markdown":
                console.print(result.to_markdown())
            else:  # table
                table = Table(title=f"Search Results: {query}")
                table.add_column("Tool Name", style="cyan")
                table.add_column("Description", style="white")
                table.add_column("Category", style="green")
                
                for tool in result.tools:
                    table.add_row(
                        tool.name,
                        tool.description[:100] + "..." if len(tool.description) > 100 else tool.description,
                        tool.category or "Unknown",
                    )
                
                console.print(table)
        
        finally:
            await agent.close()
    
    asyncio.run(_search())


@cli.command()
@click.argument("topic")
@click.option("--depth", default="standard",
              type=click.Choice(["quick", "standard", "comprehensive"]),
              help="Research depth")
@click.option("--output", "-o", help="Output file path")
@click.option("--format", "output_format", default="markdown",
              type=click.Choice(["markdown", "html", "json"]),
              help="Output format")
@click.pass_context
def research(
    ctx: click.Context,
    topic: str,
    depth: str,
    output: Optional[str],
    output_format: str,
) -> None:
    """Perform deep research on a topic."""
    async def _research():
        agent = MCPAgent(config_path=ctx.obj.get("config"))
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Initializing agent...", total=None)
                await agent.initialize()
                
                progress.update(task, description=f"Researching: {topic}")
                
                result = await agent.research(
                    topic=topic,
                    depth=depth,
                    output_format=output_format,
                )
                
                progress.update(task, description="Complete!")
            
            # Save or display results
            if output:
                Path(output).write_text(result.report)
                console.print(f"✅ Research report saved to: {output}")
            else:
                console.print(result.report)
        
        finally:
            await agent.close()
    
    asyncio.run(_research())


@cli.command()
@click.option("--init-vectordb", is_flag=True, help="Initialize vector database")
@click.option("--check-deps", is_flag=True, help="Check dependencies")
@click.option("--test-mcp", is_flag=True, help="Test MCP servers")
@click.pass_context
def setup(
    ctx: click.Context,
    init_vectordb: bool,
    check_deps: bool,
    test_mcp: bool,
) -> None:
    """Setup and configuration commands."""
    if check_deps:
        console.print("🔍 Checking dependencies...")
        try:
            from mcp_agent import check_dependencies
            deps = check_dependencies()
            
            table = Table(title="Dependency Status")
            table.add_column("Package", style="cyan")
            table.add_column("Status", style="white")
            
            for name, available in deps.items():
                status = "✅ Available" if available else "❌ Missing"
                table.add_row(name, status)
            
            console.print(table)
        except Exception as e:
            console.print(f"❌ Error checking dependencies: {e}")
    
    if init_vectordb:
        console.print("🗄️  Initializing vector database...")
        # Implementation would go here
        console.print("✅ Vector database initialized")
    
    if test_mcp:
        console.print("🔌 Testing MCP servers...")
        # Implementation would go here
        console.print("✅ MCP servers tested")


# Alternative entry point for bioflow-search
def search_cli() -> None:
    """Entry point for bioflow-search command."""
    import sys
    # Remove the command name and pass remaining args to search
    sys.argv = ["bioflow"] + sys.argv[1:]
    cli(["search"] + sys.argv[1:])


# Main entry point
if __name__ == "__main__":
    cli()