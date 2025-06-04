#!/usr/bin/env python3
"""Basic Search Example for the MCP Agent Framework.

This example demonstrates how to use the MCP Agent Framework for basic
bioinformatics tool discovery and search functionality. It shows how to
set up the agent, perform searches, and display results.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Usage:
    Basic usage:
    $ python basic_search.py

    With custom query:
    $ python basic_search.py --query "protein structure analysis"

    Interactive mode:
    $ python basic_search.py --interactive

Requirements:
    - MCP Agent Framework installed
    - Vector database populated (run populate_vectordb.py first)
    - Environment variables configured (.env file)
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from mcp_agent.main import MCPAgent
    from mcp_agent.config.settings import AgentSettings
    from mcp_agent.models.schemas import SearchQuery
    from mcp_agent.utils import get_logger, setup_logger
except ImportError as e:
    print(f"Error importing MCP Agent components: {e}")
    print("Please ensure the MCP Agent Framework is properly installed.")
    print("You may need to run: pip install -e .")
    sys.exit(1)

# Initialize console and logger
console = Console()
logger = get_logger(__name__)

# Sample search queries for demonstration
SAMPLE_QUERIES = [
    "RNA sequencing analysis tools",
    "protein structure prediction software",
    "genome assembly algorithms",
    "BLAST sequence alignment",
    "phylogenetic tree construction",
    "variant calling tools",
    "gene expression analysis",
    "metabolomics data processing",
]

# Bioinformatics categories for filtering
BIOINFORMATICS_CATEGORIES = [
    "sequence_analysis",
    "genomics", 
    "transcriptomics",
    "proteomics",
    "phylogenetics",
    "utilities",
    "quality_control",
    "visualization",
]


class BasicSearchExample:
    """Example class demonstrating basic search functionality."""
    
    def __init__(self, settings: Optional[AgentSettings] = None):
        """Initialize the search example.
        
        Args:
            settings: Optional agent settings. If None, uses defaults.
        """
        self.settings = settings
        self.agent: Optional[MCPAgent] = None
        self.search_history: List[Dict] = []
    
    async def initialize(self) -> None:
        """Initialize the MCP Agent."""
        console.print("[blue]Initializing MCP Agent...[/blue]")
        
        try:
            # Create agent with settings
            self.agent = MCPAgent(settings=self.settings)
            
            # Initialize the agent
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Starting up agent components...", total=None)
                await self.agent.initialize()
                progress.update(task, description="Agent ready!")
            
            console.print("[green]✓ MCP Agent initialized successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to initialize agent: {e}[/red]")
            raise
    
    async def simple_search(self, query: str, max_results: int = 10) -> None:
        """Perform a simple search and display results.
        
        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
        """
        if not self.agent:
            console.print("[red]Agent not initialized. Call initialize() first.[/red]")
            return
        
        console.print(f"\n[bold blue]Searching for:[/bold blue] '{query}'")
        
        try:
            # Perform search
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching tools...", total=None)
                
                result = await self.agent.search(
                    query=query,
                    max_results=max_results,
                    include_documentation=True
                )
                
                progress.update(task, description="Search complete!")
            
            # Store search in history
            self.search_history.append({
                "query": query,
                "results_count": len(result.tools),
                "timestamp": result.timestamp,
            })
            
            # Display results
            self._display_search_results(result)
            
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")
            logger.error(f"Search error: {e}")
    
    async def filtered_search(
        self, 
        query: str, 
        category: Optional[str] = None,
        organism: Optional[str] = None,
        max_results: int = 10
    ) -> None:
        """Perform a filtered search with specific criteria.
        
        Args:
            query: Search query string.
            category: Tool category filter.
            organism: Organism filter.
            max_results: Maximum number of results to return.
        """
        if not self.agent:
            console.print("[red]Agent not initialized. Call initialize() first.[/red]")
            return
        
        # Build filter description
        filter_desc = []
        if category:
            filter_desc.append(f"category: {category}")
        if organism:
            filter_desc.append(f"organism: {organism}")
        
        filter_text = f" (filters: {', '.join(filter_desc)})" if filter_desc else ""
        console.print(f"\n[bold blue]Filtered search:[/bold blue] '{query}'{filter_text}")
        
        try:
            # Build filters
            filters = {}
            if category:
                filters["category"] = category
            if organism:
                filters["organism"] = organism
            
            # Perform search
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching with filters...", total=None)
                
                result = await self.agent.search(
                    query=query,
                    max_results=max_results,
                    filters=filters,
                    include_documentation=True
                )
                
                progress.update(task, description="Filtered search complete!")
            
            # Display results
            self._display_search_results(result, show_filters=True)
            
        except Exception as e:
            console.print(f"[red]Filtered search failed: {e}[/red]")
            logger.error(f"Filtered search error: {e}")
    
    def _display_search_results(self, result, show_filters: bool = False) -> None:
        """Display search results in a formatted table.
        
        Args:
            result: Search result object.
            show_filters: Whether to show filter information.
        """
        if not result.tools:
            console.print("[yellow]No tools found for this search.[/yellow]")
            return
        
        # Create results table
        table = Table(title=f"Search Results ({len(result.tools)} tools found)")
        table.add_column("Tool Name", style="cyan", no_wrap=True)
        table.add_column("Category", style="green")
        table.add_column("Description", style="white")
        table.add_column("Language", style="yellow")
        table.add_column("Score", style="magenta")
        
        # Add results to table
        for tool in result.tools:
            # Truncate description if too long
            description = tool.description
            if len(description) > 80:
                description = description[:77] + "..."
            
            # Get similarity score if available
            score = getattr(tool, 'similarity_score', None)
            score_text = f"{score:.3f}" if score else "N/A"
            
            table.add_row(
                tool.name,
                tool.category or "Unknown",
                description,
                tool.language or "N/A",
                score_text,
            )
        
        console.print(table)
        
        # Show search metadata
        if hasattr(result, 'search_time_ms'):
            console.print(f"\n[dim]Search completed in {result.search_time_ms:.2f}ms[/dim]")
    
    async def tool_details(self, tool_name: str) -> None:
        """Show detailed information about a specific tool.
        
        Args:
            tool_name: Name of the tool to get details for.
        """
        if not self.agent:
            console.print("[red]Agent not initialized. Call initialize() first.[/red]")
            return
        
        console.print(f"\n[bold blue]Getting details for:[/bold blue] {tool_name}")
        
        try:
            # Get tool information
            tool_info = await self.agent.get_tool_info(tool_name)
            
            if tool_info:
                self._display_tool_details(tool_info)
            else:
                console.print(f"[yellow]Tool '{tool_name}' not found.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Failed to get tool details: {e}[/red]")
            logger.error(f"Tool details error: {e}")
    
    def _display_tool_details(self, tool_info: Dict) -> None:
        """Display detailed tool information.
        
        Args:
            tool_info: Tool information dictionary.
        """
        # Create details panel
        details = []
        
        # Basic information
        details.append(f"**Name:** {tool_info.get('name', 'Unknown')}")
        details.append(f"**Description:** {tool_info.get('description', 'No description available')}")
        details.append(f"**Category:** {tool_info.get('category', 'Unknown')}")
        
        if tool_info.get('subcategory'):
            details.append(f"**Subcategory:** {tool_info['subcategory']}")
        
        # Technical details
        if tool_info.get('language'):
            details.append(f"**Language:** {tool_info['language']}")
        
        if tool_info.get('license'):
            details.append(f"**License:** {tool_info['license']}")
        
        if tool_info.get('version'):
            details.append(f"**Version:** {tool_info['version']}")
        
        # Links
        if tool_info.get('url'):
            details.append(f"**Website:** {tool_info['url']}")
        
        if tool_info.get('documentation_url'):
            details.append(f"**Documentation:** {tool_info['documentation_url']}")
        
        if tool_info.get('repository_url'):
            details.append(f"**Repository:** {tool_info['repository_url']}")
        
        # Installation
        if tool_info.get('installation'):
            installation = tool_info['installation']
            details.append("**Installation:**")
            for method, command in installation.items():
                details.append(f"  - {method}: `{command}`")
        
        # Formats
        if tool_info.get('input_formats'):
            formats = ', '.join(tool_info['input_formats'])
            details.append(f"**Input Formats:** {formats}")
        
        if tool_info.get('output_formats'):
            formats = ', '.join(tool_info['output_formats'])
            details.append(f"**Output Formats:** {formats}")
        
        # Tags
        if tool_info.get('tags'):
            tags = ', '.join(tool_info['tags'])
            details.append(f"**Tags:** {tags}")
        
        # Create markdown content
        markdown_content = '\n'.join(details)
        
        # Display in panel
        console.print(Panel(
            Markdown(markdown_content),
            title=f"Tool Details: {tool_info.get('name', 'Unknown')}",
            border_style="blue"
        ))
    
    async def list_available_tools(self, category: Optional[str] = None) -> None:
        """List all available tools, optionally filtered by category.
        
        Args:
            category: Optional category to filter by.
        """
        if not self.agent:
            console.print("[red]Agent not initialized. Call initialize() first.[/red]")
            return
        
        filter_text = f" in category '{category}'" if category else ""
        console.print(f"\n[bold blue]Listing available tools{filter_text}...[/bold blue]")
        
        try:
            # Get tools list
            tools = await self.agent.list_tools(category=category)
            
            if not tools:
                console.print(f"[yellow]No tools found{filter_text}.[/yellow]")
                return
            
            # Group tools by category
            tools_by_category = {}
            for tool in tools:
                cat = tool.get('category', 'Unknown')
                if cat not in tools_by_category:
                    tools_by_category[cat] = []
                tools_by_category[cat].append(tool)
            
            # Display tools by category
            for cat_name, cat_tools in sorted(tools_by_category.items()):
                console.print(f"\n[bold green]{cat_name.title()}[/bold green] ({len(cat_tools)} tools)")
                
                for tool in sorted(cat_tools, key=lambda x: x.get('name', '')):
                    name = tool.get('name', 'Unknown')
                    description = tool.get('description', 'No description')
                    if len(description) > 60:
                        description = description[:57] + "..."
                    
                    console.print(f"  • [cyan]{name}[/cyan]: {description}")
            
            console.print(f"\n[dim]Total tools: {len(tools)}[/dim]")
            
        except Exception as e:
            console.print(f"[red]Failed to list tools: {e}[/red]")
            logger.error(f"List tools error: {e}")
    
    def show_search_history(self) -> None:
        """Display search history."""
        if not self.search_history:
            console.print("[yellow]No search history available.[/yellow]")
            return
        
        table = Table(title="Search History")
        table.add_column("Query", style="cyan")
        table.add_column("Results", style="green")
        table.add_column("Timestamp", style="dim")
        
        for search in self.search_history[-10:]:  # Show last 10 searches
            table.add_row(
                search["query"],
                str(search["results_count"]),
                search["timestamp"][:19] if search["timestamp"] else "Unknown"
            )
        
        console.print(table)
    
    async def interactive_mode(self) -> None:
        """Run interactive search mode."""
        console.print(Panel(
            "[bold blue]Interactive Bioinformatics Tool Search[/bold blue]\n\n"
            "Commands:\n"
            "• Type a search query to find tools\n"
            "• 'list' - Show all available tools\n"
            "• 'list <category>' - Show tools in a category\n"
            "• 'details <tool_name>' - Show tool details\n"
            "• 'history' - Show search history\n"
            "• 'categories' - Show available categories\n"
            "• 'help' - Show this help\n"
            "• 'quit' or 'exit' - Exit interactive mode",
            title="Welcome to MCP Agent Tool Search",
            border_style="blue"
        ))
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]Search[/bold cyan]").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[green]Goodbye![/green]")
                    break
                
                elif user_input.lower() == 'help':
                    console.print(Panel(
                        "Available commands:\n"
                        "• [cyan]<search query>[/cyan] - Search for tools\n"
                        "• [cyan]list[/cyan] - Show all tools\n"
                        "• [cyan]list <category>[/cyan] - Show tools in category\n"
                        "• [cyan]details <tool_name>[/cyan] - Show tool details\n"
                        "• [cyan]history[/cyan] - Show search history\n"
                        "• [cyan]categories[/cyan] - Show available categories\n"
                        "• [cyan]quit[/cyan] - Exit",
                        title="Help",
                        border_style="green"
                    ))
                
                elif user_input.lower() == 'history':
                    self.show_search_history()
                
                elif user_input.lower() == 'categories':
                    console.print("\n[bold green]Available Categories:[/bold green]")
                    for i, cat in enumerate(BIOINFORMATICS_CATEGORIES, 1):
                        console.print(f"  {i}. [cyan]{cat}[/cyan]")
                
                elif user_input.lower().startswith('list'):
                    parts = user_input.split(maxsplit=1)
                    category = parts[1] if len(parts) > 1 else None
                    await self.list_available_tools(category)
                
                elif user_input.lower().startswith('details'):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        await self.tool_details(parts[1])
                    else:
                        console.print("[yellow]Please specify a tool name: details <tool_name>[/yellow]")
                
                else:
                    # Treat as search query
                    await self.simple_search(user_input)
            
            except KeyboardInterrupt:
                console.print("\n[green]Goodbye![/green]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                logger.error(f"Interactive mode error: {e}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.agent:
            await self.agent.close()
            console.print("[dim]Agent resources cleaned up.[/dim]")


async def run_basic_examples(agent_example: BasicSearchExample) -> None:
    """Run a series of basic search examples.
    
    Args:
        agent_example: Initialized agent example instance.
    """
    console.print(Panel(
        "[bold blue]Running Basic Search Examples[/bold blue]\n\n"
        "This demo will show various search capabilities of the MCP Agent Framework.",
        title="MCP Agent Framework Demo",
        border_style="blue"
    ))
    
    # Example 1: Simple search
    console.print("\n[bold magenta]Example 1: Simple Search[/bold magenta]")
    await agent_example.simple_search("RNA sequencing analysis")
    
    # Wait for user
    if Confirm.ask("\nContinue to next example?", default=True):
        # Example 2: Category filtered search
        console.print("\n[bold magenta]Example 2: Category Filtered Search[/bold magenta]")
        await agent_example.filtered_search(
            "alignment tools", 
            category="sequence_analysis"
        )
    
    if Confirm.ask("\nContinue to next example?", default=True):
        # Example 3: Tool details
        console.print("\n[bold magenta]Example 3: Tool Details[/bold magenta]")
        await agent_example.tool_details("BLAST+")
    
    if Confirm.ask("\nContinue to next example?", default=True):
        # Example 4: List tools by category
        console.print("\n[bold magenta]Example 4: List Tools by Category[/bold magenta]")
        await agent_example.list_available_tools("genomics")
    
    if Confirm.ask("\nView search history?", default=True):
        console.print("\n[bold magenta]Search History[/bold magenta]")
        agent_example.show_search_history()


# CLI Interface
@click.command()
@click.option("--query", "-q", help="Search query to execute")
@click.option("--category", "-c", help="Filter by category")
@click.option("--max-results", "-n", default=10, help="Maximum results to return")
@click.option("--interactive", "-i", is_flag=True, help="Run in interactive mode")
@click.option("--demo", "-d", is_flag=True, help="Run demonstration examples")
@click.option("--tool-details", "-t", help="Show details for specific tool")
@click.option("--list-tools", "-l", is_flag=True, help="List all available tools")
@click.option("--list-categories", is_flag=True, help="List available categories")
@click.option("--config", help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(
    query: Optional[str],
    category: Optional[str], 
    max_results: int,
    interactive: bool,
    demo: bool,
    tool_details: Optional[str],
    list_tools: bool,
    list_categories: bool,
    config: Optional[str],
    verbose: bool,
):
    """Basic Search Example for MCP Agent Framework.
    
    This example demonstrates how to search for bioinformatics tools using
    the MCP Agent Framework. You can perform simple searches, filter by
    categories, get tool details, and more.
    
    Examples:
        python basic_search.py --query "protein analysis"
        python basic_search.py --interactive
        python basic_search.py --demo
        python basic_search.py --tool-details "BLAST+"
    """
    
    async def run_example():
        # Set up logging
        log_level = "DEBUG" if verbose else "INFO"
        setup_logger(level=log_level, enable_file_logging=False)
        
        # Create settings
        try:
            from mcp_agent.config import get_settings
            settings = get_settings(config)
            
            # Override for example
            settings.log_level = "WARNING"  # Reduce log noise for demo
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load settings: {e}[/yellow]")
            console.print("[yellow]Using default settings...[/yellow]")
            settings = None
        
        # Initialize example
        agent_example = BasicSearchExample(settings)
        
        try:
            # Initialize agent
            await agent_example.initialize()
            
            # Handle different modes
            if list_categories:
                console.print("\n[bold green]Available Categories:[/bold green]")
                for i, cat in enumerate(BIOINFORMATICS_CATEGORIES, 1):
                    console.print(f"  {i}. [cyan]{cat}[/cyan]")
            
            elif list_tools:
                await agent_example.list_available_tools(category)
            
            elif tool_details:
                await agent_example.tool_details(tool_details)
            
            elif query:
                if category:
                    await agent_example.filtered_search(query, category, max_results=max_results)
                else:
                    await agent_example.simple_search(query, max_results)
            
            elif demo:
                await run_basic_examples(agent_example)
            
            elif interactive:
                await agent_example.interactive_mode()
            
            else:
                # Default: show help and run a sample search
                console.print(Panel(
                    "[bold blue]MCP Agent Framework - Basic Search Example[/bold blue]\n\n"
                    "No specific action requested. Running a sample search...\n"
                    "Use --help to see all available options.",
                    title="Welcome",
                    border_style="blue"
                ))
                
                # Run sample search
                sample_query = "sequence alignment tools"
                console.print(f"\n[bold yellow]Sample Search:[/bold yellow] '{sample_query}'")
                await agent_example.simple_search(sample_query)
                
                console.print(f"\n[dim]Try running with --interactive for an interactive experience![/dim]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        
        except Exception as e:
            console.print(f"[red]Error running example: {e}[/red]")
            logger.error(f"Example error: {e}")
            sys.exit(1)
        
        finally:
            # Clean up
            await agent_example.cleanup()
    
    # Run the async example
    asyncio.run(run_example())


if __name__ == "__main__":
    main()