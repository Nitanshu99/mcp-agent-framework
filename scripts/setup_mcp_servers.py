#!/usr/bin/env python3
"""MCP Server Setup and Management Script for the MCP Agent Framework.

This script provides utilities for setting up, configuring, and managing MCP servers
for bioinformatics tool discovery and integration. It supports both local development
servers and production server configurations.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Usage:
    Setup all default servers:
    $ python setup_mcp_servers.py setup

    Setup specific server:
    $ python setup_mcp_servers.py setup --server bioinformatics

    Check server status:
    $ python setup_mcp_servers.py status

    Start development servers:
    $ python setup_mcp_servers.py start --dev

    Stop all servers:
    $ python setup_mcp_servers.py stop

Example:
    >>> from scripts.setup_mcp_servers import MCPServerManager
    >>> manager = MCPServerManager()
    >>> await manager.setup_all_servers()
    >>> await manager.check_server_health("bioinformatics")
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile
import yaml
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone

import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

# Initialize rich console
console = Console()

# Default server configurations
@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    description: str
    command: str
    args: List[str]
    transport: str = "stdio"
    url: Optional[str] = None
    port: Optional[int] = None
    env_vars: Optional[Dict[str, str]] = None
    health_check_path: Optional[str] = None
    auto_start: bool = True
    category: str = "general"


class MCPServerManager:
    """Manager for MCP server setup and operations."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the MCP server manager.
        
        Args:
            config_path: Optional path to server configuration file.
        """
        self.config_path = Path(config_path) if config_path else Path("conf.yaml")
        self.servers: Dict[str, MCPServerConfig] = {}
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.server_status: Dict[str, str] = {}
        
        # Default server configurations
        self._load_default_servers()
        
        # Try to load existing configuration
        if self.config_path.exists():
            self._load_server_config()
    
    def _load_default_servers(self) -> None:
        """Load default MCP server configurations."""
        default_servers = {
            "bioinformatics": MCPServerConfig(
                name="bioinformatics",
                description="Bioinformatics tools and databases server",
                command="python",
                args=["-m", "mcp_servers.bioinformatics"],
                transport="stdio",
                category="bioinformatics",
                env_vars={
                    "NCBI_API_KEY": os.getenv("NCBI_API_KEY", ""),
                    "UNIPROT_API_KEY": os.getenv("UNIPROT_API_KEY", ""),
                },
            ),
            "web_search": MCPServerConfig(
                name="web_search",
                description="Web search and research server",
                command="python",
                args=["-m", "mcp_servers.web_search"],
                transport="stdio",
                category="search",
                env_vars={
                    "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
                    "BRAVE_SEARCH_API_KEY": os.getenv("BRAVE_SEARCH_API_KEY", ""),
                    "SERPER_API_KEY": os.getenv("SERPER_API_KEY", ""),
                },
            ),
            "documentation": MCPServerConfig(
                name="documentation",
                description="Documentation and help server",
                command="python",
                args=["-m", "mcp_servers.documentation"],
                transport="stdio",
                category="documentation",
            ),
            "file_operations": MCPServerConfig(
                name="file_operations",
                description="File system operations server",
                command="python",
                args=["-m", "mcp_servers.file_ops"],
                transport="stdio",
                category="utilities",
            ),
            "data_analysis": MCPServerConfig(
                name="data_analysis",
                description="Data analysis and processing server",
                command="python",
                args=["-m", "mcp_servers.data_analysis"],
                transport="stdio",
                category="analysis",
            ),
            "http_example": MCPServerConfig(
                name="http_example",
                description="Example HTTP MCP server",
                command="python",
                args=["-m", "mcp_servers.http_server"],
                transport="http",
                url="http://localhost:8001/mcp",
                port=8001,
                health_check_path="/health",
                category="examples",
            ),
        }
        
        self.servers.update(default_servers)
    
    def _load_server_config(self) -> None:
        """Load server configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if "mcp_servers" in config:
                for name, server_config in config["mcp_servers"].items():
                    self.servers[name] = MCPServerConfig(
                        name=name,
                        description=server_config.get("description", f"MCP server: {name}"),
                        command=server_config.get("command", "python"),
                        args=server_config.get("args", []),
                        transport=server_config.get("transport", "stdio"),
                        url=server_config.get("url"),
                        port=server_config.get("port"),
                        env_vars=server_config.get("env_vars", {}),
                        health_check_path=server_config.get("health_check_path"),
                        auto_start=server_config.get("auto_start", True),
                        category=server_config.get("category", "custom"),
                    )
                    
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load server config: {e}[/yellow]")
    
    def save_server_config(self) -> None:
        """Save current server configuration to file."""
        config = {"mcp_servers": {}}
        
        for name, server in self.servers.items():
            config["mcp_servers"][name] = {
                "description": server.description,
                "command": server.command,
                "args": server.args,
                "transport": server.transport,
                "url": server.url,
                "port": server.port,
                "env_vars": server.env_vars or {},
                "health_check_path": server.health_check_path,
                "auto_start": server.auto_start,
                "category": server.category,
            }
        
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        console.print(f"[green]Server configuration saved to {self.config_path}[/green]")
    
    async def setup_all_servers(self, categories: Optional[List[str]] = None) -> Dict[str, bool]:
        """Set up all configured MCP servers.
        
        Args:
            categories: Optional list of categories to set up.
            
        Returns:
            Dict[str, bool]: Setup status for each server.
        """
        results = {}
        
        servers_to_setup = self.servers.values()
        if categories:
            servers_to_setup = [s for s in servers_to_setup if s.category in categories]
        
        console.print("[bold blue]Setting up MCP servers...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Setting up servers...", total=len(servers_to_setup))
            
            for server in servers_to_setup:
                progress.update(task, description=f"Setting up {server.name}...")
                
                try:
                    success = await self.setup_server(server.name)
                    results[server.name] = success
                    
                    if success:
                        console.print(f"[green]✓[/green] {server.name}: Setup successful")
                    else:
                        console.print(f"[red]✗[/red] {server.name}: Setup failed")
                        
                except Exception as e:
                    console.print(f"[red]✗[/red] {server.name}: Setup error - {e}")
                    results[server.name] = False
                
                progress.advance(task)
        
        return results
    
    async def setup_server(self, server_name: str) -> bool:
        """Set up a specific MCP server.
        
        Args:
            server_name: Name of the server to set up.
            
        Returns:
            bool: True if setup was successful.
        """
        if server_name not in self.servers:
            console.print(f"[red]Server '{server_name}' not found in configuration[/red]")
            return False
        
        server = self.servers[server_name]
        
        try:
            # Create server directory structure
            await self._create_server_structure(server)
            
            # Create server implementation if it doesn't exist
            await self._create_server_implementation(server)
            
            # Validate server configuration
            is_valid = await self._validate_server_config(server)
            
            if is_valid:
                self.server_status[server_name] = "configured"
                return True
            else:
                self.server_status[server_name] = "invalid"
                return False
                
        except Exception as e:
            console.print(f"[red]Error setting up server {server_name}: {e}[/red]")
            self.server_status[server_name] = "error"
            return False
    
    async def _create_server_structure(self, server: MCPServerConfig) -> None:
        """Create directory structure for server."""
        server_dir = Path("mcp_servers")
        server_dir.mkdir(exist_ok=True)
        
        # Create __init__.py if it doesn't exist
        init_file = server_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""MCP Servers package."""\n')
        
        # Create logs directory
        logs_dir = Path("logs") / "mcp_servers"
        logs_dir.mkdir(parents=True, exist_ok=True)
    
    async def _create_server_implementation(self, server: MCPServerConfig) -> None:
        """Create basic server implementation if it doesn't exist."""
        server_file = Path("mcp_servers") / f"{server.name}.py"
        
        if server_file.exists():
            return  # Server implementation already exists
        
        # Generate basic server template
        template = self._get_server_template(server)
        server_file.write_text(template)
        
        console.print(f"[yellow]Created basic server template: {server_file}[/yellow]")
    
    def _get_server_template(self, server: MCPServerConfig) -> str:
        """Get server implementation template."""
        if server.transport == "stdio":
            return self._get_stdio_server_template(server)
        elif server.transport == "http":
            return self._get_http_server_template(server)
        else:
            raise ValueError(f"Unsupported transport: {server.transport}")
    
    def _get_stdio_server_template(self, server: MCPServerConfig) -> str:
        """Get STDIO server template."""
        return f'''#!/usr/bin/env python3
"""
{server.description}

This is an auto-generated MCP server template for {server.name}.
Customize this implementation based on your specific requirements.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

class {server.name.title().replace('_', '')}MCPServer:
    """MCP server for {server.description.lower()}."""
    
    def __init__(self):
        self.tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, Dict[str, Any]]:
        """Register available tools."""
        return {{
            "hello": {{
                "description": "A simple hello world tool",
                "parameters": {{
                    "type": "object",
                    "properties": {{
                        "name": {{
                            "type": "string",
                            "description": "Name to greet"
                        }}
                    }},
                    "required": ["name"]
                }}
            }},
            # Add more tools here
        }}
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests."""
        method = request.get("method")
        params = request.get("params", {{}})
        
        if method == "tools/list":
            return {{
                "tools": [
                    {{"name": name, **config}}
                    for name, config in self.tools.items()
                ]
            }}
        
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {{}})
            
            if tool_name in self.tools:
                return await self._call_tool(tool_name, tool_args)
            else:
                return {{"error": f"Unknown tool: {{tool_name}}"}}
        
        else:
            return {{"error": f"Unknown method: {{method}}"}}
    
    async def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call."""
        if tool_name == "hello":
            name = args.get("name", "World")
            return {{
                "content": [
                    {{
                        "type": "text",
                        "text": f"Hello, {{name}}! This is the {{self.__class__.__name__}}."
                    }}
                ]
            }}
        
        # Add more tool implementations here
        
        return {{"error": f"Tool implementation not found: {{tool_name}}"}}
    
    async def run(self):
        """Run the MCP server."""
        while True:
            try:
                # Read JSON-RPC request from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                
                # Write JSON-RPC response to stdout
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError:
                error_response = {{"error": "Invalid JSON"}}
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                error_response = {{"error": str(e)}}
                print(json.dumps(error_response), flush=True)

if __name__ == "__main__":
    server = {server.name.title().replace('_', '')}MCPServer()
    asyncio.run(server.run())
'''
    
    def _get_http_server_template(self, server: MCPServerConfig) -> str:
        """Get HTTP server template."""
        return f'''#!/usr/bin/env python3
"""
{server.description}

This is an auto-generated HTTP MCP server template for {server.name}.
Customize this implementation based on your specific requirements.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("FastAPI and uvicorn are required for HTTP servers")
    print("Install with: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(
    title="{server.description}",
    description="MCP HTTP Server for {server.name}",
    version="1.0.0"
)

class {server.name.title().replace('_', '')}MCPServer:
    """HTTP MCP server for {server.description.lower()}."""
    
    def __init__(self):
        self.tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, Dict[str, Any]]:
        """Register available tools."""
        return {{
            "hello": {{
                "description": "A simple hello world tool",
                "parameters": {{
                    "type": "object",
                    "properties": {{
                        "name": {{
                            "type": "string",
                            "description": "Name to greet"
                        }}
                    }},
                    "required": ["name"]
                }}
            }},
            # Add more tools here
        }}
    
    async def handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests."""
        method = request.get("method")
        params = request.get("params", {{}})
        
        if method == "tools/list":
            return {{
                "tools": [
                    {{"name": name, **config}}
                    for name, config in self.tools.items()
                ]
            }}
        
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {{}})
            
            if tool_name in self.tools:
                return await self._call_tool(tool_name, tool_args)
            else:
                raise HTTPException(status_code=404, detail=f"Unknown tool: {{tool_name}}")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {{method}}")
    
    async def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call."""
        if tool_name == "hello":
            name = args.get("name", "World")
            return {{
                "content": [
                    {{
                        "type": "text",
                        "text": f"Hello, {{name}}! This is the {{self.__class__.__name__}}."
                    }}
                ]
            }}
        
        # Add more tool implementations here
        
        raise HTTPException(status_code=501, detail=f"Tool not implemented: {{tool_name}}")

# Initialize server instance
mcp_server = {server.name.title().replace('_', '')}MCPServer()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {{"status": "healthy", "timestamp": "{{datetime.now().isoformat()}}"}}

@app.post("/mcp")
async def mcp_endpoint(request: Dict[str, Any]):
    """Main MCP endpoint."""
    try:
        response = await mcp_server.handle_mcp_request(request)
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def list_tools():
    """List available tools."""
    return await mcp_server.handle_mcp_request({{"method": "tools/list"}})

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port={server.port or 8000},
        log_level="info"
    )
'''
    
    async def _validate_server_config(self, server: MCPServerConfig) -> bool:
        """Validate server configuration."""
        # Check if command exists
        try:
            result = subprocess.run(
                [server.command, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                console.print(f"[yellow]Warning: Command '{server.command}' may not be available[/yellow]")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            console.print(f"[yellow]Warning: Could not verify command '{server.command}'[/yellow]")
        
        # Check if server file exists for Python modules
        if server.command == "python" and server.args and server.args[0] == "-m":
            module_name = server.args[1]
            module_file = Path(module_name.replace(".", "/") + ".py")
            if not module_file.exists():
                console.print(f"[yellow]Warning: Server module {module_file} not found[/yellow]")
        
        return True
    
    async def start_server(self, server_name: str) -> bool:
        """Start a specific MCP server.
        
        Args:
            server_name: Name of the server to start.
            
        Returns:
            bool: True if server started successfully.
        """
        if server_name not in self.servers:
            console.print(f"[red]Server '{server_name}' not found[/red]")
            return False
        
        if server_name in self.running_processes:
            console.print(f"[yellow]Server '{server_name}' is already running[/yellow]")
            return True
        
        server = self.servers[server_name]
        
        try:
            # Prepare environment
            env = os.environ.copy()
            if server.env_vars:
                env.update(server.env_vars)
            
            # Build command
            cmd = [server.command] + server.args
            
            # Start process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment and check if process is still running
            await asyncio.sleep(1)
            
            if process.poll() is None:
                self.running_processes[server_name] = process
                self.server_status[server_name] = "running"
                console.print(f"[green]✓[/green] Server '{server_name}' started (PID: {process.pid})")
                return True
            else:
                stdout, stderr = process.communicate()
                console.print(f"[red]✗[/red] Server '{server_name}' failed to start")
                if stderr:
                    console.print(f"[red]Error: {stderr}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error starting server '{server_name}': {e}[/red]")
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """Stop a specific MCP server.
        
        Args:
            server_name: Name of the server to stop.
            
        Returns:
            bool: True if server stopped successfully.
        """
        if server_name not in self.running_processes:
            console.print(f"[yellow]Server '{server_name}' is not running[/yellow]")
            return True
        
        try:
            process = self.running_processes[server_name]
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                process.kill()
                process.wait()
            
            del self.running_processes[server_name]
            self.server_status[server_name] = "stopped"
            console.print(f"[green]✓[/green] Server '{server_name}' stopped")
            return True
            
        except Exception as e:
            console.print(f"[red]Error stopping server '{server_name}': {e}[/red]")
            return False
    
    async def stop_all_servers(self) -> Dict[str, bool]:
        """Stop all running servers.
        
        Returns:
            Dict[str, bool]: Stop status for each server.
        """
        results = {}
        
        for server_name in list(self.running_processes.keys()):
            results[server_name] = await self.stop_server(server_name)
        
        return results
    
    async def check_server_health(self, server_name: str) -> Dict[str, Any]:
        """Check health status of a server.
        
        Args:
            server_name: Name of the server to check.
            
        Returns:
            Dict[str, Any]: Health status information.
        """
        if server_name not in self.servers:
            return {"status": "unknown", "error": "Server not configured"}
        
        server = self.servers[server_name]
        health_info = {
            "server": server_name,
            "configured": True,
            "running": server_name in self.running_processes,
            "status": self.server_status.get(server_name, "unknown"),
            "transport": server.transport,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Check if process is still alive
        if server_name in self.running_processes:
            process = self.running_processes[server_name]
            if process.poll() is not None:
                # Process has terminated
                del self.running_processes[server_name]
                self.server_status[server_name] = "crashed"
                health_info["running"] = False
                health_info["status"] = "crashed"
        
        # For HTTP servers, check health endpoint
        if server.transport == "http" and server.url and health_info["running"]:
            try:
                health_url = f"{server.url.rstrip('/')}{server.health_check_path or '/health'}"
                
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(health_url)
                    
                if response.status_code == 200:
                    health_info["http_status"] = "healthy"
                    health_info["response_time_ms"] = response.elapsed.total_seconds() * 1000
                else:
                    health_info["http_status"] = "unhealthy"
                    health_info["http_code"] = response.status_code
                    
            except Exception as e:
                health_info["http_status"] = "unreachable"
                health_info["http_error"] = str(e)
        
        return health_info
    
    async def get_all_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers.
        
        Returns:
            Dict[str, Dict[str, Any]]: Status information for all servers.
        """
        status = {}
        
        for server_name in self.servers.keys():
            status[server_name] = await self.check_server_health(server_name)
        
        return status
    
    def list_servers(self, category: Optional[str] = None) -> List[MCPServerConfig]:
        """List available servers.
        
        Args:
            category: Optional category filter.
            
        Returns:
            List[MCPServerConfig]: List of server configurations.
        """
        servers = list(self.servers.values())
        
        if category:
            servers = [s for s in servers if s.category == category]
        
        return servers
    
    def add_server(self, server_config: MCPServerConfig) -> None:
        """Add a new server configuration.
        
        Args:
            server_config: Server configuration to add.
        """
        self.servers[server_config.name] = server_config
        console.print(f"[green]Added server configuration: {server_config.name}[/green]")
    
    def remove_server(self, server_name: str) -> bool:
        """Remove a server configuration.
        
        Args:
            server_name: Name of the server to remove.
            
        Returns:
            bool: True if server was removed.
        """
        if server_name in self.servers:
            # Stop server if running
            if server_name in self.running_processes:
                asyncio.create_task(self.stop_server(server_name))
            
            del self.servers[server_name]
            console.print(f"[green]Removed server configuration: {server_name}[/green]")
            return True
        else:
            console.print(f"[red]Server '{server_name}' not found[/red]")
            return False


# CLI Commands
@click.group()
@click.option("--config", help="Path to server configuration file", default="conf.yaml")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool):
    """MCP Server Setup and Management Tool."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose
    ctx.obj["manager"] = MCPServerManager(config)


@cli.command()
@click.option("--server", help="Specific server to set up")
@click.option("--category", help="Server category to set up")
@click.option("--all", "setup_all", is_flag=True, help="Set up all servers")
@click.pass_context
def setup(ctx: click.Context, server: Optional[str], category: Optional[str], setup_all: bool):
    """Set up MCP servers."""
    manager = ctx.obj["manager"]
    
    async def run_setup():
        if server:
            success = await manager.setup_server(server)
            if success:
                console.print(f"[green]Setup completed for server: {server}[/green]")
            else:
                console.print(f"[red]Setup failed for server: {server}[/red]")
                sys.exit(1)
        
        elif category:
            results = await manager.setup_all_servers([category])
            failed = [name for name, success in results.items() if not success]
            if failed:
                console.print(f"[red]Setup failed for servers: {', '.join(failed)}[/red]")
                sys.exit(1)
        
        elif setup_all:
            results = await manager.setup_all_servers()
            failed = [name for name, success in results.items() if not success]
            if failed:
                console.print(f"[red]Setup failed for servers: {', '.join(failed)}[/red]")
                sys.exit(1)
        
        else:
            console.print("[yellow]Please specify --server, --category, or --all[/yellow]")
            sys.exit(1)
        
        # Save configuration
        manager.save_server_config()
    
    asyncio.run(run_setup())


@cli.command()
@click.option("--server", help="Specific server to start")
@click.option("--category", help="Start servers in category")
@click.option("--all", "start_all", is_flag=True, help="Start all configured servers")
@click.option("--dev", is_flag=True, help="Start in development mode")
@click.pass_context
def start(ctx: click.Context, server: Optional[str], category: Optional[str], start_all: bool, dev: bool):
    """Start MCP servers."""
    manager = ctx.obj["manager"]
    
    async def run_start():
        if server:
            success = await manager.start_server(server)
            if not success:
                sys.exit(1)
        
        elif category or start_all:
            servers_to_start = manager.list_servers(category if category else None)
            
            for server_config in servers_to_start:
                if server_config.auto_start or start_all:
                    await manager.start_server(server_config.name)
        
        else:
            console.print("[yellow]Please specify --server, --category, or --all[/yellow]")
            sys.exit(1)
    
    asyncio.run(run_start())


@cli.command()
@click.option("--server", help="Specific server to stop")
@click.option("--all", "stop_all", is_flag=True, help="Stop all running servers")
@click.pass_context
def stop(ctx: click.Context, server: Optional[str], stop_all: bool):
    """Stop MCP servers."""
    manager = ctx.obj["manager"]
    
    async def run_stop():
        if server:
            await manager.stop_server(server)
        elif stop_all:
            await manager.stop_all_servers()
        else:
            console.print("[yellow]Please specify --server or --all[/yellow]")
            sys.exit(1)
    
    asyncio.run(run_stop())


@cli.command()
@click.option("--server", help="Check specific server")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def status(ctx: click.Context, server: Optional[str], output_json: bool):
    """Check server status."""
    manager = ctx.obj["manager"]
    
    async def run_status():
        if server:
            health = await manager.check_server_health(server)
            
            if output_json:
                console.print_json(json.dumps(health, indent=2))
            else:
                _display_server_status(server, health)
        
        else:
            all_status = await manager.get_all_server_status()
            
            if output_json:
                console.print_json(json.dumps(all_status, indent=2))
            else:
                _display_all_server_status(all_status)
    
    asyncio.run(run_status())


@cli.command()
@click.option("--category", help="Filter by category")
@click.pass_context
def list(ctx: click.Context, category: Optional[str]):
    """List available servers."""
    manager = ctx.obj["manager"]
    servers = manager.list_servers(category)
    
    table = Table(title="Available MCP Servers")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Transport", style="green")
    table.add_column("Category", style="yellow")
    table.add_column("Auto Start", style="blue")
    
    for server in servers:
        table.add_row(
            server.name,
            server.description,
            server.transport,
            server.category,
            "Yes" if server.auto_start else "No",
        )
    
    console.print(table)


def _display_server_status(server_name: str, health: Dict[str, Any]) -> None:
    """Display status for a single server."""
    status_color = "green" if health.get("running") else "red"
    status_text = health.get("status", "unknown").title()
    
    panel_content = f"""
[bold]Server:[/bold] {server_name}
[bold]Status:[/bold] [{status_color}]{status_text}[/{status_color}]
[bold]Transport:[/bold] {health.get("transport", "unknown")}
[bold]Running:[/bold] {"Yes" if health.get("running") else "No"}
[bold]Configured:[/bold] {"Yes" if health.get("configured") else "No"}
"""
    
    if "http_status" in health:
        panel_content += f"\n[bold]HTTP Status:[/bold] {health['http_status']}"
    
    if "response_time_ms" in health:
        panel_content += f"\n[bold]Response Time:[/bold] {health['response_time_ms']:.2f}ms"
    
    panel_content += f"\n[bold]Last Check:[/bold] {health.get('timestamp', 'unknown')}"
    
    console.print(Panel(panel_content.strip(), title=f"MCP Server Status: {server_name}"))


def _display_all_server_status(all_status: Dict[str, Dict[str, Any]]) -> None:
    """Display status for all servers."""
    table = Table(title="MCP Server Status")
    table.add_column("Server", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Transport", style="green")
    table.add_column("Running", style="yellow")
    table.add_column("Health", style="blue")
    
    for server_name, health in all_status.items():
        status = health.get("status", "unknown")
        status_color = "green" if health.get("running") else "red"
        
        running_status = "Yes" if health.get("running") else "No"
        http_status = health.get("http_status", "N/A")
        
        table.add_row(
            server_name,
            f"[{status_color}]{status}[/{status_color}]",
            health.get("transport", "unknown"),
            running_status,
            http_status,
        )
    
    console.print(table)


if __name__ == "__main__":
    cli()