# MCP Agent Framework

A basic Agent with Model Context Protocol (MCP) capabilities built using LangChain, LangGraph, and Google Gemini. This project is inspired by ByteDance's DeerFlow architecture but simplified for educational and research purposes in bioinformatics tool discovery.

## Project Overview

This framework combines the power of Google Gemini LLM with MCP servers to create a semantic search engine for bioinformatics tools. The agent can interact with external tools through MCP while maintaining a modular, extensible architecture.

## Features

- **MCP Integration**: Connect to multiple MCP servers for tool discovery
- **Google Gemini**: Powered by Google's latest LLM for intelligent reasoning
- **ChromaDB**: Vector database for semantic search capabilities
- **Modular Architecture**: Inspired by DeerFlow's multi-agent pattern
- **Type Safety**: Full type hints and type checking
- **Documentation**: Comprehensive Sphinx docstrings
- **Human-in-the-Loop**: Interactive workflow with user oversight

## Tech Stack

- **Language**: Python 3.12+
- **LLM**: Google Gemini (via langchain-google-genai)
- **Orchestration**: LangChain + LangGraph
- **MCP**: langchain-mcp-adapters
- **Vector DB**: ChromaDB
- **Type Checking**: mypy
- **Documentation**: Sphinx
- **Environment Management**: uv
- **Testing**: pytest

## Project Structure

```
mcp-agent-framework/
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── pyproject.toml              # Python project configuration
├── uv.lock                     # Dependency lock file
├── mypy.ini                    # MyPy configuration
├── docs/                       # Documentation
│   ├── conf.py                 # Sphinx configuration
│   ├── index.rst              # Main documentation index
│   └── modules/               # Auto-generated API docs
├── src/
│   └── mcp_agent/
│       ├── __init__.py         # Package initialization
│       ├── main.py             # Main application entry point
│       ├── config/
│       │   ├── __init__.py     # Config package init
│       │   └── settings.py     # Configuration management
│       ├── agents/
│       │   ├── __init__.py     # Agents package init
│       │   ├── base.py         # Base agent class
│       │   ├── coordinator.py  # Main coordinator agent
│       │   ├── researcher.py   # Research agent for tool discovery
│       │   └── reporter.py     # Report generation agent
│       ├── tools/
│       │   ├── __init__.py     # Tools package init
│       │   ├── mcp_client.py   # MCP client wrapper
│       │   ├── vector_store.py # ChromaDB integration
│       │   └── search.py       # Search utilities
│       ├── models/
│       │   ├── __init__.py     # Models package init
│       │   ├── schemas.py      # Pydantic data models
│       │   └── state.py        # LangGraph state models
│       ├── graph/
│       │   ├── __init__.py     # Graph package init
│       │   ├── workflow.py     # Main LangGraph workflow
│       │   └── nodes.py        # Individual graph nodes
│       └── utils/
│           ├── __init__.py     # Utils package init
│           ├── logger.py       # Logging configuration
│           └── helpers.py      # Utility functions
├── tests/                      # Test suite
│   ├── __init__.py            # Test package init
│   ├── conftest.py            # Pytest configuration
│   ├── test_agents/           # Agent tests
│   ├── test_tools/            # Tool tests
│   └── test_integration/      # Integration tests
├── scripts/                   # Utility scripts
│   ├── setup_mcp_servers.py  # MCP server setup
│   └── populate_vectordb.py  # Initial data population
└── examples/                  # Usage examples
    ├── basic_search.py        # Simple search example
    └── advanced_workflow.py   # Complex workflow example
```

## Installation & Setup

### Prerequisites

- Python 3.12+
- Git
- Google API key for Gemini
- Access to MCP servers (optional: we'll create local ones)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd mcp-agent-framework
```

### Step 2: Environment Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate     # On Windows
```

### Step 3: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
# GOOGLE_API_KEY=your_gemini_api_key_here
# CHROMADB_PATH=./data/chroma
# LOG_LEVEL=INFO
```

### Step 4: Initialize Vector Database

```bash
# Create ChromaDB and populate with initial data
python scripts/populate_vectordb.py
```

### Step 5: Setup MCP Servers (Optional)

```bash
# Setup local MCP servers for testing
python scripts/setup_mcp_servers.py
```

## Quick Start

### Basic Usage

```python
from mcp_agent import MCPAgent

# Initialize the agent
agent = MCPAgent()

# Perform a semantic search for bioinformatics tools
result = await agent.search("tools for RNA sequencing analysis")
print(result)
```

### Advanced Workflow

```python
from mcp_agent.graph import create_workflow

# Create and run the full workflow
workflow = create_workflow()
result = await workflow.ainvoke({
    "query": "Find Python libraries for protein structure analysis",
    "max_results": 10,
    "include_documentation": True
})
```

## Architecture Overview

### Agent Hierarchy

1. **Coordinator Agent**: Orchestrates the overall workflow and manages communication between specialized agents
2. **Researcher Agent**: Handles tool discovery, web search, and MCP server communication
3. **Reporter Agent**: Generates comprehensive reports and summaries

### MCP Integration

The framework supports multiple MCP servers:
- **Local Tool Server**: Custom bioinformatics tool catalog
- **Web Search Server**: Enhanced web search capabilities
- **Documentation Server**: Access to tool documentation and examples

### Vector Database

ChromaDB stores:
- Tool embeddings for semantic search
- Documentation chunks
- Usage examples and code snippets
- User query history and preferences

## Configuration

### Agent Configuration (`src/mcp_agent/config/settings.py`)

```python
class AgentSettings:
    """Agent configuration settings."""
    
    # LLM Settings
    llm_model: str = "gemini-1.5-pro"
    temperature: float = 0.1
    max_tokens: int = 4096
    
    # Vector Store Settings
    chroma_collection: str = "bioinformatics_tools"
    embedding_model: str = "text-embedding-3-small"
    
    # MCP Settings
    mcp_servers: Dict[str, MCPServerConfig]
```

### MCP Server Configuration

```yaml
# conf.yaml.example
mcp_servers:
  bioinformatics:
    command: "python"
    args: ["servers/bio_tools_server.py"]
    transport: "stdio"
  
  web_search:
    url: "http://localhost:8000/mcp"
    transport: "streamable_http"
```

## Development Guidelines

### Type Hints

All functions must include comprehensive type hints:

```python
from typing import List, Dict, Optional, Union
from pydantic import BaseModel

async def search_tools(
    query: str,
    max_results: int = 10,
    filters: Optional[Dict[str, str]] = None
) -> List[ToolResult]:
    """Search for bioinformatics tools based on query."""
    pass
```

### Documentation Standards

Use Sphinx-style docstrings:

```python
def process_query(query: str, context: Dict[str, Any]) -> ProcessedQuery:
    """Process a user query for tool discovery.
    
    Args:
        query: The user's search query
        context: Additional context information
        
    Returns:
        ProcessedQuery: Processed and enriched query object
        
    Raises:
        ValueError: If query is empty or invalid
        
    Examples:
        >>> result = process_query("RNA-seq tools", {"organism": "human"})
        >>> print(result.enhanced_query)
        "RNA sequencing analysis tools for human organisms"
    """
    pass
```

### Code Quality

- **Linting**: Use ruff for code formatting and linting
- **Type Checking**: Run mypy for static type analysis
- **Testing**: Maintain >90% test coverage with pytest
- **Documentation**: Generate docs with Sphinx

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/mcp_agent --cov-report=html

# Run specific test categories
pytest tests/test_agents/
pytest tests/test_integration/
```

## Usage Examples

### 1. Simple Tool Search

```python
from mcp_agent import MCPAgent

agent = MCPAgent()
results = await agent.search("BLAST alternatives for protein alignment")
```

### 2. Complex Research Workflow

```python
from mcp_agent.graph import create_workflow

workflow = create_workflow()
result = await workflow.ainvoke({
    "query": "Compare machine learning frameworks for genomics",
    "research_depth": "comprehensive",
    "output_format": "report"
})
```

### 3. Custom MCP Server Integration

```python
from mcp_agent.tools import MCPClient

client = MCPClient()
await client.add_server("custom_bio_db", {
    "command": "python",
    "args": ["custom_bio_server.py"],
    "transport": "stdio"
})
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a Pull Request

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run quality checks
make lint
make type-check
make test
```

## Roadmap

### Phase 1: Core Agent (Current)
- [x] Basic MCP integration
- [x] ChromaDB vector store
- [x] Google Gemini integration
- [ ] Simple workflow implementation

### Phase 2: Enhanced Search
- [ ] Advanced semantic search
- [ ] Multi-modal tool discovery
- [ ] Real-time index updates
- [ ] User preference learning

### Phase 3: Advanced Features
- [ ] Multi-agent collaboration
- [ ] Custom MCP server creation
- [ ] API endpoints
- [ ] Web UI interface

## Troubleshooting

### Common Issues

1. **MCP Connection Errors**
   ```bash
   # Check server status
   python scripts/check_mcp_servers.py
   ```

2. **ChromaDB Initialization**
   ```bash
   # Reset vector database
   rm -rf ./data/chroma
   python scripts/populate_vectordb.py
   ```

3. **Gemini API Issues**
   ```bash
   # Verify API key
   python -c "from langchain_google_genai import ChatGoogleGenerativeAI; print('API key valid')"
   ```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **DeerFlow**: Architecture inspiration from ByteDance's framework
- **LangChain**: Core LLM orchestration
- **Anthropic**: MCP protocol specification
- **Google**: Gemini LLM capabilities
- **ChromaDB**: Vector database implementation

## Support

For questions and support:
- GitHub Issues: [Create an issue]
- Documentation: [Read the docs]
- Community: [Join discussions]

---

