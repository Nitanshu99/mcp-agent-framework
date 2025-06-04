====================
MCP Agent Framework
====================

**A basic Agent with Model Context Protocol (MCP) capabilities for bioinformatics tool discovery**

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code Style

.. image:: https://img.shields.io/badge/type%20checked-mypy-blue.svg
   :target: https://mypy.readthedocs.io/
   :alt: Type Checked

Welcome to the MCP Agent Framework documentation! This framework combines the power of Google Gemini LLM with Model Context Protocol (MCP) servers to create a semantic search engine for bioinformatics tools and research automation.

.. note::
   This project is inspired by ByteDance's DeerFlow architecture but simplified for educational and research purposes in bioinformatics tool discovery.

Overview
========

The MCP Agent Framework is a modular, extensible system that enables intelligent research automation through:

- **Multi-Agent Architecture**: Specialized agents for coordination, research, and reporting
- **MCP Integration**: Standardized protocol for connecting to external tools and data sources
- **Vector Search**: ChromaDB-powered semantic search for bioinformatics tools
- **Google Gemini**: Advanced language model for intelligent reasoning and analysis
- **Type Safety**: Comprehensive type hints and static analysis with mypy

Key Features
============

🤖 **Multi-Agent System**
   Coordinator, Researcher, and Reporter agents working in harmony

🔌 **MCP Protocol Support**
   Connect to any MCP-compliant server for tool integration

🧬 **Bioinformatics Focus**
   Specialized for discovering and analyzing bioinformatics tools

🔍 **Semantic Search**
   Vector-based search using ChromaDB for intelligent tool discovery

📊 **Human-in-the-Loop**
   Interactive workflows with user oversight and control

📚 **Comprehensive Documentation**
   Full API documentation with examples and tutorials

Quick Start
===========

Installation
------------

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd mcp-agent-framework

   # Install with uv
   uv sync

   # Configure environment
   cp .env.example .env
   # Edit .env with your API keys

Basic Usage
-----------

.. code-block:: python

   from mcp_agent import MCPAgent

   # Initialize the agent
   agent = MCPAgent()

   # Perform a semantic search
   result = await agent.search("tools for RNA sequencing analysis")
   print(result)

Advanced Workflow
-----------------

.. code-block:: python

   from mcp_agent.graph import create_workflow

   # Create and run the full workflow
   workflow = create_workflow()
   result = await workflow.ainvoke({
       "query": "Find Python libraries for protein structure analysis",
       "max_results": 10,
       "include_documentation": True
   })

Architecture
============

The framework follows a modular multi-agent architecture:

.. code-block:: text

   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │   Coordinator   │    │   Researcher    │    │    Reporter     │
   │                 │    │                 │    │                 │
   │ • Orchestrates  │◄──►│ • Tool Discovery│◄──►│ • Generate      │
   │   workflow      │    │ • Web Search    │    │   Reports       │
   │ • Manages state │    │ • MCP Clients   │    │ • Summarize     │
   │ • User interact │    │ • Vector Search │    │ • Format Output │
   └─────────────────┘    └─────────────────┘    └─────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                           ┌─────────────────┐
                           │   Vector Store  │
                           │                 │
                           │ • ChromaDB      │
                           │ • Embeddings    │
                           │ • Similarity    │
                           │   Search        │
                           └─────────────────┘

Components Overview
===================

Agents
------

**Coordinator Agent**
   Central orchestrator that manages the overall workflow, handles user interactions,
   and coordinates between specialized agents.

**Researcher Agent**
   Specialized in tool discovery, web search, and MCP server communication.
   Handles complex research tasks and data gathering.

**Reporter Agent**
   Generates comprehensive reports, summaries, and formatted outputs
   from research findings.

Tools
-----

**MCP Client**
   Wrapper for connecting to and communicating with MCP servers.
   Supports both stdio and HTTP transports.

**Vector Store**
   ChromaDB integration for semantic search and similarity matching
   of bioinformatics tools and documentation.

**Search Utilities**
   Enhanced search capabilities including web search APIs and
   custom search algorithms.

Models
------

**Schemas**
   Pydantic models for data validation and serialization.

**State Models**
   LangGraph state management for workflow persistence and tracking.

Getting Started
===============

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   configuration
   examples

User Guide
==========

.. toctree::
   :maxdepth: 2

   user_guide/basic_usage
   user_guide/advanced_workflows
   user_guide/mcp_integration
   user_guide/customization

Developer Guide
===============

.. toctree::
   :maxdepth: 2

   developer_guide/architecture
   developer_guide/contributing
   developer_guide/testing
   developer_guide/deployment

API Reference
=============

.. toctree::
   :maxdepth: 2

   api/agents
   api/tools
   api/models
   api/graph
   api/utils

Tutorials
=========

.. toctree::
   :maxdepth: 2

   tutorials/creating_custom_agents
   tutorials/mcp_server_development
   tutorials/vector_store_optimization
   tutorials/workflow_customization

Examples
========

.. toctree::
   :maxdepth: 2

   examples/basic_search
   examples/complex_research
   examples/custom_mcp_servers
   examples/batch_processing

Configuration
=============

.. toctree::
   :maxdepth: 2

   configuration/environment_variables
   configuration/mcp_servers
   configuration/vector_database
   configuration/logging

Troubleshooting
===============

.. toctree::
   :maxdepth: 2

   troubleshooting/common_issues
   troubleshooting/debugging
   troubleshooting/performance

FAQ
===

.. toctree::
   :maxdepth: 1

   faq

Contributing
============

We welcome contributions! Please see our contributing guidelines:

.. toctree::
   :maxdepth: 2

   contributing/guidelines
   contributing/development_setup
   contributing/testing
   contributing/documentation

Changelog
=========

.. toctree::
   :maxdepth: 1

   changelog

License
=======

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
===============

This project builds upon the incredible work of the open-source community:

- **ByteDance DeerFlow**: Architecture inspiration
- **LangChain & LangGraph**: Core orchestration framework
- **Anthropic**: Model Context Protocol specification
- **Google**: Gemini LLM capabilities
- **ChromaDB**: Vector database implementation

Support
=======

- **GitHub Issues**: `Report bugs and request features <https://github.com/your-org/mcp-agent-framework/issues>`_
- **Documentation**: This documentation site
- **Community**: Join our discussions

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note::
   This documentation is automatically generated from the source code.
   For the most up-to-date information, please refer to the GitHub repository.