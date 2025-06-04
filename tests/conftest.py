"""Pytest configuration and fixtures for the MCP Agent Framework tests.

This module provides pytest configuration, shared fixtures, and test utilities
for comprehensive testing of all MCP Agent Framework components.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Using fixtures in tests:

    >>> def test_search_functionality(test_settings, mock_vector_store):
    ...     # Test implementation using fixtures
    ...     pass
    
    >>> @pytest.mark.asyncio
    ... async def test_async_workflow(event_loop, test_agent):
    ...     result = await test_agent.process("test query")
    ...     assert result["success"]
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch

import pytest
import pytest_asyncio

# Import test utilities and framework components
try:
    from tests import (
        setup_test_environment,
        cleanup_test_environment,
        create_test_settings,
        MockMCPClient,
        MockVectorStore,
        MockAgent,
        MockWorkflow,
        MockLLM,
        sample_search_query,
        sample_search_results,
        sample_tool_info,
        SAMPLE_TOOLS,
        SAMPLE_QUERIES,
    )
except ImportError:
    # Fallback imports for development
    def setup_test_environment():
        return {}
    
    def cleanup_test_environment():
        return 0
    
    def create_test_settings(**kwargs):
        return {}


# ===== Pytest Configuration =====

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "mcp: marks tests related to MCP functionality"
    )
    config.addinivalue_line(
        "markers", "llm: marks tests that require LLM API calls"
    )
    config.addinivalue_line(
        "markers", "vector: marks tests related to vector database"
    )
    config.addinivalue_line(
        "markers", "agents: marks tests related to agent functionality"
    )
    config.addinivalue_line(
        "markers", "workflows: marks tests related to workflow execution"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance/benchmark tests"
    )
    config.addinivalue_line(
        "markers", "api_required: marks tests that require external API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test characteristics."""
    for item in items:
        # Auto-mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Auto-mark tests based on file location
        test_file = str(item.fspath)
        
        if "test_agents" in test_file:
            item.add_marker(pytest.mark.agents)
        elif "test_tools" in test_file:
            item.add_marker(pytest.mark.mcp)
        elif "test_integration" in test_file:
            item.add_marker(pytest.mark.integration)
        elif "test_performance" in test_file:
            item.add_marker(pytest.mark.performance)
        
        # Auto-mark tests that use certain fixtures as requiring APIs
        if any(fixture in item.fixturenames for fixture in ["llm_client", "real_mcp_client"]):
            item.add_marker(pytest.mark.api_required)


def pytest_runtest_setup(item):
    """Setup hook for individual test runs."""
    # Skip integration tests if not explicitly enabled
    if item.get_closest_marker("integration"):
        if not item.config.getoption("--run-integration", default=False):
            pytest.skip("Integration tests disabled (use --run-integration)")
    
    # Skip performance tests if not explicitly enabled
    if item.get_closest_marker("performance"):
        if not item.config.getoption("--run-performance", default=False):
            pytest.skip("Performance tests disabled (use --run-performance)")
    
    # Skip API tests if no API keys available
    if item.get_closest_marker("api_required"):
        required_keys = ["GOOGLE_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            pytest.skip(f"API tests require environment variables: {missing_keys}")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--run-performance",
        action="store_true", 
        default=False,
        help="Run performance tests"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--test-env",
        action="store",
        default="test",
        choices=["test", "development", "ci"],
        help="Test environment configuration"
    )


# ===== Session-level Fixtures =====

@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment for the entire test session."""
    env_info = setup_test_environment()
    yield env_info
    cleanup_test_environment()


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config(request):
    """Provide test configuration based on command line options."""
    return {
        "environment": request.config.getoption("--test-env"),
        "run_integration": request.config.getoption("--run-integration"),
        "run_performance": request.config.getoption("--run-performance"),
        "run_slow": request.config.getoption("--run-slow"),
        "temp_dir": tempfile.gettempdir(),
    }


# ===== Core Component Fixtures =====

@pytest.fixture
def test_settings(test_environment):
    """Provide test-optimized settings."""
    return create_test_settings(
        chromadb_path=str(Path(test_environment["temp_dir"]) / "test_chroma"),
        log_level="WARNING",
        enable_file_logging=False,
    )


@pytest.fixture
def test_settings_with_logging(test_environment):
    """Provide test settings with logging enabled for debugging."""
    return create_test_settings(
        chromadb_path=str(Path(test_environment["temp_dir"]) / "test_chroma"),
        log_level="DEBUG",
        enable_file_logging=True,
        log_file=str(Path(test_environment["logs_dir"]) / "test_debug.log"),
    )


# ===== Mock Component Fixtures =====

@pytest.fixture
async def mock_mcp_client():
    """Provide mock MCP client for testing."""
    client = MockMCPClient()
    await client.initialize()
    yield client
    await client.close()


@pytest.fixture
async def mock_vector_store():
    """Provide mock vector store for testing."""
    store = MockVectorStore()
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def mock_llm():
    """Provide mock LLM for testing."""
    return MockLLM()


@pytest.fixture
async def mock_coordinator_agent(test_settings, mock_mcp_client, mock_vector_store):
    """Provide mock coordinator agent."""
    agent = MockAgent("coordinator")
    await agent.initialize()
    yield agent
    await agent.shutdown()


@pytest.fixture
async def mock_researcher_agent(test_settings, mock_mcp_client, mock_vector_store):
    """Provide mock researcher agent."""
    agent = MockAgent("researcher")
    await agent.initialize()
    yield agent
    await agent.shutdown()


@pytest.fixture
async def mock_reporter_agent(test_settings):
    """Provide mock reporter agent."""
    agent = MockAgent("reporter")
    await agent.initialize()
    yield agent
    await agent.shutdown()


@pytest.fixture
def mock_workflow():
    """Provide mock workflow for testing."""
    return MockWorkflow()


# ===== Real Component Fixtures (for integration tests) =====

@pytest.fixture
@pytest.mark.api_required
async def real_mcp_client(test_settings):
    """Provide real MCP client for integration testing."""
    try:
        from mcp_agent.tools.mcp_client import MCPClient
        
        client = MCPClient(test_settings)
        await client.initialize()
        yield client
        await client.close()
        
    except ImportError:
        pytest.skip("MCP client not available")


@pytest.fixture
@pytest.mark.integration
async def real_vector_store(test_settings, test_environment):
    """Provide real vector store for integration testing."""
    try:
        from mcp_agent.tools.vector_store import VectorStore
        
        # Use test-specific path
        test_settings.chromadb_path = str(
            Path(test_environment["temp_dir"]) / "integration_chroma"
        )
        
        store = VectorStore(test_settings)
        await store.initialize()
        
        # Add some test data
        for tool in SAMPLE_TOOLS:
            await store.add_document({
                "id": tool["name"],
                "content": f"{tool['name']}: {tool['description']}",
                "metadata": tool,
            })
        
        yield store
        await store.close()
        
    except ImportError:
        pytest.skip("Vector store not available")


@pytest.fixture
@pytest.mark.api_required
async def real_llm_client(test_settings):
    """Provide real LLM client for API testing."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("Requires GOOGLE_API_KEY environment variable")
        
        llm = ChatGoogleGenerativeAI(
            model=test_settings.llm_model,
            temperature=test_settings.llm_temperature,
            max_tokens=test_settings.llm_max_tokens,
        )
        
        yield llm
        
    except ImportError:
        pytest.skip("Google Generative AI not available")


# ===== Data Fixtures =====

@pytest.fixture
def sample_query():
    """Provide sample search query."""
    return sample_search_query()


@pytest.fixture
def sample_queries():
    """Provide multiple sample queries."""
    return [sample_search_query(text=query) for query in SAMPLE_QUERIES[:5]]


@pytest.fixture
def sample_results():
    """Provide sample search results."""
    return sample_search_results()


@pytest.fixture
def sample_tool():
    """Provide sample tool information."""
    return sample_tool_info()


@pytest.fixture
def sample_tools():
    """Provide multiple sample tools."""
    return SAMPLE_TOOLS.copy()


@pytest.fixture(params=SAMPLE_QUERIES[:3])
def parametrized_query(request):
    """Parametrized fixture for testing with different queries."""
    return sample_search_query(text=request.param)


@pytest.fixture(params=["genomics", "proteomics", "transcriptomics"])
def parametrized_category(request):
    """Parametrized fixture for testing with different categories."""
    return request.param


# ===== File System Fixtures =====

@pytest.fixture
def temp_dir():
    """Provide temporary directory that auto-cleans."""
    temp_path = Path(tempfile.mkdtemp(prefix="mcp_test_"))
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def test_data_dir(temp_dir):
    """Provide test data directory with sample files."""
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    
    # Create sample files
    (data_dir / "sample_config.yaml").write_text("""
    llm_model: gemini-1.5-flash
    log_level: DEBUG
    max_search_results: 5
    """)
    
    (data_dir / "sample_tools.json").write_text(
        json.dumps(SAMPLE_TOOLS, indent=2)
    )
    
    yield data_dir


@pytest.fixture
def mock_file_system(temp_dir):
    """Provide mock file system with predictable structure."""
    fs_structure = {
        "config": {
            "agent_config.yaml": "llm_model: test-model",
            "mcp_servers.yaml": "servers: []",
        },
        "data": {
            "tools.json": json.dumps(SAMPLE_TOOLS),
            "cache": {},
        },
        "logs": {},
    }
    
    def create_structure(base_path: Path, structure: dict):
        for name, content in structure.items():
            path = base_path / name
            if isinstance(content, dict):
                path.mkdir(exist_ok=True)
                create_structure(path, content)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(content))
    
    create_structure(temp_dir, fs_structure)
    yield temp_dir


# ===== Database Fixtures =====

@pytest.fixture
async def empty_vector_db(test_settings, temp_dir):
    """Provide empty vector database for testing."""
    try:
        from mcp_agent.tools.vector_store import VectorStore
        
        test_settings.chromadb_path = str(temp_dir / "empty_chroma")
        store = VectorStore(test_settings)
        await store.initialize()
        
        yield store
        await store.close()
        
    except ImportError:
        # Fallback to mock
        store = MockVectorStore()
        store.documents = []  # Ensure it's empty
        await store.initialize()
        yield store
        await store.close()


@pytest.fixture
async def populated_vector_db(test_settings, temp_dir):
    """Provide vector database populated with test data."""
    try:
        from mcp_agent.tools.vector_store import VectorStore
        
        test_settings.chromadb_path = str(temp_dir / "populated_chroma")
        store = VectorStore(test_settings)
        await store.initialize()
        
        # Add test documents
        for i, tool in enumerate(SAMPLE_TOOLS):
            await store.add_document({
                "id": f"tool_{i}",
                "content": f"{tool['name']}: {tool['description']}",
                "metadata": tool,
            })
        
        yield store
        await store.close()
        
    except ImportError:
        # Fallback to mock
        store = MockVectorStore()
        await store.initialize()
        yield store
        await store.close()


# ===== Network and API Fixtures =====

@pytest.fixture
def mock_api_responses():
    """Provide mock API responses for external services."""
    from unittest.mock import patch
    
    responses = {
        "google_gemini": {
            "generate_content": "This is a mock response from Google Gemini API.",
        },
        "tavily_search": {
            "search": {
                "results": [
                    {
                        "title": "Mock Search Result",
                        "url": "https://example.com/mock",
                        "content": "Mock search result content",
                    }
                ]
            }
        },
    }
    
    with patch.multiple(
        "requests",
        get=Mock(return_value=Mock(json=lambda: responses)),
        post=Mock(return_value=Mock(json=lambda: responses)),
    ):
        yield responses


@pytest.fixture
def network_disabled():
    """Disable network access for testing offline behavior."""
    import socket
    
    def guard(*args, **kwargs):
        raise OSError("Network access disabled in tests")
    
    with patch.object(socket, "socket", side_effect=guard):
        yield


# ===== Performance Testing Fixtures =====

@pytest.fixture
def performance_monitor():
    """Provide performance monitoring for tests."""
    from tests import ProfiledTest
    
    class PerformanceMonitor:
        def __init__(self):
            self.measurements = []
        
        def measure(self, operation_name: str):
            return ProfiledTest(operation_name)
        
        def assert_threshold(self, duration_ms: float, threshold_ms: float):
            assert duration_ms <= threshold_ms, (
                f"Performance threshold exceeded: {duration_ms:.2f}ms > {threshold_ms:.2f}ms"
            )
    
    return PerformanceMonitor()


@pytest.fixture
def memory_monitor():
    """Provide memory usage monitoring for tests."""
    try:
        import psutil
        import os
        
        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.baseline = self.current_usage()
            
            def current_usage(self) -> float:
                """Get current memory usage in MB."""
                return self.process.memory_info().rss / (1024 * 1024)
            
            def usage_since_baseline(self) -> float:
                """Get memory usage change since baseline in MB."""
                return self.current_usage() - self.baseline
            
            def assert_no_memory_leak(self, threshold_mb: float = 50.0):
                """Assert that memory usage hasn't increased significantly."""
                delta = self.usage_since_baseline()
                assert delta <= threshold_mb, (
                    f"Potential memory leak detected: {delta:.2f}MB increase"
                )
        
        return MemoryMonitor()
        
    except ImportError:
        # Fallback mock monitor
        class MockMemoryMonitor:
            def current_usage(self):
                return 0.0
            
            def usage_since_baseline(self):
                return 0.0
            
            def assert_no_memory_leak(self, threshold_mb=50.0):
                pass
        
        return MockMemoryMonitor()


# ===== Error Simulation Fixtures =====

@pytest.fixture
def simulate_network_errors():
    """Simulate various network errors for testing resilience."""
    import random
    from unittest.mock import patch
    import httpx
    
    def error_generator(*args, **kwargs):
        error_types = [
            httpx.TimeoutException("Mock timeout"),
            httpx.NetworkError("Mock network error"),
            httpx.HTTPStatusError("Mock HTTP error", request=None, response=None),
        ]
        raise random.choice(error_types)
    
    with patch("httpx.AsyncClient.get", side_effect=error_generator), \
         patch("httpx.AsyncClient.post", side_effect=error_generator):
        yield


@pytest.fixture
def simulate_mcp_errors():
    """Simulate MCP server errors for testing error handling."""
    from unittest.mock import patch
    
    def mcp_error(*args, **kwargs):
        raise ConnectionError("Mock MCP server connection error")
    
    with patch("mcp_agent.tools.mcp_client.MCPClient.call_tool", side_effect=mcp_error):
        yield


# ===== Specialized Test Fixtures =====

@pytest.fixture
def bioinformatics_queries():
    """Provide bioinformatics-specific test queries."""
    return [
        "RNA-seq differential expression analysis",
        "protein structure prediction AlphaFold",
        "genome variant calling GATK",
        "phylogenetic tree maximum likelihood",
        "metabolomics pathway analysis",
        "CRISPR guide RNA design",
        "protein-protein interaction networks",
        "single-cell RNA sequencing clustering",
    ]


@pytest.fixture
def mock_bioinformatics_databases():
    """Provide mock responses for bioinformatics databases."""
    return {
        "pubmed": {
            "search_results": [
                {
                    "pmid": "12345678",
                    "title": "Mock PubMed Article",
                    "abstract": "This is a mock abstract for testing.",
                    "authors": ["Mock A.", "Test B."],
                }
            ]
        },
        "uniprot": {
            "protein_info": {
                "id": "P12345",
                "name": "Mock Protein",
                "sequence": "MOCKSEQUENCE",
                "organism": "Test organism",
            }
        },
        "ensembl": {
            "gene_info": {
                "id": "ENSG00000000001",
                "symbol": "MOCK1",
                "description": "Mock gene for testing",
                "biotype": "protein_coding",
            }
        },
    }


# ===== Cleanup and Teardown =====

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatically cleanup after each test."""
    yield
    
    # Clean up any global state
    import gc
    gc.collect()
    
    # Reset any environment variables that might have been modified
    test_env_vars = [
        "MCP_AGENT_TEST_MODE",
        "MOCK_API_RESPONSES", 
        "DISABLE_EXTERNAL_CALLS",
    ]
    
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


def pytest_sessionfinish(session, exitstatus):
    """Cleanup after entire test session."""
    # Final cleanup
    cleanup_test_environment()
    
    # Print test summary if verbose
    if session.config.getoption("verbose") > 0:
        print(f"\nTest session completed with exit status: {exitstatus}")


# ===== Utility Functions for Tests =====

def skip_if_no_api_key(key_name: str):
    """Skip test if API key is not available."""
    return pytest.mark.skipif(
        not os.getenv(key_name),
        reason=f"Requires {key_name} environment variable"
    )


def requires_slow_tests():
    """Skip test unless slow tests are enabled."""
    return pytest.mark.skipif(
        not pytest.config.getoption("--run-slow", default=False),
        reason="Slow test (use --run-slow to enable)"
    )


# Import the test utilities to make them available
import json

# Mark this module as providing pytest configuration
__all__ = [
    "test_environment",
    "test_settings", 
    "mock_mcp_client",
    "mock_vector_store",
    "mock_llm",
    "sample_query",
    "sample_results",
    "temp_dir",
    "performance_monitor",
    "memory_monitor",
]