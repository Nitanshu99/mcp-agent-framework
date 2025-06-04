"""Test package for the MCP Agent Framework.

This package provides testing utilities, fixtures, and helper functions for testing
all components of the MCP Agent Framework including agents, tools, models, and workflows.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic test setup:

    >>> from tests import create_test_settings, MockMCPClient
    >>> settings = create_test_settings()
    >>> mock_client = MockMCPClient()
    
    Using test fixtures:
    
    >>> from tests import sample_search_results, mock_vector_store
    >>> results = sample_search_results()
    >>> vector_store = mock_vector_store()
"""

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock, MagicMock
import uuid

# Import test dependencies
try:
    import pytest
    import pytest_asyncio
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

# Test configuration and utilities
__all__ = [
    # Test configuration
    "create_test_settings",
    "get_test_config",
    "setup_test_environment",
    "cleanup_test_environment",
    
    # Mock objects
    "MockMCPClient",
    "MockVectorStore", 
    "MockAgent",
    "MockWorkflow",
    "MockLLM",
    
    # Test data generators
    "sample_search_query",
    "sample_search_results",
    "sample_tool_info",
    "sample_agent_response",
    "generate_test_data",
    
    # Test utilities
    "async_test",
    "timeout_test",
    "assert_valid_search_result",
    "assert_valid_tool_info",
    "compare_search_results",
    
    # Test fixtures
    "temp_directory",
    "test_database",
    "mock_api_responses",
    
    # Performance testing
    "measure_test_performance",
    "assert_performance_threshold",
    "ProfiledTest",
    
    # Integration testing
    "integration_test",
    "requires_api_key",
    "requires_network",
    
    # Constants
    "TEST_DATA_DIR",
    "SAMPLE_QUERIES",
    "SAMPLE_TOOLS",
]

# Test constants
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEMP_TEST_DIR = Path(tempfile.gettempdir()) / "mcp_agent_tests"

# Sample test data
SAMPLE_QUERIES = [
    "RNA sequencing analysis tools",
    "protein structure prediction software", 
    "genome assembly algorithms",
    "BLAST sequence alignment",
    "phylogenetic tree construction",
    "gene expression analysis",
    "variant calling tools",
    "metabolomics data processing",
]

SAMPLE_TOOLS = [
    {
        "name": "BLAST+",
        "description": "Basic Local Alignment Search Tool for sequence similarity search",
        "category": "sequence_analysis",
        "organism": "universal",
        "url": "https://blast.ncbi.nlm.nih.gov/",
        "language": "C++",
        "license": "Public Domain",
    },
    {
        "name": "BWA",
        "description": "Burrows-Wheeler Aligner for mapping DNA sequences",
        "category": "sequence_alignment", 
        "organism": "universal",
        "url": "http://bio-bwa.sourceforge.net/",
        "language": "C",
        "license": "MIT",
    },
    {
        "name": "SAMtools",
        "description": "Tools for manipulating alignments in SAM/BAM format",
        "category": "file_manipulation",
        "organism": "universal", 
        "url": "http://samtools.sourceforge.net/",
        "language": "C",
        "license": "MIT",
    },
]


def create_test_settings(**overrides: Any) -> Any:
    """Create test-optimized settings configuration.
    
    Args:
        **overrides: Settings to override from defaults.
        
    Returns:
        AgentSettings: Test-configured settings object.
        
    Example:
        >>> settings = create_test_settings(log_level="DEBUG")
        >>> settings = create_test_settings(chromadb_path="./test_data/chroma")
    """
    try:
        from mcp_agent.config.settings import AgentSettings
        
        # Test-optimized defaults
        test_defaults = {
            # Use faster models for testing
            "llm_model": "gemini-1.5-flash",
            "llm_temperature": 0.0,  # Deterministic responses
            "llm_max_tokens": 1024,
            
            # Test database settings
            "chromadb_path": str(TEMP_TEST_DIR / "chroma"),
            "chromadb_collection": "test_bioinformatics_tools",
            
            # Reduced timeouts for faster tests
            "mcp_timeout": 10,
            "mcp_max_retries": 1,
            
            # Minimal search settings
            "max_search_results": 5,
            "search_similarity_threshold": 0.5,
            
            # Test logging
            "log_level": "WARNING",
            "enable_file_logging": False,
            
            # Disable external services
            "enable_hybrid_search": False,
            "cache_enabled": False,
            
            # Performance settings for tests
            "max_concurrent_requests": 2,
            "batch_size": 8,
        }
        
        # Merge with overrides
        test_defaults.update(overrides)
        
        return AgentSettings(**test_defaults)
        
    except ImportError:
        # Return mock settings for development
        class MockSettings:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        return MockSettings(**overrides)


def get_test_config() -> Dict[str, Any]:
    """Get test configuration dictionary.
    
    Returns:
        Dict[str, Any]: Test configuration settings.
        
    Example:
        >>> config = get_test_config()
        >>> print(config["test_timeout"])
    """
    return {
        "test_timeout": 30,
        "async_test_timeout": 60,
        "performance_threshold_ms": 1000,
        "max_memory_usage_mb": 512,
        "temp_dir": str(TEMP_TEST_DIR),
        "test_data_dir": str(TEST_DATA_DIR),
        "mock_api_responses": True,
        "enable_integration_tests": False,
        "enable_performance_tests": False,
    }


def setup_test_environment() -> Dict[str, Any]:
    """Set up test environment with temporary directories and configuration.
    
    Returns:
        Dict[str, Any]: Environment setup information.
        
    Example:
        >>> env_info = setup_test_environment()
        >>> print(f"Test directory: {env_info['temp_dir']}")
    """
    # Create temporary directories
    TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)
    (TEMP_TEST_DIR / "data").mkdir(exist_ok=True)
    (TEMP_TEST_DIR / "logs").mkdir(exist_ok=True)
    (TEMP_TEST_DIR / "cache").mkdir(exist_ok=True)
    (TEMP_TEST_DIR / "chroma").mkdir(exist_ok=True)
    
    # Set up test logging
    try:
        from mcp_agent.utils import setup_logger
        setup_logger(
            level="WARNING",
            log_file=str(TEMP_TEST_DIR / "logs" / "test.log"),
            enable_file_logging=True,
            enable_json_logging=False,
        )
    except ImportError:
        import logging
        logging.basicConfig(level=logging.WARNING)
    
    return {
        "temp_dir": str(TEMP_TEST_DIR),
        "data_dir": str(TEMP_TEST_DIR / "data"),
        "logs_dir": str(TEMP_TEST_DIR / "logs"),
        "cache_dir": str(TEMP_TEST_DIR / "cache"),
        "chroma_dir": str(TEMP_TEST_DIR / "chroma"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def cleanup_test_environment() -> int:
    """Clean up test environment and temporary files.
    
    Returns:
        int: Number of files/directories cleaned up.
        
    Example:
        >>> cleanup_count = cleanup_test_environment()
        >>> print(f"Cleaned up {cleanup_count} items")
    """
    import shutil
    
    cleanup_count = 0
    
    if TEMP_TEST_DIR.exists():
        try:
            shutil.rmtree(TEMP_TEST_DIR)
            cleanup_count = 1
        except Exception:
            # Try to clean individual files
            for item in TEMP_TEST_DIR.rglob("*"):
                try:
                    if item.is_file():
                        item.unlink()
                        cleanup_count += 1
                    elif item.is_dir() and not any(item.iterdir()):
                        item.rmdir()
                        cleanup_count += 1
                except Exception:
                    continue
    
    return cleanup_count


# Mock Objects
class MockMCPClient:
    """Mock MCP client for testing.
    
    Example:
        >>> client = MockMCPClient()
        >>> await client.initialize()
        >>> tools = await client.list_tools()
    """
    
    def __init__(self, **kwargs):
        self.servers = {}
        self.tools = SAMPLE_TOOLS.copy()
        self.initialized = False
        self.call_count = 0
    
    async def initialize(self) -> None:
        """Mock initialization."""
        self.initialized = True
    
    async def add_server(self, name: str, config: Dict[str, Any]) -> bool:
        """Mock server addition."""
        self.servers[name] = config
        return True
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """Mock tool listing."""
        self.call_count += 1
        return self.tools
    
    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Mock tool calling."""
        self.call_count += 1
        return {
            "tool": tool_name,
            "result": f"Mock result for {tool_name}",
            "success": True,
            "kwargs": kwargs,
        }
    
    async def close(self) -> None:
        """Mock cleanup."""
        self.initialized = False


class MockVectorStore:
    """Mock vector store for testing.
    
    Example:
        >>> store = MockVectorStore()
        >>> await store.initialize()
        >>> results = await store.search("RNA-seq")
    """
    
    def __init__(self, **kwargs):
        self.documents = []
        self.initialized = False
        self.search_count = 0
    
    async def initialize(self) -> None:
        """Mock initialization."""
        self.initialized = True
        # Add sample documents
        for tool in SAMPLE_TOOLS:
            self.documents.append({
                "id": tool["name"],
                "content": f"{tool['name']}: {tool['description']}",
                "metadata": tool,
            })
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Mock search."""
        self.search_count += 1
        
        # Simple mock search based on query keywords
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            content_lower = doc["content"].lower()
            if any(word in content_lower for word in query_lower.split()):
                results.append({
                    "document": doc,
                    "score": 0.8,  # Mock similarity score
                    "metadata": doc["metadata"],
                })
                
                if len(results) >= max_results:
                    break
        
        return results
    
    async def add_document(self, document: Dict[str, Any]) -> str:
        """Mock document addition."""
        doc_id = document.get("id", str(uuid.uuid4()))
        self.documents.append(document)
        return doc_id
    
    async def close(self) -> None:
        """Mock cleanup."""
        self.initialized = False


class MockAgent:
    """Mock agent for testing.
    
    Example:
        >>> agent = MockAgent("coordinator")
        >>> response = await agent.process("search for tools")
    """
    
    def __init__(self, agent_type: str = "mock", **kwargs):
        self.agent_type = agent_type
        self.initialized = False
        self.process_count = 0
    
    async def initialize(self) -> None:
        """Mock initialization."""
        self.initialized = True
    
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """Mock processing."""
        self.process_count += 1
        return {
            "agent": self.agent_type,
            "input": input_data,
            "result": f"Mock result from {self.agent_type}",
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    async def shutdown(self) -> None:
        """Mock shutdown."""
        self.initialized = False


class MockWorkflow:
    """Mock workflow for testing.
    
    Example:
        >>> workflow = MockWorkflow()
        >>> result = await workflow.ainvoke({"query": "test"})
    """
    
    def __init__(self, **kwargs):
        self.execution_count = 0
        self.last_input = None
        self.last_config = None
    
    async def ainvoke(
        self,
        input_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mock workflow execution."""
        self.execution_count += 1
        self.last_input = input_data
        self.last_config = config
        
        return {
            "input": input_data,
            "config": config,
            "result": "Mock workflow result",
            "execution_count": self.execution_count,
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class MockLLM:
    """Mock LLM for testing.
    
    Example:
        >>> llm = MockLLM()
        >>> response = await llm.agenerate([["What are RNA-seq tools?"]])
    """
    
    def __init__(self, **kwargs):
        self.call_count = 0
        self.responses = [
            "RNA-seq tools include STAR, HISAT2, and TopHat for alignment.",
            "Popular tools for protein analysis include BLAST, PSI-BLAST, and HMMER.",
            "Genome assembly tools include SPAdes, MEGAHIT, and Canu.",
        ]
    
    async def agenerate(self, prompts: List[List[str]], **kwargs) -> Any:
        """Mock LLM generation."""
        self.call_count += 1
        
        # Return mock response
        response_text = self.responses[self.call_count % len(self.responses)]
        
        # Mock LangChain LLMResult structure
        class MockGeneration:
            def __init__(self, text: str):
                self.text = text
        
        class MockLLMResult:
            def __init__(self, generations: List[List[MockGeneration]]):
                self.generations = generations
        
        return MockLLMResult([[MockGeneration(response_text)]])


# Test Data Generators
def sample_search_query(**overrides: Any) -> Dict[str, Any]:
    """Generate sample search query for testing.
    
    Args:
        **overrides: Fields to override in the sample query.
        
    Returns:
        Dict[str, Any]: Sample search query data.
        
    Example:
        >>> query = sample_search_query(text="custom query")
        >>> query = sample_search_query(max_results=20)
    """
    query_data = {
        "text": "RNA sequencing analysis tools",
        "max_results": 10,
        "filters": {"category": "genomics"},
        "include_documentation": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": "test_user",
    }
    
    query_data.update(overrides)
    return query_data


def sample_search_results(num_results: int = 3) -> Dict[str, Any]:
    """Generate sample search results for testing.
    
    Args:
        num_results: Number of results to generate.
        
    Returns:
        Dict[str, Any]: Sample search results data.
        
    Example:
        >>> results = sample_search_results(5)
        >>> print(len(results["tools"]))  # 5
    """
    tools = SAMPLE_TOOLS[:num_results] if num_results <= len(SAMPLE_TOOLS) else SAMPLE_TOOLS
    
    return {
        "query": "RNA sequencing analysis tools",
        "tools": [sample_tool_info(**tool) for tool in tools],
        "total_results": num_results,
        "search_time_ms": 150.5,
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "search_type": "vector_similarity",
            "model": "text-embedding-3-small",
            "threshold": 0.7,
        },
    }


def sample_tool_info(**overrides: Any) -> Dict[str, Any]:
    """Generate sample tool information for testing.
    
    Args:
        **overrides: Fields to override in the sample tool.
        
    Returns:
        Dict[str, Any]: Sample tool information.
        
    Example:
        >>> tool = sample_tool_info(name="Custom Tool")
        >>> tool = sample_tool_info(category="proteomics")
    """
    tool_data = {
        "name": "BLAST+",
        "description": "Basic Local Alignment Search Tool for sequence similarity search",
        "category": "sequence_analysis",
        "organism": "universal",
        "url": "https://blast.ncbi.nlm.nih.gov/",
        "language": "C++",
        "license": "Public Domain",
        "version": "2.14.0",
        "publication_year": 1990,
        "citations": 50000,
        "documentation_url": "https://blast.ncbi.nlm.nih.gov/doc/blast-help/",
        "installation": {
            "conda": "conda install -c bioconda blast",
            "pip": None,
            "docker": "ncbi/blast",
        },
        "tags": ["alignment", "sequence", "similarity", "ncbi"],
        "similarity_score": 0.85,
    }
    
    tool_data.update(overrides)
    return tool_data


def sample_agent_response(**overrides: Any) -> Dict[str, Any]:
    """Generate sample agent response for testing.
    
    Args:
        **overrides: Fields to override in the sample response.
        
    Returns:
        Dict[str, Any]: Sample agent response data.
        
    Example:
        >>> response = sample_agent_response(agent="researcher")
        >>> response = sample_agent_response(success=False)
    """
    response_data = {
        "agent": "coordinator",
        "task": "search_tools",
        "result": {
            "summary": "Found 5 relevant bioinformatics tools for RNA sequencing analysis",
            "tools_found": 5,
            "categories": ["sequence_analysis", "alignment", "quality_control"],
            "recommendations": ["STAR", "HISAT2", "FastQC"],
        },
        "success": True,
        "error": None,
        "execution_time_ms": 250.7,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "workflow_id": "wf_12345",
            "step": 1,
            "total_steps": 3,
        },
    }
    
    response_data.update(overrides)
    return response_data


def generate_test_data(data_type: str, count: int = 1, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate test data of specified type.
    
    Args:
        data_type: Type of data to generate (query, results, tool, response).
        count: Number of items to generate.
        **kwargs: Additional parameters for data generation.
        
    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]: Generated test data.
        
    Example:
        >>> tools = generate_test_data("tool", count=5)
        >>> queries = generate_test_data("query", count=3, category="genomics")
    """
    generators = {
        "query": sample_search_query,
        "results": lambda **kw: sample_search_results(kw.get("num_results", 3)),
        "tool": sample_tool_info,
        "response": sample_agent_response,
    }
    
    if data_type not in generators:
        raise ValueError(f"Unknown data type: {data_type}. Available: {list(generators.keys())}")
    
    generator = generators[data_type]
    
    if count == 1:
        return generator(**kwargs)
    else:
        return [generator(**{**kwargs, "id": i}) for i in range(count)]


# Test Utilities
def async_test(timeout: float = 30.0):
    """Decorator for async test functions with timeout.
    
    Args:
        timeout: Test timeout in seconds.
        
    Example:
        >>> @async_test(timeout=60.0)
        ... async def test_long_operation():
        ...     await long_async_operation()
    """
    def decorator(func):
        if PYTEST_AVAILABLE:
            @pytest.mark.asyncio
            @pytest.mark.timeout(timeout)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
        else:
            async def wrapper(*args, **kwargs):
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


def timeout_test(seconds: float):
    """Decorator to add timeout to test functions.
    
    Args:
        seconds: Timeout in seconds.
        
    Example:
        >>> @timeout_test(30.0)
        ... def test_slow_operation():
        ...     # Test implementation
        ...     pass
    """
    def decorator(func):
        if PYTEST_AVAILABLE:
            return pytest.mark.timeout(seconds)(func)
        else:
            # Simple timeout wrapper for non-pytest environments
            import functools
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Test timed out after {seconds} seconds")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(seconds))
                
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
            
            return wrapper
    
    return decorator


# Assertion Helpers
def assert_valid_search_result(result: Dict[str, Any]) -> None:
    """Assert that a search result has valid structure and content.
    
    Args:
        result: Search result to validate.
        
    Raises:
        AssertionError: If result is invalid.
        
    Example:
        >>> result = sample_search_results()
        >>> assert_valid_search_result(result)
    """
    assert isinstance(result, dict), "Search result must be a dictionary"
    assert "query" in result, "Search result must have 'query' field"
    assert "tools" in result, "Search result must have 'tools' field"
    assert "success" in result, "Search result must have 'success' field"
    assert "timestamp" in result, "Search result must have 'timestamp' field"
    
    assert isinstance(result["tools"], list), "Tools must be a list"
    assert isinstance(result["success"], bool), "Success must be a boolean"
    
    # Validate individual tools
    for tool in result["tools"]:
        assert_valid_tool_info(tool)


def assert_valid_tool_info(tool: Dict[str, Any]) -> None:
    """Assert that tool information has valid structure.
    
    Args:
        tool: Tool information to validate.
        
    Raises:
        AssertionError: If tool info is invalid.
        
    Example:
        >>> tool = sample_tool_info()
        >>> assert_valid_tool_info(tool)
    """
    assert isinstance(tool, dict), "Tool must be a dictionary"
    
    required_fields = ["name", "description"]
    for field in required_fields:
        assert field in tool, f"Tool must have '{field}' field"
        assert isinstance(tool[field], str), f"Tool '{field}' must be a string"
        assert len(tool[field]) > 0, f"Tool '{field}' cannot be empty"
    
    # Optional but recommended fields
    recommended_fields = ["category", "url", "language"]
    for field in recommended_fields:
        if field in tool:
            assert isinstance(tool[field], str), f"Tool '{field}' must be a string"


def compare_search_results(result1: Dict[str, Any], result2: Dict[str, Any], tolerance: float = 0.1) -> Dict[str, Any]:
    """Compare two search results and return differences.
    
    Args:
        result1: First search result.
        result2: Second search result.
        tolerance: Tolerance for numeric comparisons.
        
    Returns:
        Dict[str, Any]: Comparison results with differences.
        
    Example:
        >>> r1 = sample_search_results(3)
        >>> r2 = sample_search_results(5)
        >>> diff = compare_search_results(r1, r2)
        >>> print(diff["tools_count_diff"])
    """
    comparison = {
        "identical": True,
        "differences": [],
        "similarities": [],
    }
    
    # Compare basic fields
    if result1.get("query") != result2.get("query"):
        comparison["identical"] = False
        comparison["differences"].append({
            "field": "query",
            "result1": result1.get("query"),
            "result2": result2.get("query"),
        })
    
    # Compare tool counts
    tools1_count = len(result1.get("tools", []))
    tools2_count = len(result2.get("tools", []))
    
    if tools1_count != tools2_count:
        comparison["identical"] = False
        comparison["differences"].append({
            "field": "tools_count",
            "result1": tools1_count,
            "result2": tools2_count,
        })
    
    # Compare timing if available
    time1 = result1.get("search_time_ms", 0)
    time2 = result2.get("search_time_ms", 0)
    
    if abs(time1 - time2) > tolerance * max(time1, time2, 1):
        comparison["differences"].append({
            "field": "search_time_ms",
            "result1": time1,
            "result2": time2,
            "difference": abs(time1 - time2),
        })
    
    # Compare tool names
    tools1_names = {tool.get("name") for tool in result1.get("tools", [])}
    tools2_names = {tool.get("name") for tool in result2.get("tools", [])}
    
    common_tools = tools1_names & tools2_names
    only_in_1 = tools1_names - tools2_names
    only_in_2 = tools2_names - tools1_names
    
    if common_tools:
        comparison["similarities"].append({
            "field": "common_tools",
            "tools": list(common_tools),
            "count": len(common_tools),
        })
    
    if only_in_1:
        comparison["identical"] = False
        comparison["differences"].append({
            "field": "tools_only_in_result1",
            "tools": list(only_in_1),
        })
    
    if only_in_2:
        comparison["identical"] = False
        comparison["differences"].append({
            "field": "tools_only_in_result2", 
            "tools": list(only_in_2),
        })
    
    return comparison


# Test Fixtures
def temp_directory() -> Generator[Path, None, None]:
    """Create temporary directory for testing.
    
    Yields:
        Path: Path to temporary directory.
        
    Example:
        >>> with temp_directory() as temp_dir:
        ...     test_file = temp_dir / "test.txt"
        ...     test_file.write_text("test data")
    """
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp(prefix="mcp_agent_test_"))
    try:
        yield temp_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


async def test_database() -> AsyncGenerator[MockVectorStore, None]:
    """Create test database for testing.
    
    Yields:
        MockVectorStore: Initialized mock vector store.
        
    Example:
        >>> async with test_database() as db:
        ...     results = await db.search("test query")
    """
    db = MockVectorStore()
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def mock_api_responses() -> Dict[str, Any]:
    """Get mock API responses for testing.
    
    Returns:
        Dict[str, Any]: Mock API responses by endpoint.
        
    Example:
        >>> responses = mock_api_responses()
        >>> gemini_response = responses["gemini"]["chat_completion"]
    """
    return {
        "gemini": {
            "chat_completion": {
                "choices": [{
                    "message": {
                        "content": "This is a mock response from Gemini API for testing purposes."
                    }
                }],
                "usage": {"total_tokens": 50},
            }
        },
        "mcp_server": {
            "list_tools": SAMPLE_TOOLS,
            "call_tool": {
                "result": "Mock tool execution result",
                "success": True,
            }
        },
        "vector_search": {
            "similarity_search": sample_search_results(3)["tools"],
        }
    }


# Performance Testing
class ProfiledTest:
    """Context manager for profiling test performance.
    
    Example:
        >>> with ProfiledTest("search_operation") as profiler:
        ...     # Perform operation
        ...     result = perform_search()
        >>> print(f"Duration: {profiler.duration_ms:.2f}ms")
    """
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_end = None
    
    def __enter__(self):
        import time
        
        self.start_time = time.time()
        
        # Try to measure memory
        try:
            from mcp_agent.utils.helpers import measure_memory_usage
            self.memory_start = measure_memory_usage()
        except (ImportError, NameError):
            pass
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        
        self.end_time = time.time()
        
        # Try to measure memory
        try:
            from mcp_agent.utils.helpers import measure_memory_usage
            self.memory_end = measure_memory_usage()
        except (ImportError, NameError):
            pass
    
    @property
    def duration_ms(self) -> float:
        """Get test duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0
    
    @property
    def memory_delta_mb(self) -> float:
        """Get memory usage change in MB."""
        if self.memory_start is not None and self.memory_end is not None:
            return self.memory_end - self.memory_start
        return 0.0


def measure_test_performance(func: Any):
    """Decorator to measure test function performance.
    
    Args:
        func: Test function to measure.
        
    Example:
        >>> @measure_test_performance
        ... def test_search_performance():
        ...     # Test implementation
        ...     pass
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with ProfiledTest(func.__name__) as profiler:
            result = func(*args, **kwargs)
        
        # Store performance metrics
        if not hasattr(wrapper, "_performance_metrics"):
            wrapper._performance_metrics = []
        
        wrapper._performance_metrics.append({
            "duration_ms": profiler.duration_ms,
            "memory_delta_mb": profiler.memory_delta_mb,
            "test_name": func.__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        return result
    
    return wrapper


def assert_performance_threshold(duration_ms: float, threshold_ms: float) -> None:
    """Assert that operation duration is within performance threshold.
    
    Args:
        duration_ms: Actual duration in milliseconds.
        threshold_ms: Maximum allowed duration in milliseconds.
        
    Raises:
        AssertionError: If duration exceeds threshold.
        
    Example:
        >>> with ProfiledTest("operation") as profiler:
        ...     # Perform operation
        ...     pass
        >>> assert_performance_threshold(profiler.duration_ms, 1000.0)
    """
    assert duration_ms <= threshold_ms, (
        f"Performance threshold exceeded: {duration_ms:.2f}ms > {threshold_ms:.2f}ms"
    )


# Integration Testing Decorators
def integration_test(func: Any):
    """Mark test as integration test.
    
    Example:
        >>> @integration_test
        ... def test_full_workflow():
        ...     # Integration test implementation
        ...     pass
    """
    if PYTEST_AVAILABLE:
        return pytest.mark.integration(func)
    else:
        func._is_integration_test = True
        return func


def requires_api_key(api_key_name: str):
    """Skip test if required API key is not available.
    
    Args:
        api_key_name: Name of required API key environment variable.
        
    Example:
        >>> @requires_api_key("GOOGLE_API_KEY")
        ... def test_gemini_integration():
        ...     # Test that requires Google API key
        ...     pass
    """
    def decorator(func):
        import os
        
        if PYTEST_AVAILABLE:
            return pytest.mark.skipif(
                not os.getenv(api_key_name),
                reason=f"Requires {api_key_name} environment variable"
            )(func)
        else:
            import functools
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not os.getenv(api_key_name):
                    print(f"Skipping {func.__name__}: requires {api_key_name}")
                    return
                return func(*args, **kwargs)
            
            return wrapper
    
    return decorator


def requires_network(func: Any):
    """Skip test if network is not available.
    
    Example:
        >>> @requires_network
        ... def test_external_api():
        ...     # Test that requires network access
        ...     pass
    """
    if PYTEST_AVAILABLE:
        return pytest.mark.skipif(
            not _check_network_available(),
            reason="Requires network access"
        )(func)
    else:
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _check_network_available():
                print(f"Skipping {func.__name__}: requires network access")
                return
            return func(*args, **kwargs)
        
        return wrapper


def _check_network_available() -> bool:
    """Check if network is available."""
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


# Package constants
TESTS_VERSION = "0.1.0"
DEFAULT_TEST_TIMEOUT = 30.0
DEFAULT_PERFORMANCE_THRESHOLD_MS = 1000.0

# Auto-setup test environment if not in pytest
import os
if not PYTEST_AVAILABLE and os.getenv("AUTO_SETUP_TESTS", "false").lower() == "true":
    try:
        setup_test_environment()
    except Exception:
        pass