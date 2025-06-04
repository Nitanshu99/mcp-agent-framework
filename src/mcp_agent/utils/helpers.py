"""Helper utilities and common functions for the MCP Agent Framework.

This module provides utility functions for data validation, text processing,
file operations, and other common tasks used throughout the framework.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic utility usage:

    >>> from mcp_agent.utils.helpers import sanitize_query, validate_email
    >>> clean_query = sanitize_query("RNA-seq tools & methods")
    >>> is_valid = validate_email("user@example.com")
    
    Async utilities:
    
    >>> from mcp_agent.utils.helpers import safe_async_call, timeout_after
    >>> result = await safe_async_call(some_async_function, "arg1", timeout=30)
    >>> 
    >>> @timeout_after(60)
    ... async def long_running_task():
    ...     # Task implementation
    ...     pass
"""

import asyncio
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple
from urllib.parse import urlparse
import unicodedata

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


# Text Processing Utilities
def sanitize_query(query: str, max_length: int = 1000) -> str:
    """Sanitize user input query for safe processing.
    
    Args:
        query: Raw user input query.
        max_length: Maximum allowed query length.
        
    Returns:
        str: Sanitized query string.
        
    Example:
        >>> sanitize_query("RNA-seq tools & <script>alert('xss')</script>")
        'RNA-seq tools & '
        >>> sanitize_query("protein   structure    analysis")  
        'protein structure analysis'
    """
    if not isinstance(query, str):
        query = str(query)
    
    # Remove HTML tags and potentially dangerous content
    query = re.sub(r'<[^>]+>', '', query)
    
    # Remove potentially dangerous characters but keep scientific notation
    query = re.sub(r'[<>"\';{}()\\]', '', query)
    
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Normalize unicode characters
    query = unicodedata.normalize('NFKC', query)
    
    # Truncate to max length
    if len(query) > max_length:
        query = query[:max_length].rsplit(' ', 1)[0]
    
    return query


def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
    """Extract relevant keywords from text for search enhancement.
    
    Args:
        text: Input text to extract keywords from.
        min_length: Minimum keyword length.
        max_keywords: Maximum number of keywords to return.
        
    Returns:
        List[str]: List of extracted keywords.
        
    Example:
        >>> keywords = extract_keywords("RNA sequencing analysis tools for genomics")
        >>> print(keywords)
        ['RNA', 'sequencing', 'analysis', 'tools', 'genomics']
    """
    if not text:
        return []
    
    # Common bioinformatics stop words to exclude
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
        'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'analysis', 'tool', 'tools', 'method',
        'methods', 'software', 'program', 'application', 'using', 'used'
    }
    
    # Extract words and filter
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9-]*\b', text.lower())
    keywords = [
        word for word in words 
        if len(word) >= min_length and word not in stop_words
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords[:max_keywords]


def format_tool_name(name: str) -> str:
    """Format tool names consistently.
    
    Args:
        name: Raw tool name.
        
    Returns:
        str: Formatted tool name.
        
    Example:
        >>> format_tool_name("blast_plus")
        'BLAST+'
        >>> format_tool_name("r-bioconductor")
        'R-Bioconductor'
    """
    if not name:
        return ""
    
    # Handle common bioinformatics tool naming conventions
    replacements = {
        'blast_plus': 'BLAST+',
        'blast+': 'BLAST+',
        'r-': 'R-',
        'python-': 'Python-',
        'bio': 'Bio',
        'rna': 'RNA',
        'dna': 'DNA',
        'seq': 'Seq',
        'fastq': 'FASTQ',
        'fasta': 'FASTA',
        'sam': 'SAM',
        'bam': 'BAM',
        'vcf': 'VCF',
        'gtf': 'GTF',
        'gff': 'GFF',
    }
    
    # Convert to lowercase for processing
    formatted = name.lower()
    
    # Apply specific replacements
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    # Capitalize first letter of each word
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    return formatted


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix.
    
    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to append when truncating.
        
    Returns:
        str: Truncated text.
        
    Example:
        >>> truncate_text("This is a very long description...", 20)
        'This is a very lo...'
    """
    if not text or len(text) <= max_length:
        return text
    
    # Account for suffix length
    actual_length = max_length - len(suffix)
    if actual_length <= 0:
        return suffix[:max_length]
    
    # Try to break at word boundary
    truncated = text[:actual_length]
    if ' ' in truncated:
        truncated = truncated.rsplit(' ', 1)[0]
    
    return truncated + suffix


# Validation Utilities
def validate_email(email: str) -> bool:
    """Validate email address format.
    
    Args:
        email: Email address to validate.
        
    Returns:
        bool: True if email is valid.
        
    Example:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid-email")
        False
    """
    if not email or not isinstance(email, str):
        return False
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> bool:
    """Validate URL format and scheme.
    
    Args:
        url: URL to validate.
        allowed_schemes: List of allowed schemes (default: ['http', 'https']).
        
    Returns:
        bool: True if URL is valid.
        
    Example:
        >>> validate_url("https://example.com/api")
        True
        >>> validate_url("ftp://example.com", ["ftp", "sftp"])
        True
    """
    if not url or not isinstance(url, str):
        return False
    
    if allowed_schemes is None:
        allowed_schemes = ['http', 'https']
    
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in allowed_schemes and
            parsed.netloc and
            len(parsed.netloc) > 0
        )
    except Exception:
        return False


def validate_api_key(api_key: str, min_length: int = 10) -> bool:
    """Validate API key format.
    
    Args:
        api_key: API key to validate.
        min_length: Minimum required length.
        
    Returns:
        bool: True if API key appears valid.
        
    Example:
        >>> validate_api_key("sk-1234567890abcdef")
        True
        >>> validate_api_key("short")
        False
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Remove whitespace
    api_key = api_key.strip()
    
    return (
        len(api_key) >= min_length and
        api_key.isprintable() and
        not api_key.isspace()
    )


def validate_file_path(file_path: Union[str, Path], must_exist: bool = False) -> bool:
    """Validate file path format and existence.
    
    Args:
        file_path: File path to validate.
        must_exist: Whether the file must exist.
        
    Returns:
        bool: True if path is valid.
        
    Example:
        >>> validate_file_path("./data/tools.json")
        True
        >>> validate_file_path("/invalid/<>path")
        False
    """
    try:
        path = Path(file_path)
        
        # Check for invalid characters (basic check)
        if any(char in str(path) for char in '<>"|?*'):
            return False
        
        if must_exist:
            return path.exists()
        
        # Check if parent directory could be created
        return True
    except (TypeError, ValueError, OSError):
        return False


# Time and Date Utilities
def get_timestamp(include_microseconds: bool = False) -> str:
    """Get current timestamp in ISO format.
    
    Args:
        include_microseconds: Whether to include microseconds.
        
    Returns:
        str: ISO formatted timestamp.
        
    Example:
        >>> timestamp = get_timestamp()
        >>> print(timestamp)  # '2024-01-15T10:30:45Z'
    """
    now = datetime.now(timezone.utc)
    if include_microseconds:
        return now.isoformat()
    else:
        return now.replace(microsecond=0).isoformat()


def parse_timestamp(timestamp: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime object.
    
    Args:
        timestamp: ISO formatted timestamp string.
        
    Returns:
        Optional[datetime]: Parsed datetime or None if invalid.
        
    Example:
        >>> dt = parse_timestamp("2024-01-15T10:30:45Z")
        >>> print(dt.year)  # 2024
    """
    try:
        # Handle different timestamp formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ", 
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
        ]:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
        
        # Try fromisoformat for modern Python
        return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        str: Human-readable duration.
        
    Example:
        >>> format_duration(3661.5)
        '1h 1m 1.5s'
        >>> format_duration(45.123)
        '45.1s'
    """
    if seconds < 0:
        return "0s"
    
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        
        if remaining_seconds > 0:
            return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"
        elif remaining_minutes > 0:
            return f"{hours}h {remaining_minutes}m"
        else:
            return f"{hours}h"


# Data Utilities
def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary.
        dict2: Second dictionary (takes precedence).
        
    Returns:
        Dict[str, Any]: Merged dictionary.
        
    Example:
        >>> d1 = {"a": {"b": 1, "c": 2}}
        >>> d2 = {"a": {"c": 3, "d": 4}}
        >>> result = deep_merge_dicts(d1, d2)
        >>> print(result)  # {"a": {"b": 1, "c": 3, "d": 4}}
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(nested_dict: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Flatten nested dictionary with separator.
    
    Args:
        nested_dict: Nested dictionary to flatten.
        separator: Separator for keys.
        
    Returns:
        Dict[str, Any]: Flattened dictionary.
        
    Example:
        >>> nested = {"a": {"b": {"c": 1}}, "d": 2}
        >>> flat = flatten_dict(nested)
        >>> print(flat)  # {"a.b.c": 1, "d": 2}
    """
    def _flatten(obj: Any, parent_key: str = "") -> Dict[str, Any]:
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                
                if isinstance(value, dict):
                    items.extend(_flatten(value, new_key).items())
                else:
                    items.append((new_key, value))
        else:
            items.append((parent_key, obj))
        
        return dict(items)
    
    return _flatten(nested_dict)


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string with fallback.
    
    Args:
        json_str: JSON string to parse.
        default: Default value if parsing fails.
        
    Returns:
        Any: Parsed JSON or default value.
        
    Example:
        >>> data = safe_json_loads('{"key": "value"}', {})
        >>> bad_data = safe_json_loads('invalid json', {})
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Safely serialize data to JSON string with fallback.
    
    Args:
        data: Data to serialize.
        default: Default value if serialization fails.
        
    Returns:
        str: JSON string or default value.
    """
    try:
        return json.dumps(data, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return default


# Hashing and ID Utilities
def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate unique ID with optional prefix.
    
    Args:
        prefix: Optional prefix for the ID.
        length: Length of the random part.
        
    Returns:
        str: Generated unique ID.
        
    Example:
        >>> agent_id = generate_id("agent", 12)
        >>> print(agent_id)  # "agent_a1b2c3d4e5f6"
    """
    random_part = uuid.uuid4().hex[:length]
    return f"{prefix}_{random_part}" if prefix else random_part


def hash_string(text: str, algorithm: str = "sha256") -> str:
    """Generate hash of string using specified algorithm.
    
    Args:
        text: Text to hash.
        algorithm: Hashing algorithm (md5, sha1, sha256, etc.).
        
    Returns:
        str: Hexadecimal hash string.
        
    Example:
        >>> hash_val = hash_string("search query", "sha256")
        >>> print(len(hash_val))  # 64 (for SHA256)
    """
    if not text:
        text = ""
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()


def generate_cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate cache key from arguments.
    
    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.
        
    Returns:
        str: Cache key.
        
    Example:
        >>> key = generate_cache_key("search", "RNA-seq", max_results=10)
        >>> print(key)  # Hash of the combined arguments
    """
    # Combine all arguments into a string
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    combined = "|".join(key_parts)
    
    return hash_string(combined, "sha256")[:16]


# Async Utilities
async def safe_async_call(
    func: Callable[..., Any],
    *args: Any,
    timeout: Optional[float] = None,
    default: Any = None,
    **kwargs: Any
) -> Any:
    """Safely call async function with timeout and error handling.
    
    Args:
        func: Async function to call.
        *args: Positional arguments.
        timeout: Timeout in seconds.
        default: Default value on error.
        **kwargs: Keyword arguments.
        
    Returns:
        Any: Function result or default value.
        
    Example:
        >>> result = await safe_async_call(
        ...     some_api_call,
        ...     "param1",
        ...     timeout=30,
        ...     default={}
        ... )
    """
    try:
        if timeout:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        else:
            return await func(*args, **kwargs)
    except Exception:
        return default


def timeout_after(seconds: float):
    """Decorator to add timeout to async functions.
    
    Args:
        seconds: Timeout in seconds.
        
    Returns:
        Decorator function.
        
    Example:
        >>> @timeout_after(30)
        ... async def long_running_task():
        ...     await asyncio.sleep(60)  # Will timeout after 30 seconds
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
        return wrapper
    return decorator


async def gather_with_limit(limit: int, *coroutines) -> List[Any]:
    """Run coroutines with concurrency limit.
    
    Args:
        limit: Maximum number of concurrent coroutines.
        *coroutines: Coroutines to run.
        
    Returns:
        List[Any]: Results from all coroutines.
        
    Example:
        >>> results = await gather_with_limit(
        ...     5,  # Max 5 concurrent
        ...     *[fetch_data(url) for url in urls]
        ... )
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def limited_coroutine(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[limited_coroutine(coro) for coro in coroutines])


# File Utilities
def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path.
        
    Returns:
        Path: Path object for the directory.
        
    Example:
        >>> data_dir = ensure_directory("./data/cache")
        >>> print(data_dir.exists())  # True
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_read_file(file_path: Union[str, Path], encoding: str = "utf-8") -> Optional[str]:
    """Safely read file content with error handling.
    
    Args:
        file_path: Path to file.
        encoding: File encoding.
        
    Returns:
        Optional[str]: File content or None if error.
        
    Example:
        >>> content = safe_read_file("config.yaml")
        >>> if content:
        ...     print("File read successfully")
    """
    try:
        return Path(file_path).read_text(encoding=encoding)
    except Exception:
        return None


def safe_write_file(
    file_path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = True
) -> bool:
    """Safely write content to file with error handling.
    
    Args:
        file_path: Path to file.
        content: Content to write.
        encoding: File encoding.
        create_dirs: Whether to create parent directories.
        
    Returns:
        bool: True if successful.
        
    Example:
        >>> success = safe_write_file("output.txt", "Hello, World!")
        >>> print(success)  # True or False
    """
    try:
        path_obj = Path(file_path)
        
        if create_dirs:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        path_obj.write_text(content, encoding=encoding)
        return True
    except Exception:
        return False


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Get file size in megabytes.
    
    Args:
        file_path: Path to file.
        
    Returns:
        float: File size in MB or 0 if file doesn't exist.
        
    Example:
        >>> size = get_file_size_mb("large_dataset.csv")
        >>> print(f"File size: {size:.2f} MB")
    """
    try:
        return Path(file_path).stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


# Error Handling Utilities
class RetryError(Exception):
    """Exception raised when retry attempts are exhausted."""
    pass


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
    **kwargs: Any
) -> Any:
    """Retry async function with exponential backoff.
    
    Args:
        func: Async function to retry.
        *args: Positional arguments.
        max_attempts: Maximum retry attempts.
        delay: Initial delay between retries.
        backoff_factor: Exponential backoff factor.
        exceptions: Exception types to catch and retry.
        **kwargs: Keyword arguments.
        
    Returns:
        Any: Function result.
        
    Raises:
        RetryError: If all retry attempts fail.
        
    Example:
        >>> result = await retry_async(
        ...     api_call,
        ...     "param",
        ...     max_attempts=3,
        ...     delay=1.0,
        ...     exceptions=(ConnectionError, TimeoutError)
        ... )
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            
            if attempt == max_attempts - 1:
                break
            
            await asyncio.sleep(current_delay)
            current_delay *= backoff_factor
    
    raise RetryError(f"Failed after {max_attempts} attempts: {last_exception}")


# Performance Utilities
class Timer:
    """Context manager for timing operations.
    
    Example:
        >>> with Timer() as timer:
        ...     # Some operation
        ...     time.sleep(1)
        >>> print(f"Operation took {timer.elapsed:.2f} seconds")
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time or time.time()
        return end - self.start_time


def measure_memory_usage():
    """Get current memory usage in MB.
    
    Returns:
        float: Memory usage in MB or 0 if unavailable.
        
    Example:
        >>> memory_mb = measure_memory_usage()
        >>> print(f"Memory usage: {memory_mb:.2f} MB")
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


# Environment Utilities
def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable.
    
    Args:
        key: Environment variable key.
        default: Default value if not set.
        
    Returns:
        bool: Boolean value.
        
    Example:
        >>> debug_enabled = get_env_bool("DEBUG", False)
        >>> print(debug_enabled)
    """
    import os
    
    value = os.getenv(key, "").lower()
    return value in ("true", "1", "yes", "on", "enabled")


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer value from environment variable.
    
    Args:
        key: Environment variable key.
        default: Default value if not set or invalid.
        
    Returns:
        int: Integer value.
        
    Example:
        >>> max_workers = get_env_int("MAX_WORKERS", 4)
        >>> print(max_workers)
    """
    import os
    
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_list(key: str, separator: str = ",", default: Optional[List[str]] = None) -> List[str]:
    """Get list value from environment variable.
    
    Args:
        key: Environment variable key.
        separator: List item separator.
        default: Default value if not set.
        
    Returns:
        List[str]: List of values.
        
    Example:
        >>> servers = get_env_list("MCP_SERVERS", ",", ["default"])
        >>> print(servers)
    """
    import os
    
    if default is None:
        default = []
    
    value = os.getenv(key, "")
    if not value:
        return default
    
    return [item.strip() for item in value.split(separator) if item.strip()]


# Package constants
HELPERS_VERSION = "0.1.0"
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0