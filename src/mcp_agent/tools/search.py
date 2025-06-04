"""Search utilities and providers for the MCP Agent Framework.

This module provides various search providers and utilities for web search,
bioinformatics database search, and other specialized search functionality.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic search usage:

    >>> web_search = WebSearchEngine(settings)
    >>> await web_search.initialize()
    >>> results = await web_search.search("bioinformatics tools")
    
    >>> bio_search = BioinformaticsSearchEngine(settings)
    >>> await bio_search.initialize()
    >>> results = await bio_search.search_pubmed("CRISPR RNA sequencing")
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote_plus, urljoin

import httpx
from pydantic import BaseModel, Field

try:
    from mcp_agent.config.settings import AgentSettings
    from mcp_agent.utils.logger import get_logger
except ImportError:
    # Mock imports for development
    class AgentSettings:
        pass
    def get_logger(name: str):
        import logging
        return logging.getLogger(name)


class SearchResult(BaseModel):
    """Represents a search result from any search provider.
    
    Attributes:
        title: Result title.
        url: Result URL.
        snippet: Result description/snippet.
        score: Relevance score (0-1).
        source: Search provider name.
        metadata: Additional metadata.
        
    Example:
        >>> result = SearchResult(
        ...     title="BLAST Tool",
        ...     url="https://blast.ncbi.nlm.nih.gov/",
        ...     snippet="Basic Local Alignment Search Tool",
        ...     score=0.95,
        ...     source="pubmed"
        ... )
    """
    
    title: str = Field(description="Result title")
    url: str = Field(description="Result URL")
    snippet: str = Field(description="Result description or snippet")
    score: float = Field(default=0.0, description="Relevance score (0-1)")
    source: str = Field(description="Search provider name")
    published_date: Optional[str] = Field(default=None, description="Publication date")
    authors: List[str] = Field(default_factory=list, description="Authors (for academic results)")
    doi: Optional[str] = Field(default=None, description="DOI (for academic results)")
    abstract: Optional[str] = Field(default=None, description="Abstract (for academic results)")
    keywords: List[str] = Field(default_factory=list, description="Keywords/tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()


class SearchProvider(ABC):
    """Abstract base class for search providers.
    
    This class defines the interface that all search providers must implement.
    
    Attributes:
        name: Provider name.
        settings: Configuration settings.
        rate_limit: Rate limit in requests per minute.
        timeout: Request timeout in seconds.
        
    Example:
        >>> class CustomSearchProvider(SearchProvider):
        ...     async def _perform_search(self, query, **kwargs):
        ...         # Implementation here
        ...         pass
    """
    
    def __init__(
        self,
        name: str,
        settings: AgentSettings,
        rate_limit: int = 60,
        timeout: int = 30,
    ) -> None:
        """Initialize the search provider.
        
        Args:
            name: Provider name.
            settings: Configuration settings.
            rate_limit: Rate limit in requests per minute.
            timeout: Request timeout in seconds.
        """
        self.name = name
        self.settings = settings
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.logger = get_logger(f"SearchProvider.{name}")
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        # Rate limiting
        self._last_request_time = 0.0
        self._request_count = 0
        self._rate_limit_window_start = time.time()
        
        # Caching
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 3600  # 1 hour
        self._max_cache_size = 1000
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.cached_requests = 0
        self.failed_requests = 0
        
    async def initialize(self) -> None:
        """Initialize the search provider.
        
        Raises:
            RuntimeError: If initialization fails.
        """
        try:
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "User-Agent": "MCP-Agent-Framework/1.0.0",
                    "Accept": "application/json",
                },
            )
            
            await self._provider_specific_init()
            
            self.logger.info(f"Search provider '{self.name}' initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize search provider '{self.name}': {e}")
            raise RuntimeError(f"Search provider initialization failed: {e}") from e
    
    async def _provider_specific_init(self) -> None:
        """Provider-specific initialization. Override in subclasses."""
        pass
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> List[SearchResult]:
        """Perform a search query.
        
        Args:
            query: Search query.
            max_results: Maximum number of results.
            **kwargs: Additional search parameters.
            
        Returns:
            List[SearchResult]: Search results.
            
        Raises:
            ValueError: If query is empty.
            RuntimeError: If search fails.
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty")
        
        self.total_requests += 1
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, max_results, kwargs)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                self.cached_requests += 1
                return [SearchResult(**result) for result in cached_result]
            
            # Apply rate limiting
            await self._apply_rate_limit()
            
            # Perform search
            results = await self._perform_search(query, max_results, **kwargs)
            
            # Cache results
            self._cache_results(cache_key, [result.to_dict() for result in results])
            
            self.successful_requests += 1
            self.logger.info(f"Search for '{query}' returned {len(results)} results")
            
            return results
            
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"Search failed for query '{query}': {e}")
            raise RuntimeError(f"Search failed: {e}") from e
    
    @abstractmethod
    async def _perform_search(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> List[SearchResult]:
        """Perform the actual search. Must be implemented by subclasses.
        
        Args:
            query: Search query.
            max_results: Maximum number of results.
            **kwargs: Additional search parameters.
            
        Returns:
            List[SearchResult]: Search results.
        """
        pass
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to requests."""
        current_time = time.time()
        
        # Reset rate limit window if needed
        if current_time - self._rate_limit_window_start >= 60:
            self._request_count = 0
            self._rate_limit_window_start = current_time
        
        # Check rate limit
        if self._request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self._rate_limit_window_start)
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                self._request_count = 0
                self._rate_limit_window_start = time.time()
        
        # Minimum delay between requests
        time_since_last = current_time - self._last_request_time
        min_delay = 60 / self.rate_limit  # Minimum seconds between requests
        
        if time_since_last < min_delay:
            await asyncio.sleep(min_delay - time_since_last)
        
        self._request_count += 1
        self._last_request_time = time.time()
    
    def _generate_cache_key(
        self,
        query: str,
        max_results: int,
        kwargs: Dict[str, Any],
    ) -> str:
        """Generate cache key for search results."""
        cache_data = {
            "provider": self.name,
            "query": query.lower().strip(),
            "max_results": max_results,
            "kwargs": kwargs,
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        if cache_key not in self._cache:
            return None
        
        cached_data = self._cache[cache_key]
        
        # Check if cache is still valid
        if time.time() - cached_data["timestamp"] > self._cache_ttl:
            del self._cache[cache_key]
            return None
        
        return cached_data["results"]
    
    def _cache_results(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """Cache search results."""
        self._cache[cache_key] = {
            "results": results,
            "timestamp": time.time(),
        }
        
        # Clean up old cache entries
        self._cleanup_cache()
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""
        if len(self._cache) <= self._max_cache_size:
            return
        
        # Remove expired entries first
        current_time = time.time()
        expired_keys = [
            key for key, data in self._cache.items()
            if current_time - data["timestamp"] > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        # Remove oldest entries if still over limit
        if len(self._cache) > self._max_cache_size:
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1]["timestamp"]
            )
            
            excess = len(self._cache) - self._max_cache_size + 100
            for i in range(excess):
                if i < len(sorted_items):
                    del self._cache[sorted_items[i][0]]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get search provider metrics."""
        cache_hit_rate = 0.0
        if self.total_requests > 0:
            cache_hit_rate = (self.cached_requests / self.total_requests) * 100
        
        success_rate = 0.0
        if self.total_requests > 0:
            success_rate = (self.successful_requests / self.total_requests) * 100
        
        return {
            "provider": self.name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "cached_requests": self.cached_requests,
            "cache_hit_rate": cache_hit_rate,
            "success_rate": success_rate,
            "cache_size": len(self._cache),
            "rate_limit": self.rate_limit,
        }
    
    async def close(self) -> None:
        """Close the search provider and clean up resources."""
        if self.client:
            await self.client.aclose()
        
        self._cache.clear()
        self.logger.info(f"Search provider '{self.name}' closed")


class WebSearchEngine(SearchProvider):
    """Web search engine with support for multiple providers.
    
    Supports Tavily, Brave Search, and Serper APIs for comprehensive web search.
    
    Attributes:
        providers: Dictionary of available search providers.
        default_provider: Default provider to use.
        
    Example:
        >>> web_search = WebSearchEngine(settings)
        >>> await web_search.initialize()
        >>> results = await web_search.search("bioinformatics tools")
    """
    
    def __init__(self, settings: AgentSettings) -> None:
        """Initialize the web search engine.
        
        Args:
            settings: Configuration settings.
        """
        super().__init__(
            name="web_search",
            settings=settings,
            rate_limit=getattr(settings, 'search_rate_limit', 60),
            timeout=getattr(settings, 'search_timeout', 30),
        )
        
        # Available providers and their API keys
        self.providers: Dict[str, Optional[str]] = {
            "tavily": getattr(settings, 'tavily_api_key', None),
            "brave": getattr(settings, 'brave_search_api_key', None),
            "serper": getattr(settings, 'serper_api_key', None),
        }
        
        # Filter out providers without API keys
        self.available_providers = {
            name: key for name, key in self.providers.items() if key
        }
        
        self.default_provider = "tavily" if "tavily" in self.available_providers else None
        if self.available_providers:
            self.default_provider = next(iter(self.available_providers))
    
    async def _provider_specific_init(self) -> None:
        """Initialize web search providers."""
        if not self.available_providers:
            self.logger.warning("No web search API keys available")
        else:
            self.logger.info(
                f"Web search initialized with providers: {list(self.available_providers.keys())}"
            )
    
    async def _perform_search(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> List[SearchResult]:
        """Perform web search using available providers."""
        provider = kwargs.get("provider", self.default_provider)
        search_type = kwargs.get("search_type", "general")
        
        if not provider or provider not in self.available_providers:
            if not self.available_providers:
                raise RuntimeError("No web search providers available")
            provider = self.default_provider
        
        try:
            if provider == "tavily":
                return await self._search_tavily(query, max_results, search_type)
            elif provider == "brave":
                return await self._search_brave(query, max_results)
            elif provider == "serper":
                return await self._search_serper(query, max_results)
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        except Exception as e:
            # Try fallback providers
            self.logger.warning(f"Provider {provider} failed: {e}")
            
            for fallback_provider in self.available_providers:
                if fallback_provider != provider:
                    try:
                        self.logger.info(f"Trying fallback provider: {fallback_provider}")
                        return await self._perform_search(
                            query, max_results, provider=fallback_provider, **kwargs
                        )
                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback provider {fallback_provider} failed: {fallback_error}")
            
            raise RuntimeError(f"All web search providers failed: {e}")
    
    async def _search_tavily(
        self,
        query: str,
        max_results: int,
        search_type: str,
    ) -> List[SearchResult]:
        """Search using Tavily API."""
        try:
            url = "https://api.tavily.com/search"
            
            payload = {
                "api_key": self.available_providers["tavily"],
                "query": query,
                "max_results": min(max_results, 20),
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
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    score=item.get("score", 0.5),
                    source="tavily",
                    published_date=item.get("published_date"),
                    metadata={
                        "raw_content": item.get("raw_content", ""),
                        "relevance_score": item.get("relevance_score", 0),
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Tavily search failed: {e}")
            raise
    
    async def _search_brave(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Brave Search API."""
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.available_providers["brave"],
            }
            
            params = {
                "q": query,
                "count": min(max_results, 20),
                "offset": 0,
                "mkt": "en-US",
                "safesearch": "moderate",
                "text_decorations": False,
            }
            
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("web", {}).get("results", []):
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    score=0.7,  # Brave doesn't provide relevance scores
                    source="brave",
                    published_date=item.get("age"),
                    metadata={
                        "language": item.get("language"),
                        "family_friendly": item.get("family_friendly", True),
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Brave search failed: {e}")
            raise
    
    async def _search_serper(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Serper API."""
        try:
            url = "https://google.serper.dev/search"
            
            headers = {
                "X-API-KEY": self.available_providers["serper"],
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
            
            for i, item in enumerate(data.get("organic", [])):
                # Convert position to score (higher position = lower score)
                position_score = max(0.1, 1.0 - (i / max_results))
                
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    score=position_score,
                    source="serper",
                    published_date=item.get("date"),
                    metadata={
                        "position": i + 1,
                        "displayed_link": item.get("displayedLink"),
                        "snippet_highlighted_words": item.get("snippetHighlightedWords", []),
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Serper search failed: {e}")
            raise


class BioinformaticsSearchEngine(SearchProvider):
    """Specialized search engine for bioinformatics databases and resources.
    
    Provides search functionality for PubMed, Bioconductor, BioStars,
    and other bioinformatics-specific databases.
    
    Attributes:
        ncbi_api_key: Optional NCBI API key for enhanced rate limits.
        
    Example:
        >>> bio_search = BioinformaticsSearchEngine(settings)
        >>> await bio_search.initialize()
        >>> results = await bio_search.search_pubmed("CRISPR genome editing")
    """
    
    def __init__(self, settings: AgentSettings) -> None:
        """Initialize the bioinformatics search engine.
        
        Args:
            settings: Configuration settings.
        """
        super().__init__(
            name="bioinformatics_search",
            settings=settings,
            rate_limit=10,  # Lower rate limit for NCBI
            timeout=30,
        )
        
        self.ncbi_api_key = getattr(settings, 'ncbi_api_key', None)
        self.ncbi_email = getattr(settings, 'ncbi_email', 'user@example.com')
        
        # Database configurations
        self.databases = {
            "pubmed": {
                "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                "search_endpoint": "esearch.fcgi",
                "fetch_endpoint": "efetch.fcgi",
                "summary_endpoint": "esummary.fcgi",
            },
            "bioconductor": {
                "base_url": "https://bioconductor.org/",
                "packages_api": "packages/json/",
            },
            "biostars": {
                "base_url": "https://www.biostars.org/",
                "api_endpoint": "api/",
            },
        }
    
    async def _provider_specific_init(self) -> None:
        """Initialize bioinformatics search databases."""
        if self.ncbi_api_key:
            self.logger.info("NCBI API key available for enhanced rate limits")
        else:
            self.logger.warning("No NCBI API key provided, using default rate limits")
    
    async def _perform_search(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> List[SearchResult]:
        """Perform bioinformatics search."""
        database = kwargs.get("database", "pubmed")
        search_type = kwargs.get("search_type", "general")
        
        if database == "pubmed":
            return await self.search_pubmed(query, max_results, search_type)
        elif database == "bioconductor":
            return await self.search_bioconductor(query, max_results)
        elif database == "biostars":
            return await self.search_biostars(query, max_results)
        else:
            raise ValueError(f"Unknown database: {database}")
    
    async def search_pubmed(
        self,
        query: str,
        max_results: int = 20,
        search_type: str = "general",
    ) -> List[SearchResult]:
        """Search PubMed for scientific articles.
        
        Args:
            query: Search query.
            max_results: Maximum number of results.
            search_type: Type of search ('general', 'review', 'clinical').
            
        Returns:
            List[SearchResult]: PubMed search results.
        """
        try:
            # Step 1: Search for PMIDs
            pmids = await self._search_pubmed_ids(query, max_results, search_type)
            
            if not pmids:
                return []
            
            # Step 2: Fetch article summaries
            results = await self._fetch_pubmed_summaries(pmids)
            
            return results
            
        except Exception as e:
            self.logger.error(f"PubMed search failed: {e}")
            raise
    
    async def _search_pubmed_ids(
        self,
        query: str,
        max_results: int,
        search_type: str,
    ) -> List[str]:
        """Search PubMed for article IDs."""
        base_url = self.databases["pubmed"]["base_url"]
        endpoint = self.databases["pubmed"]["search_endpoint"]
        
        # Enhance query based on search type
        enhanced_query = query
        if search_type == "review":
            enhanced_query += " AND Review[ptyp]"
        elif search_type == "clinical":
            enhanced_query += " AND Clinical Trial[ptyp]"
        
        params = {
            "db": "pubmed",
            "term": enhanced_query,
            "retmax": min(max_results, 100),
            "retmode": "json",
            "sort": "relevance",
            "tool": "mcp_agent",
            "email": self.ncbi_email,
        }
        
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key
        
        url = urljoin(base_url, endpoint)
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        
        return pmids
    
    async def _fetch_pubmed_summaries(self, pmids: List[str]) -> List[SearchResult]:
        """Fetch article summaries for PMIDs."""
        if not pmids:
            return []
        
        base_url = self.databases["pubmed"]["base_url"]
        endpoint = self.databases["pubmed"]["summary_endpoint"]
        
        # Process in batches to avoid URL length limits
        batch_size = 20
        all_results = []
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            params = {
                "db": "pubmed",
                "id": ",".join(batch_pmids),
                "retmode": "json",
                "tool": "mcp_agent",
                "email": self.ncbi_email,
            }
            
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key
            
            url = urljoin(base_url, endpoint)
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("result", {})
            
            for pmid in batch_pmids:
                if pmid in results:
                    article = results[pmid]
                    
                    # Extract authors
                    authors = []
                    for author in article.get("authors", []):
                        if "name" in author:
                            authors.append(author["name"])
                    
                    # Create search result
                    result = SearchResult(
                        title=article.get("title", ""),
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        snippet=article.get("abstract", ""),
                        score=0.8,  # PubMed doesn't provide relevance scores
                        source="pubmed",
                        published_date=article.get("pubdate", ""),
                        authors=authors,
                        doi=article.get("elocationid", ""),
                        metadata={
                            "pmid": pmid,
                            "journal": article.get("fulljournalname", ""),
                            "volume": article.get("volume", ""),
                            "issue": article.get("issue", ""),
                            "pages": article.get("pages", ""),
                            "publication_types": article.get("pubtype", []),
                        }
                    )
                    all_results.append(result)
            
            # Rate limiting between batches
            if i + batch_size < len(pmids):
                await asyncio.sleep(0.5)
        
        return all_results
    
    async def search_bioconductor(
        self,
        query: str,
        max_results: int = 20,
    ) -> List[SearchResult]:
        """Search Bioconductor packages.
        
        Args:
            query: Search query.
            max_results: Maximum number of results.
            
        Returns:
            List[SearchResult]: Bioconductor package results.
        """
        try:
            # Note: This is a simplified implementation
            # In practice, you'd use the actual Bioconductor API or web scraping
            
            base_url = "https://bioconductor.org/packages/search/"
            
            params = {
                "q": query,
                "biocVersion": "3.18",  # Latest version
            }
            
            # For now, return mock results
            # In a real implementation, you'd parse the search results page
            results = [
                SearchResult(
                    title=f"Bioconductor Package for {query}",
                    url=f"https://bioconductor.org/packages/{query.lower()}/",
                    snippet=f"R package for {query} analysis in bioinformatics",
                    score=0.7,
                    source="bioconductor",
                    metadata={
                        "package_type": "software",
                        "bioc_version": "3.18",
                        "r_version": "4.3",
                    }
                )
            ]
            
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Bioconductor search failed: {e}")
            raise
    
    async def search_biostars(
        self,
        query: str,
        max_results: int = 20,
    ) -> List[SearchResult]:
        """Search BioStars Q&A site.
        
        Args:
            query: Search query.
            max_results: Maximum number of results.
            
        Returns:
            List[SearchResult]: BioStars question/answer results.
        """
        try:
            # Note: This is a simplified implementation
            # BioStars doesn't have a public API, so this would require web scraping
            
            # For now, return mock results
            results = [
                SearchResult(
                    title=f"BioStars Question about {query}",
                    url=f"https://www.biostars.org/p/{hash(query) % 100000}/",
                    snippet=f"Community discussion about {query} in bioinformatics",
                    score=0.6,
                    source="biostars",
                    metadata={
                        "question_type": "discussion",
                        "tags": query.split(),
                        "community": "biostars",
                    }
                )
            ]
            
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"BioStars search failed: {e}")
            raise
    
    async def search_all_databases(
        self,
        query: str,
        max_results_per_db: int = 10,
    ) -> Dict[str, List[SearchResult]]:
        """Search across all bioinformatics databases.
        
        Args:
            query: Search query.
            max_results_per_db: Maximum results per database.
            
        Returns:
            Dict[str, List[SearchResult]]: Results grouped by database.
        """
        results = {}
        
        # Search each database
        for db_name in self.databases.keys():
            try:
                db_results = await self._perform_search(
                    query,
                    max_results_per_db,
                    database=db_name
                )
                results[db_name] = db_results
                
            except Exception as e:
                self.logger.warning(f"Search failed for database {db_name}: {e}")
                results[db_name] = []
        
        return results


class SearchAggregator:
    """Aggregates and ranks results from multiple search providers.
    
    This class combines results from different search providers and applies
    ranking algorithms to provide the best overall results.
    
    Example:
        >>> aggregator = SearchAggregator([web_search, bio_search])
        >>> results = await aggregator.search("CRISPR tools")
    """
    
    def __init__(self, providers: List[SearchProvider]) -> None:
        """Initialize the search aggregator.
        
        Args:
            providers: List of search providers to aggregate.
        """
        self.providers = providers
        self.logger = get_logger(self.__class__.__name__)
    
    async def search(
        self,
        query: str,
        max_results: int = 20,
        provider_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """Perform aggregated search across all providers.
        
        Args:
            query: Search query.
            max_results: Maximum number of results to return.
            provider_weights: Weights for different providers.
            **kwargs: Additional search parameters.
            
        Returns:
            List[SearchResult]: Ranked and deduplicated results.
        """
        if not self.providers:
            return []
        
        # Default weights
        if provider_weights is None:
            provider_weights = {provider.name: 1.0 for provider in self.providers}
        
        # Search all providers concurrently
        search_tasks = []
        for provider in self.providers:
            task = asyncio.create_task(
                self._search_provider_safe(provider, query, max_results, **kwargs)
            )
            search_tasks.append((provider.name, task))
        
        # Collect results
        all_results = []
        for provider_name, task in search_tasks:
            try:
                provider_results = await task
                
                # Apply provider weight
                weight = provider_weights.get(provider_name, 1.0)
                for result in provider_results:
                    result.score *= weight
                
                all_results.extend(provider_results)
                
            except Exception as e:
                self.logger.warning(f"Provider {provider_name} failed: {e}")
        
        # Deduplicate and rank results
        deduplicated_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(deduplicated_results)
        
        return ranked_results[:max_results]
    
    async def _search_provider_safe(
        self,
        provider: SearchProvider,
        query: str,
        max_results: int,
        **kwargs,
    ) -> List[SearchResult]:
        """Safely search a provider with error handling."""
        try:
            return await provider.search(query, max_results, **kwargs)
        except Exception as e:
            self.logger.warning(f"Provider {provider.name} search failed: {e}")
            return []
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL similarity."""
        seen_urls = set()
        deduplicated = []
        
        for result in results:
            # Normalize URL for comparison
            normalized_url = result.url.lower().rstrip('/')
            
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                deduplicated.append(result)
            else:
                # If we've seen this URL, keep the one with higher score
                for i, existing_result in enumerate(deduplicated):
                    if existing_result.url.lower().rstrip('/') == normalized_url:
                        if result.score > existing_result.score:
                            deduplicated[i] = result
                        break
        
        return deduplicated
    
    def _rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank results using multiple factors."""
        # Apply ranking factors
        for result in results:
            # Base score from provider
            final_score = result.score
            
            # Boost academic sources
            if result.source in ["pubmed", "bioconductor"]:
                final_score *= 1.2
            
            # Boost results with DOI
            if result.doi:
                final_score *= 1.1
            
            # Boost recent results (if date available)
            if result.published_date:
                try:
                    # Simple date parsing - could be enhanced
                    if "2023" in result.published_date or "2024" in result.published_date:
                        final_score *= 1.05
                except:
                    pass
            
            # Apply title relevance (simple keyword matching)
            # In practice, you'd use more sophisticated text similarity
            
            result.score = min(1.0, final_score)  # Cap at 1.0
        
        # Sort by final score
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get metrics from all providers."""
        metrics = {}
        
        for provider in self.providers:
            metrics[provider.name] = provider.get_metrics()
        
        return metrics