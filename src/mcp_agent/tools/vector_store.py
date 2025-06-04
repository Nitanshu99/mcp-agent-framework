"""Vector database implementation using ChromaDB for semantic search.

This module provides a comprehensive vector store implementation for storing
and searching bioinformatics tools using semantic embeddings.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic vector store usage:

    >>> vector_store = VectorStore(settings)
    >>> await vector_store.initialize()
    >>> await vector_store.add_tool("BLAST", "sequence alignment tool", {...})
    >>> results = await vector_store.search("protein alignment")
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from sentence_transformers import SentenceTransformer
    
    from mcp_agent.config.settings import AgentSettings, VectorStoreConfig
    from mcp_agent.utils.logger import get_logger
except ImportError as e:
    # Mock imports for development
    import warnings
    warnings.warn(f"Vector store dependencies not available: {e}", ImportWarning)
    
    class SentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name
        def encode(self, texts):
            return [[0.1] * 384 for _ in texts]
    
    class chromadb:
        @staticmethod
        def PersistentClient(*args, **kwargs):
            return MockChromaClient()
        
        class Settings:
            def __init__(self, **kwargs):
                pass
    
    ChromaSettings = chromadb.Settings
    
    class MockChromaClient:
        def get_or_create_collection(self, *args, **kwargs):
            return MockCollection()
    
    class MockCollection:
        def add(self, *args, **kwargs):
            pass
        def query(self, *args, **kwargs):
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
        def update(self, *args, **kwargs):
            pass
        def delete(self, *args, **kwargs):
            pass
        def get(self, *args, **kwargs):
            return {"ids": [], "metadatas": [], "documents": []}
        def count(self):
            return 0
    
    class AgentSettings:
        pass
    class VectorStoreConfig:
        pass
    def get_logger(name: str):
        import logging
        return logging.getLogger(name)


class ToolDocument(BaseModel):
    """Represents a tool document in the vector store.
    
    Attributes:
        id: Unique identifier for the tool.
        name: Tool name.
        description: Tool description.
        content: Full text content for embedding.
        category: Tool category.
        tags: List of tags.
        metadata: Additional metadata.
        
    Example:
        >>> doc = ToolDocument(
        ...     id="blast_001",
        ...     name="BLAST",
        ...     description="Basic Local Alignment Search Tool",
        ...     content="BLAST is a tool for sequence alignment...",
        ...     category="sequence_analysis"
        ... )
    """
    
    id: str = Field(description="Unique tool identifier")
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    content: str = Field(description="Full text content")
    category: Optional[str] = Field(default=None, description="Tool category")
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    url: Optional[str] = Field(default=None, description="Tool URL")
    installation: Optional[str] = Field(default=None, description="Installation command")
    documentation: Optional[str] = Field(default=None, description="Documentation URL")
    organism: Optional[str] = Field(default=None, description="Target organism")
    data_types: List[str] = Field(default_factory=list, description="Supported data types")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_embedding_text(self) -> str:
        """Generate text for embedding creation."""
        parts = [self.name, self.description]
        
        if self.category:
            parts.append(f"Category: {self.category}")
        
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        
        if self.organism:
            parts.append(f"Organism: {self.organism}")
        
        if self.data_types:
            parts.append(f"Data types: {', '.join(self.data_types)}")
        
        return " ".join(parts)
    
    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata format."""
        metadata = {
            "name": self.name,
            "description": self.description[:500],  # Limit for ChromaDB
            "category": self.category or "unknown",
            "tags": ",".join(self.tags),
            "url": self.url or "",
            "installation": self.installation or "",
            "organism": self.organism or "",
            "data_types": ",".join(self.data_types),
        }
        
        # Add custom metadata (flatten if necessary)
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[f"meta_{key}"] = value
            else:
                metadata[f"meta_{key}"] = str(value)
        
        return metadata


class EmbeddingProvider:
    """Provider for generating embeddings using various models.
    
    Attributes:
        model_name: Name of the embedding model.
        model: Loaded embedding model.
        dimension: Embedding dimension.
        
    Example:
        >>> provider = EmbeddingProvider("text-embedding-3-small")
        >>> await provider.initialize()
        >>> embeddings = await provider.encode(["text1", "text2"])
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding provider.
        
        Args:
            model_name: Name of the embedding model to use.
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.logger = get_logger(self.__class__.__name__)
        
        # Model mappings
        self.model_mappings = {
            "text-embedding-3-small": "all-MiniLM-L6-v2",
            "text-embedding-3-large": "all-mpnet-base-v2", 
            "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2": "all-mpnet-base-v2",
        }
        
        # Update dimension based on model
        if "mpnet" in model_name.lower():
            self.dimension = 768
        elif "large" in model_name.lower():
            self.dimension = 768
    
    async def initialize(self) -> None:
        """Initialize the embedding model.
        
        Raises:
            RuntimeError: If model initialization fails.
        """
        try:
            # Map model name
            actual_model = self.model_mappings.get(self.model_name, self.model_name)
            
            self.logger.info(f"Loading embedding model: {actual_model}")
            
            # Load model in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, SentenceTransformer, actual_model
            )
            
            # Update dimension if available
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                self.dimension = self.model.get_sentence_embedding_dimension()
            
            self.logger.info(f"Embedding model loaded (dimension: {self.dimension})")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}") from e
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode.
            
        Returns:
            List[List[float]]: List of embedding vectors.
            
        Raises:
            RuntimeError: If model is not initialized.
        """
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            # Encode in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self.model.encode, texts
            )
            
            # Convert to list format
            return embeddings.tolist()
            
        except Exception as e:
            self.logger.error(f"Embedding encoding failed: {e}")
            raise RuntimeError(f"Embedding encoding failed: {e}") from e
    
    async def encode_single(self, text: str) -> List[float]:
        """Encode a single text to embedding.
        
        Args:
            text: Text to encode.
            
        Returns:
            List[float]: Embedding vector.
        """
        embeddings = await self.encode([text])
        return embeddings[0]


class VectorStore:
    """Vector database implementation using ChromaDB for semantic search.
    
    This class provides a comprehensive vector store for storing and searching
    bioinformatics tools using semantic embeddings.
    
    Attributes:
        settings: Configuration settings.
        config: Vector store specific configuration.
        client: ChromaDB client.
        collection: ChromaDB collection.
        embedding_provider: Embedding generation provider.
        
    Example:
        >>> store = VectorStore(settings)
        >>> await store.initialize()
        >>> await store.add_tool("BLAST", "sequence alignment tool")
        >>> results = await store.search("protein alignment", limit=10)
    """
    
    def __init__(self, settings: AgentSettings) -> None:
        """Initialize the vector store.
        
        Args:
            settings: Configuration settings.
        """
        self.settings = settings
        self.config = settings.vector_store
        self.logger = get_logger(self.__class__.__name__)
        
        # ChromaDB components
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[Any] = None
        
        # Embedding provider
        self.embedding_provider = EmbeddingProvider(self.config.embedding_model)
        
        # Cache and metrics
        self._cache: Dict[str, Any] = {}
        self._cache_max_size = 1000
        self.total_documents = 0
        self.total_searches = 0
        self.cache_hits = 0
        
        # Performance tracking
        self.last_activity = None
        self.avg_search_time = 0.0
        self.avg_add_time = 0.0
        
        self.logger.info(f"VectorStore initialized with model: {self.config.embedding_model}")
    
    async def initialize(self) -> None:
        """Initialize the vector store and its components.
        
        Raises:
            RuntimeError: If initialization fails.
            
        Example:
            >>> await store.initialize()
        """
        self.logger.info("Initializing vector store...")
        
        try:
            # Initialize embedding provider
            await self.embedding_provider.initialize()
            
            # Initialize ChromaDB
            await self._initialize_chromadb()
            
            # Load existing documents count
            self.total_documents = await self._get_document_count()
            
            self.logger.info(
                f"Vector store initialized with {self.total_documents} documents"
            )
            
        except Exception as e:
            self.logger.error(f"Vector store initialization failed: {e}")
            raise RuntimeError(f"Vector store initialization failed: {e}") from e
    
    async def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Create data directory
            self.config.path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            chroma_settings = ChromaSettings(
                persist_directory=str(self.config.path),
                anonymized_telemetry=False,
            )
            
            self.client = chromadb.PersistentClient(
                path=str(self.config.path),
                settings=chroma_settings,
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={
                    "description": "Bioinformatics tools vector database",
                    "embedding_model": self.config.embedding_model,
                    "embedding_dimension": self.embedding_provider.dimension,
                }
            )
            
            self.logger.info(f"ChromaDB collection '{self.config.collection_name}' ready")
            
        except Exception as e:
            self.logger.error(f"ChromaDB initialization failed: {e}")
            raise
    
    async def _get_document_count(self) -> int:
        """Get the current number of documents in the collection."""
        try:
            if self.collection:
                return self.collection.count()
            return 0
        except Exception as e:
            self.logger.warning(f"Failed to get document count: {e}")
            return 0
    
    async def add_tool(
        self,
        name: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Add a tool to the vector store.
        
        Args:
            name: Tool name.
            description: Tool description.
            metadata: Additional metadata.
            **kwargs: Additional tool attributes.
            
        Returns:
            str: Document ID of the added tool.
            
        Example:
            >>> doc_id = await store.add_tool(
            ...     "BLAST",
            ...     "Basic Local Alignment Search Tool",
            ...     {"category": "sequence_analysis", "organism": "any"}
            ... )
        """
        start_time = time.time()
        
        try:
            # Create document ID
            doc_id = self._generate_document_id(name, description)
            
            # Create tool document
            tool_doc = ToolDocument(
                id=doc_id,
                name=name,
                description=description,
                content=f"{name} {description}",
                metadata=metadata or {},
                **kwargs
            )
            
            # Generate embedding
            embedding_text = tool_doc.to_embedding_text()
            embedding = await self.embedding_provider.encode_single(embedding_text)
            
            # Add to ChromaDB
            await self._add_document_to_collection(tool_doc, embedding)
            
            # Update metrics
            self.total_documents += 1
            execution_time = time.time() - start_time
            self.avg_add_time = (self.avg_add_time + execution_time) / 2
            self.last_activity = datetime.now()
            
            # Clear cache for this document
            self._invalidate_cache(doc_id)
            
            self.logger.info(f"Added tool '{name}' with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Failed to add tool '{name}': {e}")
            raise
    
    async def _add_document_to_collection(
        self,
        doc: ToolDocument,
        embedding: List[float],
    ) -> None:
        """Add a document to the ChromaDB collection."""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.collection.add(
                    ids=[doc.id],
                    embeddings=[embedding],
                    documents=[doc.content],
                    metadatas=[doc.to_chroma_metadata()],
                )
            )
        except Exception as e:
            self.logger.error(f"Failed to add document to collection: {e}")
            raise
    
    async def add_tools_batch(self, tools: List[Dict[str, Any]]) -> List[str]:
        """Add multiple tools in batch for better performance.
        
        Args:
            tools: List of tool dictionaries.
            
        Returns:
            List[str]: List of document IDs.
            
        Example:
            >>> tools = [
            ...     {"name": "BLAST", "description": "Sequence alignment"},
            ...     {"name": "BWA", "description": "Read alignment"}
            ... ]
            >>> doc_ids = await store.add_tools_batch(tools)
        """
        start_time = time.time()
        
        try:
            # Prepare documents
            documents = []
            embeddings_texts = []
            
            for tool_data in tools:
                name = tool_data.get("name", "")
                description = tool_data.get("description", "")
                
                if not name or not description:
                    continue
                
                doc_id = self._generate_document_id(name, description)
                tool_doc = ToolDocument(
                    id=doc_id,
                    name=name,
                    description=description,
                    content=f"{name} {description}",
                    **{k: v for k, v in tool_data.items() if k not in ["name", "description"]}
                )
                
                documents.append(tool_doc)
                embeddings_texts.append(tool_doc.to_embedding_text())
            
            if not documents:
                return []
            
            # Generate embeddings in batch
            embeddings = await self.embedding_provider.encode(embeddings_texts)
            
            # Add to ChromaDB in batch
            ids = [doc.id for doc in documents]
            contents = [doc.content for doc in documents]
            metadatas = [doc.to_chroma_metadata() for doc in documents]
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=metadatas,
                )
            )
            
            # Update metrics
            self.total_documents += len(documents)
            execution_time = time.time() - start_time
            self.avg_add_time = (self.avg_add_time + execution_time) / 2
            self.last_activity = datetime.now()
            
            # Clear cache
            self._clear_cache()
            
            self.logger.info(f"Added {len(documents)} tools in batch")
            return ids
            
        except Exception as e:
            self.logger.error(f"Batch add failed: {e}")
            raise
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for tools using semantic similarity.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            filters: Metadata filters.
            min_score: Minimum similarity score.
            
        Returns:
            List[Dict[str, Any]]: Search results with scores.
            
        Example:
            >>> results = await store.search(
            ...     "protein sequence alignment",
            ...     limit=5,
            ...     filters={"category": "sequence_analysis"}
            ... )
        """
        start_time = time.time()
        self.total_searches += 1
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(query, limit, filters, min_score)
            if cache_key in self._cache:
                self.cache_hits += 1
                return self._cache[cache_key]
            
            # Generate query embedding
            query_embedding = await self.embedding_provider.encode_single(query)
            
            # Build where clause for filters
            where_clause = self._build_where_clause(filters)
            
            # Perform search
            results = await self._search_collection(
                query_embedding=query_embedding,
                limit=limit,
                where_clause=where_clause,
            )
            
            # Process and filter results
            processed_results = self._process_search_results(results, min_score)
            
            # Cache results
            self._cache[cache_key] = processed_results
            self._cleanup_cache()
            
            # Update metrics
            execution_time = time.time() - start_time
            self.avg_search_time = (self.avg_search_time + execution_time) / 2
            self.last_activity = datetime.now()
            
            self.logger.info(
                f"Search for '{query}' returned {len(processed_results)} results"
            )
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {e}")
            raise
    
    async def _search_collection(
        self,
        query_embedding: List[float],
        limit: int,
        where_clause: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform search on ChromaDB collection."""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": limit,
            }
            
            if where_clause:
                search_kwargs["where"] = where_clause
            
            results = await loop.run_in_executor(
                None,
                lambda: self.collection.query(**search_kwargs)
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Collection search failed: {e}")
            raise
    
    def _build_where_clause(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from filters."""
        if not filters:
            return None
        
        where_conditions = {}
        
        for key, value in filters.items():
            if key == "category" and value:
                where_conditions["category"] = value
            elif key == "organism" and value:
                where_conditions["organism"] = value
            elif key == "tags" and value:
                # For tags, we might want to check if the tag exists in the comma-separated list
                where_conditions["tags"] = {"$contains": value}
            elif key.startswith("meta_") and value:
                where_conditions[key] = value
        
        return where_conditions if where_conditions else None
    
    def _process_search_results(
        self,
        results: Dict[str, Any],
        min_score: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Process raw search results from ChromaDB."""
        processed = []
        
        if not results.get("ids") or not results["ids"][0]:
            return processed
        
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]
        
        for i, doc_id in enumerate(ids):
            # Convert distance to similarity score (ChromaDB uses L2 distance)
            distance = distances[i]
            similarity_score = max(0, 1 / (1 + distance))
            
            # Filter by minimum score
            if min_score and similarity_score < min_score:
                continue
            
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            
            # Parse tags back to list
            tags_str = metadata.get("tags", "")
            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
            
            # Parse data types back to list
            data_types_str = metadata.get("data_types", "")
            data_types = [dt.strip() for dt in data_types_str.split(",") if dt.strip()]
            
            result = {
                "id": doc_id,
                "name": metadata.get("name", ""),
                "description": metadata.get("description", ""),
                "category": metadata.get("category", ""),
                "tags": tags,
                "url": metadata.get("url", ""),
                "installation": metadata.get("installation", ""),
                "organism": metadata.get("organism", ""),
                "data_types": data_types,
                "score": similarity_score,
                "distance": distance,
                "content": document,
            }
            
            # Add custom metadata
            for key, value in metadata.items():
                if key.startswith("meta_"):
                    result[key[5:]] = value  # Remove "meta_" prefix
            
            processed.append(result)
        
        # Sort by similarity score (descending)
        processed.sort(key=lambda x: x["score"], reverse=True)
        
        return processed
    
    async def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific tool by ID.
        
        Args:
            tool_id: Tool document ID.
            
        Returns:
            Optional[Dict[str, Any]]: Tool information or None if not found.
            
        Example:
            >>> tool = await store.get_tool("blast_001")
            >>> if tool:
            ...     print(f"Tool: {tool['name']}")
        """
        try:
            # Check cache first
            cache_key = f"tool_{tool_id}"
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Query ChromaDB
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.collection.get(ids=[tool_id])
            )
            
            if not results.get("ids") or not results["ids"]:
                return None
            
            # Process result
            metadata = results["metadatas"][0] if results["metadatas"] else {}
            document = results["documents"][0] if results["documents"] else ""
            
            # Parse tags and data types
            tags_str = metadata.get("tags", "")
            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
            
            data_types_str = metadata.get("data_types", "")
            data_types = [dt.strip() for dt in data_types_str.split(",") if dt.strip()]
            
            tool_data = {
                "id": tool_id,
                "name": metadata.get("name", ""),
                "description": metadata.get("description", ""),
                "category": metadata.get("category", ""),
                "tags": tags,
                "url": metadata.get("url", ""),
                "installation": metadata.get("installation", ""),
                "organism": metadata.get("organism", ""),
                "data_types": data_types,
                "content": document,
            }
            
            # Add custom metadata
            for key, value in metadata.items():
                if key.startswith("meta_"):
                    tool_data[key[5:]] = value
            
            # Cache result
            self._cache[cache_key] = tool_data
            
            return tool_data
            
        except Exception as e:
            self.logger.error(f"Failed to get tool {tool_id}: {e}")
            return None
    
    async def update_tool(
        self,
        tool_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update a tool in the vector store.
        
        Args:
            tool_id: Tool document ID.
            updates: Fields to update.
            
        Returns:
            bool: True if update was successful.
            
        Example:
            >>> success = await store.update_tool(
            ...     "blast_001",
            ...     {"description": "Updated description", "category": "alignment"}
            ... )
        """
        try:
            # Get existing tool
            existing_tool = await self.get_tool(tool_id)
            if not existing_tool:
                self.logger.warning(f"Tool {tool_id} not found for update")
                return False
            
            # Merge updates
            updated_tool = existing_tool.copy()
            updated_tool.update(updates)
            
            # Create updated document
            tool_doc = ToolDocument(
                id=tool_id,
                name=updated_tool.get("name", ""),
                description=updated_tool.get("description", ""),
                content=f"{updated_tool.get('name', '')} {updated_tool.get('description', '')}",
                category=updated_tool.get("category"),
                tags=updated_tool.get("tags", []),
                url=updated_tool.get("url"),
                installation=updated_tool.get("installation"),
                organism=updated_tool.get("organism"),
                data_types=updated_tool.get("data_types", []),
                metadata={k: v for k, v in updated_tool.items() 
                         if k not in ["id", "name", "description", "content", "category", 
                                    "tags", "url", "installation", "organism", "data_types"]}
            )
            
            # Generate new embedding
            embedding_text = tool_doc.to_embedding_text()
            embedding = await self.embedding_provider.encode_single(embedding_text)
            
            # Update in ChromaDB
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.collection.update(
                    ids=[tool_id],
                    embeddings=[embedding],
                    documents=[tool_doc.content],
                    metadatas=[tool_doc.to_chroma_metadata()],
                )
            )
            
            # Invalidate cache
            self._invalidate_cache(tool_id)
            
            self.logger.info(f"Updated tool {tool_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update tool {tool_id}: {e}")
            return False
    
    async def delete_tool(self, tool_id: str) -> bool:
        """Delete a tool from the vector store.
        
        Args:
            tool_id: Tool document ID.
            
        Returns:
            bool: True if deletion was successful.
            
        Example:
            >>> success = await store.delete_tool("blast_001")
        """
        try:
            # Delete from ChromaDB
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.collection.delete(ids=[tool_id])
            )
            
            # Update metrics
            self.total_documents = max(0, self.total_documents - 1)
            
            # Invalidate cache
            self._invalidate_cache(tool_id)
            
            self.logger.info(f"Deleted tool {tool_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete tool {tool_id}: {e}")
            return False
    
    async def list_tools(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List tools with optional filters.
        
        Args:
            filters: Metadata filters.
            limit: Maximum number of results.
            
        Returns:
            List[Dict[str, Any]]: List of tools.
            
        Example:
            >>> tools = await store.list_tools(
            ...     filters={"category": "sequence_analysis"},
            ...     limit=50
            ... )
        """
        try:
            # Build where clause
            where_clause = self._build_where_clause(filters)
            
            # Query ChromaDB
            loop = asyncio.get_event_loop()
            
            query_kwargs = {"limit": limit}
            if where_clause:
                query_kwargs["where"] = where_clause
            
            results = await loop.run_in_executor(
                None,
                lambda: self.collection.get(**query_kwargs)
            )
            
            # Process results
            tools = []
            if results.get("ids"):
                for i, tool_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i] if i < len(results["metadatas"]) else {}
                    document = results["documents"][i] if i < len(results["documents"]) else ""
                    
                    # Parse tags and data types
                    tags_str = metadata.get("tags", "")
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
                    
                    data_types_str = metadata.get("data_types", "")
                    data_types = [dt.strip() for dt in data_types_str.split(",") if dt.strip()]
                    
                    tool_data = {
                        "id": tool_id,
                        "name": metadata.get("name", ""),
                        "description": metadata.get("description", ""),
                        "category": metadata.get("category", ""),
                        "tags": tags,
                        "url": metadata.get("url", ""),
                        "installation": metadata.get("installation", ""),
                        "organism": metadata.get("organism", ""),
                        "data_types": data_types,
                        "content": document,
                    }
                    
                    # Add custom metadata
                    for key, value in metadata.items():
                        if key.startswith("meta_"):
                            tool_data[key[5:]] = value
                    
                    tools.append(tool_data)
            
            return tools
            
        except Exception as e:
            self.logger.error(f"Failed to list tools: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dict[str, Any]: Collection statistics.
            
        Example:
            >>> stats = await store.get_collection_stats()
            >>> print(f"Total documents: {stats['document_count']}")
        """
        try:
            document_count = await self._get_document_count()
            
            # Get sample of documents for category analysis
            sample_tools = await self.list_tools(limit=1000)
            
            categories = {}
            organisms = {}
            data_types = {}
            
            for tool in sample_tools:
                # Count categories
                category = tool.get("category", "unknown")
                categories[category] = categories.get(category, 0) + 1
                
                # Count organisms
                organism = tool.get("organism", "unknown")
                if organism:
                    organisms[organism] = organisms.get(organism, 0) + 1
                
                # Count data types
                for data_type in tool.get("data_types", []):
                    if data_type:
                        data_types[data_type] = data_types.get(data_type, 0) + 1
            
            return {
                "document_count": document_count,
                "collection_name": self.config.collection_name,
                "embedding_model": self.config.embedding_model,
                "embedding_dimension": self.embedding_provider.dimension,
                "categories": categories,
                "organisms": organisms,
                "data_types": data_types,
                "sample_size": len(sample_tools),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def _generate_document_id(self, name: str, description: str) -> str:
        """Generate a unique document ID."""
        content = f"{name}_{description}_{int(time.time())}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_cache_key(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        min_score: Optional[float],
    ) -> str:
        """Generate cache key for search results."""
        cache_data = {
            "query": query,
            "limit": limit,
            "filters": filters,
            "min_score": min_score,
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _invalidate_cache(self, tool_id: str) -> None:
        """Invalidate cache entries related to a tool."""
        keys_to_remove = []
        for key in self._cache:
            if key.startswith(f"tool_{tool_id}"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]
    
    def _clear_cache(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def _cleanup_cache(self) -> None:
        """Clean up cache to maintain size limit."""
        if len(self._cache) > self._cache_max_size:
            # Remove oldest entries (simple FIFO for now)
            excess = len(self._cache) - self._cache_max_size + 100
            keys_to_remove = list(self._cache.keys())[:excess]
            for key in keys_to_remove:
                del self._cache[key]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get vector store metrics.
        
        Returns:
            Dict[str, Any]: Store metrics.
            
        Example:
            >>> metrics = store.get_metrics()
            >>> print(f"Cache hit rate: {metrics['cache_hit_rate']:.1f}%")
        """
        cache_hit_rate = 0.0
        if self.total_searches > 0:
            cache_hit_rate = (self.cache_hits / self.total_searches) * 100
        
        return {
            "total_documents": self.total_documents,
            "total_searches": self.total_searches,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
            "avg_search_time": self.avg_search_time,
            "avg_add_time": self.avg_add_time,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "embedding_model": self.config.embedding_model,
            "embedding_dimension": self.embedding_provider.dimension,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector store.
        
        Returns:
            Dict[str, Any]: Health status.
            
        Example:
            >>> health = await store.health_check()
            >>> print(f"Status: {health['status']}")
        """
        try:
            # Check basic connectivity
            document_count = await self._get_document_count()
            
            # Test embedding generation
            test_embedding = await self.embedding_provider.encode_single("test")
            
            # Test search functionality
            test_results = await self.search("test", limit=1)
            
            status = "healthy"
            issues = []
            
            if document_count == 0:
                issues.append("No documents in collection")
            
            if len(test_embedding) != self.embedding_provider.dimension:
                issues.append("Embedding dimension mismatch")
                status = "warning"
            
            if status == "healthy" and issues:
                status = "warning"
            
            return {
                "status": status,
                "document_count": document_count,
                "embedding_dimension": len(test_embedding),
                "search_functional": len(test_results) >= 0,
                "issues": issues,
                "metrics": self.get_metrics(),
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Health check failed: {e}"],
            }
    
    async def close(self) -> None:
        """Close the vector store and clean up resources.
        
        Example:
            >>> await store.close()
        """
        self.logger.info("Shutting down vector store...")
        
        # Clear cache
        self._clear_cache()
        
        # ChromaDB client cleanup (if needed)
        self.client = None
        self.collection = None
        
        self.logger.info("Vector store shutdown complete")
    
    def __repr__(self) -> str:
        """String representation of the vector store."""
        return (
            f"VectorStore("
            f"collection='{self.config.collection_name}', "
            f"documents={self.total_documents}, "
            f"model='{self.config.embedding_model}'"
            f")"
        )