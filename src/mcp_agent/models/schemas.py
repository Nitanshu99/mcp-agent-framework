"""Core data models and schemas for the MCP Agent Framework.

This module contains Pydantic models that define the data structures used
throughout the framework for search queries, results, tools, tasks, and responses.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic model usage:

    >>> query = SearchQuery(text="RNA-seq tools", max_results=10)
    >>> tool = ToolInfo(name="BLAST", description="Sequence alignment tool")
    >>> result = SearchResult(query="RNA-seq", tools=[tool], total_results=1)
"""

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator, model_validator


class TaskType(str, Enum):
    """Enumeration of task types for agent operations."""
    
    SEARCH = "search"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    REPORT = "report"
    COMPARISON = "comparison"
    WORKFLOW = "workflow"
    COORDINATION = "coordination"
    VALIDATION = "validation"


class TaskStatus(str, Enum):
    """Enumeration of task execution statuses."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ToolCategory(str, Enum):
    """Enumeration of bioinformatics tool categories."""
    
    SEQUENCE_ANALYSIS = "sequence_analysis"
    ALIGNMENT = "alignment"
    ASSEMBLY = "assembly"
    ANNOTATION = "annotation"
    PHYLOGENETICS = "phylogenetics"
    GENOMICS = "genomics"
    TRANSCRIPTOMICS = "transcriptomics"
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"
    STRUCTURAL_BIOLOGY = "structural_biology"
    MACHINE_LEARNING = "machine_learning"
    STATISTICS = "statistics"
    VISUALIZATION = "visualization"
    DATABASE = "database"
    WORKFLOW = "workflow"
    UTILITY = "utility"


class DataType(str, Enum):
    """Enumeration of biological data types."""
    
    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"
    GENOME = "genome"
    TRANSCRIPTOME = "transcriptome"
    PROTEOME = "proteome"
    METABOLOME = "metabolome"
    MICROBIOME = "microbiome"
    EPIGENOME = "epigenome"
    STRUCTURE = "structure"
    PATHWAY = "pathway"
    PHENOTYPE = "phenotype"
    CLINICAL = "clinical"


class SearchQuery(BaseModel):
    """Represents a search query for bioinformatics tools.
    
    Attributes:
        text: The search query text.
        max_results: Maximum number of results to return.
        filters: Optional filters to apply.
        include_documentation: Whether to include documentation.
        search_id: Unique identifier for this search.
        timestamp: When the search was created.
        
    Example:
        >>> query = SearchQuery(
        ...     text="RNA sequencing analysis tools",
        ...     max_results=20,
        ...     filters={"category": "transcriptomics", "organism": "human"}
        ... )
    """
    
    text: str = Field(
        description="Search query text",
        min_length=1,
        max_length=1000
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return",
        ge=1,
        le=100
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional filters to apply to the search"
    )
    include_documentation: bool = Field(
        default=True,
        description="Whether to include documentation snippets"
    )
    search_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this search"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the search query was created"
    )
    
    @validator("text")
    def validate_text(cls, v: str) -> str:
        """Validate and clean search text."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Search text cannot be empty")
        return cleaned
    
    def to_cache_key(self) -> str:
        """Generate a cache key for this search query."""
        data = {
            "text": self.text.lower().strip(),
            "max_results": self.max_results,
            "filters": sorted(self.filters.items()) if self.filters else [],
            "include_documentation": self.include_documentation,
        }
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


class ToolInfo(BaseModel):
    """Represents information about a bioinformatics tool.
    
    Attributes:
        name: Tool name.
        description: Tool description.
        category: Tool category.
        url: Tool URL or homepage.
        installation: Installation instructions.
        documentation: Documentation URL.
        version: Tool version.
        license: Tool license.
        authors: Tool authors.
        citations: Citations or papers.
        tags: List of tags.
        supported_formats: Supported file formats.
        organisms: Supported organisms.
        data_types: Supported data types.
        dependencies: Tool dependencies.
        platforms: Supported platforms.
        metadata: Additional metadata.
        
    Example:
        >>> tool = ToolInfo(
        ...     name="BLAST",
        ...     description="Basic Local Alignment Search Tool",
        ...     category=ToolCategory.SEQUENCE_ANALYSIS,
        ...     url="https://blast.ncbi.nlm.nih.gov/",
        ...     organisms=["any"],
        ...     data_types=[DataType.DNA, DataType.PROTEIN]
        ... )
    """
    
    name: str = Field(description="Tool name", min_length=1)
    description: str = Field(description="Tool description", min_length=1)
    category: Optional[ToolCategory] = Field(default=None, description="Tool category")
    url: Optional[str] = Field(default=None, description="Tool URL or homepage")
    installation: Optional[str] = Field(default=None, description="Installation instructions")
    documentation: Optional[str] = Field(default=None, description="Documentation URL")
    version: Optional[str] = Field(default=None, description="Tool version")
    license: Optional[str] = Field(default=None, description="Tool license")
    authors: List[str] = Field(default_factory=list, description="Tool authors")
    citations: List[str] = Field(default_factory=list, description="Citations or papers")
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    supported_formats: List[str] = Field(default_factory=list, description="Supported file formats")
    organisms: List[str] = Field(default_factory=list, description="Supported organisms")
    data_types: List[DataType] = Field(default_factory=list, description="Supported data types")
    dependencies: List[str] = Field(default_factory=list, description="Tool dependencies")
    platforms: List[str] = Field(default_factory=list, description="Supported platforms (Linux, Windows, macOS)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_embedding_text(self) -> str:
        """Generate text suitable for embedding creation."""
        parts = [self.name, self.description]
        
        if self.category:
            parts.append(f"Category: {self.category.value}")
        
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        
        if self.organisms:
            parts.append(f"Organisms: {', '.join(self.organisms)}")
        
        if self.data_types:
            data_type_names = [dt.value for dt in self.data_types]
            parts.append(f"Data types: {', '.join(data_type_names)}")
        
        return " ".join(parts)


class SearchResult(BaseModel):
    """Represents the results of a search query.
    
    Attributes:
        query: The original search query text.
        tools: List of found tools.
        total_results: Total number of results found.
        summary: Summary of the search results.
        execution_time: Time taken to execute the search.
        search_id: Unique identifier linking to the original query.
        metadata: Additional search metadata.
        
    Example:
        >>> result = SearchResult(
        ...     query="RNA-seq analysis",
        ...     tools=[tool1, tool2],
        ...     total_results=2,
        ...     summary="Found 2 tools for RNA-seq analysis"
        ... )
    """
    
    query: str = Field(description="Original search query text")
    tools: List[ToolInfo] = Field(description="List of found tools")
    total_results: int = Field(description="Total number of results found", ge=0)
    summary: Optional[str] = Field(default=None, description="Summary of search results")
    execution_time: Optional[float] = Field(default=None, description="Search execution time in seconds")
    search_id: Optional[str] = Field(default=None, description="Unique identifier for the search")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional search metadata")
    
    @model_validator(mode='after')
    def validate_results_consistency(self):
        """Validate that total_results matches the number of tools."""
        if len(self.tools) != self.total_results:
            # Allow total_results to be higher (truncated results)
            if self.total_results < len(self.tools):
                self.total_results = len(self.tools)
        return self
    
    def to_markdown(self) -> str:
        """Convert search results to Markdown format."""
        md = f"# Search Results: {self.query}\n\n"
        
        if self.summary:
            md += f"**Summary**: {self.summary}\n\n"
        
        md += f"**Total Results**: {self.total_results}\n\n"
        
        if self.tools:
            md += "## Tools Found\n\n"
            for i, tool in enumerate(self.tools, 1):
                md += f"### {i}. {tool.name}\n\n"
                md += f"**Description**: {tool.description}\n\n"
                
                if tool.category:
                    md += f"**Category**: {tool.category.value}\n\n"
                
                if tool.url:
                    md += f"**URL**: [{tool.url}]({tool.url})\n\n"
                
                if tool.installation:
                    md += f"**Installation**: `{tool.installation}`\n\n"
        
        return md


class AgentTask(BaseModel):
    """Represents a task for an agent to execute.
    
    Attributes:
        task_id: Unique task identifier.
        type: Type of task to execute.
        parameters: Task parameters.
        priority: Task priority.
        timeout: Task timeout in seconds.
        created_at: When the task was created.
        started_at: When task execution started.
        completed_at: When task execution completed.
        status: Current task status.
        progress: Task progress percentage.
        result: Task execution result.
        error: Error information if task failed.
        metadata: Additional task metadata.
        
    Example:
        >>> task = AgentTask(
        ...     type=TaskType.SEARCH,
        ...     parameters={"query": "BLAST tools", "max_results": 10},
        ...     priority=1
        ... )
    """
    
    task_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique task identifier"
    )
    type: TaskType = Field(description="Type of task to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task parameters"
    )
    priority: int = Field(
        default=0,
        description="Task priority (higher numbers = higher priority)",
        ge=0,
        le=10
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Task timeout in seconds"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the task was created"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="When task execution started"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When task execution completed"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current task status"
    )
    progress: float = Field(
        default=0.0,
        description="Task progress percentage",
        ge=0.0,
        le=100.0
    )
    result: Optional[Any] = Field(
        default=None,
        description="Task execution result"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error information if task failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata"
    )
    
    def mark_started(self) -> None:
        """Mark the task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
    
    def mark_completed(self, result: Any = None) -> None:
        """Mark the task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 100.0
        if result is not None:
            self.result = result
    
    def mark_failed(self, error: str) -> None:
        """Mark the task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error
    
    def get_duration(self) -> Optional[float]:
        """Get task execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class AgentResponse(BaseModel):
    """Represents a response from an agent.
    
    Attributes:
        success: Whether the operation was successful.
        content: Response content.
        data: Additional response data.
        error: Error information if operation failed.
        agent_id: ID of the agent that generated this response.
        task_id: ID of the task this response relates to.
        timestamp: When the response was generated.
        execution_time: Time taken to generate the response.
        metadata: Additional response metadata.
        
    Example:
        >>> response = AgentResponse(
        ...     success=True,
        ...     content="Search completed successfully",
        ...     data={"results": search_results},
        ...     agent_id="researcher_001"
        ... )
    """
    
    success: bool = Field(description="Whether the operation was successful")
    content: str = Field(description="Response content")
    data: Optional[Any] = Field(default=None, description="Additional response data")
    error: Optional[str] = Field(default=None, description="Error information if operation failed")
    agent_id: Optional[str] = Field(default=None, description="ID of the agent that generated this response")
    task_id: Optional[str] = Field(default=None, description="ID of the task this response relates to")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the response was generated"
    )
    execution_time: Optional[float] = Field(
        default=None,
        description="Time taken to generate the response in seconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata"
    )
    
    @model_validator(mode='after')
    def validate_error_consistency(self):
        """Validate error consistency with success flag."""
        if not self.success and not self.error:
            self.error = "Operation failed (no error details provided)"
        elif self.success and self.error:
            # If marked as success but has error, prioritize the error
            self.success = False
        return self


class ResearchData(BaseModel):
    """Represents research data collected during analysis.
    
    Attributes:
        topic: Research topic.
        sources: Data sources used.
        findings: Research findings.
        tools_identified: Tools identified during research.
        publications: Relevant publications.
        synthesis: Research synthesis.
        confidence_score: Confidence in the research quality.
        research_id: Unique research identifier.
        timestamp: When the research was conducted.
        metadata: Additional research metadata.
        
    Example:
        >>> research = ResearchData(
        ...     topic="CRISPR gene editing tools",
        ...     sources=["PubMed", "Bioconductor"],
        ...     findings=["Multiple CRISPR tools available", "Cas9 most common"],
        ...     confidence_score=0.85
        ... )
    """
    
    topic: str = Field(description="Research topic", min_length=1)
    sources: List[str] = Field(description="Data sources used in research")
    findings: List[str] = Field(description="Key research findings")
    tools_identified: List[ToolInfo] = Field(
        default_factory=list,
        description="Tools identified during research"
    )
    publications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant publications"
    )
    synthesis: Optional[str] = Field(
        default=None,
        description="Research synthesis and summary"
    )
    confidence_score: float = Field(
        default=0.0,
        description="Confidence in research quality (0-1)",
        ge=0.0,
        le=1.0
    )
    research_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique research identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the research was conducted"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional research metadata"
    )


class ReportSection(BaseModel):
    """Represents a section within a report.
    
    Attributes:
        title: Section title.
        content: Section content.
        order: Section order in the report.
        subsections: List of subsections.
        metadata: Section metadata.
        
    Example:
        >>> section = ReportSection(
        ...     title="Tool Analysis",
        ...     content="Analysis of discovered tools...",
        ...     order=2
        ... )
    """
    
    title: str = Field(description="Section title", min_length=1)
    content: str = Field(description="Section content")
    order: int = Field(description="Section order in the report", ge=0)
    subsections: List['ReportSection'] = Field(
        default_factory=list,
        description="List of subsections"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Section metadata"
    )


class ReportTemplate(BaseModel):
    """Represents a template for generating reports.
    
    Attributes:
        name: Template name.
        description: Template description.
        sections: Required sections.
        format: Output format.
        metadata: Template metadata.
        
    Example:
        >>> template = ReportTemplate(
        ...     name="Tool Search Report",
        ...     description="Template for tool search results",
        ...     sections=["summary", "tools", "recommendations"]
        ... )
    """
    
    name: str = Field(description="Template name", min_length=1)
    description: str = Field(description="Template description")
    sections: List[str] = Field(description="Required sections")
    format: str = Field(default="markdown", description="Output format")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Template metadata"
    )


class ReportMetadata(BaseModel):
    """Represents metadata for a generated report.
    
    Attributes:
        title: Report title.
        author: Report author.
        generated_at: When the report was generated.
        template_used: Template used for generation.
        data_sources: Data sources used.
        version: Report version.
        tags: Report tags.
        metadata: Additional metadata.
        
    Example:
        >>> metadata = ReportMetadata(
        ...     title="RNA-seq Tools Analysis",
        ...     author="MCP Agent Framework",
        ...     template_used="research_report"
        ... )
    """
    
    title: str = Field(description="Report title", min_length=1)
    author: str = Field(default="MCP Agent Framework", description="Report author")
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the report was generated"
    )
    template_used: Optional[str] = Field(
        default=None,
        description="Template used for generation"
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used"
    )
    version: str = Field(default="1.0", description="Report version")
    tags: List[str] = Field(default_factory=list, description="Report tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class UserQuery(BaseModel):
    """Represents a query from a user.
    
    Attributes:
        text: User query text.
        intent: Detected user intent.
        context: Query context.
        user_id: User identifier.
        session_id: Session identifier.
        timestamp: When the query was made.
        preferences: User preferences.
        metadata: Additional query metadata.
        
    Example:
        >>> query = UserQuery(
        ...     text="Find tools for protein structure prediction",
        ...     intent="tool_search",
        ...     user_id="user_123"
        ... )
    """
    
    text: str = Field(description="User query text", min_length=1, max_length=2000)
    intent: Optional[str] = Field(default=None, description="Detected user intent")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Query context"
    )
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the query was made"
    )
    preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="User preferences"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional query metadata"
    )


class UserPreferences(BaseModel):
    """Represents user preferences for search and interaction.
    
    Attributes:
        preferred_organisms: Preferred organisms for research.
        preferred_data_types: Preferred data types.
        preferred_categories: Preferred tool categories.
        max_results: Preferred maximum results.
        output_format: Preferred output format.
        include_documentation: Whether to include documentation.
        language: Preferred language.
        expertise_level: User expertise level.
        notification_settings: Notification preferences.
        metadata: Additional preferences.
        
    Example:
        >>> prefs = UserPreferences(
        ...     preferred_organisms=["human", "mouse"],
        ...     preferred_data_types=[DataType.RNA, DataType.PROTEIN],
        ...     expertise_level="intermediate"
        ... )
    """
    
    preferred_organisms: List[str] = Field(
        default_factory=list,
        description="Preferred organisms for research"
    )
    preferred_data_types: List[DataType] = Field(
        default_factory=list,
        description="Preferred data types"
    )
    preferred_categories: List[ToolCategory] = Field(
        default_factory=list,
        description="Preferred tool categories"
    )
    max_results: int = Field(
        default=10,
        description="Preferred maximum results",
        ge=1,
        le=100
    )
    output_format: str = Field(
        default="markdown",
        description="Preferred output format"
    )
    include_documentation: bool = Field(
        default=True,
        description="Whether to include documentation"
    )
    language: str = Field(default="en", description="Preferred language")
    expertise_level: str = Field(
        default="intermediate",
        description="User expertise level (beginner, intermediate, advanced)"
    )
    notification_settings: Dict[str, bool] = Field(
        default_factory=dict,
        description="Notification preferences"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional preferences"
    )


class BioinformaticsToolInfo(ToolInfo):
    """Extended tool information specific to bioinformatics.
    
    Attributes:
        input_formats: Supported input formats.
        output_formats: Supported output formats.
        algorithm_type: Type of algorithm used.
        computational_requirements: Computational requirements.
        use_cases: Common use cases.
        benchmarks: Performance benchmarks.
        tutorials: Available tutorials.
        docker_image: Docker image information.
        conda_package: Conda package information.
        
    Example:
        >>> tool = BioinformaticsToolInfo(
        ...     name="TopHat",
        ...     description="RNA-Seq read alignment",
        ...     input_formats=["FASTQ", "FASTA"],
        ...     output_formats=["BAM", "SAM"],
        ...     algorithm_type="splice-aware alignment"
        ... )
    """
    
    input_formats: List[str] = Field(
        default_factory=list,
        description="Supported input file formats"
    )
    output_formats: List[str] = Field(
        default_factory=list,
        description="Supported output file formats"
    )
    algorithm_type: Optional[str] = Field(
        default=None,
        description="Type of algorithm used"
    )
    computational_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Computational requirements (CPU, memory, etc.)"
    )
    use_cases: List[str] = Field(
        default_factory=list,
        description="Common use cases"
    )
    benchmarks: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance benchmarks"
    )
    tutorials: List[str] = Field(
        default_factory=list,
        description="Available tutorials"
    )
    docker_image: Optional[str] = Field(
        default=None,
        description="Docker image name/tag"
    )
    conda_package: Optional[str] = Field(
        default=None,
        description="Conda package name"
    )


class DatabaseInfo(BaseModel):
    """Represents information about a biological database.
    
    Attributes:
        name: Database name.
        description: Database description.
        url: Database URL.
        data_types: Types of data stored.
        organisms: Covered organisms.
        access_methods: Methods to access the database.
        update_frequency: How often the database is updated.
        size_info: Information about database size.
        contact_info: Contact information.
        metadata: Additional database metadata.
        
    Example:
        >>> db = DatabaseInfo(
        ...     name="NCBI GenBank",
        ...     description="Genetic sequence database",
        ...     data_types=[DataType.DNA, DataType.RNA],
        ...     organisms=["all"]
        ... )
    """
    
    name: str = Field(description="Database name", min_length=1)
    description: str = Field(description="Database description")
    url: Optional[str] = Field(default=None, description="Database URL")
    data_types: List[DataType] = Field(description="Types of data stored")
    organisms: List[str] = Field(description="Covered organisms")
    access_methods: List[str] = Field(
        default_factory=list,
        description="Methods to access the database (API, web, FTP, etc.)"
    )
    update_frequency: Optional[str] = Field(
        default=None,
        description="How often the database is updated"
    )
    size_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information about database size"
    )
    contact_info: Dict[str, str] = Field(
        default_factory=dict,
        description="Contact information"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional database metadata"
    )


class PublicationInfo(BaseModel):
    """Represents information about a scientific publication.
    
    Attributes:
        title: Publication title.
        authors: List of authors.
        journal: Journal name.
        year: Publication year.
        doi: Digital Object Identifier.
        pmid: PubMed ID.
        abstract: Publication abstract.
        keywords: Publication keywords.
        url: Publication URL.
        citation_count: Number of citations.
        tools_mentioned: Tools mentioned in the publication.
        metadata: Additional publication metadata.
        
    Example:
        >>> pub = PublicationInfo(
        ...     title="BLAST: a new generation of protein database search programs",
        ...     authors=["Altschul SF", "Gish W"],
        ...     journal="Nucleic Acids Research",
        ...     year=1997,
        ...     doi="10.1093/nar/25.17.3389"
        ... )
    """
    
    title: str = Field(description="Publication title", min_length=1)
    authors: List[str] = Field(description="List of authors")
    journal: Optional[str] = Field(default=None, description="Journal name")
    year: Optional[int] = Field(default=None, description="Publication year")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    pmid: Optional[str] = Field(default=None, description="PubMed ID")
    abstract: Optional[str] = Field(default=None, description="Publication abstract")
    keywords: List[str] = Field(
        default_factory=list,
        description="Publication keywords"
    )
    url: Optional[str] = Field(default=None, description="Publication URL")
    citation_count: Optional[int] = Field(
        default=None,
        description="Number of citations"
    )
    tools_mentioned: List[str] = Field(
        default_factory=list,
        description="Tools mentioned in the publication"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional publication metadata"
    )


class ValidationResult(BaseModel):
    """Represents the result of a validation operation.
    
    Attributes:
        valid: Whether the validation passed.
        errors: List of validation errors.
        warnings: List of validation warnings.
        score: Validation score (0-1).
        details: Detailed validation information.
        timestamp: When the validation was performed.
        
    Example:
        >>> result = ValidationResult(
        ...     valid=False,
        ...     errors=["Missing required field: name"],
        ...     warnings=["URL format may be invalid"],
        ...     score=0.7
        ... )
    """
    
    valid: bool = Field(description="Whether the validation passed")
    errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warnings"
    )
    score: float = Field(
        default=0.0,
        description="Validation score (0-1)",
        ge=0.0,
        le=1.0
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed validation information"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the validation was performed"
    )


class ErrorInfo(BaseModel):
    """Represents detailed error information.
    
    Attributes:
        error_id: Unique error identifier.
        error_type: Type of error.
        message: Error message.
        details: Detailed error information.
        stack_trace: Stack trace if available.
        timestamp: When the error occurred.
        context: Error context.
        severity: Error severity level.
        
    Example:
        >>> error = ErrorInfo(
        ...     error_type="ValidationError",
        ...     message="Invalid tool name format",
        ...     severity="high"
        ... )
    """
    
    error_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique error identifier"
    )
    error_type: str = Field(description="Type of error")
    message: str = Field(description="Error message")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed error information"
    )
    stack_trace: Optional[str] = Field(
        default=None,
        description="Stack trace if available"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the error occurred"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Error context"
    )
    severity: str = Field(
        default="medium",
        description="Error severity level (low, medium, high, critical)"
    )


# Update forward references
ReportSection.model_rebuild()