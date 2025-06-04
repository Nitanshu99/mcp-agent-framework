#!/usr/bin/env python3
"""Vector Database Population Script for the MCP Agent Framework.

This script provides utilities for populating ChromaDB with bioinformatics tools data,
including tool metadata, embeddings generation, and search optimization for the
MCP Agent Framework.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Usage:
    Populate with default data:
    $ python populate_vectordb.py populate

    Load from specific file:
    $ python populate_vectordb.py load --file tools_data.json

    Update existing data:
    $ python populate_vectordb.py update --source bioconductor

    Check database status:
    $ python populate_vectordb.py status

Example:
    >>> from scripts.populate_vectordb import VectorDBPopulator
    >>> populator = VectorDBPopulator()
    >>> await populator.populate_default_data()
    >>> await populator.optimize_search()
"""

import asyncio
import json
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone
import tempfile
import aiohttp
import aiofiles
from dataclasses import dataclass, asdict

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.prompt import Confirm

# Initialize rich console
console = Console()

# Tool data structure
@dataclass
class BioinformaticsTool:
    """Data structure for a bioinformatics tool."""
    name: str
    description: str
    category: str
    subcategory: Optional[str] = None
    organism: Optional[str] = None
    url: Optional[str] = None
    documentation_url: Optional[str] = None
    repository_url: Optional[str] = None
    language: Optional[str] = None
    license: Optional[str] = None
    version: Optional[str] = None
    publication_year: Optional[int] = None
    citations: Optional[int] = None
    installation: Optional[Dict[str, str]] = None
    dependencies: Optional[List[str]] = None
    input_formats: Optional[List[str]] = None
    output_formats: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    complexity: Optional[str] = None  # beginner, intermediate, advanced
    maintenance_status: Optional[str] = None  # active, inactive, deprecated
    last_updated: Optional[str] = None
    platform: Optional[List[str]] = None  # linux, windows, macos, web
    gui_available: Optional[bool] = None
    command_line: Optional[bool] = None
    api_available: Optional[bool] = None
    docker_available: Optional[bool] = None
    singularity_available: Optional[bool] = None


class VectorDBPopulator:
    """Populator for ChromaDB vector database with bioinformatics tools."""
    
    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        collection_name: str = "bioinformatics_tools",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize the vector database populator.
        
        Args:
            db_path: Path to ChromaDB database.
            collection_name: Name of the collection to populate.
            embedding_model: Name of the embedding model to use.
        """
        self.db_path = Path(db_path) if db_path else Path("./data/chroma")
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        self.client = None
        self.collection = None
        self.tools_data: List[BioinformaticsTool] = []
        
        # Data sources
        self.data_sources = {
            "default": self._get_default_tools_data,
            "bioconductor": self._fetch_bioconductor_data,
            "biotools": self._fetch_biotools_data,
            "galaxy": self._fetch_galaxy_tools,
            "ncbi": self._fetch_ncbi_tools,
            "ebi": self._fetch_ebi_tools,
        }
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Ensure database directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self._get_embedding_function(),
                )
                console.print(f"[green]Connected to existing collection: {self.collection_name}[/green]")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self._get_embedding_function(),
                    metadata={"description": "Bioinformatics tools and resources"}
                )
                console.print(f"[green]Created new collection: {self.collection_name}[/green]")
                
        except ImportError:
            console.print("[red]ChromaDB not available. Install with: pip install chromadb[/red]")
            raise
        except Exception as e:
            console.print(f"[red]Failed to initialize ChromaDB: {e}[/red]")
            raise
    
    def _get_embedding_function(self):
        """Get embedding function for ChromaDB."""
        try:
            from chromadb.utils import embedding_functions
            
            # Try sentence-transformers first
            try:
                return embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model
                )
            except Exception:
                # Fallback to default embedding function
                console.print("[yellow]Using default embedding function[/yellow]")
                return embedding_functions.DefaultEmbeddingFunction()
                
        except ImportError:
            console.print("[yellow]Using ChromaDB default embedding function[/yellow]")
            return None
    
    async def populate_default_data(self) -> int:
        """Populate database with default bioinformatics tools data.
        
        Returns:
            int: Number of tools added.
        """
        console.print("[blue]Loading default bioinformatics tools data...[/blue]")
        
        tools = await self._get_default_tools_data()
        return await self._add_tools_to_db(tools)
    
    async def load_from_file(self, file_path: Union[str, Path]) -> int:
        """Load tools data from file.
        
        Args:
            file_path: Path to data file (JSON, CSV, or YAML).
            
        Returns:
            int: Number of tools loaded.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            return 0
        
        console.print(f"[blue]Loading data from {file_path}...[/blue]")
        
        try:
            if file_path.suffix.lower() == '.json':
                tools = await self._load_json_file(file_path)
            elif file_path.suffix.lower() == '.csv':
                tools = await self._load_csv_file(file_path)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                tools = await self._load_yaml_file(file_path)
            else:
                console.print(f"[red]Unsupported file format: {file_path.suffix}[/red]")
                return 0
            
            return await self._add_tools_to_db(tools)
            
        except Exception as e:
            console.print(f"[red]Error loading file: {e}[/red]")
            return 0
    
    async def load_from_source(self, source_name: str) -> int:
        """Load tools data from external source.
        
        Args:
            source_name: Name of the data source.
            
        Returns:
            int: Number of tools loaded.
        """
        if source_name not in self.data_sources:
            console.print(f"[red]Unknown data source: {source_name}[/red]")
            console.print(f"Available sources: {', '.join(self.data_sources.keys())}")
            return 0
        
        console.print(f"[blue]Loading data from source: {source_name}...[/blue]")
        
        try:
            loader_func = self.data_sources[source_name]
            tools = await loader_func()
            return await self._add_tools_to_db(tools)
            
        except Exception as e:
            console.print(f"[red]Error loading from source {source_name}: {e}[/red]")
            return 0
    
    async def _add_tools_to_db(self, tools: List[BioinformaticsTool]) -> int:
        """Add tools to the database.
        
        Args:
            tools: List of tools to add.
            
        Returns:
            int: Number of tools successfully added.
        """
        if not tools:
            console.print("[yellow]No tools to add[/yellow]")
            return 0
        
        added_count = 0
        batch_size = 50
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Adding tools to database...", total=len(tools))
            
            # Process tools in batches
            for i in range(0, len(tools), batch_size):
                batch = tools[i:i + batch_size]
                
                try:
                    # Prepare batch data
                    ids = []
                    documents = []
                    metadatas = []
                    
                    for tool in batch:
                        # Create unique ID
                        tool_id = f"{tool.category}_{tool.name}".replace(" ", "_").lower()
                        ids.append(tool_id)
                        
                        # Create document text for embedding
                        doc_text = self._create_document_text(tool)
                        documents.append(doc_text)
                        
                        # Prepare metadata
                        metadata = self._create_metadata(tool)
                        metadatas.append(metadata)
                    
                    # Add batch to collection
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                    )
                    
                    added_count += len(batch)
                    progress.update(task, advance=len(batch))
                    
                except Exception as e:
                    console.print(f"[red]Error adding batch: {e}[/red]")
                    continue
        
        console.print(f"[green]Successfully added {added_count} tools to database[/green]")
        return added_count
    
    def _create_document_text(self, tool: BioinformaticsTool) -> str:
        """Create document text for embedding.
        
        Args:
            tool: Tool information.
            
        Returns:
            str: Document text for embedding.
        """
        # Combine relevant fields for better search
        text_parts = [
            f"Tool: {tool.name}",
            f"Description: {tool.description}",
            f"Category: {tool.category}",
        ]
        
        if tool.subcategory:
            text_parts.append(f"Subcategory: {tool.subcategory}")
        
        if tool.organism:
            text_parts.append(f"Organism: {tool.organism}")
        
        if tool.language:
            text_parts.append(f"Language: {tool.language}")
        
        if tool.tags:
            text_parts.append(f"Tags: {', '.join(tool.tags)}")
        
        if tool.keywords:
            text_parts.append(f"Keywords: {', '.join(tool.keywords)}")
        
        if tool.input_formats:
            text_parts.append(f"Input formats: {', '.join(tool.input_formats)}")
        
        if tool.output_formats:
            text_parts.append(f"Output formats: {', '.join(tool.output_formats)}")
        
        return " | ".join(text_parts)
    
    def _create_metadata(self, tool: BioinformaticsTool) -> Dict[str, Any]:
        """Create metadata dictionary for ChromaDB.
        
        Args:
            tool: Tool information.
            
        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        # Convert tool to dict and filter out None values
        metadata = {}
        tool_dict = asdict(tool)
        
        for key, value in tool_dict.items():
            if value is not None:
                # Convert lists to comma-separated strings for ChromaDB
                if isinstance(value, list):
                    metadata[key] = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    # Convert dicts to JSON strings
                    metadata[key] = json.dumps(value)
                else:
                    metadata[key] = str(value)
        
        # Add timestamp
        metadata["added_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return metadata
    
    async def _get_default_tools_data(self) -> List[BioinformaticsTool]:
        """Get default bioinformatics tools data.
        
        Returns:
            List[BioinformaticsTool]: List of default tools.
        """
        # Comprehensive list of bioinformatics tools across categories
        default_tools = [
            # Sequence Analysis Tools
            BioinformaticsTool(
                name="BLAST+",
                description="Basic Local Alignment Search Tool for sequence similarity searching",
                category="sequence_analysis",
                subcategory="alignment",
                organism="universal",
                url="https://blast.ncbi.nlm.nih.gov/",
                documentation_url="https://blast.ncbi.nlm.nih.gov/doc/blast-help/",
                repository_url="https://github.com/ncbi/blast",
                language="C++",
                license="Public Domain",
                version="2.14.0",
                publication_year=1990,
                citations=50000,
                installation={
                    "conda": "conda install -c bioconda blast",
                    "docker": "ncbi/blast",
                    "source": "ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/"
                },
                input_formats=["FASTA", "GenBank"],
                output_formats=["XML", "tabular", "ASN.1"],
                tags=["alignment", "similarity", "database_search"],
                keywords=["sequence", "alignment", "homology", "similarity"],
                complexity="intermediate",
                maintenance_status="active",
                platform=["linux", "windows", "macos"],
                command_line=True,
                api_available=True,
                docker_available=True,
            ),
            
            BioinformaticsTool(
                name="BWA",
                description="Burrows-Wheeler Aligner for mapping DNA sequences against reference genome",
                category="sequence_analysis",
                subcategory="alignment",
                organism="universal",
                url="http://bio-bwa.sourceforge.net/",
                repository_url="https://github.com/lh3/bwa",
                language="C",
                license="MIT",
                version="0.7.17",
                publication_year=2009,
                citations=15000,
                installation={
                    "conda": "conda install -c bioconda bwa",
                    "apt": "apt-get install bwa"
                },
                input_formats=["FASTA", "FASTQ"],
                output_formats=["SAM"],
                tags=["alignment", "mapping", "reference_genome"],
                keywords=["DNA", "mapping", "alignment", "BWA"],
                complexity="intermediate",
                maintenance_status="active",
                platform=["linux", "macos"],
                command_line=True,
            ),
            
            # RNA-seq Analysis Tools
            BioinformaticsTool(
                name="STAR",
                description="Spliced Transcripts Alignment to a Reference",
                category="transcriptomics",
                subcategory="alignment",
                organism="universal",
                url="https://github.com/alexdobin/STAR",
                documentation_url="https://physiology.med.cornell.edu/faculty/skrabanek/lab/angsd/lecture_notes/STARmanual.pdf",
                repository_url="https://github.com/alexdobin/STAR",
                language="C++",
                license="MIT",
                version="2.7.10a",
                publication_year=2013,
                citations=8000,
                installation={
                    "conda": "conda install -c bioconda star",
                    "docker": "quay.io/biocontainers/star"
                },
                input_formats=["FASTQ", "FASTA"],
                output_formats=["SAM", "BAM"],
                tags=["RNA-seq", "alignment", "splicing"],
                keywords=["RNA", "transcriptome", "splicing", "alignment"],
                complexity="advanced",
                maintenance_status="active",
                platform=["linux", "macos"],
                command_line=True,
                docker_available=True,
            ),
            
            BioinformaticsTool(
                name="DESeq2",
                description="Differential gene expression analysis based on negative binomial distribution",
                category="transcriptomics",
                subcategory="differential_expression",
                organism="universal",
                url="https://bioconductor.org/packages/DESeq2/",
                documentation_url="https://bioconductor.org/packages/devel/bioc/vignettes/DESeq2/inst/doc/DESeq2.html",
                repository_url="https://github.com/mikelove/DESeq2",
                language="R",
                license="LGPL",
                version="1.40.0",
                publication_year=2014,
                citations=12000,
                installation={
                    "bioconductor": "BiocManager::install('DESeq2')",
                    "conda": "conda install -c bioconductor bioconductor-deseq2"
                },
                input_formats=["count_matrix", "SummarizedExperiment"],
                output_formats=["R_objects", "CSV", "plots"],
                tags=["differential_expression", "statistics", "RNA-seq"],
                keywords=["differential", "expression", "statistics", "RNA-seq"],
                complexity="intermediate",
                maintenance_status="active",
                platform=["linux", "windows", "macos"],
                gui_available=False,
                command_line=False,
                api_available=True,
            ),
            
            # Protein Analysis Tools
            BioinformaticsTool(
                name="AlphaFold",
                description="AI system for protein structure prediction",
                category="proteomics",
                subcategory="structure_prediction",
                organism="universal",
                url="https://alphafold.ebi.ac.uk/",
                documentation_url="https://github.com/deepmind/alphafold",
                repository_url="https://github.com/deepmind/alphafold",
                language="Python",
                license="Apache 2.0",
                version="2.3.0",
                publication_year=2021,
                citations=5000,
                installation={
                    "docker": "deepmind/alphafold",
                    "conda": "conda install -c conda-forge alphafold2"
                },
                input_formats=["FASTA"],
                output_formats=["PDB", "mmCIF"],
                tags=["protein_structure", "AI", "deep_learning"],
                keywords=["protein", "structure", "prediction", "AI"],
                complexity="advanced",
                maintenance_status="active",
                platform=["linux"],
                command_line=True,
                docker_available=True,
            ),
            
            BioinformaticsTool(
                name="HMMER",
                description="Profile HMMs for protein sequence analysis",
                category="proteomics",
                subcategory="sequence_analysis",
                organism="universal",
                url="http://hmmer.org/",
                documentation_url="http://hmmer.org/documentation.html",
                repository_url="https://github.com/EddyRivasLab/hmmer",
                language="C",
                license="BSD",
                version="3.3.2",
                publication_year=1998,
                citations=8000,
                installation={
                    "conda": "conda install -c bioconda hmmer",
                    "apt": "apt-get install hmmer"
                },
                input_formats=["FASTA", "Stockholm"],
                output_formats=["tabular", "alignment"],
                tags=["HMM", "protein_domains", "homology"],
                keywords=["HMM", "protein", "domains", "homology"],
                complexity="intermediate",
                maintenance_status="active",
                platform=["linux", "macos", "windows"],
                command_line=True,
            ),
            
            # Genomics Tools
            BioinformaticsTool(
                name="GATK",
                description="Genome Analysis Toolkit for variant discovery and genotyping",
                category="genomics",
                subcategory="variant_calling",
                organism="universal",
                url="https://gatk.broadinstitute.org/",
                documentation_url="https://gatk.broadinstitute.org/hc/en-us",
                repository_url="https://github.com/broadinstitute/gatk",
                language="Java",
                license="BSD",
                version="4.4.0.0",
                publication_year=2010,
                citations=15000,
                installation={
                    "conda": "conda install -c bioconda gatk4",
                    "docker": "broadinstitute/gatk"
                },
                input_formats=["SAM", "BAM", "CRAM", "VCF"],
                output_formats=["VCF", "GVCF", "BAM"],
                tags=["variant_calling", "SNP", "indel"],
                keywords=["variant", "SNP", "indel", "genotyping"],
                complexity="advanced",
                maintenance_status="active",
                platform=["linux", "macos", "windows"],
                command_line=True,
                docker_available=True,
            ),
            
            BioinformaticsTool(
                name="SPAdes",
                description="St. Petersburg genome assembler",
                category="genomics",
                subcategory="assembly",
                organism="universal",
                url="https://cab.spbu.ru/software/spades/",
                documentation_url="https://cab.spbu.ru/files/release3.15.5/manual.html",
                repository_url="https://github.com/ablab/spades",
                language="C++",
                license="GPLv2",
                version="3.15.5",
                publication_year=2012,
                citations=5000,
                installation={
                    "conda": "conda install -c bioconda spades",
                    "docker": "quay.io/biocontainers/spades"
                },
                input_formats=["FASTQ", "FASTA"],
                output_formats=["FASTA", "GFA"],
                tags=["assembly", "genome", "scaffolding"],
                keywords=["assembly", "genome", "contigs", "scaffolds"],
                complexity="advanced",
                maintenance_status="active",
                platform=["linux", "macos"],
                command_line=True,
                docker_available=True,
            ),
            
            # Phylogenetics Tools
            BioinformaticsTool(
                name="RAxML",
                description="Randomized Axelerated Maximum Likelihood for phylogenetic inference",
                category="phylogenetics",
                subcategory="tree_building",
                organism="universal",
                url="https://cme.h-its.org/exelixis/web/software/raxml/",
                documentation_url="https://cme.h-its.org/exelixis/resource/download/NewManual.pdf",
                repository_url="https://github.com/stamatak/standard-RAxML",
                language="C",
                license="GPLv3",
                version="8.2.12",
                publication_year=2006,
                citations=8000,
                installation={
                    "conda": "conda install -c bioconda raxml",
                    "apt": "apt-get install raxml"
                },
                input_formats=["PHYLIP", "FASTA"],
                output_formats=["Newick", "phylip"],
                tags=["phylogeny", "maximum_likelihood", "evolution"],
                keywords=["phylogeny", "tree", "evolution", "ML"],
                complexity="advanced",
                maintenance_status="active",
                platform=["linux", "macos", "windows"],
                command_line=True,
            ),
            
            BioinformaticsTool(
                name="BEAST2",
                description="Bayesian Evolutionary Analysis Sampling Trees",
                category="phylogenetics",
                subcategory="bayesian_inference",
                organism="universal",
                url="https://www.beast2.org/",
                documentation_url="https://www.beast2.org/tutorials/",
                repository_url="https://github.com/CompEvol/beast2",
                language="Java",
                license="LGPL",
                version="2.7.3",
                publication_year=2014,
                citations=3000,
                installation={
                    "download": "https://www.beast2.org/managing-packages/",
                    "conda": "conda install -c conda-forge beast2"
                },
                input_formats=["XML", "NEXUS"],
                output_formats=["log", "trees", "XML"],
                tags=["bayesian", "phylogeny", "mcmc"],
                keywords=["bayesian", "phylogeny", "MCMC", "dating"],
                complexity="advanced",
                maintenance_status="active",
                platform=["linux", "macos", "windows"],
                gui_available=True,
                command_line=True,
            ),
            
            # Data Processing Tools
            BioinformaticsTool(
                name="SAMtools",
                description="Tools for manipulating alignments in SAM/BAM format",
                category="utilities",
                subcategory="file_processing",
                organism="universal",
                url="http://samtools.sourceforge.net/",
                documentation_url="http://www.htslib.org/doc/samtools.html",
                repository_url="https://github.com/samtools/samtools",
                language="C",
                license="MIT",
                version="1.17",
                publication_year=2009,
                citations=20000,
                installation={
                    "conda": "conda install -c bioconda samtools",
                    "apt": "apt-get install samtools"
                },
                input_formats=["SAM", "BAM", "CRAM"],
                output_formats=["SAM", "BAM", "CRAM", "FASTA"],
                tags=["alignment", "file_processing", "format_conversion"],
                keywords=["SAM", "BAM", "alignment", "processing"],
                complexity="beginner",
                maintenance_status="active",
                platform=["linux", "macos", "windows"],
                command_line=True,
            ),
            
            BioinformaticsTool(
                name="BEDTools",
                description="Toolset for genome arithmetic and interval operations",
                category="utilities",
                subcategory="genome_arithmetic",
                organism="universal",
                url="https://bedtools.readthedocs.io/",
                documentation_url="https://bedtools.readthedocs.io/en/latest/",
                repository_url="https://github.com/arq5x/bedtools2",
                language="C++",
                license="MIT",
                version="2.31.0",
                publication_year=2010,
                citations=5000,
                installation={
                    "conda": "conda install -c bioconda bedtools",
                    "apt": "apt-get install bedtools"
                },
                input_formats=["BED", "GFF", "GTF", "VCF"],
                output_formats=["BED", "GFF", "GTF"],
                tags=["genome_arithmetic", "intervals", "annotations"],
                keywords=["BED", "intervals", "genome", "arithmetic"],
                complexity="intermediate",
                maintenance_status="active",
                platform=["linux", "macos"],
                command_line=True,
            ),
            
            # Quality Control Tools
            BioinformaticsTool(
                name="FastQC",
                description="Quality control tool for high throughput sequence data",
                category="quality_control",
                subcategory="sequence_qc",
                organism="universal",
                url="https://www.bioinformatics.babraham.ac.uk/projects/fastqc/",
                documentation_url="https://www.bioinformatics.babraham.ac.uk/projects/fastqc/Help/",
                repository_url="https://github.com/s-andrews/FastQC",
                language="Java",
                license="GPLv3",
                version="0.12.1",
                publication_year=2010,
                citations=8000,
                installation={
                    "conda": "conda install -c bioconda fastqc",
                    "apt": "apt-get install fastqc"
                },
                input_formats=["FASTQ", "SAM", "BAM"],
                output_formats=["HTML", "ZIP"],
                tags=["quality_control", "sequencing", "reports"],
                keywords=["quality", "QC", "sequencing", "FASTQ"],
                complexity="beginner",
                maintenance_status="active",
                platform=["linux", "macos", "windows"],
                gui_available=True,
                command_line=True,
            ),
            
            # Visualization Tools
            BioinformaticsTool(
                name="IGV",
                description="Integrative Genomics Viewer for visualization of genomic data",
                category="visualization",
                subcategory="genome_browser",
                organism="universal",
                url="https://software.broadinstitute.org/software/igv/",
                documentation_url="https://software.broadinstitute.org/software/igv/UserGuide",
                repository_url="https://github.com/igvteam/igv",
                language="Java",
                license="MIT",
                version="2.16.2",
                publication_year=2011,
                citations=4000,
                installation={
                    "download": "https://software.broadinstitute.org/software/igv/download",
                    "conda": "conda install -c bioconda igv"
                },
                input_formats=["BAM", "VCF", "BED", "GFF", "BigWig"],
                output_formats=["SVG", "PNG", "session_files"],
                tags=["visualization", "genome_browser", "interactive"],
                keywords=["visualization", "genome", "browser", "interactive"],
                complexity="beginner",
                maintenance_status="active",
                platform=["linux", "macos", "windows"],
                gui_available=True,
                api_available=True,
            ),
        ]
        
        return default_tools
    
    async def _load_json_file(self, file_path: Path) -> List[BioinformaticsTool]:
        """Load tools from JSON file."""
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            data = json.loads(content)
        
        tools = []
        if isinstance(data, list):
            for item in data:
                tools.append(BioinformaticsTool(**item))
        elif isinstance(data, dict) and 'tools' in data:
            for item in data['tools']:
                tools.append(BioinformaticsTool(**item))
        
        return tools
    
    async def _load_csv_file(self, file_path: Path) -> List[BioinformaticsTool]:
        """Load tools from CSV file."""
        tools = []
        
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            
        # Parse CSV
        reader = csv.DictReader(content.splitlines())
        for row in reader:
            # Convert string representations back to appropriate types
            processed_row = {}
            for key, value in row.items():
                if value == '':
                    processed_row[key] = None
                elif key in ['tags', 'keywords', 'input_formats', 'output_formats', 'dependencies', 'platform']:
                    # Convert comma-separated strings back to lists
                    processed_row[key] = [item.strip() for item in value.split(',') if item.strip()]
                elif key == 'installation':
                    # Convert JSON string back to dict
                    try:
                        processed_row[key] = json.loads(value) if value else None
                    except json.JSONDecodeError:
                        processed_row[key] = None
                elif key in ['publication_year', 'citations']:
                    # Convert to int
                    try:
                        processed_row[key] = int(value) if value else None
                    except ValueError:
                        processed_row[key] = None
                elif key in ['gui_available', 'command_line', 'api_available', 'docker_available', 'singularity_available']:
                    # Convert to bool
                    processed_row[key] = value.lower() == 'true' if value else None
                else:
                    processed_row[key] = value
            
            tools.append(BioinformaticsTool(**processed_row))
        
        return tools
    
    async def _load_yaml_file(self, file_path: Path) -> List[BioinformaticsTool]:
        """Load tools from YAML file."""
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            data = yaml.safe_load(content)
        
        tools = []
        if isinstance(data, list):
            for item in data:
                tools.append(BioinformaticsTool(**item))
        elif isinstance(data, dict) and 'tools' in data:
            for item in data['tools']:
                tools.append(BioinformaticsTool(**item))
        
        return tools
    
    async def _fetch_bioconductor_data(self) -> List[BioinformaticsTool]:
        """Fetch tools data from Bioconductor API."""
        tools = []
        
        try:
            url = "https://bioconductor.org/packages/json/3.18/bioc/packages.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process a subset of packages to avoid overwhelming the database
                        package_names = list(data.keys())[:100]  # Limit to first 100
                        
                        for package_name in package_names:
                            package_info = data[package_name]
                            
                            tool = BioinformaticsTool(
                                name=package_name,
                                description=package_info.get('Title', ''),
                                category="bioconductor_package",
                                organism="universal",
                                url=f"https://bioconductor.org/packages/{package_name}/",
                                language="R",
                                license=package_info.get('License', ''),
                                version=package_info.get('Version', ''),
                                installation={"bioconductor": f"BiocManager::install('{package_name}')"},
                                tags=["bioconductor", "R_package"],
                                maintenance_status="active",
                                platform=["linux", "macos", "windows"],
                            )
                            tools.append(tool)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch Bioconductor data: {e}[/yellow]")
        
        return tools
    
    async def _fetch_biotools_data(self) -> List[BioinformaticsTool]:
        """Fetch tools data from bio.tools registry."""
        tools = []
        
        try:
            # bio.tools API endpoint
            url = "https://bio.tools/api/tool/"
            params = {"format": "json", "page_size": 100}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for tool_data in data.get('list', []):
                            # Extract relevant information
                            name = tool_data.get('name', '')
                            description = tool_data.get('description', '')
                            
                            # Extract topics as categories
                            topics = tool_data.get('topic', [])
                            category = topics[0].get('term', 'general') if topics else 'general'
                            
                            # Extract operations as subcategories
                            operations = tool_data.get('operation', [])
                            subcategory = operations[0].get('term', '') if operations else None
                            
                            # Extract links
                            links = tool_data.get('link', [])
                            url = None
                            repo_url = None
                            doc_url = None
                            
                            for link in links:
                                link_type = link.get('type', '')
                                if link_type == 'Browser' and not url:
                                    url = link.get('url')
                                elif link_type == 'Repository':
                                    repo_url = link.get('url')
                                elif link_type == 'Helpdesk':
                                    doc_url = link.get('url')
                            
                            tool = BioinformaticsTool(
                                name=name,
                                description=description,
                                category=category.lower().replace(' ', '_'),
                                subcategory=subcategory.lower().replace(' ', '_') if subcategory else None,
                                url=url,
                                repository_url=repo_url,
                                documentation_url=doc_url,
                                tags=["biotools"],
                                maintenance_status="active",
                            )
                            tools.append(tool)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch bio.tools data: {e}[/yellow]")
        
        return tools
    
    async def _fetch_galaxy_tools(self) -> List[BioinformaticsTool]:
        """Fetch tools from Galaxy Tool Shed."""
        tools = []
        
        try:
            # Galaxy Tool Shed API
            url = "https://toolshed.g2.bx.psu.edu/api/repositories"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process a subset to avoid overwhelming
                        for tool_data in data[:50]:  # Limit to first 50
                            name = tool_data.get('name', '')
                            description = tool_data.get('description', '')
                            
                            tool = BioinformaticsTool(
                                name=name,
                                description=description,
                                category="galaxy_tool",
                                url=f"https://toolshed.g2.bx.psu.edu/view/{tool_data.get('owner', '')}/{name}",
                                tags=["galaxy", "workflow_tool"],
                                platform=["web"],
                                gui_available=True,
                                maintenance_status="active",
                            )
                            tools.append(tool)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch Galaxy tools data: {e}[/yellow]")
        
        return tools
    
    async def _fetch_ncbi_tools(self) -> List[BioinformaticsTool]:
        """Fetch NCBI tools information."""
        # Static list of major NCBI tools since there's no comprehensive API
        ncbi_tools = [
            BioinformaticsTool(
                name="BLAST+",
                description="Basic Local Alignment Search Tool",
                category="sequence_analysis",
                subcategory="alignment",
                url="https://blast.ncbi.nlm.nih.gov/",
                organization="NCBI",
                tags=["ncbi", "alignment", "sequence"],
                maintenance_status="active",
            ),
            BioinformaticsTool(
                name="Entrez Programming Utilities",
                description="Tools for accessing NCBI databases programmatically",
                category="database_access",
                url="https://www.ncbi.nlm.nih.gov/books/NBK25501/",
                language="Various",
                tags=["ncbi", "database", "api"],
                api_available=True,
                maintenance_status="active",
            ),
            BioinformaticsTool(
                name="NCBI Datasets",
                description="Access to NCBI's genome, transcriptome and protein data",
                category="data_retrieval",
                url="https://www.ncbi.nlm.nih.gov/datasets/",
                tags=["ncbi", "genome", "data"],
                command_line=True,
                api_available=True,
                maintenance_status="active",
            ),
        ]
        
        return ncbi_tools
    
    async def _fetch_ebi_tools(self) -> List[BioinformaticsTool]:
        """Fetch EBI tools information."""
        # Static list of major EBI tools
        ebi_tools = [
            BioinformaticsTool(
                name="InterPro",
                description="Protein sequence analysis and classification",
                category="proteomics",
                subcategory="protein_analysis",
                url="https://www.ebi.ac.uk/interpro/",
                organization="EBI",
                tags=["ebi", "protein", "domains"],
                api_available=True,
                maintenance_status="active",
            ),
            BioinformaticsTool(
                name="Ensembl",
                description="Genome browser and annotation database",
                category="genomics",
                subcategory="genome_annotation",
                url="https://www.ensembl.org/",
                organization="EBI",
                tags=["ebi", "genome", "annotation"],
                gui_available=True,
                api_available=True,
                maintenance_status="active",
            ),
            BioinformaticsTool(
                name="UniProt",
                description="Protein sequence and annotation database",
                category="proteomics",
                subcategory="protein_database",
                url="https://www.uniprot.org/",
                organization="EBI",
                tags=["ebi", "protein", "database"],
                gui_available=True,
                api_available=True,
                maintenance_status="active",
            ),
        ]
        
        return ebi_tools
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dict[str, Any]: Database statistics.
        """
        if not self.collection:
            await self.initialize()
        
        try:
            # Get collection info
            collection_count = self.collection.count()
            
            # Get some sample documents to analyze categories
            sample_docs = self.collection.get(limit=1000)
            
            # Analyze categories
            categories = {}
            languages = {}
            maintenance_status = {}
            
            for metadata in sample_docs['metadatas']:
                # Count categories
                category = metadata.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                
                # Count languages
                language = metadata.get('language', 'unknown')
                languages[language] = languages.get(language, 0) + 1
                
                # Count maintenance status
                status = metadata.get('maintenance_status', 'unknown')
                maintenance_status[status] = maintenance_status.get(status, 0) + 1
            
            return {
                "total_tools": collection_count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "categories": categories,
                "languages": languages,
                "maintenance_status": maintenance_status,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            
        except Exception as e:
            console.print(f"[red]Error getting database stats: {e}[/red]")
            return {"error": str(e)}
    
    async def search_tools(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search tools in the database.
        
        Args:
            query: Search query.
            n_results: Number of results to return.
            
        Returns:
            List[Dict[str, Any]]: Search results.
        """
        if not self.collection:
            await self.initialize()
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None,
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            console.print(f"[red]Error searching tools: {e}[/red]")
            return []
    
    async def update_tool(self, tool_id: str, tool: BioinformaticsTool) -> bool:
        """Update a tool in the database.
        
        Args:
            tool_id: ID of the tool to update.
            tool: Updated tool information.
            
        Returns:
            bool: True if update was successful.
        """
        if not self.collection:
            await self.initialize()
        
        try:
            # Delete existing entry
            self.collection.delete(ids=[tool_id])
            
            # Add updated entry
            doc_text = self._create_document_text(tool)
            metadata = self._create_metadata(tool)
            
            self.collection.add(
                ids=[tool_id],
                documents=[doc_text],
                metadatas=[metadata],
            )
            
            console.print(f"[green]Updated tool: {tool_id}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error updating tool {tool_id}: {e}[/red]")
            return False
    
    async def delete_tool(self, tool_id: str) -> bool:
        """Delete a tool from the database.
        
        Args:
            tool_id: ID of the tool to delete.
            
        Returns:
            bool: True if deletion was successful.
        """
        if not self.collection:
            await self.initialize()
        
        try:
            self.collection.delete(ids=[tool_id])
            console.print(f"[green]Deleted tool: {tool_id}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error deleting tool {tool_id}: {e}[/red]")
            return False
    
    async def optimize_database(self) -> None:
        """Optimize database for better search performance."""
        if not self.collection:
            await self.initialize()
        
        console.print("[blue]Optimizing database for search performance...[/blue]")
        
        # ChromaDB doesn't require explicit optimization, but we can provide tips
        stats = await self.get_database_stats()
        
        console.print(f"[green]Database contains {stats.get('total_tools', 0)} tools[/green]")
        console.print("[green]ChromaDB automatically optimizes embeddings for search[/green]")
    
    async def export_tools(self, output_file: Union[str, Path], format: str = "json") -> bool:
        """Export tools data from database.
        
        Args:
            output_file: Output file path.
            format: Export format (json, csv, yaml).
            
        Returns:
            bool: True if export was successful.
        """
        if not self.collection:
            await self.initialize()
        
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            # Convert to tools format
            tools_data = []
            for i, metadata in enumerate(all_docs['metadatas']):
                # Convert back to tool format
                tool_data = {}
                for key, value in metadata.items():
                    if key == 'added_timestamp':
                        continue
                    
                    # Convert back from string format
                    if key in ['tags', 'keywords', 'input_formats', 'output_formats', 'dependencies', 'platform']:
                        tool_data[key] = [item.strip() for item in value.split(',') if item.strip()] if value else None
                    elif key == 'installation':
                        try:
                            tool_data[key] = json.loads(value) if value else None
                        except json.JSONDecodeError:
                            tool_data[key] = None
                    elif key in ['publication_year', 'citations']:
                        try:
                            tool_data[key] = int(value) if value else None
                        except ValueError:
                            tool_data[key] = None
                    elif key in ['gui_available', 'command_line', 'api_available', 'docker_available', 'singularity_available']:
                        tool_data[key] = value.lower() == 'true' if value else None
                    else:
                        tool_data[key] = value if value else None
                
                tools_data.append(tool_data)
            
            # Export in requested format
            output_path = Path(output_file)
            
            if format.lower() == 'json':
                async with aiofiles.open(output_path, 'w') as f:
                    await f.write(json.dumps({"tools": tools_data}, indent=2))
            
            elif format.lower() == 'csv':
                # Get all unique fields
                all_fields = set()
                for tool in tools_data:
                    all_fields.update(tool.keys())
                
                # Write CSV
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(all_fields))
                    writer.writeheader()
                    
                    for tool in tools_data:
                        # Convert lists and dicts to strings for CSV
                        csv_row = {}
                        for field in all_fields:
                            value = tool.get(field)
                            if isinstance(value, list):
                                csv_row[field] = ', '.join(str(v) for v in value)
                            elif isinstance(value, dict):
                                csv_row[field] = json.dumps(value)
                            else:
                                csv_row[field] = value
                        writer.writerow(csv_row)
            
            elif format.lower() in ['yaml', 'yml']:
                async with aiofiles.open(output_path, 'w') as f:
                    await f.write(yaml.dump({"tools": tools_data}, default_flow_style=False))
            
            else:
                console.print(f"[red]Unsupported export format: {format}[/red]")
                return False
            
            console.print(f"[green]Exported {len(tools_data)} tools to {output_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error exporting tools: {e}[/red]")
            return False


# CLI Commands
@click.group()
@click.option("--db-path", help="Path to ChromaDB database", default="./data/chroma")
@click.option("--collection", help="Collection name", default="bioinformatics_tools")
@click.option("--embedding-model", help="Embedding model name", default="all-MiniLM-L6-v2")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, db_path: str, collection: str, embedding_model: str, verbose: bool):
    """Vector Database Population Tool for MCP Agent Framework."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path
    ctx.obj["collection"] = collection
    ctx.obj["embedding_model"] = embedding_model
    ctx.obj["verbose"] = verbose
    ctx.obj["populator"] = VectorDBPopulator(db_path, collection, embedding_model)


@cli.command()
@click.option("--source", help="Data source to load", default="default")
@click.pass_context
def populate(ctx: click.Context, source: str):
    """Populate database with tools data."""
    populator = ctx.obj["populator"]
    
    async def run_populate():
        await populator.initialize()
        
        if source == "default":
            count = await populator.populate_default_data()
        else:
            count = await populator.load_from_source(source)
        
        if count > 0:
            console.print(f"[green]Successfully populated database with {count} tools[/green]")
        else:
            console.print("[red]No tools were added to the database[/red]")
    
    asyncio.run(run_populate())


@cli.command()
@click.option("--file", "file_path", required=True, help="Path to data file")
@click.pass_context
def load(ctx: click.Context, file_path: str):
    """Load tools data from file."""
    populator = ctx.obj["populator"]
    
    async def run_load():
        await populator.initialize()
        count = await populator.load_from_file(file_path)
        
        if count > 0:
            console.print(f"[green]Successfully loaded {count} tools from {file_path}[/green]")
        else:
            console.print(f"[red]No tools were loaded from {file_path}[/red]")
    
    asyncio.run(run_load())


@cli.command()
@click.option("--query", required=True, help="Search query")
@click.option("--limit", default=10, help="Number of results")
@click.pass_context
def search(ctx: click.Context, query: str, limit: int):
    """Search tools in the database."""
    populator = ctx.obj["populator"]
    
    async def run_search():
        await populator.initialize()
        results = await populator.search_tools(query, limit)
        
        if results:
            table = Table(title=f"Search Results for: '{query}'")
            table.add_column("Tool", style="cyan")
            table.add_column("Category", style="green")
            table.add_column("Description", style="white")
            table.add_column("Distance", style="yellow")
            
            for result in results:
                metadata = result["metadata"]
                table.add_row(
                    metadata.get("name", "Unknown"),
                    metadata.get("category", "Unknown"),
                    metadata.get("description", "")[:80] + "..." if len(metadata.get("description", "")) > 80 else metadata.get("description", ""),
                    f"{result.get('distance', 0):.3f}" if result.get('distance') else "N/A",
                )
            
            console.print(table)
        else:
            console.print(f"[yellow]No results found for query: {query}[/yellow]")
    
    asyncio.run(run_search())


@cli.command()
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def status(ctx: click.Context, output_json: bool):
    """Check database status and statistics."""
    populator = ctx.obj["populator"]
    
    async def run_status():
        await populator.initialize()
        stats = await populator.get_database_stats()
        
        if output_json:
            console.print_json(json.dumps(stats, indent=2))
        else:
            # Display formatted status
            panel_content = f"""
[bold]Database Path:[/bold] {ctx.obj['db_path']}
[bold]Collection:[/bold] {stats.get('collection_name', 'Unknown')}
[bold]Total Tools:[/bold] {stats.get('total_tools', 0)}
[bold]Embedding Model:[/bold] {stats.get('embedding_model', 'Unknown')}
[bold]Last Updated:[/bold] {stats.get('last_updated', 'Unknown')}
"""
            
            console.print(Panel(panel_content.strip(), title="Database Status"))
            
            # Categories table
            if 'categories' in stats and stats['categories']:
                cat_table = Table(title="Tools by Category")
                cat_table.add_column("Category", style="cyan")
                cat_table.add_column("Count", style="green")
                
                for category, count in sorted(stats['categories'].items()):
                    cat_table.add_row(category, str(count))
                
                console.print(cat_table)
            
            # Languages table
            if 'languages' in stats and stats['languages']:
                lang_table = Table(title="Tools by Language")
                lang_table.add_column("Language", style="cyan")
                lang_table.add_column("Count", style="green")
                
                for language, count in sorted(stats['languages'].items()):
                    lang_table.add_row(language, str(count))
                
                console.print(lang_table)
    
    asyncio.run(run_status())


@cli.command()
@click.option("--output", required=True, help="Output file path")
@click.option("--format", "export_format", default="json", 
              type=click.Choice(["json", "csv", "yaml"]),
              help="Export format")
@click.pass_context
def export(ctx: click.Context, output: str, export_format: str):
    """Export tools data from database."""
    populator = ctx.obj["populator"]
    
    async def run_export():
        await populator.initialize()
        success = await populator.export_tools(output, export_format)
        
        if success:
            console.print(f"[green]Export completed: {output}[/green]")
        else:
            console.print(f"[red]Export failed[/red]")
    
    asyncio.run(run_export())


@cli.command()
@click.pass_context
def optimize(ctx: click.Context):
    """Optimize database for better performance."""
    populator = ctx.obj["populator"]
    
    async def run_optimize():
        await populator.initialize()
        await populator.optimize_database()
    
    asyncio.run(run_optimize())


if __name__ == "__main__":
    cli()