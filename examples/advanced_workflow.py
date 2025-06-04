#!/usr/bin/env python3
"""Advanced Workflow Example for the MCP Agent Framework.

This example demonstrates complex multi-agent workflows for bioinformatics research
automation, including comparative analysis, report generation, and human-in-the-loop
decision making using the MCP Agent Framework.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Usage:
    Run comparative analysis:
    $ python advanced_workflow.py comparative --topic "RNA-seq vs microarray"

    Research workflow:
    $ python advanced_workflow.py research --topic "CRISPR gene editing tools"

    Tool evaluation workflow:
    $ python advanced_workflow.py evaluate --tools "STAR,HISAT2,TopHat"

    Interactive workflow:
    $ python advanced_workflow.py interactive

Requirements:
    - MCP Agent Framework fully installed
    - Vector database populated
    - MCP servers configured
    - API keys for external services (optional)
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.markdown import Markdown
from rich.tree import Tree
from rich.status import Status

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from mcp_agent.main import MCPAgent
    from mcp_agent.config.settings import AgentSettings
    from mcp_agent.models.schemas import SearchQuery, AgentResponse
    from mcp_agent.graph.workflow import create_workflow
    from mcp_agent.utils import get_logger, setup_logger, Timer, format_duration
except ImportError as e:
    print(f"Error importing MCP Agent components: {e}")
    print("Please ensure the MCP Agent Framework is properly installed.")
    sys.exit(1)

# Initialize console and logger
console = Console()
logger = get_logger(__name__)


class WorkflowType(Enum):
    """Types of advanced workflows."""
    COMPARATIVE_ANALYSIS = "comparative"
    RESEARCH_AUTOMATION = "research"
    TOOL_EVALUATION = "evaluation"
    COMPREHENSIVE_REVIEW = "review"
    METHODOLOGY_COMPARISON = "methodology"
    LITERATURE_SYNTHESIS = "literature"


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""
    workflow_type: WorkflowType
    topic: str
    max_tools: int = 20
    include_documentation: bool = True
    include_citations: bool = True
    include_code_examples: bool = True
    output_format: str = "markdown"
    depth: str = "comprehensive"  # quick, standard, comprehensive
    enable_human_input: bool = True
    save_intermediate: bool = True
    performance_monitoring: bool = True


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    workflow_id: str
    config: WorkflowConfig
    success: bool
    execution_time_ms: float
    steps_completed: int
    total_steps: int
    final_report: str
    intermediate_results: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class AdvancedWorkflowOrchestrator:
    """Orchestrator for advanced multi-agent workflows."""
    
    def __init__(self, settings: Optional[AgentSettings] = None):
        """Initialize the workflow orchestrator.
        
        Args:
            settings: Optional agent settings.
        """
        self.settings = settings
        self.agent: Optional[MCPAgent] = None
        self.workflow = None
        self.execution_history: List[WorkflowResult] = []
        self.current_workflow_id: Optional[str] = None
        
        # Workflow templates
        self.workflow_templates = {
            WorkflowType.COMPARATIVE_ANALYSIS: self._comparative_analysis_workflow,
            WorkflowType.RESEARCH_AUTOMATION: self._research_automation_workflow,
            WorkflowType.TOOL_EVALUATION: self._tool_evaluation_workflow,
            WorkflowType.COMPREHENSIVE_REVIEW: self._comprehensive_review_workflow,
            WorkflowType.METHODOLOGY_COMPARISON: self._methodology_comparison_workflow,
            WorkflowType.LITERATURE_SYNTHESIS: self._literature_synthesis_workflow,
        }
    
    async def initialize(self) -> None:
        """Initialize the MCP Agent and workflow components."""
        console.print("[blue]Initializing Advanced Workflow Orchestrator...[/blue]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Initialize agent
                task1 = progress.add_task("Initializing MCP Agent...", total=None)
                self.agent = MCPAgent(settings=self.settings)
                await self.agent.initialize()
                progress.update(task1, description="✓ MCP Agent ready")
                
                # Initialize workflow
                task2 = progress.add_task("Setting up workflow engine...", total=None)
                self.workflow = create_workflow()
                progress.update(task2, description="✓ Workflow engine ready")
                
                # Verify connections
                task3 = progress.add_task("Verifying system health...", total=None)
                await self._verify_system_health()
                progress.update(task3, description="✓ System verification complete")
            
            console.print("[green]✓ Advanced Workflow Orchestrator initialized successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to initialize orchestrator: {e}[/red]")
            raise
    
    async def _verify_system_health(self) -> None:
        """Verify that all system components are healthy."""
        # Check agent health
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        
        # Test vector database
        try:
            test_results = await self.agent.search("test query", max_results=1)
            logger.info("Vector database health check passed")
        except Exception as e:
            logger.warning(f"Vector database health check failed: {e}")
        
        # Test MCP servers (if available)
        try:
            tools = await self.agent.list_tools()
            logger.info(f"MCP servers health check passed ({len(tools)} tools available)")
        except Exception as e:
            logger.warning(f"MCP servers health check failed: {e}")
    
    async def execute_workflow(self, config: WorkflowConfig) -> WorkflowResult:
        """Execute a specific workflow based on configuration.
        
        Args:
            config: Workflow configuration.
            
        Returns:
            WorkflowResult: Execution result.
        """
        # Generate workflow ID
        workflow_id = f"{config.workflow_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_workflow_id = workflow_id
        
        console.print(f"\n[bold blue]Executing Workflow:[/bold blue] {config.workflow_type.value}")
        console.print(f"[bold blue]Topic:[/bold blue] {config.topic}")
        console.print(f"[bold blue]Workflow ID:[/bold blue] {workflow_id}")
        
        # Initialize result
        result = WorkflowResult(
            workflow_id=workflow_id,
            config=config,
            success=False,
            execution_time_ms=0.0,
            steps_completed=0,
            total_steps=0,
            final_report="",
            intermediate_results={},
            performance_metrics={},
        )
        
        try:
            with Timer() as timer:
                # Get workflow template
                if config.workflow_type not in self.workflow_templates:
                    raise ValueError(f"Unknown workflow type: {config.workflow_type}")
                
                workflow_func = self.workflow_templates[config.workflow_type]
                
                # Execute workflow
                workflow_result = await workflow_func(config)
                
                # Update result
                result.success = workflow_result.get("success", False)
                result.steps_completed = workflow_result.get("steps_completed", 0)
                result.total_steps = workflow_result.get("total_steps", 0)
                result.final_report = workflow_result.get("final_report", "")
                result.intermediate_results = workflow_result.get("intermediate_results", {})
                result.performance_metrics = workflow_result.get("performance_metrics", {})
            
            result.execution_time_ms = timer.elapsed * 1000
            
            if result.success:
                console.print(f"[green]✓ Workflow completed successfully in {format_duration(timer.elapsed)}[/green]")
            else:
                console.print(f"[yellow]⚠ Workflow completed with issues in {format_duration(timer.elapsed)}[/yellow]")
        
        except Exception as e:
            result.error_message = str(e)
            console.print(f"[red]✗ Workflow failed: {e}[/red]")
            logger.error(f"Workflow execution error: {e}")
        
        # Store result
        self.execution_history.append(result)
        
        # Save intermediate results if enabled
        if config.save_intermediate:
            await self._save_workflow_result(result)
        
        return result
    
    async def _comparative_analysis_workflow(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Execute comparative analysis workflow.
        
        Args:
            config: Workflow configuration.
            
        Returns:
            Dict[str, Any]: Workflow execution result.
        """
        console.print("\n[bold magenta]Starting Comparative Analysis Workflow[/bold magenta]")
        
        steps = [
            "Analyzing topic and extracting comparison criteria",
            "Searching for relevant tools and methods",
            "Categorizing and filtering results",
            "Performing detailed comparison analysis",
            "Generating comparative report",
            "Synthesizing recommendations"
        ]
        
        intermediate_results = {}
        performance_metrics = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            main_task = progress.add_task("Comparative Analysis", total=len(steps))
            
            # Step 1: Topic Analysis
            progress.update(main_task, description=steps[0])
            
            # Extract comparison terms from topic
            comparison_terms = await self._extract_comparison_terms(config.topic)
            intermediate_results["comparison_terms"] = comparison_terms
            
            if config.enable_human_input and comparison_terms:
                console.print(f"\n[yellow]Detected comparison terms:[/yellow] {', '.join(comparison_terms)}")
                if not Confirm.ask("Proceed with these terms?", default=True):
                    custom_terms = Prompt.ask("Enter custom comparison terms (comma-separated)")
                    comparison_terms = [term.strip() for term in custom_terms.split(",")]
                    intermediate_results["comparison_terms"] = comparison_terms
            
            progress.advance(main_task)
            
            # Step 2: Search for tools/methods
            progress.update(main_task, description=steps[1])
            
            all_results = {}
            for term in comparison_terms:
                search_results = await self.agent.search(
                    query=f"{term} {config.topic}",
                    max_results=config.max_tools // len(comparison_terms),
                    include_documentation=config.include_documentation
                )
                all_results[term] = search_results.tools
            
            intermediate_results["search_results"] = all_results
            progress.advance(main_task)
            
            # Step 3: Categorization
            progress.update(main_task, description=steps[2])
            
            categorized_results = await self._categorize_tools(all_results)
            intermediate_results["categorized_results"] = categorized_results
            progress.advance(main_task)
            
            # Step 4: Detailed comparison
            progress.update(main_task, description=steps[3])
            
            comparison_matrix = await self._create_comparison_matrix(categorized_results, comparison_terms)
            intermediate_results["comparison_matrix"] = comparison_matrix
            progress.advance(main_task)
            
            # Step 5: Generate report
            progress.update(main_task, description=steps[4])
            
            report = await self._generate_comparative_report(
                config.topic,
                comparison_terms,
                comparison_matrix,
                config.output_format
            )
            progress.advance(main_task)
            
            # Step 6: Synthesize recommendations
            progress.update(main_task, description=steps[5])
            
            recommendations = await self._generate_recommendations(comparison_matrix, config.topic)
            final_report = f"{report}\n\n{recommendations}"
            progress.advance(main_task)
        
        # Performance metrics
        performance_metrics = {
            "total_tools_analyzed": sum(len(tools) for tools in all_results.values()),
            "categories_identified": len(categorized_results),
            "comparison_criteria": len(comparison_terms),
            "report_length": len(final_report),
        }
        
        return {
            "success": True,
            "steps_completed": len(steps),
            "total_steps": len(steps),
            "final_report": final_report,
            "intermediate_results": intermediate_results,
            "performance_metrics": performance_metrics,
        }
    
    async def _research_automation_workflow(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Execute research automation workflow.
        
        Args:
            config: Workflow configuration.
            
        Returns:
            Dict[str, Any]: Workflow execution result.
        """
        console.print("\n[bold magenta]Starting Research Automation Workflow[/bold magenta]")
        
        steps = [
            "Research topic analysis and decomposition",
            "Literature and tool discovery",
            "Expert knowledge synthesis",
            "Methodology evaluation",
            "Best practices compilation",
            "Comprehensive report generation"
        ]
        
        intermediate_results = {}
        performance_metrics = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            main_task = progress.add_task("Research Automation", total=len(steps))
            
            # Step 1: Topic Analysis
            progress.update(main_task, description=steps[0])
            
            # Use the workflow system for complex research
            research_input = {
                "topic": config.topic,
                "depth": config.depth,
                "max_results": config.max_tools,
                "include_documentation": config.include_documentation,
                "include_code_examples": config.include_code_examples,
                "output_format": config.output_format,
            }
            
            if self.workflow:
                # Execute the workflow using LangGraph
                workflow_result = await self.workflow.ainvoke(research_input)
                intermediate_results["workflow_result"] = workflow_result
            else:
                # Fallback to direct agent calls
                workflow_result = await self.agent.research(
                    topic=config.topic,
                    depth=config.depth,
                    output_format=config.output_format,
                    include_code_examples=config.include_code_examples
                )
            
            progress.advance(main_task)
            
            # Step 2: Tool Discovery
            progress.update(main_task, description=steps[1])
            
            tool_search = await self.agent.search(
                query=config.topic,
                max_results=config.max_tools,
                include_documentation=config.include_documentation
            )
            intermediate_results["discovered_tools"] = tool_search.tools
            progress.advance(main_task)
            
            # Step 3: Knowledge Synthesis
            progress.update(main_task, description=steps[2])
            
            synthesized_knowledge = await self._synthesize_expert_knowledge(
                config.topic,
                tool_search.tools,
                workflow_result
            )
            intermediate_results["synthesized_knowledge"] = synthesized_knowledge
            progress.advance(main_task)
            
            # Step 4: Methodology Evaluation
            progress.update(main_task, description=steps[3])
            
            methodology_analysis = await self._evaluate_methodologies(
                config.topic,
                tool_search.tools
            )
            intermediate_results["methodology_analysis"] = methodology_analysis
            progress.advance(main_task)
            
            # Step 5: Best Practices
            progress.update(main_task, description=steps[4])
            
            best_practices = await self._compile_best_practices(
                config.topic,
                tool_search.tools,
                methodology_analysis
            )
            intermediate_results["best_practices"] = best_practices
            progress.advance(main_task)
            
            # Step 6: Final Report
            progress.update(main_task, description=steps[5])
            
            final_report = await self._generate_research_report(
                config.topic,
                workflow_result,
                synthesized_knowledge,
                methodology_analysis,
                best_practices,
                config.output_format
            )
            progress.advance(main_task)
        
        # Performance metrics
        performance_metrics = {
            "tools_discovered": len(tool_search.tools),
            "knowledge_sources": len(synthesized_knowledge),
            "methodologies_evaluated": len(methodology_analysis),
            "best_practices_compiled": len(best_practices),
            "report_sections": final_report.count("##") if "##" in final_report else final_report.count("\n\n"),
        }
        
        return {
            "success": True,
            "steps_completed": len(steps),
            "total_steps": len(steps),
            "final_report": final_report,
            "intermediate_results": intermediate_results,
            "performance_metrics": performance_metrics,
        }
    
    async def _tool_evaluation_workflow(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Execute tool evaluation workflow.
        
        Args:
            config: Workflow configuration.
            
        Returns:
            Dict[str, Any]: Workflow execution result.
        """
        console.print("\n[bold magenta]Starting Tool Evaluation Workflow[/bold magenta]")
        
        # Parse tools from topic (assuming comma-separated tool names)
        if "," in config.topic:
            tool_names = [name.strip() for name in config.topic.split(",")]
        else:
            # If not comma-separated, search for tools related to the topic
            search_results = await self.agent.search(config.topic, max_results=5)
            tool_names = [tool.name for tool in search_results.tools]
        
        if config.enable_human_input:
            console.print(f"\n[yellow]Tools to evaluate:[/yellow] {', '.join(tool_names)}")
            if not Confirm.ask("Proceed with these tools?", default=True):
                custom_tools = Prompt.ask("Enter tool names (comma-separated)")
                tool_names = [name.strip() for name in custom_tools.split(",")]
        
        steps = [
            "Tool information gathering",
            "Feature analysis",
            "Performance comparison",
            "User experience evaluation",
            "Compatibility assessment",
            "Final evaluation report"
        ]
        
        intermediate_results = {}
        performance_metrics = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            main_task = progress.add_task("Tool Evaluation", total=len(steps))
            
            # Step 1: Information Gathering
            progress.update(main_task, description=steps[0])
            
            tool_details = {}
            for tool_name in tool_names:
                details = await self.agent.get_tool_info(tool_name)
                if details:
                    tool_details[tool_name] = details
                else:
                    # Search for the tool if not found directly
                    search_result = await self.agent.search(tool_name, max_results=1)
                    if search_result.tools:
                        tool_details[tool_name] = search_result.tools[0].__dict__
            
            intermediate_results["tool_details"] = tool_details
            progress.advance(main_task)
            
            # Step 2: Feature Analysis
            progress.update(main_task, description=steps[1])
            
            feature_analysis = await self._analyze_tool_features(tool_details)
            intermediate_results["feature_analysis"] = feature_analysis
            progress.advance(main_task)
            
            # Step 3: Performance Comparison
            progress.update(main_task, description=steps[2])
            
            performance_comparison = await self._compare_tool_performance(tool_details)
            intermediate_results["performance_comparison"] = performance_comparison
            progress.advance(main_task)
            
            # Step 4: User Experience
            progress.update(main_task, description=steps[3])
            
            ux_evaluation = await self._evaluate_user_experience(tool_details)
            intermediate_results["ux_evaluation"] = ux_evaluation
            progress.advance(main_task)
            
            # Step 5: Compatibility Assessment
            progress.update(main_task, description=steps[4])
            
            compatibility_assessment = await self._assess_compatibility(tool_details)
            intermediate_results["compatibility_assessment"] = compatibility_assessment
            progress.advance(main_task)
            
            # Step 6: Final Report
            progress.update(main_task, description=steps[5])
            
            final_report = await self._generate_evaluation_report(
                tool_names,
                tool_details,
                feature_analysis,
                performance_comparison,
                ux_evaluation,
                compatibility_assessment,
                config.output_format
            )
            progress.advance(main_task)
        
        # Performance metrics
        performance_metrics = {
            "tools_evaluated": len(tool_details),
            "features_analyzed": sum(len(features) for features in feature_analysis.values()),
            "comparison_criteria": len(performance_comparison),
            "compatibility_checks": len(compatibility_assessment),
        }
        
        return {
            "success": True,
            "steps_completed": len(steps),
            "total_steps": len(steps),
            "final_report": final_report,
            "intermediate_results": intermediate_results,
            "performance_metrics": performance_metrics,
        }
    
    async def _comprehensive_review_workflow(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Execute comprehensive review workflow."""
        # Combine elements from multiple workflow types
        comparative_result = await self._comparative_analysis_workflow(config)
        research_result = await self._research_automation_workflow(config)
        
        # Synthesize comprehensive review
        final_report = await self._generate_comprehensive_review(
            config.topic,
            comparative_result,
            research_result,
            config.output_format
        )
        
        return {
            "success": True,
            "steps_completed": comparative_result["steps_completed"] + research_result["steps_completed"],
            "total_steps": comparative_result["total_steps"] + research_result["total_steps"],
            "final_report": final_report,
            "intermediate_results": {
                "comparative_analysis": comparative_result["intermediate_results"],
                "research_automation": research_result["intermediate_results"],
            },
            "performance_metrics": {
                **comparative_result["performance_metrics"],
                **research_result["performance_metrics"],
            },
        }
    
    async def _methodology_comparison_workflow(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Execute methodology comparison workflow."""
        # Extract methodologies from topic
        methodologies = await self._extract_methodologies(config.topic)
        
        # Compare methodologies
        comparison_result = await self._compare_methodologies(methodologies, config)
        
        return comparison_result
    
    async def _literature_synthesis_workflow(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Execute literature synthesis workflow."""
        # Search for literature and synthesize
        literature_results = await self._search_literature(config.topic)
        synthesis = await self._synthesize_literature(literature_results, config)
        
        return synthesis
    
    # Helper methods for workflow steps
    
    async def _extract_comparison_terms(self, topic: str) -> List[str]:
        """Extract comparison terms from topic."""
        # Simple keyword extraction for comparison
        comparison_keywords = ["vs", "versus", "compared to", "against", "alternative"]
        terms = []
        
        topic_lower = topic.lower()
        for keyword in comparison_keywords:
            if keyword in topic_lower:
                parts = topic_lower.split(keyword)
                for part in parts:
                    cleaned = part.strip().replace(" and ", " ").replace(",", "")
                    if cleaned:
                        terms.extend(cleaned.split())
        
        # If no comparison keywords found, extract main terms
        if not terms:
            # Remove common words and extract key terms
            stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            words = topic.lower().split()
            terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Limit to most relevant terms
        return list(set(terms))[:5]
    
    async def _categorize_tools(self, search_results: Dict[str, List]) -> Dict[str, List]:
        """Categorize tools by their properties."""
        categories = {}
        
        for term, tools in search_results.items():
            for tool in tools:
                category = getattr(tool, 'category', 'unknown')
                if category not in categories:
                    categories[category] = []
                categories[category].append({
                    "name": tool.name,
                    "term": term,
                    "description": tool.description,
                    "language": getattr(tool, 'language', None),
                    "license": getattr(tool, 'license', None),
                })
        
        return categories
    
    async def _create_comparison_matrix(self, categorized_results: Dict, comparison_terms: List[str]) -> Dict[str, Any]:
        """Create a comparison matrix for analysis."""
        matrix = {
            "categories": list(categorized_results.keys()),
            "terms": comparison_terms,
            "comparisons": [],
        }
        
        for category, tools in categorized_results.items():
            category_comparison = {
                "category": category,
                "tool_count": len(tools),
                "languages": list(set(tool.get("language") for tool in tools if tool.get("language"))),
                "licenses": list(set(tool.get("license") for tool in tools if tool.get("license"))),
                "tools": tools,
            }
            matrix["comparisons"].append(category_comparison)
        
        return matrix
    
    async def _generate_comparative_report(self, topic: str, terms: List[str], matrix: Dict, format: str) -> str:
        """Generate comparative analysis report."""
        if format.lower() == "markdown":
            report = f"# Comparative Analysis: {topic}\n\n"
            report += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += f"**Comparison Terms:** {', '.join(terms)}\n\n"
            
            report += "## Summary\n\n"
            report += f"This analysis compared tools and methods related to: {topic}\n"
            report += f"Total categories analyzed: {len(matrix['categories'])}\n"
            report += f"Total tools examined: {sum(comp['tool_count'] for comp in matrix['comparisons'])}\n\n"
            
            report += "## Category Analysis\n\n"
            for comparison in matrix["comparisons"]:
                report += f"### {comparison['category'].replace('_', ' ').title()}\n\n"
                report += f"- **Tool Count:** {comparison['tool_count']}\n"
                report += f"- **Languages:** {', '.join(comparison['languages']) if comparison['languages'] else 'Not specified'}\n"
                report += f"- **Licenses:** {', '.join(comparison['licenses']) if comparison['licenses'] else 'Not specified'}\n\n"
                
                report += "**Tools in this category:**\n"
                for tool in comparison['tools'][:5]:  # Limit to top 5
                    report += f"- **{tool['name']}:** {tool['description'][:100]}...\n"
                report += "\n"
            
            return report
        else:
            # Plain text format
            return f"Comparative Analysis Report for: {topic}\n" + "="*50 + "\n" + str(matrix)
    
    async def _generate_recommendations(self, comparison_matrix: Dict, topic: str) -> str:
        """Generate recommendations based on comparison."""
        recommendations = "## Recommendations\n\n"
        
        # Find most popular category
        most_tools_category = max(
            comparison_matrix["comparisons"],
            key=lambda x: x["tool_count"]
        )
        
        recommendations += f"### Primary Recommendation\n\n"
        recommendations += f"Based on the analysis, the **{most_tools_category['category'].replace('_', ' ').title()}** "
        recommendations += f"category has the most available tools ({most_tools_category['tool_count']} tools), "
        recommendations += f"suggesting it's a well-established area for {topic}.\n\n"
        
        # Language recommendations
        all_languages = []
        for comp in comparison_matrix["comparisons"]:
            all_languages.extend(comp["languages"])
        
        if all_languages:
            from collections import Counter
            language_counts = Counter(all_languages)
            top_language = language_counts.most_common(1)[0][0]
            
            recommendations += f"### Technology Recommendation\n\n"
            recommendations += f"**{top_language}** appears to be the most commonly used language "
            recommendations += f"across tools in this domain, making it a good choice for development.\n\n"
        
        recommendations += f"### Next Steps\n\n"
        recommendations += f"1. Investigate tools in the {most_tools_category['category'].replace('_', ' ').title()} category\n"
        recommendations += f"2. Consider the licensing implications of your tool choices\n"
        recommendations += f"3. Evaluate the specific requirements of your {topic} use case\n"
        recommendations += f"4. Test multiple tools to determine the best fit for your workflow\n\n"
        
        return recommendations
    
    async def _synthesize_expert_knowledge(self, topic: str, tools: List, workflow_result: Any) -> Dict[str, Any]:
        """Synthesize expert knowledge from various sources."""
        return {
            "topic_summary": f"Expert knowledge synthesis for {topic}",
            "key_tools": [tool.name for tool in tools[:10]],
            "workflow_insights": getattr(workflow_result, 'summary', 'Analysis complete'),
            "expert_recommendations": ["Use established tools", "Consider compatibility", "Follow best practices"],
        }
    
    async def _evaluate_methodologies(self, topic: str, tools: List) -> Dict[str, Any]:
        """Evaluate methodologies related to the topic."""
        methodologies = {}
        
        # Group tools by category to identify methodologies
        for tool in tools:
            category = getattr(tool, 'category', 'unknown')
            if category not in methodologies:
                methodologies[category] = {
                    "tools": [],
                    "approach": f"{category.replace('_', ' ').title()} methodology",
                    "pros": ["Established approach", "Good tool support"],
                    "cons": ["May have limitations", "Consider alternatives"],
                }
            methodologies[category]["tools"].append(tool.name)
        
        return methodologies
    
    async def _compile_best_practices(self, topic: str, tools: List, methodology_analysis: Dict) -> List[str]:
        """Compile best practices based on analysis."""
        practices = [
            f"Choose tools appropriate for your {topic} requirements",
            "Consider the learning curve and documentation quality",
            "Evaluate performance characteristics for your data size",
            "Ensure compatibility with your existing workflow",
            "Plan for data format conversions between tools",
            "Consider licensing and cost implications",
            "Implement proper quality control measures",
            "Document your methodology for reproducibility",
        ]
        
        # Add methodology-specific practices
        for methodology, details in methodology_analysis.items():
            if details["tools"]:
                practices.append(f"For {methodology}: Consider using {', '.join(details['tools'][:3])}")
        
        return practices
    
    async def _generate_research_report(
        self,
        topic: str,
        workflow_result: Any,
        knowledge: Dict,
        methodologies: Dict,
        practices: List[str],
        format: str
    ) -> str:
        """Generate comprehensive research report."""
        if format.lower() == "markdown":
            report = f"# Research Report: {topic}\n\n"
            report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            report += "## Executive Summary\n\n"
            report += f"This report provides a comprehensive analysis of {topic}, "
            report += f"including tool recommendations, methodological considerations, and best practices.\n\n"
            
            report += "## Key Findings\n\n"
            report += f"- **Tools Analyzed:** {len(knowledge.get('key_tools', []))}\n"
            report += f"- **Methodologies Identified:** {len(methodologies)}\n"
            report += f"- **Best Practices Compiled:** {len(practices)}\n\n"
            
            report += "## Methodology Analysis\n\n"
            for method, details in methodologies.items():
                report += f"### {details['approach']}\n\n"
                report += f"**Available Tools:** {', '.join(details['tools'])}\n\n"
                report += "**Advantages:**\n"
                for pro in details['pros']:
                    report += f"- {pro}\n"
                report += "\n**Considerations:**\n"
                for con in details['cons']:
                    report += f"- {con}\n"
                report += "\n"
            
            report += "## Best Practices\n\n"
            for i, practice in enumerate(practices, 1):
                report += f"{i}. {practice}\n"
            
            report += "\n## Conclusion\n\n"
            report += f"The analysis of {topic} reveals a rich ecosystem of tools and methodologies. "
            report += "Success depends on careful tool selection, proper methodology application, "
            report += "and adherence to established best practices.\n\n"
            
            return report
        else:
            return f"Research Report: {topic}\n" + "="*50 + "\n" + str(knowledge)
    
    async def _analyze_tool_features(self, tool_details: Dict) -> Dict[str, List[str]]:
        """Analyze features of tools."""
        features = {}
        
        for tool_name, details in tool_details.items():
            tool_features = []
            
            # Extract features from tool details
            if details.get('input_formats'):
                tool_features.append(f"Input: {', '.join(details['input_formats'])}")
            if details.get('output_formats'):
                tool_features.append(f"Output: {', '.join(details['output_formats'])}")
            if details.get('language'):
                tool_features.append(f"Language: {details['language']}")
            if details.get('gui_available'):
                tool_features.append("GUI Available")
            if details.get('command_line'):
                tool_features.append("Command Line Interface")
            if details.get('api_available'):
                tool_features.append("API Available")
            if details.get('docker_available'):
                tool_features.append("Docker Support")
            
            features[tool_name] = tool_features
        
        return features
    
    async def _compare_tool_performance(self, tool_details: Dict) -> Dict[str, Any]:
        """Compare performance characteristics of tools."""
        comparison = {
            "criteria": ["Language", "Platform Support", "Interface Options", "Container Support"],
            "tools": {},
        }
        
        for tool_name, details in tool_details.items():
            comparison["tools"][tool_name] = {
                "language": details.get('language', 'Unknown'),
                "platforms": details.get('platform', ['Unknown']),
                "interfaces": [],
                "containers": [],
            }
            
            # Interface options
            if details.get('gui_available'):
                comparison["tools"][tool_name]["interfaces"].append("GUI")
            if details.get('command_line'):
                comparison["tools"][tool_name]["interfaces"].append("CLI")
            if details.get('api_available'):
                comparison["tools"][tool_name]["interfaces"].append("API")
            
            # Container support
            if details.get('docker_available'):
                comparison["tools"][tool_name]["containers"].append("Docker")
            if details.get('singularity_available'):
                comparison["tools"][tool_name]["containers"].append("Singularity")
        
        return comparison
    
    async def _evaluate_user_experience(self, tool_details: Dict) -> Dict[str, str]:
        """Evaluate user experience aspects of tools."""
        ux_evaluation = {}
        
        for tool_name, details in tool_details.items():
            ux_score = []
            
            # Documentation
            if details.get('documentation_url'):
                ux_score.append("Good documentation")
            
            # Installation options
            installation = details.get('installation', {})
            if len(installation) > 1:
                ux_score.append("Multiple installation options")
            
            # Interface options
            interfaces = 0
            if details.get('gui_available'):
                interfaces += 1
            if details.get('command_line'):
                interfaces += 1
            if details.get('api_available'):
                interfaces += 1
            
            if interfaces > 1:
                ux_score.append("Multiple interface options")
            
            # Maintenance status
            if details.get('maintenance_status') == 'active':
                ux_score.append("Actively maintained")
            
            ux_evaluation[tool_name] = "; ".join(ux_score) if ux_score else "Limited information"
        
        return ux_evaluation
    
    async def _assess_compatibility(self, tool_details: Dict) -> Dict[str, List[str]]:
        """Assess compatibility between tools."""
        compatibility = {}
        
        # Find common formats
        all_input_formats = set()
        all_output_formats = set()
        
        for details in tool_details.values():
            if details.get('input_formats'):
                all_input_formats.update(details['input_formats'])
            if details.get('output_formats'):
                all_output_formats.update(details['output_formats'])
        
        # Check compatibility
        for tool_name, details in tool_details.items():
            compatible_tools = []
            tool_outputs = set(details.get('output_formats', []))
            
            for other_name, other_details in tool_details.items():
                if other_name != tool_name:
                    other_inputs = set(other_details.get('input_formats', []))
                    if tool_outputs & other_inputs:  # Intersection
                        compatible_tools.append(other_name)
            
            compatibility[tool_name] = compatible_tools
        
        return compatibility
    
    async def _generate_evaluation_report(
        self,
        tool_names: List[str],
        tool_details: Dict,
        features: Dict,
        performance: Dict,
        ux: Dict,
        compatibility: Dict,
        format: str
    ) -> str:
        """Generate tool evaluation report."""
        if format.lower() == "markdown":
            report = f"# Tool Evaluation Report\n\n"
            report += f"**Tools Evaluated:** {', '.join(tool_names)}\n"
            report += f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            report += "## Tool Comparison Matrix\n\n"
            report += "| Tool | Language | Interfaces | UX Rating | Compatible Tools |\n"
            report += "|------|----------|------------|-----------|------------------|\n"
            
            for tool_name in tool_names:
                if tool_name in tool_details:
                    language = tool_details[tool_name].get('language', 'Unknown')
                    interfaces = len(performance.get('tools', {}).get(tool_name, {}).get('interfaces', []))
                    ux_rating = ux.get(tool_name, 'Not evaluated')[:30] + "..." if len(ux.get(tool_name, '')) > 30 else ux.get(tool_name, 'Not evaluated')
                    compatible = len(compatibility.get(tool_name, []))
                    
                    report += f"| {tool_name} | {language} | {interfaces} | {ux_rating} | {compatible} |\n"
            
            report += "\n## Detailed Analysis\n\n"
            
            for tool_name in tool_names:
                if tool_name in tool_details:
                    report += f"### {tool_name}\n\n"
                    
                    # Features
                    tool_features = features.get(tool_name, [])
                    if tool_features:
                        report += "**Key Features:**\n"
                        for feature in tool_features:
                            report += f"- {feature}\n"
                        report += "\n"
                    
                    # User Experience
                    report += f"**User Experience:** {ux.get(tool_name, 'Not evaluated')}\n\n"
                    
                    # Compatibility
                    compatible_tools = compatibility.get(tool_name, [])
                    if compatible_tools:
                        report += f"**Compatible Tools:** {', '.join(compatible_tools)}\n\n"
                    else:
                        report += "**Compatible Tools:** None identified\n\n"
            
            report += "## Recommendations\n\n"
            report += "Based on this evaluation:\n\n"
            
            # Find best overall tool
            best_tool = max(tool_names, key=lambda t: len(compatibility.get(t, [])))
            report += f"1. **{best_tool}** appears to have the best compatibility with other tools\n"
            report += f"2. Consider your specific workflow requirements when making the final selection\n"
            report += f"3. Evaluate the learning curve and documentation quality for your team\n\n"
            
            return report
        else:
            return f"Tool Evaluation Report\n" + "="*30 + "\n" + str(tool_details)
    
    async def _generate_comprehensive_review(
        self,
        topic: str,
        comparative_result: Dict,
        research_result: Dict,
        format: str
    ) -> str:
        """Generate comprehensive review combining multiple analyses."""
        if format.lower() == "markdown":
            report = f"# Comprehensive Review: {topic}\n\n"
            report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            report += "## Overview\n\n"
            report += f"This comprehensive review synthesizes comparative analysis and research automation "
            report += f"results to provide a complete picture of {topic}.\n\n"
            
            report += "## Comparative Analysis Summary\n\n"
            comp_metrics = comparative_result.get("performance_metrics", {})
            report += f"- Tools analyzed: {comp_metrics.get('total_tools_analyzed', 'Unknown')}\n"
            report += f"- Categories identified: {comp_metrics.get('categories_identified', 'Unknown')}\n"
            report += f"- Comparison criteria: {comp_metrics.get('comparison_criteria', 'Unknown')}\n\n"
            
            report += "## Research Automation Summary\n\n"
            research_metrics = research_result.get("performance_metrics", {})
            report += f"- Tools discovered: {research_metrics.get('tools_discovered', 'Unknown')}\n"
            report += f"- Knowledge sources: {research_metrics.get('knowledge_sources', 'Unknown')}\n"
            report += f"- Best practices compiled: {research_metrics.get('best_practices_compiled', 'Unknown')}\n\n"
            
            report += "## Integrated Recommendations\n\n"
            report += "Based on both comparative analysis and comprehensive research:\n\n"
            report += f"1. The field of {topic} shows significant tool diversity\n"
            report += f"2. Multiple methodological approaches are available\n"
            report += f"3. Consider both established and emerging tools\n"
            report += f"4. Evaluate tools based on your specific requirements\n\n"
            
            report += "## Detailed Findings\n\n"
            report += comparative_result.get("final_report", "No comparative analysis available")
            report += "\n\n"
            report += research_result.get("final_report", "No research analysis available")
            
            return report
        else:
            return f"Comprehensive Review: {topic}\n" + "="*50
    
    async def _extract_methodologies(self, topic: str) -> List[str]:
        """Extract methodologies from topic."""
        # Simplified methodology extraction
        return [topic, f"alternative to {topic}", f"improved {topic}"]
    
    async def _compare_methodologies(self, methodologies: List[str], config: WorkflowConfig) -> Dict[str, Any]:
        """Compare different methodologies."""
        return {
            "success": True,
            "steps_completed": 3,
            "total_steps": 3,
            "final_report": f"Methodology comparison for: {', '.join(methodologies)}",
            "intermediate_results": {"methodologies": methodologies},
            "performance_metrics": {"methodologies_compared": len(methodologies)},
        }
    
    async def _search_literature(self, topic: str) -> List[Dict]:
        """Search for literature related to topic."""
        # Mock literature search
        return [
            {"title": f"Literature review of {topic}", "authors": ["Smith, J.", "Doe, A."]},
            {"title": f"Advances in {topic}", "authors": ["Johnson, B."]},
        ]
    
    async def _synthesize_literature(self, literature: List[Dict], config: WorkflowConfig) -> Dict[str, Any]:
        """Synthesize literature findings."""
        return {
            "success": True,
            "steps_completed": 2,
            "total_steps": 2,
            "final_report": f"Literature synthesis of {len(literature)} papers",
            "intermediate_results": {"literature": literature},
            "performance_metrics": {"papers_synthesized": len(literature)},
        }
    
    async def _save_workflow_result(self, result: WorkflowResult) -> None:
        """Save workflow result to file."""
        try:
            output_dir = Path("workflow_results")
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"{result.workflow_id}.json"
            
            # Convert result to dict for JSON serialization
            result_dict = asdict(result)
            
            with open(output_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            console.print(f"[green]Workflow result saved to: {output_file}[/green]")
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save workflow result: {e}[/yellow]")
    
    def display_workflow_result(self, result: WorkflowResult) -> None:
        """Display workflow result in rich format."""
        # Result summary panel
        status_color = "green" if result.success else "red"
        status_text = "✓ Success" if result.success else "✗ Failed"
        
        summary_content = f"""
[bold]Workflow ID:[/bold] {result.workflow_id}
[bold]Type:[/bold] {result.config.workflow_type.value}
[bold]Topic:[/bold] {result.config.topic}
[bold]Status:[/bold] [{status_color}]{status_text}[/{status_color}]
[bold]Execution Time:[/bold] {format_duration(result.execution_time_ms / 1000)}
[bold]Steps:[/bold] {result.steps_completed}/{result.total_steps}
"""
        
        if result.error_message:
            summary_content += f"\n[bold]Error:[/bold] [red]{result.error_message}[/red]"
        
        console.print(Panel(
            summary_content.strip(),
            title="Workflow Execution Summary",
            border_style=status_color
        ))
        
        # Performance metrics
        if result.performance_metrics:
            metrics_table = Table(title="Performance Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="white")
            
            for metric, value in result.performance_metrics.items():
                metrics_table.add_row(metric.replace("_", " ").title(), str(value))
            
            console.print(metrics_table)
        
        # Show report preview
        if result.final_report:
            preview = result.final_report[:500] + "..." if len(result.final_report) > 500 else result.final_report
            console.print(Panel(
                Markdown(preview),
                title="Report Preview",
                border_style="blue"
            ))
    
    def show_execution_history(self) -> None:
        """Display execution history."""
        if not self.execution_history:
            console.print("[yellow]No workflow execution history available.[/yellow]")
            return
        
        history_table = Table(title="Workflow Execution History")
        history_table.add_column("Workflow ID", style="cyan")
        history_table.add_column("Type", style="green")
        history_table.add_column("Topic", style="white")
        history_table.add_column("Status", style="white")
        history_table.add_column("Duration", style="yellow")
        history_table.add_column("Timestamp", style="dim")
        
        for result in self.execution_history[-10:]:  # Show last 10
            status = "✓" if result.success else "✗"
            status_color = "green" if result.success else "red"
            
            history_table.add_row(
                result.workflow_id,
                result.config.workflow_type.value,
                result.config.topic[:30] + "..." if len(result.config.topic) > 30 else result.config.topic,
                f"[{status_color}]{status}[/{status_color}]",
                format_duration(result.execution_time_ms / 1000),
                result.timestamp[:19] if result.timestamp else "Unknown"
            )
        
        console.print(history_table)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.agent:
            await self.agent.close()
            console.print("[dim]Workflow orchestrator resources cleaned up.[/dim]")


async def interactive_workflow_mode(orchestrator: AdvancedWorkflowOrchestrator) -> None:
    """Run interactive workflow mode."""
    console.print(Panel(
        "[bold blue]Advanced Workflow Orchestrator - Interactive Mode[/bold blue]\n\n"
        "Available workflow types:\n"
        "• [cyan]comparative[/cyan] - Compare tools, methods, or approaches\n"
        "• [cyan]research[/cyan] - Comprehensive research automation\n"
        "• [cyan]evaluation[/cyan] - Detailed tool evaluation\n"
        "• [cyan]review[/cyan] - Comprehensive review (combines multiple workflows)\n"
        "• [cyan]methodology[/cyan] - Compare methodological approaches\n"
        "• [cyan]literature[/cyan] - Literature synthesis\n\n"
        "Commands:\n"
        "• [cyan]run <type> <topic>[/cyan] - Execute workflow\n"
        "• [cyan]history[/cyan] - Show execution history\n"
        "• [cyan]help[/cyan] - Show this help\n"
        "• [cyan]quit[/cyan] - Exit",
        title="Welcome to Advanced Workflow Orchestrator",
        border_style="blue"
    ))
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]Workflow[/bold cyan]").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("[green]Goodbye![/green]")
                break
            
            elif user_input.lower() == 'help':
                console.print(Panel(
                    "Available commands:\n"
                    "• [cyan]run <type> <topic>[/cyan] - Execute workflow\n"
                    "  Example: run comparative 'RNA-seq vs microarray'\n"
                    "• [cyan]history[/cyan] - Show execution history\n"
                    "• [cyan]quit[/cyan] - Exit",
                    title="Help",
                    border_style="green"
                ))
            
            elif user_input.lower() == 'history':
                orchestrator.show_execution_history()
            
            elif user_input.lower().startswith('run'):
                parts = user_input.split(maxsplit=2)
                if len(parts) >= 3:
                    workflow_type_str = parts[1]
                    topic = parts[2].strip('"\'')
                    
                    try:
                        workflow_type = WorkflowType(workflow_type_str)
                        
                        # Get additional configuration
                        console.print(f"\n[yellow]Configuring {workflow_type.value} workflow for:[/yellow] {topic}")
                        
                        max_tools = IntPrompt.ask("Maximum tools to analyze", default=20)
                        depth = Prompt.ask("Analysis depth", choices=["quick", "standard", "comprehensive"], default="standard")
                        enable_human = Confirm.ask("Enable human-in-the-loop interactions", default=True)
                        
                        config = WorkflowConfig(
                            workflow_type=workflow_type,
                            topic=topic,
                            max_tools=max_tools,
                            depth=depth,
                            enable_human_input=enable_human,
                        )
                        
                        # Execute workflow
                        result = await orchestrator.execute_workflow(config)
                        orchestrator.display_workflow_result(result)
                        
                    except ValueError:
                        console.print(f"[red]Unknown workflow type: {workflow_type_str}[/red]")
                        console.print("Available types: comparative, research, evaluation, review, methodology, literature")
                
                else:
                    console.print("[yellow]Usage: run <type> <topic>[/yellow]")
                    console.print("Example: run comparative 'RNA-seq vs microarray'")
            
            else:
                console.print("[yellow]Unknown command. Type 'help' for available commands.[/yellow]")
        
        except KeyboardInterrupt:
            console.print("\n[green]Goodbye![/green]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# CLI Interface
@click.group()
@click.option("--config", help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool):
    """Advanced Workflow Orchestrator for MCP Agent Framework."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option("--topic", "-t", required=True, help="Research topic for comparative analysis")
@click.option("--max-tools", "-n", default=20, help="Maximum tools to analyze")
@click.option("--output", "-o", help="Output file path")
@click.pass_context
def comparative(ctx: click.Context, topic: str, max_tools: int, output: Optional[str]):
    """Run comparative analysis workflow."""
    
    async def run_comparative():
        # Setup
        setup_logger("DEBUG" if ctx.obj["verbose"] else "INFO")
        
        try:
            from mcp_agent.config import get_settings
            settings = get_settings(ctx.obj["config"])
        except Exception:
            settings = None
        
        orchestrator = AdvancedWorkflowOrchestrator(settings)
        
        try:
            await orchestrator.initialize()
            
            config = WorkflowConfig(
                workflow_type=WorkflowType.COMPARATIVE_ANALYSIS,
                topic=topic,
                max_tools=max_tools,
                enable_human_input=True,
            )
            
            result = await orchestrator.execute_workflow(config)
            orchestrator.display_workflow_result(result)
            
            # Save output if requested
            if output and result.final_report:
                Path(output).write_text(result.final_report)
                console.print(f"[green]Report saved to: {output}[/green]")
        
        finally:
            await orchestrator.cleanup()
    
    asyncio.run(run_comparative())


@cli.command()
@click.option("--topic", "-t", required=True, help="Research topic")
@click.option("--depth", default="comprehensive", type=click.Choice(["quick", "standard", "comprehensive"]))
@click.option("--max-tools", "-n", default=20, help="Maximum tools to discover")
@click.option("--output", "-o", help="Output file path")
@click.pass_context
def research(ctx: click.Context, topic: str, depth: str, max_tools: int, output: Optional[str]):
    """Run research automation workflow."""
    
    async def run_research():
        setup_logger("DEBUG" if ctx.obj["verbose"] else "INFO")
        
        try:
            from mcp_agent.config import get_settings
            settings = get_settings(ctx.obj["config"])
        except Exception:
            settings = None
        
        orchestrator = AdvancedWorkflowOrchestrator(settings)
        
        try:
            await orchestrator.initialize()
            
            config = WorkflowConfig(
                workflow_type=WorkflowType.RESEARCH_AUTOMATION,
                topic=topic,
                max_tools=max_tools,
                depth=depth,
                enable_human_input=True,
            )
            
            result = await orchestrator.execute_workflow(config)
            orchestrator.display_workflow_result(result)
            
            if output and result.final_report:
                Path(output).write_text(result.final_report)
                console.print(f"[green]Report saved to: {output}[/green]")
        
        finally:
            await orchestrator.cleanup()
    
    asyncio.run(run_research())


@cli.command()
@click.option("--tools", "-t", required=True, help="Comma-separated list of tools to evaluate")
@click.option("--output", "-o", help="Output file path")
@click.pass_context
def evaluate(ctx: click.Context, tools: str, output: Optional[str]):
    """Run tool evaluation workflow."""
    
    async def run_evaluation():
        setup_logger("DEBUG" if ctx.obj["verbose"] else "INFO")
        
        try:
            from mcp_agent.config import get_settings
            settings = get_settings(ctx.obj["config"])
        except Exception:
            settings = None
        
        orchestrator = AdvancedWorkflowOrchestrator(settings)
        
        try:
            await orchestrator.initialize()
            
            config = WorkflowConfig(
                workflow_type=WorkflowType.TOOL_EVALUATION,
                topic=tools,  # Pass tools as topic
                enable_human_input=True,
            )
            
            result = await orchestrator.execute_workflow(config)
            orchestrator.display_workflow_result(result)
            
            if output and result.final_report:
                Path(output).write_text(result.final_report)
                console.print(f"[green]Report saved to: {output}[/green]")
        
        finally:
            await orchestrator.cleanup()
    
    asyncio.run(run_evaluation())


@cli.command()
@click.pass_context
def interactive(ctx: click.Context):
    """Run interactive workflow mode."""
    
    async def run_interactive():
        setup_logger("DEBUG" if ctx.obj["verbose"] else "INFO")
        
        try:
            from mcp_agent.config import get_settings
            settings = get_settings(ctx.obj["config"])
        except Exception:
            settings = None
        
        orchestrator = AdvancedWorkflowOrchestrator(settings)
        
        try:
            await orchestrator.initialize()
            await interactive_workflow_mode(orchestrator)
        finally:
            await orchestrator.cleanup()
    
    asyncio.run(run_interactive())


if __name__ == "__main__":
    cli()