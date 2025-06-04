"""Reporter agent for report generation and content formatting.

The ReporterAgent specializes in generating comprehensive reports, summaries,
and formatted content from research data and search results.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic reporter usage:

    >>> reporter = ReporterAgent(settings)
    >>> await reporter.initialize()
    >>> task = AgentTask(type="generate_report", parameters={"topic": "RNA-seq"})
    >>> result = await reporter.execute_task(task)
"""

import asyncio
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from mcp_agent.agents.base import BaseAgent, AgentCapability, AgentState
    from mcp_agent.config.settings import AgentSettings
    from mcp_agent.models.schemas import (
        AgentTask,
        AgentResponse,
        ToolInfo,
        SearchResult,
        ReportTemplate,
        ReportMetadata,
    )
    from mcp_agent.utils.logger import get_logger
except ImportError:
    # Mock imports for development
    class BaseAgent:
        pass
    class AgentCapability:
        REPORT_GENERATION = "report_generation"
        CONTENT_FORMATTING = "content_formatting"
        SUMMARY_CREATION = "summary_creation"
        CODE_GENERATION = "code_generation"
        AUDIO_PROCESSING = "audio_processing"
        MULTI_MODAL_PROCESSING = "multi_modal_processing"
    class AgentState:
        pass
    class AgentSettings:
        pass
    class AgentTask:
        pass
    class AgentResponse:
        pass
    class ToolInfo:
        pass
    class SearchResult:
        pass
    class ReportTemplate:
        pass
    class ReportMetadata:
        pass
    class ChatGoogleGenerativeAI:
        pass
    def get_logger(name: str):
        import logging
        return logging.getLogger(name)


class ReportFormatter:
    """Utility class for formatting content in different output formats."""
    
    @staticmethod
    def to_markdown(content: Dict[str, Any]) -> str:
        """Convert structured content to Markdown format."""
        md = ""
        
        # Title
        if "title" in content:
            md += f"# {content['title']}\n\n"
        
        # Metadata
        if "metadata" in content:
            md += "## Report Information\n\n"
            for key, value in content["metadata"].items():
                md += f"- **{key.replace('_', ' ').title()}**: {value}\n"
            md += "\n"
        
        # Executive Summary
        if "summary" in content:
            md += "## Executive Summary\n\n"
            md += f"{content['summary']}\n\n"
        
        # Main Sections
        if "sections" in content:
            for section in content["sections"]:
                if isinstance(section, dict):
                    md += f"## {section.get('title', 'Section')}\n\n"
                    md += f"{section.get('content', '')}\n\n"
                    
                    # Subsections
                    if "subsections" in section:
                        for subsection in section["subsections"]:
                            md += f"### {subsection.get('title', 'Subsection')}\n\n"
                            md += f"{subsection.get('content', '')}\n\n"
        
        # Tools/Results
        if "tools" in content:
            md += "## Tools Found\n\n"
            for i, tool in enumerate(content["tools"], 1):
                md += f"### {i}. {tool.get('name', 'Unknown Tool')}\n\n"
                md += f"**Description**: {tool.get('description', 'No description available')}\n\n"
                
                if tool.get('category'):
                    md += f"**Category**: {tool['category']}\n\n"
                
                if tool.get('url'):
                    md += f"**URL**: [{tool['url']}]({tool['url']})\n\n"
                
                if tool.get('installation'):
                    md += f"**Installation**: `{tool['installation']}`\n\n"
        
        # Code Examples
        if "code_examples" in content:
            md += "## Code Examples\n\n"
            for example in content["code_examples"]:
                md += f"### {example.get('title', 'Example')}\n\n"
                md += f"```{example.get('language', 'bash')}\n"
                md += f"{example.get('code', '')}\n"
                md += "```\n\n"
                
                if example.get('description'):
                    md += f"{example['description']}\n\n"
        
        # References
        if "references" in content:
            md += "## References\n\n"
            for i, ref in enumerate(content["references"], 1):
                if isinstance(ref, dict):
                    md += f"{i}. {ref.get('title', 'Reference')}"
                    if ref.get('url'):
                        md += f" - [{ref['url']}]({ref['url']})"
                    md += "\n"
                else:
                    md += f"{i}. {ref}\n"
            md += "\n"
        
        return md
    
    @staticmethod
    def to_html(content: Dict[str, Any]) -> str:
        """Convert structured content to HTML format."""
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                h3 {{ color: #2c3e50; }}
                .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .tool-card {{ border: 1px solid #bdc3c7; border-radius: 5px; padding: 15px; margin: 10px 0; }}
                .tool-card h3 {{ margin-top: 0; color: #e74c3c; }}
                pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                code {{ background: #ecf0f1; padding: 2px 5px; border-radius: 3px; }}
                .summary {{ background: #d5e8d4; padding: 15px; border-left: 4px solid #27ae60; margin: 20px 0; }}
                .references {{ background: #fdf2e9; padding: 15px; border-left: 4px solid #f39c12; }}
            </style>
        </head>
        <body>
        """.format(title=html.escape(content.get("title", "Report")))
        
        # Title
        if "title" in content:
            html_content += f"<h1>{html.escape(content['title'])}</h1>\n"
        
        # Metadata
        if "metadata" in content:
            html_content += '<div class="metadata">\n<h2>Report Information</h2>\n<ul>\n'
            for key, value in content["metadata"].items():
                html_content += f"<li><strong>{html.escape(key.replace('_', ' ').title())}</strong>: {html.escape(str(value))}</li>\n"
            html_content += "</ul>\n</div>\n"
        
        # Summary
        if "summary" in content:
            html_content += f'<div class="summary">\n<h2>Executive Summary</h2>\n<p>{html.escape(content["summary"])}</p>\n</div>\n'
        
        # Sections
        if "sections" in content:
            for section in content["sections"]:
                if isinstance(section, dict):
                    html_content += f"<h2>{html.escape(section.get('title', 'Section'))}</h2>\n"
                    html_content += f"<p>{html.escape(section.get('content', ''))}</p>\n"
        
        # Tools
        if "tools" in content:
            html_content += "<h2>Tools Found</h2>\n"
            for tool in content["tools"]:
                html_content += '<div class="tool-card">\n'
                html_content += f"<h3>{html.escape(tool.get('name', 'Unknown Tool'))}</h3>\n"
                html_content += f"<p><strong>Description:</strong> {html.escape(tool.get('description', 'No description'))}</p>\n"
                
                if tool.get('category'):
                    html_content += f"<p><strong>Category:</strong> {html.escape(tool['category'])}</p>\n"
                
                if tool.get('url'):
                    html_content += f'<p><strong>URL:</strong> <a href="{html.escape(tool["url"])}" target="_blank">{html.escape(tool["url"])}</a></p>\n'
                
                html_content += "</div>\n"
        
        # Code Examples
        if "code_examples" in content:
            html_content += "<h2>Code Examples</h2>\n"
            for example in content["code_examples"]:
                html_content += f"<h3>{html.escape(example.get('title', 'Example'))}</h3>\n"
                html_content += f"<pre><code>{html.escape(example.get('code', ''))}</code></pre>\n"
                if example.get('description'):
                    html_content += f"<p>{html.escape(example['description'])}</p>\n"
        
        # References
        if "references" in content:
            html_content += '<div class="references">\n<h2>References</h2>\n<ol>\n'
            for ref in content["references"]:
                if isinstance(ref, dict):
                    html_content += f"<li>{html.escape(ref.get('title', 'Reference'))}"
                    if ref.get('url'):
                        html_content += f' - <a href="{html.escape(ref["url"])}" target="_blank">{html.escape(ref["url"])}</a>'
                    html_content += "</li>\n"
                else:
                    html_content += f"<li>{html.escape(str(ref))}</li>\n"
            html_content += "</ol>\n</div>\n"
        
        html_content += "</body>\n</html>"
        return html_content
    
    @staticmethod
    def to_json(content: Dict[str, Any]) -> str:
        """Convert content to formatted JSON."""
        return json.dumps(content, indent=2, ensure_ascii=False)


class AudioGenerator:
    """Audio generation utility for creating spoken reports."""
    
    def __init__(self, settings: AgentSettings) -> None:
        """Initialize audio generator.
        
        Args:
            settings: Agent settings containing audio configuration.
        """
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)
        
    async def generate_audio_report(
        self,
        content: str,
        output_path: Optional[Path] = None,
        voice: str = "en-US-Standard-A",
        speed: float = 1.0,
    ) -> Optional[Path]:
        """Generate audio version of a report.
        
        Args:
            content: Text content to convert to speech.
            output_path: Optional output file path.
            voice: Voice to use for TTS.
            speed: Speech speed multiplier.
            
        Returns:
            Optional[Path]: Path to generated audio file or None if failed.
        """
        if not getattr(self.settings, 'enable_audio_reports', False):
            self.logger.info("Audio reports disabled in settings")
            return None
        
        try:
            # This would integrate with a TTS service like Google Cloud TTS
            # For now, we'll mock the implementation
            self.logger.info(f"Generating audio report ({len(content)} characters)")
            
            # Mock audio generation
            if output_path is None:
                output_path = Path(f"audio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
            
            # In a real implementation, this would call a TTS API
            # await self._call_tts_api(content, output_path, voice, speed)
            
            self.logger.info(f"Audio report would be saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Audio generation failed: {e}")
            return None


class ReporterAgent(BaseAgent):
    """Specialized agent for report generation and content formatting.
    
    The ReporterAgent handles tasks related to:
    - Generating comprehensive reports from research data
    - Converting data to multiple output formats (Markdown, HTML, JSON)
    - Creating summaries and executive overviews
    - Formatting content for readability
    - Generating code examples and documentation
    - Creating audio versions of reports (optional)
    
    Attributes:
        settings: Configuration settings for the agent.
        llm: Language model for content generation.
        formatter: Content formatter utility.
        audio_generator: Audio generation utility.
        report_templates: Predefined report templates.
        
    Example:
        >>> reporter = ReporterAgent(settings)
        >>> await reporter.initialize()
        >>> task = AgentTask(
        ...     type="generate_report",
        ...     parameters={"topic": "Machine Learning Tools", "data": research_data}
        ... )
        >>> result = await reporter.execute_task(task)
    """
    
    def __init__(
        self,
        settings: AgentSettings,
        name: str = "reporter",
    ) -> None:
        """Initialize the reporter agent.
        
        Args:
            settings: Configuration settings.
            name: Agent name (default: "reporter").
        """
        super().__init__(
            name=name,
            settings=settings,
            capabilities=[
                AgentCapability.REPORT_GENERATION,
                AgentCapability.CONTENT_FORMATTING,
                AgentCapability.SUMMARY_CREATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.AUDIO_PROCESSING,
                AgentCapability.MULTI_MODAL_PROCESSING,
                AgentCapability.NATURAL_LANGUAGE_PROCESSING,
            ]
        )
        
        # Core components
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.formatter = ReportFormatter()
        self.audio_generator = AudioGenerator(settings)
        
        # Report templates and configuration
        self.report_templates: Dict[str, ReportTemplate] = {}
        self.default_output_format = getattr(settings, 'default_output_format', 'markdown')
        self.max_report_length = 50000  # Maximum characters per report
        
        # Report generation settings
        self.include_metadata = True
        self.include_timestamps = True
        self.include_references = True
        
        self.logger.info(f"ReporterAgent initialized with {len(self.capabilities)} capabilities")
    
    async def _initialize(self) -> None:
        """Initialize the reporter agent and its components."""
        self.logger.info("Initializing ReporterAgent...")
        
        try:
            # Initialize LLM
            self._initialize_llm()
            
            # Load report templates
            self._load_report_templates()
            
            self.logger.info("ReporterAgent initialization complete")
            
        except Exception as e:
            self.logger.error(f"ReporterAgent initialization failed: {e}")
            raise
    
    def _initialize_llm(self) -> None:
        """Initialize the language model for content generation."""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.settings.llm.model,
                temperature=self.settings.llm.temperature,
                max_tokens=self.settings.llm.max_tokens,
                google_api_key=self.settings.google_api_key,
            )
            self.logger.info(f"Report LLM initialized: {self.settings.llm.model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize report LLM: {e}")
            raise
    
    def _load_report_templates(self) -> None:
        """Load predefined report templates."""
        # Define standard templates
        self.report_templates = {
            "search_report": {
                "title": "Tool Search Report",
                "sections": [
                    "executive_summary",
                    "search_methodology",
                    "tools_found",
                    "tool_analysis",
                    "recommendations",
                    "code_examples",
                    "references"
                ]
            },
            "research_report": {
                "title": "Research Analysis Report",
                "sections": [
                    "executive_summary",
                    "background",
                    "methodology",
                    "findings",
                    "analysis",
                    "tools_and_resources",
                    "recommendations",
                    "future_work",
                    "references"
                ]
            },
            "tool_comparison": {
                "title": "Tool Comparison Report",
                "sections": [
                    "executive_summary",
                    "comparison_methodology",
                    "tool_overview",
                    "feature_comparison",
                    "performance_analysis",
                    "recommendations",
                    "implementation_guide",
                    "references"
                ]
            },
            "bioinformatics_workflow": {
                "title": "Bioinformatics Workflow Report",
                "sections": [
                    "executive_summary",
                    "workflow_overview",
                    "data_requirements",
                    "tool_pipeline",
                    "implementation_steps",
                    "code_examples",
                    "validation",
                    "troubleshooting",
                    "references"
                ]
            }
        }
        
        self.logger.info(f"Loaded {len(self.report_templates)} report templates")
    
    async def _execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a reporter task.
        
        Args:
            task: Task to execute.
            
        Returns:
            AgentResponse: Task execution result.
        """
        self.logger.info(f"Executing reporter task: {task.type}")
        
        try:
            # Route task based on type
            if task.type == "generate_report":
                return await self._handle_generate_report(task)
            elif task.type == "format_content":
                return await self._handle_format_content(task)
            elif task.type == "create_summary":
                return await self._handle_create_summary(task)
            elif task.type == "generate_comparison":
                return await self._handle_generate_comparison(task)
            elif task.type == "create_workflow":
                return await self._handle_create_workflow(task)
            elif task.type == "generate_audio":
                return await self._handle_generate_audio(task)
            else:
                raise ValueError(f"Unknown task type: {task.type}")
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return AgentResponse(
                success=False,
                content=f"Report generation failed: {e}",
                error=str(e),
                agent_id=self.agent_id,
            )
    
    async def _handle_generate_report(self, task: AgentTask) -> AgentResponse:
        """Handle comprehensive report generation tasks."""
        topic = task.parameters.get("topic")
        if not topic:
            raise ValueError("Report generation requires 'topic' parameter")
        
        # Extract parameters
        search_results = task.parameters.get("search_results", {})
        enhanced_results = task.parameters.get("enhanced_results", {})
        output_format = task.parameters.get("output_format", self.default_output_format)
        template_name = task.parameters.get("template", "research_report")
        include_code_examples = task.parameters.get("include_code_examples", True)
        include_audio = task.parameters.get("include_audio", False)
        
        # Generate structured content
        structured_content = await self._generate_structured_content(
            topic=topic,
            search_results=search_results,
            enhanced_results=enhanced_results,
            template_name=template_name,
            include_code_examples=include_code_examples,
        )
        
        # Format content according to output format
        formatted_content = await self._format_structured_content(
            structured_content,
            output_format
        )
        
        # Generate audio if requested
        audio_path = None
        if include_audio:
            audio_path = await self.audio_generator.generate_audio_report(
                self._extract_text_for_audio(structured_content)
            )
        
        return AgentResponse(
            success=True,
            content=formatted_content,
            data={
                "structured_content": structured_content,
                "output_format": output_format,
                "template_used": template_name,
                "audio_path": str(audio_path) if audio_path else None,
                "word_count": len(formatted_content.split()),
                "character_count": len(formatted_content),
            },
            agent_id=self.agent_id,
        )
    
    async def _handle_format_content(self, task: AgentTask) -> AgentResponse:
        """Handle content formatting tasks."""
        content = task.parameters.get("content")
        if not content:
            raise ValueError("Content formatting requires 'content' parameter")
        
        input_format = task.parameters.get("input_format", "json")
        output_format = task.parameters.get("output_format", "markdown")
        
        # Parse input content
        if input_format == "json" and isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON content: {e}")
        
        # Format content
        formatted_content = await self._format_structured_content(content, output_format)
        
        return AgentResponse(
            success=True,
            content=formatted_content,
            data={
                "input_format": input_format,
                "output_format": output_format,
                "character_count": len(formatted_content),
            },
            agent_id=self.agent_id,
        )
    
    async def _handle_create_summary(self, task: AgentTask) -> AgentResponse:
        """Handle summary creation tasks."""
        data = task.parameters.get("data")
        if not data:
            raise ValueError("Summary creation requires 'data' parameter")
        
        summary_type = task.parameters.get("summary_type", "executive")
        max_length = task.parameters.get("max_length", 500)
        key_points = task.parameters.get("key_points", 5)
        
        # Generate summary using LLM
        summary = await self._generate_summary_with_llm(
            data=data,
            summary_type=summary_type,
            max_length=max_length,
            key_points=key_points,
        )
        
        return AgentResponse(
            success=True,
            content=summary,
            data={
                "summary_type": summary_type,
                "word_count": len(summary.split()),
                "character_count": len(summary),
                "key_points": key_points,
            },
            agent_id=self.agent_id,
        )
    
    async def _handle_generate_comparison(self, task: AgentTask) -> AgentResponse:
        """Handle tool comparison report generation."""
        tools = task.parameters.get("tools", [])
        if not tools:
            raise ValueError("Comparison generation requires 'tools' parameter")
        
        comparison_criteria = task.parameters.get("criteria", [
            "functionality", "ease_of_use", "performance", "documentation", "community"
        ])
        output_format = task.parameters.get("output_format", "markdown")
        
        # Generate comparison content
        comparison_content = await self._generate_tool_comparison(
            tools=tools,
            criteria=comparison_criteria,
        )
        
        # Format the comparison
        formatted_content = await self._format_structured_content(
            comparison_content,
            output_format
        )
        
        return AgentResponse(
            success=True,
            content=formatted_content,
            data={
                "tools_compared": len(tools),
                "criteria_used": comparison_criteria,
                "output_format": output_format,
            },
            agent_id=self.agent_id,
        )
    
    async def _handle_create_workflow(self, task: AgentTask) -> AgentResponse:
        """Handle workflow documentation generation."""
        workflow_data = task.parameters.get("workflow_data")
        if not workflow_data:
            raise ValueError("Workflow creation requires 'workflow_data' parameter")
        
        workflow_type = task.parameters.get("workflow_type", "bioinformatics")
        include_code = task.parameters.get("include_code", True)
        output_format = task.parameters.get("output_format", "markdown")
        
        # Generate workflow documentation
        workflow_content = await self._generate_workflow_documentation(
            workflow_data=workflow_data,
            workflow_type=workflow_type,
            include_code=include_code,
        )
        
        # Format the workflow
        formatted_content = await self._format_structured_content(
            workflow_content,
            output_format
        )
        
        return AgentResponse(
            success=True,
            content=formatted_content,
            data={
                "workflow_type": workflow_type,
                "include_code": include_code,
                "output_format": output_format,
            },
            agent_id=self.agent_id,
        )
    
    async def _handle_generate_audio(self, task: AgentTask) -> AgentResponse:
        """Handle audio report generation."""
        content = task.parameters.get("content")
        if not content:
            raise ValueError("Audio generation requires 'content' parameter")
        
        voice = task.parameters.get("voice", "en-US-Standard-A")
        speed = task.parameters.get("speed", 1.0)
        output_path = task.parameters.get("output_path")
        
        if output_path:
            output_path = Path(output_path)
        
        # Generate audio
        audio_path = await self.audio_generator.generate_audio_report(
            content=content,
            output_path=output_path,
            voice=voice,
            speed=speed,
        )
        
        if audio_path:
            return AgentResponse(
                success=True,
                content=f"Audio report generated: {audio_path}",
                data={
                    "audio_path": str(audio_path),
                    "voice": voice,
                    "speed": speed,
                    "text_length": len(content),
                },
                agent_id=self.agent_id,
            )
        else:
            return AgentResponse(
                success=False,
                content="Audio generation failed",
                error="Audio generation not available or failed",
                agent_id=self.agent_id,
            )
    
    async def _generate_structured_content(
        self,
        topic: str,
        search_results: Dict[str, Any],
        enhanced_results: Dict[str, Any],
        template_name: str,
        include_code_examples: bool,
    ) -> Dict[str, Any]:
        """Generate structured content for a report."""
        if not self.llm:
            raise RuntimeError("LLM not available for content generation")
        
        # Get template
        template = self.report_templates.get(template_name, self.report_templates["research_report"])
        
        # Build content generation prompt
        content_prompt = f"""
        Generate a comprehensive report on the topic: {topic}
        
        Template: {template_name}
        Required sections: {', '.join(template['sections'])}
        
        Search Results:
        {json.dumps(search_results, indent=2)}
        
        Enhanced Research Data:
        {json.dumps(enhanced_results, indent=2)}
        
        Please generate a structured report with the following requirements:
        1. Create an engaging and informative report
        2. Include all required sections from the template
        3. Provide detailed analysis and insights
        4. Include specific tool recommendations
        5. Add practical implementation guidance
        {"6. Include relevant code examples and usage instructions" if include_code_examples else ""}
        
        Format the response as structured JSON with the following structure:
        {{
            "title": "Report Title",
            "metadata": {{
                "topic": "{topic}",
                "generated_date": "{datetime.now().isoformat()}",
                "template": "{template_name}",
                "tools_analyzed": number,
                "data_sources": []
            }},
            "summary": "Executive summary text",
            "sections": [
                {{
                    "title": "Section Title",
                    "content": "Section content",
                    "subsections": [...]
                }}
            ],
            "tools": [
                {{
                    "name": "Tool Name",
                    "description": "Description",
                    "category": "Category",
                    "url": "URL",
                    "installation": "Installation command"
                }}
            ],
            {"\"code_examples\": [...],\n" if include_code_examples else ""}
            "references": [...]
        }}
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert technical writer specializing in bioinformatics and computational biology. Create comprehensive, well-structured reports."),
                HumanMessage(content=content_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse the structured response
            try:
                structured_content = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback: create structured content from text response
                structured_content = {
                    "title": f"Report: {topic}",
                    "metadata": {
                        "topic": topic,
                        "generated_date": datetime.now().isoformat(),
                        "template": template_name,
                    },
                    "summary": response.content[:500] + "..." if len(response.content) > 500 else response.content,
                    "sections": [
                        {
                            "title": "Generated Content",
                            "content": response.content
                        }
                    ],
                    "tools": [],
                    "references": []
                }
            
            return structured_content
            
        except Exception as e:
            self.logger.error(f"Content generation failed: {e}")
            raise
    
    async def _format_structured_content(
        self,
        content: Dict[str, Any],
        output_format: str,
    ) -> str:
        """Format structured content to the specified output format."""
        try:
            if output_format.lower() == "markdown":
                return self.formatter.to_markdown(content)
            elif output_format.lower() == "html":
                return self.formatter.to_html(content)
            elif output_format.lower() == "json":
                return self.formatter.to_json(content)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            self.logger.error(f"Content formatting failed: {e}")
            raise
    
    async def _generate_summary_with_llm(
        self,
        data: Any,
        summary_type: str,
        max_length: int,
        key_points: int,
    ) -> str:
        """Generate a summary using the LLM."""
        if not self.llm:
            raise RuntimeError("LLM not available for summary generation")
        
        summary_prompt = f"""
        Create a {summary_type} summary of the following data:
        
        {json.dumps(data, indent=2) if isinstance(data, dict) else str(data)}
        
        Requirements:
        - Maximum length: {max_length} words
        - Include {key_points} key points
        - Focus on the most important insights and findings
        - Write in a clear, professional tone
        - Highlight actionable recommendations
        
        Summary:
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert analyst who creates clear, concise summaries that capture the essential information."),
                HumanMessage(content=summary_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            raise
    
    async def _generate_tool_comparison(
        self,
        tools: List[Dict[str, Any]],
        criteria: List[str],
    ) -> Dict[str, Any]:
        """Generate a comprehensive tool comparison."""
        if not self.llm:
            raise RuntimeError("LLM not available for comparison generation")
        
        comparison_prompt = f"""
        Create a comprehensive comparison of the following tools:
        
        Tools:
        {json.dumps(tools, indent=2)}
        
        Comparison Criteria:
        {', '.join(criteria)}
        
        Please provide:
        1. An overview of each tool
        2. A detailed comparison matrix
        3. Pros and cons for each tool
        4. Use case recommendations
        5. Overall rankings and recommendations
        
        Format as structured JSON with clear sections.
        """
        
        try:
            messages = [
                SystemMessage(content="You are a technical expert who evaluates and compares bioinformatics tools to help researchers make informed decisions."),
                HumanMessage(content=comparison_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                comparison_content = json.loads(response.content)
            except json.JSONDecodeError:
                comparison_content = {
                    "title": "Tool Comparison Report",
                    "metadata": {
                        "tools_compared": len(tools),
                        "criteria": criteria,
                        "generated_date": datetime.now().isoformat(),
                    },
                    "summary": "Tool comparison analysis",
                    "sections": [
                        {
                            "title": "Comparison Analysis",
                            "content": response.content
                        }
                    ],
                    "tools": tools,
                }
            
            return comparison_content
            
        except Exception as e:
            self.logger.error(f"Tool comparison generation failed: {e}")
            raise
    
    async def _generate_workflow_documentation(
        self,
        workflow_data: Dict[str, Any],
        workflow_type: str,
        include_code: bool,
    ) -> Dict[str, Any]:
        """Generate workflow documentation."""
        if not self.llm:
            raise RuntimeError("LLM not available for workflow generation")
        
        workflow_prompt = f"""
        Create comprehensive workflow documentation for a {workflow_type} workflow:
        
        Workflow Data:
        {json.dumps(workflow_data, indent=2)}
        
        Please provide:
        1. Workflow overview and objectives
        2. Step-by-step procedures
        3. Required tools and dependencies
        4. Data requirements and formats
        5. Expected outputs
        {"6. Code examples and implementation details" if include_code else ""}
        7. Troubleshooting guide
        8. Best practices and recommendations
        
        Format as structured JSON with clear sections.
        """
        
        try:
            messages = [
                SystemMessage(content="You are a bioinformatics workflow expert who creates detailed, practical documentation for complex analytical pipelines."),
                HumanMessage(content=workflow_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                workflow_content = json.loads(response.content)
            except json.JSONDecodeError:
                workflow_content = {
                    "title": f"{workflow_type.title()} Workflow Documentation",
                    "metadata": {
                        "workflow_type": workflow_type,
                        "generated_date": datetime.now().isoformat(),
                        "include_code": include_code,
                    },
                    "summary": f"Comprehensive {workflow_type} workflow guide",
                    "sections": [
                        {
                            "title": "Workflow Documentation",
                            "content": response.content
                        }
                    ],
                }
            
            return workflow_content
            
        except Exception as e:
            self.logger.error(f"Workflow documentation generation failed: {e}")
            raise
    
    def _extract_text_for_audio(self, content: Dict[str, Any]) -> str:
        """Extract readable text from structured content for audio generation."""
        text_parts = []
        
        # Title
        if "title" in content:
            text_parts.append(f"Title: {content['title']}")
        
        # Summary
        if "summary" in content:
            text_parts.append(f"Summary: {content['summary']}")
        
        # Main sections
        if "sections" in content:
            for section in content["sections"]:
                if isinstance(section, dict):
                    text_parts.append(f"Section: {section.get('title', 'Untitled')}")
                    text_parts.append(section.get('content', ''))
        
        # Clean up text for audio
        full_text = " ".join(text_parts)
        
        # Remove markdown formatting
        full_text = re.sub(r'[#*`]', '', full_text)
        full_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', full_text)  # Remove links
        full_text = re.sub(r'\s+', ' ', full_text)  # Normalize whitespace
        
        # Limit length for audio
        if len(full_text) > 10000:  # ~10 minutes of speech
            full_text = full_text[:10000] + "... Content truncated for audio version."
        
        return full_text.strip()
    
    async def _shutdown(self) -> None:
        """Shutdown reporter agent and clean up resources."""
        self.logger.info("Shutting down ReporterAgent...")
        
        # Clear templates and cache
        self.report_templates.clear()
        
        self.logger.info("ReporterAgent shutdown complete")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available report templates.
        
        Returns:
            List[str]: List of template names.
        """
        return list(self.report_templates.keys())
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats.
        
        Returns:
            List[str]: List of format names.
        """
        return ["markdown", "html", "json"]
    
    def validate_template(self, template_name: str) -> bool:
        """Validate that a template exists.
        
        Args:
            template_name: Name of template to validate.
            
        Returns:
            bool: True if template exists.
        """
        return template_name in self.report_templates