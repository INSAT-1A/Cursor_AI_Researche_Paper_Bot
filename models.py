from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SourceResult(BaseModel):
    title: str = Field(..., description="Title of the source item")
    url: str = Field(..., description="Canonical source URL")
    snippet: str = Field(..., description="Short snippet for quick context")
    source: str = Field(..., description="Agent/source name")


class ToolResponse(BaseModel):
    tool_name: str
    query: str
    items: List[SourceResult] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class DuckDuckGoToolResponse(ToolResponse):
    tool_name: str = "duckduckgo"
    abstract: Optional[str] = None


class WikipediaToolResponse(ToolResponse):
    tool_name: str = "wikipedia"
    top_page_title: Optional[str] = None


class AgentResponse(BaseModel):
    agent_name: str
    query: str
    optimized_query: str
    summary: str
    items: List[SourceResult] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class CriticReport(BaseModel):
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    verdict: str


class OrchestrationResult(BaseModel):
    query: str
    combined_summary: str
    duckduckgo_result: AgentResponse
    wiki_result: AgentResponse
    critic_report: CriticReport
    consolidated_urls: List[str] = Field(default_factory=list)
