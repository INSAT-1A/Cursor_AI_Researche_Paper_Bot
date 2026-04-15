from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

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


class FallbackSynthesis(BaseModel):
    short_answer: str
    detailed_answer: str
    key_points: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)


class OrchestrationResult(BaseModel):
    query: str
    combined_summary: str
    search_result: AgentResponse
    wiki_result: AgentResponse
    critic_report: CriticReport
    consolidated_urls: List[str] = Field(default_factory=list)
    fallback_synthesis: Optional[FallbackSynthesis] = None
    orchestration_error: Optional[str] = Field(
        default=None,
        description="Populated when orchestration hits an unexpected failure",
    )


# =========================
# Research paper bot models
# =========================


class PaperAuthor(BaseModel):
    name: str


class Paper(BaseModel):
    id: str = Field(..., description="Stable id (arXiv id, S2 paperId, or derived)")
    title: str
    abstract: str = ""
    authors: List[PaperAuthor] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    venue: Optional[str] = None
    fields_of_study: List[str] = Field(default_factory=list)

    pdf_url: Optional[str] = None
    source: str = Field(..., description="arxiv|semantic_scholar|merged")

    url: Optional[str] = Field(default=None, description="Landing page URL")
    doi: Optional[str] = None

    citation_count: Optional[int] = Field(default=None, ge=0)

    extra: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")


class PaperCard(BaseModel):
    paper: Paper
    score: float = Field(..., ge=0.0)
    rationale: str = ""


class PaperSearchResponse(BaseModel):
    query: str
    total_candidates: int = Field(..., ge=0)
    ranked_cards: List[PaperCard] = Field(default_factory=list)
    top_score: float = Field(default=0.0, ge=0.0)
    avg_top3_score: float = Field(default=0.0, ge=0.0)
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class PaperSummary(BaseModel):
    paper_id: str
    tldr: str
    simplified: str
    key_contributions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    suggested_followups: List[str] = Field(default_factory=list)
    source_used: str = Field(..., description="pdf|abstract")
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
