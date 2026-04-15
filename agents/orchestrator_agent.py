from __future__ import annotations

import json
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from agents.base_agent import BaseResearchAgent
from agents.critic_agent import CriticAgent
from agents.search_agent import SearchAgent
from agents.wiki_agent import WikiAgent
from models import (
    AgentResponse,
    CriticReport,
    FallbackSynthesis,
    OrchestrationResult,
    SourceResult,
)


class OrchestratorAgent(BaseResearchAgent):
    def __init__(self) -> None:
        """Initialize the orchestrator and its specialist agents."""
        self.name = "OrchestratorAgent"
        try:
            self.llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.0)
        except Exception as exc:
            # Logger not yet available from BaseResearchAgent
            import logging

            logging.getLogger("OrchestratorAgent").exception("ChatAnthropic initialization failed")
            raise RuntimeError(
                "Could not initialize orchestrator LLM (ChatAnthropic). "
                "Set ANTHROPIC_API_KEY and verify the model id."
            ) from exc
        super().__init__(name=self.name, llm=self.llm)
        try:
            self.search_agent = SearchAgent()
            self.wiki_agent = WikiAgent()
            self.critic_agent = CriticAgent()
        except Exception as exc:
            self.logger.exception("Failed to construct specialist agents")
            raise RuntimeError("Could not initialize one or more specialist agents.") from exc
        self.logger.info(
            f"{self.name} initialized | roster=[SearchAgent, WikiAgent, CriticAgent]"
        )

    def _merge_summaries(self, responses: List[AgentResponse]) -> str:
        valid = [r.summary for r in responses if r and r.summary]
        if not valid:
            return "No summary could be generated."
        return " ".join(valid[:2])

    def _synthesize_fallback_answer(
        self,
        query: str,
        search_result: AgentResponse,
        wiki_result: AgentResponse,
        critic_report: CriticReport,
        consolidated_urls: List[str],
    ) -> FallbackSynthesis:
        # Keep prompt small and source-grounded.
        snippets: List[str] = []
        for item in (search_result.items + wiki_result.items)[:8]:
            snippets.append(f"- {item.title}: {item.snippet[:260]}")
        snippet_block = "\n".join(snippets) if snippets else "- No snippets available."
        urls_block = "\n".join([f"- {u}" for u in consolidated_urls[:10]]) or "- No URLs available."
        prompt = f"""
You are a careful research assistant. Generate a richer fallback answer for a non-technical query.
Use ONLY provided evidence; do not invent facts.

Return ONLY valid JSON with keys:
{{
  "short_answer": "<1 paragraph>",
  "detailed_answer": "<2-4 paragraphs>",
  "key_points": ["..."],
  "caveats": ["..."],
  "sources_used": ["url1", "url2"]
}}

User query: {query}
Search summary: {search_result.summary}
Wiki summary: {wiki_result.summary}
Critic verdict: {critic_report.verdict}
Critic confidence: {critic_report.confidence_score}

Evidence snippets:
{snippet_block}

Available URLs:
{urls_block}
""".strip()
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw = getattr(response, "content", "")
            if isinstance(raw, list):
                joined = []
                for block in raw:
                    if isinstance(block, dict):
                        joined.append(str(block.get("text", "")))
                    else:
                        joined.append(str(block))
                raw = "".join(joined)
            text = str(raw).strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
            data = json.loads(text)
            synthesis = FallbackSynthesis.model_validate(data)
            if not synthesis.sources_used:
                synthesis.sources_used = consolidated_urls[:5]
            return synthesis
        except Exception as exc:
            self.logger.warning("Fallback synthesis failed, using template: %s", exc)
            return FallbackSynthesis(
                short_answer=search_result.summary or wiki_result.summary or "Limited fallback evidence available.",
                detailed_answer=(
                    f"{search_result.summary} {wiki_result.summary} "
                    f"Critic verdict: {critic_report.verdict}"
                ).strip(),
                key_points=[
                    *(critic_report.strengths[:3]),
                    *(critic_report.recommendations[:2]),
                ],
                caveats=critic_report.weaknesses[:4],
                sources_used=consolidated_urls[:5],
            )

    @staticmethod
    def _fallback_query_variants(query: str) -> List[str]:
        base = (query or "").strip()
        variants = [base, f"{base} wikipedia", f"{base} overview"]
        out: List[str] = []
        for v in variants:
            v = v.strip()
            if v and v not in out:
                out.append(v)
        return out

    @staticmethod
    def _merge_agent_results(agent_name: str, query: str, responses: List[AgentResponse]) -> AgentResponse:
        if not responses:
            return AgentResponse(
                agent_name=agent_name,
                query=query,
                optimized_query=query,
                summary=f"{agent_name} returned no responses.",
                items=[],
                source_urls=[],
                success=False,
                error="no responses",
            )
        combined_items: List[SourceResult] = []
        for r in responses:
            combined_items.extend(r.items or [])
        # dedupe URLs while preserving order
        seen = set()
        dedup_items: List[SourceResult] = []
        for item in combined_items:
            u = item.url
            if not u or u in seen:
                continue
            seen.add(u)
            dedup_items.append(item)
        source_urls = [i.url for i in dedup_items]
        success = any(r.success for r in responses)
        summary = (
            responses[0].summary
            if dedup_items
            else f"{agent_name} returned no items for fallback query variants."
        )
        optimized_used = " | ".join([r.optimized_query for r in responses if r.optimized_query])
        errors = [r.error for r in responses if r.error]
        return AgentResponse(
            agent_name=agent_name,
            query=query,
            optimized_query=optimized_used or query,
            summary=summary,
            items=dedup_items,
            source_urls=source_urls,
            success=success,
            error="; ".join(errors) if errors else None,
        )

    @staticmethod
    def _empty_search_failure(query: str, optimized: str, err: str) -> AgentResponse:
        return AgentResponse(
            agent_name="search-agent",
            query=query,
            optimized_query=optimized,
            summary="Search step failed during orchestration.",
            items=[],
            source_urls=[],
            success=False,
            error=err,
        )

    @staticmethod
    def _empty_wiki_failure(query: str, optimized: str, err: str) -> AgentResponse:
        return AgentResponse(
            agent_name="wiki-agent",
            query=query,
            optimized_query=optimized,
            summary="Wikipedia step failed during orchestration.",
            items=[],
            source_urls=[],
            success=False,
            error=err,
        )

    @staticmethod
    def _minimal_critic() -> CriticReport:
        return CriticReport(
            confidence_score=0.0,
            strengths=[],
            weaknesses=["Orchestration could not produce a critic report."],
            recommendations=["Re-run with logging enabled to capture the failure."],
            verdict="Incomplete orchestration.",
        )

    def invoke(self, query: str) -> OrchestrationResult:
        q_display = query if isinstance(query, str) else str(query)
        optimized_query = self._optimize_quuery(q_display)
        self.logger.info(
            "Orchestrating research for query='%s' optimized='%s'",
            q_display,
            optimized_query,
        )

        try:
            try:
                # Adaptive fallback queries improve recall on broad/non-technical topics.
                variants = self._fallback_query_variants(q_display)
                search_runs = [self.search_agent.invoke(v) for v in variants]
                search_result = self._merge_agent_results("search-agent", q_display, search_runs)
            except Exception as exc:
                self.logger.exception("Search agent raised during orchestration")
                search_result = self._empty_search_failure(
                    q_display, optimized_query, str(exc)
                )

            try:
                variants = self._fallback_query_variants(q_display)
                wiki_runs = [self.wiki_agent.invoke(v) for v in variants]
                wiki_result = self._merge_agent_results("wiki-agent", q_display, wiki_runs)
            except Exception as exc:
                self.logger.exception("Wiki agent raised during orchestration")
                wiki_result = self._empty_wiki_failure(q_display, optimized_query, str(exc))

            try:
                critic_report = self.critic_agent.invoke(
                    q_display, [search_result, wiki_result]
                )
            except Exception as exc:
                self.logger.exception("Critic agent raised during orchestration")
                critic_report = CriticAgent._fallback_report(str(exc))

            combined_items: List[SourceResult] = []
            try:
                combined_items = (search_result.items or []) + (wiki_result.items or [])
            except Exception as exc:
                self.logger.warning("Could not merge source items: %s", exc)

            consolidated_urls = self._extract_source_urls(combined_items)
            combined_summary = self._merge_summaries([search_result, wiki_result])
            fallback_synthesis = self._synthesize_fallback_answer(
                query=q_display,
                search_result=search_result,
                wiki_result=wiki_result,
                critic_report=critic_report,
                consolidated_urls=consolidated_urls,
            )

            return OrchestrationResult(
                query=q_display,
                combined_summary=combined_summary,
                search_result=search_result,
                wiki_result=wiki_result,
                critic_report=critic_report,
                consolidated_urls=consolidated_urls,
                fallback_synthesis=fallback_synthesis,
                orchestration_error=None,
            )
        except Exception as exc:
            self.logger.exception("Fatal orchestration failure")
            err_msg = str(exc)
            return OrchestrationResult(
                query=q_display,
                combined_summary="Orchestration failed before a full result could be built.",
                search_result=self._empty_search_failure(
                    q_display, optimized_query, err_msg
                ),
                wiki_result=self._empty_wiki_failure(
                    q_display, optimized_query, err_msg
                ),
                critic_report=self._minimal_critic(),
                consolidated_urls=[],
                fallback_synthesis=None,
                orchestration_error=err_msg,
            )
