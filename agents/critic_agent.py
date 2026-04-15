from __future__ import annotations

from typing import List, Optional

from langchain_anthropic import ChatAnthropic

from agents.base_agent import BaseResearchAgent
from models import AgentResponse, CriticReport, SourceResult


class CriticAgent(BaseResearchAgent):
    def __init__(self, llm: Optional[ChatAnthropic] = None) -> None:
        super().__init__(name="critic-agent", llm=llm)

    def _estimate_confidence(self, responses: List[AgentResponse]) -> float:
        if not responses:
            return 0.0
        success_count = sum(1 for r in responses if r.success)
        source_count = sum(len(r.source_urls) for r in responses)
        score = (success_count / len(responses)) * 0.7 + min(source_count / 10, 0.3)
        return round(min(score, 1.0), 2)

    def _extract_source_urls(self, items: List[SourceResult]) -> List[str]:
        return super()._extract_source_urls(items)

    @staticmethod
    def _fallback_report(exc: str) -> CriticReport:
        return CriticReport(
            confidence_score=0.0,
            strengths=[],
            weaknesses=[f"Critic agent failed: {exc}"],
            recommendations=["Retry the research query or check logs for details."],
            verdict="Critic could not complete analysis.",
        )

    def invoke(self, query: str, responses: List[AgentResponse]) -> CriticReport:
        try:
            if responses is None:
                responses = []
            if not isinstance(responses, list):
                raise TypeError(f"responses must be a list, got {type(responses).__name__}")

            optimized_query = self._optimize_quuery(query)
            self.logger.info("Running critic analysis for query: %s", optimized_query)

            strengths: List[str] = []
            weaknesses: List[str] = []
            recommendations: List[str] = []

            valid_responses: List[AgentResponse] = []
            for response in responses:
                if response is None:
                    weaknesses.append("Received null agent response in critic input.")
                    continue
                if not isinstance(response, AgentResponse):
                    weaknesses.append(
                        f"Invalid response type in critic input: {type(response).__name__}"
                    )
                    continue
                valid_responses.append(response)

            for response in valid_responses:
                if response.success:
                    strengths.append(
                        f"{response.agent_name} succeeded with {len(response.source_urls)} sources."
                    )
                else:
                    weaknesses.append(
                        f"{response.agent_name} failed with error: {response.error or 'unknown error'}"
                    )

            if sum(len(r.source_urls) for r in valid_responses) < 3:
                weaknesses.append("Low source coverage across the available results.")
                recommendations.append("Retry with narrower or more specific keywords.")
            elif valid_responses:
                strengths.append("Cross-source evidence is available.")
                recommendations.append("Prioritize URLs present in both search and wiki results.")

            if not recommendations:
                recommendations.append("Keep monitoring for contradictory claims in sources.")

            confidence = self._estimate_confidence(valid_responses)
            verdict = (
                "Research quality is acceptable for a first-pass summary."
                if confidence >= 0.6
                else "Research quality is weak; additional evidence gathering is recommended."
            )

            return CriticReport(
                confidence_score=confidence,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                verdict=verdict,
            )
        except Exception as exc:
            self.logger.exception("Critic agent failed")
            return self._fallback_report(str(exc))
