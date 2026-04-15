from __future__ import annotations

from typing import List

from agents.base_agent import BaseResearchAgent
from models import AgentResponse, CriticReport, SourceResult


class CriticAgent(BaseResearchAgent):
    def __init__(self) -> None:
        super().__init__(name="critic-agent")

    def _estimate_confidence(self, responses: List[AgentResponse]) -> float:
        if not responses:
            return 0.0
        success_count = sum(1 for r in responses if r.success)
        source_count = sum(len(r.source_urls) for r in responses)
        score = (success_count / len(responses)) * 0.7 + min(source_count / 10, 0.3)
        return round(min(score, 1.0), 2)

    def _extract_source_urls(self, items: List[SourceResult]) -> List[str]:
        # Maintains required method contract for all agents.
        return super()._extract_source_urls(items)

    def invoke(self, query: str, responses: List[AgentResponse]) -> CriticReport:
        optimized_query = self._optimize_quuery(query)
        self.logger.info("Running critic analysis for query: %s", optimized_query)

        strengths: List[str] = []
        weaknesses: List[str] = []
        recommendations: List[str] = []

        for response in responses:
            if response.success:
                strengths.append(
                    f"{response.agent_name} succeeded with {len(response.source_urls)} sources."
                )
            else:
                weaknesses.append(
                    f"{response.agent_name} failed with error: {response.error or 'unknown error'}"
                )

        if sum(len(r.source_urls) for r in responses) < 3:
            weaknesses.append("Low source coverage across the available results.")
            recommendations.append("Retry with narrower or more specific keywords.")
        else:
            strengths.append("Cross-source evidence is available.")
            recommendations.append("Prioritize URLs present in both search and wiki results.")

        if not recommendations:
            recommendations.append("Keep monitoring for contradictory claims in sources.")

        confidence = self._estimate_confidence(responses)
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
