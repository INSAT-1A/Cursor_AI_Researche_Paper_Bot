from __future__ import annotations

from typing import List

from agents.base_agent import BaseResearchAgent
from agents.critic_agent import CriticAgent
from agents.duckduckgo_agent import DuckDuckGoAgent
from agents.wiki_agent import WikiAgent
from models import AgentResponse, OrchestrationResult, SourceResult


class OrchestratorAgent(BaseResearchAgent):
    def __init__(self) -> None:
        super().__init__(name="orchestrator-agent")
        self.duckduckgo_agent = DuckDuckGoAgent()
        self.wiki_agent = WikiAgent()
        self.critic_agent = CriticAgent()

    def _merge_summaries(self, responses: List[AgentResponse]) -> str:
        valid = [r.summary for r in responses if r.summary]
        if not valid:
            return "No summary could be generated."
        return " ".join(valid[:2])

    def invoke(self, query: str) -> OrchestrationResult:
        optimized_query = self._optimize_quuery(query)
        self.logger.info("Orchestrating research for query: %s", optimized_query)

        duck_result = self.duckduckgo_agent.invoke(optimized_query)
        wiki_result = self.wiki_agent.invoke(optimized_query)
        critic_report = self.critic_agent.invoke(optimized_query, [duck_result, wiki_result])

        combined_items: List[SourceResult] = duck_result.items + wiki_result.items
        consolidated_urls = self._extract_source_urls(combined_items)
        combined_summary = self._merge_summaries([duck_result, wiki_result])

        return OrchestrationResult(
            query=query,
            combined_summary=combined_summary,
            duckduckgo_result=duck_result,
            wiki_result=wiki_result,
            critic_report=critic_report,
            consolidated_urls=consolidated_urls,
        )
