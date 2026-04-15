from __future__ import annotations

from typing import List

import requests

from agents.base_agent import BaseResearchAgent
from models import AgentResponse, SourceResult


class DuckDuckGoAgent(BaseResearchAgent):
    def __init__(self) -> None:
        super().__init__(name="duckduckgo-agent")

    def _normalize_related_topics(self, payload: dict) -> List[SourceResult]:
        results: List[SourceResult] = []
        for topic in payload.get("RelatedTopics", []):
            if "Topics" in topic:
                candidates = topic["Topics"]
            else:
                candidates = [topic]
            for item in candidates:
                text = item.get("Text", "")
                first_url = item.get("FirstURL", "")
                if not first_url:
                    continue
                title = text.split(" - ")[0] if text else "DuckDuckGo Result"
                results.append(
                    SourceResult(
                        title=title,
                        url=first_url,
                        snippet=text or "No snippet available.",
                        source="duckduckgo",
                    )
                )
                if len(results) >= 8:
                    return results
        return results

    def invoke(self, query: str) -> AgentResponse:
        optimized_query = self._optimize_quuery(query)
        url = "https://api.duckduckgo.com/"
        params = {
            "q": optimized_query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
        }
        self.logger.info("Calling DuckDuckGo API for query: %s", optimized_query)
        try:
            response = requests.get(url, params=params, timeout=12)
            response.raise_for_status()
            payload = response.json()
            items = self._normalize_related_topics(payload)

            abstract_url = payload.get("AbstractURL")
            if abstract_url:
                items.insert(
                    0,
                    SourceResult(
                        title=payload.get("Heading") or "DuckDuckGo Abstract",
                        url=abstract_url,
                        snippet=payload.get("AbstractText") or "No abstract text provided.",
                        source="duckduckgo",
                    ),
                )

            summary = payload.get("AbstractText") or (
                f"Found {len(items)} candidate DuckDuckGo sources for: {optimized_query}."
            )
            urls = self._extract_source_urls(items)
            return AgentResponse(
                agent_name=self.name,
                query=query,
                optimized_query=optimized_query,
                summary=summary,
                items=items,
                source_urls=urls,
                success=True,
            )
        except Exception as exc:
            self.logger.exception("DuckDuckGo agent failed")
            fallback = self._fallback_query(query)
            return AgentResponse(
                agent_name=self.name,
                query=query,
                optimized_query=fallback,
                summary="DuckDuckGo lookup failed. Returning fallback response.",
                items=[],
                source_urls=[],
                success=False,
                error=str(exc),
            )
