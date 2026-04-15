from __future__ import annotations

from typing import List
from urllib.parse import quote

import requests

from agents.base_agent import BaseResearchAgent
from models import AgentResponse, SourceResult


class WikiAgent(BaseResearchAgent):
    def __init__(self) -> None:
        super().__init__(name="wiki-agent")

    def _fetch_summary_item(self, title: str) -> SourceResult | None:
        page_title = quote(title)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
        response = requests.get(url, timeout=12)
        if response.status_code != 200:
            return None
        data = response.json()
        canonical_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        if not canonical_url:
            return None
        return SourceResult(
            title=data.get("title") or title,
            url=canonical_url,
            snippet=data.get("extract") or "No summary available.",
            source="wikipedia",
        )

    def _wiki_search(self, query: str) -> List[SourceResult]:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
        }
        response = requests.get(search_url, params=params, timeout=12)
        response.raise_for_status()
        results = response.json().get("query", {}).get("search", [])

        items: List[SourceResult] = []
        for entry in results:
            title = entry.get("title", "")
            if not title:
                continue
            summary_item = self._fetch_summary_item(title)
            if summary_item:
                items.append(summary_item)
        return items

    def invoke(self, query: str) -> AgentResponse:
        optimized_query = self._optimize_quuery(query)
        self.logger.info("Calling Wikipedia APIs for query: %s", optimized_query)
        try:
            items = self._wiki_search(optimized_query)
            summary = (
                items[0].snippet
                if items
                else f"No direct Wikipedia entries found for: {optimized_query}."
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
            self.logger.exception("Wiki agent failed")
            fallback = self._fallback_query(query)
            return AgentResponse(
                agent_name=self.name,
                query=query,
                optimized_query=fallback,
                summary="Wikipedia lookup failed. Returning fallback response.",
                items=[],
                source_urls=[],
                success=False,
                error=str(exc),
            )
