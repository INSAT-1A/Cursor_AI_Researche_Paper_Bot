from __future__ import annotations

from functools import lru_cache
from typing import List, Optional
from urllib.parse import quote

from langchain_anthropic import ChatAnthropic
from requests import exceptions as req_exc

from agents.base_agent import BaseResearchAgent
from http_utils import parse_response_json_dict, request_with_retries
from models import AgentResponse, SourceResult

WIKI_HEADERS = {
    "User-Agent": "research-paper-bot/1.0 (https://local.app)",
    "Accept": "application/json",
}


class WikiAgent(BaseResearchAgent):
    def __init__(self, llm: Optional[ChatAnthropic] = None) -> None:
        super().__init__(name="wiki-agent", llm=llm)

    def _fetch_summary_item(self, title: str) -> SourceResult | None:
        page_title = quote(title, safe="")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
        try:
            response = request_with_retries(
                logger=self.logger,
                method="GET",
                url=url,
                headers=WIKI_HEADERS,
                timeout=12,
                retries=2,
                retry_statuses=(403, 429, 500, 502, 503, 504),
                context=f"wiki summary {title}",
            )
            if response.status_code != 200:
                self.logger.debug("Summary HTTP %s for title=%s", response.status_code, title)
                return None
            data = parse_response_json_dict(response, self.logger, f"wiki summary {title}")
            if data is None:
                return None
            canonical_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            if not canonical_url:
                return None
            return SourceResult(
                title=data.get("title") or title,
                url=canonical_url,
                snippet=data.get("extract") or "No summary available.",
                source="wikipedia",
            )
        except req_exc.Timeout:
            self.logger.warning("Wikipedia summary timeout for title=%s", title)
            return None
        except req_exc.RequestException as exc:
            self.logger.warning("Wikipedia summary request failed for title=%s: %s", title, exc)
            return None
        except (TypeError, ValueError, KeyError) as exc:
            self.logger.warning("Wikipedia summary parse error for title=%s: %s", title, exc)
            return None

    @lru_cache(maxsize=128)
    def _wiki_search_cached(self, query: str) -> tuple[SourceResult, ...]:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
        }
        response = request_with_retries(
            logger=self.logger,
            method="GET",
            url=search_url,
            headers=WIKI_HEADERS,
            params=params,
            timeout=12,
            retries=2,
            retry_statuses=(403, 429, 500, 502, 503, 504),
            context=f"wiki search {query}",
        )
        root = parse_response_json_dict(response, self.logger, "wiki search")
        if root is None:
            return tuple()
        query_block = root.get("query") or {}
        if not isinstance(query_block, dict):
            return tuple()
        results = query_block.get("search") or []
        if not isinstance(results, list):
            return tuple()

        items: List[SourceResult] = []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            title = entry.get("title", "") or ""
            if not title:
                continue
            summary_item = self._fetch_summary_item(title)
            if summary_item:
                items.append(summary_item)
        return tuple(items)

    def _wiki_search(self, query: str) -> List[SourceResult]:
        return list(self._wiki_search_cached(query))

    def invoke(self, query: str) -> AgentResponse:
        optimized_query = self._optimize_quuery(query)
        query_candidates: List[str] = []
        for candidate in [str(query or "").strip(), optimized_query]:
            if candidate and candidate not in query_candidates:
                query_candidates.append(candidate)
        self.logger.info("Calling Wikipedia APIs for query candidates: %s", query_candidates)
        q_display = query if isinstance(query, str) else str(query)
        try:
            items: List[SourceResult] = []
            used_query = optimized_query
            for q in query_candidates:
                used_query = q
                items = self._wiki_search(q)
                if items:
                    break
            summary = (
                items[0].snippet
                if items
                else f"No direct Wikipedia entries found for: {used_query}."
            )
            urls = self._extract_source_urls(items)
            return AgentResponse(
                agent_name=self.name,
                query=q_display,
                optimized_query=used_query,
                summary=summary,
                items=items,
                source_urls=urls,
                success=True,
            )
        except req_exc.Timeout as exc:
            self.logger.error("Wiki agent timeout: %s", exc)
            fallback = self._fallback_query(query)
            return AgentResponse(
                agent_name=self.name,
                query=q_display,
                optimized_query=fallback,
                summary="Wikipedia request timed out.",
                items=[],
                source_urls=[],
                success=False,
                error=f"timeout: {exc}",
            )
        except req_exc.RequestException as exc:
            self.logger.error("Wiki agent HTTP error: %s", exc)
            fallback = self._fallback_query(query)
            return AgentResponse(
                agent_name=self.name,
                query=q_display,
                optimized_query=fallback,
                summary="Wikipedia HTTP request failed.",
                items=[],
                source_urls=[],
                success=False,
                error=f"http: {exc}",
            )
        except (ValueError, TypeError, KeyError) as exc:
            self.logger.exception("Wiki agent data error")
            fallback = self._fallback_query(query)
            return AgentResponse(
                agent_name=self.name,
                query=q_display,
                optimized_query=fallback,
                summary="Wikipedia response could not be parsed.",
                items=[],
                source_urls=[],
                success=False,
                error=f"data: {exc}",
            )
        except Exception as exc:
            self.logger.exception("Wiki agent unexpected error")
            fallback = self._fallback_query(query)
            return AgentResponse(
                agent_name=self.name,
                query=q_display,
                optimized_query=fallback,
                summary="Wikipedia lookup failed. Returning fallback response.",
                items=[],
                source_urls=[],
                success=False,
                error=str(exc),
            )
