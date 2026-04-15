from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Optional

from langchain_anthropic import ChatAnthropic
from requests import exceptions as req_exc

from agents.base_agent import BaseResearchAgent
from http_utils import parse_response_json_dict, request_with_retries
from models import AgentResponse, SourceResult


class SearchAgent(BaseResearchAgent):
    def __init__(self, llm: Optional[ChatAnthropic] = None) -> None:
        super().__init__(name="search-agent", llm=llm)

    def _normalize_related_topics(self, payload: dict[str, Any]) -> List[SourceResult]:
        results: List[SourceResult] = []
        topics = payload.get("RelatedTopics") or []
        if not isinstance(topics, list):
            return results

        for topic in topics:
            if not isinstance(topic, dict):
                continue
            if "Topics" in topic:
                raw_candidates = topic.get("Topics")
                candidates = raw_candidates if isinstance(raw_candidates, list) else []
            else:
                candidates = [topic]
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                text = item.get("Text", "") or ""
                first_url = item.get("FirstURL", "") or ""
                if not first_url:
                    continue
                title = text.split(" - ")[0] if text else "DuckDuckGo Result"
                try:
                    results.append(
                        SourceResult(
                            title=title,
                            url=first_url,
                            snippet=text or "No snippet available.",
                            source="duckduckgo",
                        )
                    )
                except Exception as exc:
                    self.logger.warning("Skipping malformed DuckDuckGo item: %s", exc)
                    continue
                if len(results) >= 8:
                    return results
        return results

    @lru_cache(maxsize=128)
    def _search_cached(self, query: str) -> tuple[SourceResult, ...]:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
        }
        self.logger.info("Calling DuckDuckGo API for query: %s", query)
        response = request_with_retries(
            logger=self.logger,
            method="GET",
            url=url,
            headers={"User-Agent": "research-paper-bot/1.0 (https://local.app)"},
            params=params,
            timeout=12,
            retries=2,
            retry_statuses=(429, 500, 502, 503, 504),
            context=f"duckduckgo search {query}",
        )
        payload = parse_response_json_dict(response, self.logger, "duckduckgo")
        if payload is None:
            raise ValueError("DuckDuckGo returned a non-object JSON body.")

        items = self._normalize_related_topics(payload)

        abstract_url = payload.get("AbstractURL")
        if abstract_url:
            try:
                items.insert(
                    0,
                    SourceResult(
                        title=payload.get("Heading") or "DuckDuckGo Abstract",
                        url=abstract_url,
                        snippet=payload.get("AbstractText") or "No abstract text provided.",
                        source="duckduckgo",
                    ),
                )
            except Exception as exc:
                self.logger.warning("Could not add DuckDuckGo abstract to items: %s", exc)
        return tuple(items)

    def invoke(self, query: str) -> AgentResponse:
        optimized_query = self._optimize_quuery(query)
        query_candidates: List[str] = []
        for candidate in [str(query or "").strip(), optimized_query]:
            if candidate and candidate not in query_candidates:
                query_candidates.append(candidate)
        try:
            items: List[SourceResult] = []
            used_query = optimized_query
            for q in query_candidates:
                used_query = q
                items = list(self._search_cached(q))
                if items:
                    break
            summary = f"Found {len(items)} candidate DuckDuckGo sources for: {used_query}."
            urls = self._extract_source_urls(items)
            return AgentResponse(
                agent_name=self.name,
                query=query if isinstance(query, str) else str(query),
                optimized_query=used_query,
                summary=summary,
                items=items,
                source_urls=urls,
                success=True,
            )
        except req_exc.Timeout as exc:
            self.logger.error("Search agent timeout: %s", exc)
            fallback = self._fallback_query(query)
            return AgentResponse(
                agent_name=self.name,
                query=query if isinstance(query, str) else str(query),
                optimized_query=fallback,
                summary="DuckDuckGo request timed out.",
                items=[],
                source_urls=[],
                success=False,
                error=f"timeout: {exc}",
            )
        except req_exc.RequestException as exc:
            self.logger.error("Search agent HTTP error: %s", exc)
            fallback = self._fallback_query(query)
            return AgentResponse(
                agent_name=self.name,
                query=query if isinstance(query, str) else str(query),
                optimized_query=fallback,
                summary="DuckDuckGo HTTP request failed.",
                items=[],
                source_urls=[],
                success=False,
                error=f"http: {exc}",
            )
        except (ValueError, TypeError, KeyError) as exc:
            self.logger.exception("Search agent data error")
            fallback = self._fallback_query(query)
            return AgentResponse(
                agent_name=self.name,
                query=query if isinstance(query, str) else str(query),
                optimized_query=fallback,
                summary="DuckDuckGo response could not be parsed.",
                items=[],
                source_urls=[],
                success=False,
                error=f"data: {exc}",
            )
        except Exception as exc:
            self.logger.exception("Search agent unexpected error")
            fallback = self._fallback_query(query)
            return AgentResponse(
                agent_name=self.name,
                query=query if isinstance(query, str) else str(query),
                optimized_query=fallback,
                summary="DuckDuckGo lookup failed. Returning fallback response.",
                items=[],
                source_urls=[],
                success=False,
                error=str(exc),
            )
