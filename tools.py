from __future__ import annotations

import logging
from typing import List
from urllib.parse import quote

import requests

from models import DuckDuckGoToolResponse, SourceResult, WikipediaToolResponse


logger = logging.getLogger("research-tools")


def _extract_source_urls(items: List[SourceResult]) -> List[str]:
    urls: List[str] = []
    for item in items:
        if item.url and item.url not in urls:
            urls.append(item.url)
    return urls


def duckduckgo_search(query: str, max_items: int = 8) -> DuckDuckGoToolResponse:
    """
    Query DuckDuckGo Instant Answer API and return validated Pydantic data.
    """
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query.strip(),
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
    }
    logger.info("Running duckduckgo_search for query: %s", query)

    try:
        response = requests.get(url, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()

        items: List[SourceResult] = []

        abstract_url = payload.get("AbstractURL", "")
        abstract_text = payload.get("AbstractText", "")
        if abstract_url:
            items.append(
                SourceResult(
                    title=payload.get("Heading") or "DuckDuckGo Abstract",
                    url=abstract_url,
                    snippet=abstract_text or "No abstract text available.",
                    source="duckduckgo",
                )
            )

        for topic in payload.get("RelatedTopics", []):
            nested = topic.get("Topics")
            candidates = nested if isinstance(nested, list) else [topic]
            for candidate in candidates:
                first_url = candidate.get("FirstURL", "")
                text = candidate.get("Text", "")
                if not first_url:
                    continue
                items.append(
                    SourceResult(
                        title=text.split(" - ")[0] if text else "DuckDuckGo Result",
                        url=first_url,
                        snippet=text or "No snippet available.",
                        source="duckduckgo",
                    )
                )
                if len(items) >= max_items:
                    break
            if len(items) >= max_items:
                break

        source_urls = _extract_source_urls(items)
        return DuckDuckGoToolResponse(
            query=query,
            items=items,
            source_urls=source_urls,
            abstract=abstract_text or None,
            success=True,
        )
    except Exception as exc:
        logger.exception("duckduckgo_search failed")
        return DuckDuckGoToolResponse(
            query=query,
            items=[],
            source_urls=[],
            success=False,
            error=str(exc),
        )


def _fetch_wikipedia_summary(title: str) -> SourceResult | None:
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    response = requests.get(summary_url, timeout=12)
    if response.status_code != 200:
        return None
    payload = response.json()
    page_url = payload.get("content_urls", {}).get("desktop", {}).get("page", "")
    if not page_url:
        return None
    return SourceResult(
        title=payload.get("title") or title,
        url=page_url,
        snippet=payload.get("extract") or "No summary available.",
        source="wikipedia",
    )


def wikipedia_search(query: str, max_items: int = 5) -> WikipediaToolResponse:
    """
    Query Wikipedia and return validated Pydantic data.
    """
    logger.info("Running wikipedia_search for query: %s", query)
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query.strip(),
        "srlimit": max_items,
    }

    try:
        response = requests.get(search_url, params=params, timeout=12)
        response.raise_for_status()
        raw_items = response.json().get("query", {}).get("search", [])

        items: List[SourceResult] = []
        for raw in raw_items:
            title = raw.get("title", "")
            if not title:
                continue
            summary_item = _fetch_wikipedia_summary(title)
            if summary_item:
                items.append(summary_item)

        source_urls = _extract_source_urls(items)
        return WikipediaToolResponse(
            query=query,
            items=items,
            source_urls=source_urls,
            top_page_title=items[0].title if items else None,
            success=True,
        )
    except Exception as exc:
        logger.exception("wikipedia_search failed")
        return WikipediaToolResponse(
            query=query,
            items=[],
            source_urls=[],
            success=False,
            error=str(exc),
        )
