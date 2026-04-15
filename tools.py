from __future__ import annotations

import logging
from typing import Any, List

import requests
from requests import exceptions as req_exc

from http_utils import parse_response_json_dict
from models import DuckDuckGoToolResponse, SourceResult, WikipediaToolResponse

logger = logging.getLogger("research-tools")


def _extract_source_urls(items: List[SourceResult]) -> List[str]:
    urls: List[str] = []
    for item in items:
        try:
            if item.url and item.url not in urls:
                urls.append(item.url)
        except (TypeError, AttributeError):
            continue
    return urls


def duckduckgo_search(query: str, max_items: int = 8) -> DuckDuckGoToolResponse:
    """
    Query DuckDuckGo Instant Answer API and return validated Pydantic data.
    """
    if not isinstance(query, str):
        query = str(query)
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
        payload = parse_response_json_dict(response, logger, "duckduckgo tool")
        if payload is None:
            raise ValueError("DuckDuckGo returned invalid JSON root.")

        items: List[SourceResult] = []

        abstract_url = payload.get("AbstractURL", "") or ""
        abstract_text = payload.get("AbstractText", "") or ""
        if abstract_url:
            try:
                items.append(
                    SourceResult(
                        title=payload.get("Heading") or "DuckDuckGo Abstract",
                        url=abstract_url,
                        snippet=abstract_text or "No abstract text available.",
                        source="duckduckgo",
                    )
                )
            except Exception as exc:
                logger.warning("Skipping DuckDuckGo abstract item: %s", exc)

        for topic in payload.get("RelatedTopics", []) or []:
            if not isinstance(topic, dict):
                continue
            nested = topic.get("Topics")
            candidates = nested if isinstance(nested, list) else [topic]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                first_url = candidate.get("FirstURL", "") or ""
                text = candidate.get("Text", "") or ""
                if not first_url:
                    continue
                try:
                    items.append(
                        SourceResult(
                            title=text.split(" - ")[0] if text else "DuckDuckGo Result",
                            url=first_url,
                            snippet=text or "No snippet available.",
                            source="duckduckgo",
                        )
                    )
                except Exception as exc:
                    logger.warning("Skipping DuckDuckGo related topic: %s", exc)
                    continue
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
    except req_exc.Timeout as exc:
        logger.error("duckduckgo_search timeout: %s", exc)
        return DuckDuckGoToolResponse(
            query=query,
            items=[],
            source_urls=[],
            success=False,
            error=f"timeout: {exc}",
        )
    except req_exc.RequestException as exc:
        logger.error("duckduckgo_search HTTP error: %s", exc)
        return DuckDuckGoToolResponse(
            query=query,
            items=[],
            source_urls=[],
            success=False,
            error=f"http: {exc}",
        )
    except (ValueError, TypeError, KeyError) as exc:
        logger.exception("duckduckgo_search data error")
        return DuckDuckGoToolResponse(
            query=query,
            items=[],
            source_urls=[],
            success=False,
            error=f"data: {exc}",
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
    from urllib.parse import quote

    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}"
    try:
        response = requests.get(summary_url, timeout=12)
        if response.status_code != 200:
            return None
        payload = parse_response_json_dict(response, logger, f"wiki summary {title}")
        if payload is None:
            return None
        page_url = payload.get("content_urls", {}).get("desktop", {}).get("page", "")
        if not page_url:
            return None
        return SourceResult(
            title=payload.get("title") or title,
            url=page_url,
            snippet=payload.get("extract") or "No summary available.",
            source="wikipedia",
        )
    except req_exc.RequestException:
        return None
    except (TypeError, ValueError, KeyError):
        return None


def wikipedia_search(query: str, max_items: int = 5) -> WikipediaToolResponse:
    """
    Query Wikipedia and return validated Pydantic data.
    """
    if not isinstance(query, str):
        query = str(query)
    logger.info("Running wikipedia_search for query: %s", query)
    search_url = "https://en.wikipedia.org/w/api.php"
    params: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query.strip(),
        "srlimit": max_items,
    }

    try:
        response = requests.get(search_url, params=params, timeout=12)
        response.raise_for_status()
        root = parse_response_json_dict(response, logger, "wikipedia tool search")
        if root is None:
            raise ValueError("Wikipedia search returned invalid JSON root.")
        query_block = root.get("query") or {}
        if not isinstance(query_block, dict):
            raw_items: list[Any] = []
        else:
            raw = query_block.get("search") or []
            raw_items = raw if isinstance(raw, list) else []

        items: List[SourceResult] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            title = raw_item.get("title", "") or ""
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
    except req_exc.Timeout as exc:
        logger.error("wikipedia_search timeout: %s", exc)
        return WikipediaToolResponse(
            query=query,
            items=[],
            source_urls=[],
            success=False,
            error=f"timeout: {exc}",
        )
    except req_exc.RequestException as exc:
        logger.error("wikipedia_search HTTP error: %s", exc)
        return WikipediaToolResponse(
            query=query,
            items=[],
            source_urls=[],
            success=False,
            error=f"http: {exc}",
        )
    except (ValueError, TypeError, KeyError) as exc:
        logger.exception("wikipedia_search data error")
        return WikipediaToolResponse(
            query=query,
            items=[],
            source_urls=[],
            success=False,
            error=f"data: {exc}",
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
