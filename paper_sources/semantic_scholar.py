from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests
from requests import exceptions as req_exc

from http_utils import parse_response_json_dict, request_with_retries
from models import Paper, PaperAuthor

logger = logging.getLogger("paper_sources.semantic_scholar")


S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "abstract",
        "authors",
        "year",
        "venue",
        "fieldsOfStudy",
        "citationCount",
        "url",
        "externalIds",
        "openAccessPdf",
    ]
)


def _headers() -> Dict[str, str]:
    api_key = os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers: Dict[str, str] = {
        "User-Agent": "research-paper-bot/1.0 (https://local.app)",
        "Accept": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _parse_dt_year(year: Any) -> Optional[int]:
    try:
        y = int(year)
        if 1900 <= y <= 2100:
            return y
    except Exception:
        return None
    return None


def _paper_from_s2(item: Dict[str, Any]) -> Optional[Paper]:
    paper_id = item.get("paperId") or item.get("externalIds", {}).get("ArXiv")
    title = (item.get("title") or "").strip()
    if not paper_id and not title:
        return None

    authors_raw = item.get("authors") or []
    authors: List[PaperAuthor] = []
    if isinstance(authors_raw, list):
        for a in authors_raw:
            if isinstance(a, dict) and a.get("name"):
                authors.append(PaperAuthor(name=str(a["name"]).strip()))

    year = _parse_dt_year(item.get("year"))
    open_access = item.get("openAccessPdf") or {}
    pdf_url = None
    if isinstance(open_access, dict):
        pdf_url = open_access.get("url")

    external_ids = item.get("externalIds") or {}
    doi = external_ids.get("DOI") if isinstance(external_ids, dict) else None

    return Paper(
        id=str(paper_id or title),
        title=title or "Untitled",
        abstract=(item.get("abstract") or "").strip(),
        authors=authors,
        published_at=None,
        year=year,
        venue=(item.get("venue") or None),
        fields_of_study=list(item.get("fieldsOfStudy") or []),
        pdf_url=pdf_url,
        url=item.get("url") or None,
        doi=doi,
        citation_count=item.get("citationCount") if item.get("citationCount") is not None else None,
        source="semantic_scholar",
        extra={"externalIds": external_ids},
    )


@lru_cache(maxsize=128)
def _search_semantic_scholar_cached(query: str, limit: int) -> tuple[Paper, ...]:
    q = (query or "").strip()
    if not q:
        return tuple()

    url = f"{S2_BASE}/paper/search"
    params = {"query": q, "limit": max(1, min(limit, 50)), "fields": S2_FIELDS}
    logger.info("Semantic Scholar search: %s", q)

    try:
        resp = request_with_retries(
            logger=logger,
            method="GET",
            url=url,
            headers=_headers(),
            params=params,
            timeout=15,
            retries=2,
            retry_statuses=(429, 500, 502, 503, 504),
            context="semantic_scholar search",
        )
        root = parse_response_json_dict(resp, logger, "semantic_scholar search")
        if root is None:
            return tuple()
        data = root.get("data") or []
        if not isinstance(data, list):
            return tuple()
    except req_exc.Timeout as exc:
        logger.error("Semantic Scholar timeout: %s", exc)
        return tuple()
    except req_exc.RequestException as exc:
        logger.error("Semantic Scholar HTTP error: %s", exc)
        return tuple()
    except Exception as exc:
        logger.exception("Semantic Scholar search failed")
        return tuple()

    papers: List[Paper] = []
    for item in data[:limit]:
        if not isinstance(item, dict):
            continue
        paper = _paper_from_s2(item)
        if paper:
            papers.append(paper)
    return tuple(papers)


def search_semantic_scholar(query: str, limit: int = 20) -> List[Paper]:
    return list(_search_semantic_scholar_cached(query, limit))


def enrich_paper(paper: Paper) -> Paper:
    """
    Optionally enrich a Paper using S2 paper endpoint, if we have a S2 paper id.
    Returns the original paper on failure.
    """
    paper_id = paper.id
    if not paper_id:
        return paper

    url = f"{S2_BASE}/paper/{paper_id}"
    params = {"fields": S2_FIELDS}
    try:
        resp = request_with_retries(
            logger=logger,
            method="GET",
            url=url,
            headers=_headers(),
            params=params,
            timeout=15,
            retries=1,
            retry_statuses=(429, 500, 502, 503, 504),
            context="semantic_scholar enrich",
        )
        if resp.status_code != 200:
            return paper
        root = parse_response_json_dict(resp, logger, "semantic_scholar enrich")
        if root is None:
            return paper
        if not isinstance(root, dict):
            return paper
        enriched = _paper_from_s2(root)
        return enriched or paper
    except Exception:
        return paper

