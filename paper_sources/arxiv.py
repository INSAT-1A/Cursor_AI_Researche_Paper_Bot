from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlencode

import feedparser
import requests
from requests import exceptions as req_exc

from models import Paper, PaperAuthor

logger = logging.getLogger("paper_sources.arxiv")


def _to_dt(value: str) -> Optional[datetime]:
    # arXiv uses RFC3339-ish like 2024-01-01T00:00:00Z
    if not value:
        return None
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _guess_year(dt: Optional[datetime]) -> Optional[int]:
    if not dt:
        return None
    try:
        return dt.year
    except Exception:
        return None


def _arxiv_id_from_entry_id(entry_id: str) -> Optional[str]:
    # e.g. http://arxiv.org/abs/1706.03762v5
    if not entry_id:
        return None
    if "/abs/" in entry_id:
        return entry_id.split("/abs/")[-1]
    return None


def search_arxiv(query: str, limit: int = 20) -> List[Paper]:
    """
    Search arXiv via its Atom API. No API key required.
    Returns a list of Paper models (typed + validated by Pydantic).
    """
    q = (query or "").strip()
    if not q:
        return []

    base = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{q}",
        "start": 0,
        "max_results": max(1, min(limit, 50)),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{base}?{urlencode(params)}"
    logger.info("arXiv search: %s", q)

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except req_exc.Timeout as exc:
        logger.error("arXiv timeout: %s", exc)
        return []
    except req_exc.RequestException as exc:
        logger.error("arXiv HTTP error: %s", exc)
        return []

    feed = feedparser.parse(resp.text)
    entries = getattr(feed, "entries", None) or []
    papers: List[Paper] = []

    for entry in entries[:limit]:
        try:
            title = (getattr(entry, "title", "") or "").replace("\n", " ").strip()
            abstract = (getattr(entry, "summary", "") or "").replace("\n", " ").strip()
            entry_id = getattr(entry, "id", "") or ""
            published = _to_dt(getattr(entry, "published", "") or "")
            year = _guess_year(published)

            authors_raw = getattr(entry, "authors", None) or []
            authors = [PaperAuthor(name=a.get("name", "").strip()) for a in authors_raw if a.get("name")]

            links = getattr(entry, "links", None) or []
            pdf_url = None
            landing_url = None
            for link in links:
                href = link.get("href")
                rel = link.get("rel")
                link_type = link.get("type")
                if rel == "alternate" and href:
                    landing_url = href
                if (link_type == "application/pdf" or (href and href.endswith(".pdf"))) and href:
                    pdf_url = href

            arxiv_id = _arxiv_id_from_entry_id(entry_id) or (landing_url.split("/")[-1] if landing_url else None)
            paper_id = arxiv_id or entry_id or title

            papers.append(
                Paper(
                    id=str(paper_id),
                    title=title or "Untitled",
                    abstract=abstract,
                    authors=authors,
                    published_at=published,
                    year=year,
                    pdf_url=pdf_url,
                    url=landing_url,
                    source="arxiv",
                    extra={"arxiv_id": arxiv_id, "raw_id": entry_id},
                )
            )
        except Exception as exc:
            logger.warning("Skipping malformed arXiv entry: %s", exc)
            continue

    return papers

