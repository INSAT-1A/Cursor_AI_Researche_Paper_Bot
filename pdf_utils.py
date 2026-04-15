from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from pypdf import PdfReader
from requests import exceptions as req_exc

logger = logging.getLogger("pdf_utils")


@dataclass(frozen=True)
class PdfFetchResult:
    success: bool
    path: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class PdfTextResult:
    success: bool
    text: str = ""
    pages_used: int = 0
    error: Optional[str] = None


def _cache_dir() -> Path:
    base = os.getenv("PAPERBOT_CACHE_DIR") or ".cache/papers"
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def fetch_pdf(pdf_url: str, timeout_s: int = 20) -> PdfFetchResult:
    """
    Downloads PDF into a local cache and returns the file path.
    """
    if not pdf_url or not isinstance(pdf_url, str):
        return PdfFetchResult(success=False, error="missing pdf_url")

    cache = _cache_dir()
    filename = f"{_hash_url(pdf_url)}.pdf"
    target = cache / filename

    if target.exists() and target.stat().st_size > 0:
        return PdfFetchResult(success=True, path=str(target))

    try:
        logger.info("Downloading PDF: %s", pdf_url)
        with requests.get(pdf_url, stream=True, timeout=timeout_s) as resp:
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").lower()
            if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
                logger.warning("PDF content-type suspicious: %s", content_type)
            with open(target, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
        if target.stat().st_size <= 0:
            return PdfFetchResult(success=False, error="downloaded empty file")
        return PdfFetchResult(success=True, path=str(target))
    except req_exc.Timeout as exc:
        return PdfFetchResult(success=False, error=f"timeout: {exc}")
    except req_exc.RequestException as exc:
        return PdfFetchResult(success=False, error=f"http: {exc}")
    except OSError as exc:
        return PdfFetchResult(success=False, error=f"os: {exc}")


def extract_pdf_text(
    pdf_path: str,
    max_pages: int = 12,
    max_chars: int = 120_000,
) -> PdfTextResult:
    """
    Extract text from the first `max_pages` pages (or fewer) and cap output to `max_chars`.
    """
    if not pdf_path:
        return PdfTextResult(success=False, error="missing pdf_path")
    try:
        reader = PdfReader(pdf_path)
        pages_total = len(reader.pages)
        pages_used = min(max_pages, pages_total)
        text_parts = []
        total = 0
        for i in range(pages_used):
            page = reader.pages[i]
            page_text = page.extract_text() or ""
            if page_text:
                remaining = max_chars - total
                if remaining <= 0:
                    break
                page_text = page_text[:remaining]
                text_parts.append(page_text)
                total += len(page_text)
        text = "\n\n".join(text_parts).strip()
        if not text:
            return PdfTextResult(success=False, error="no extractable text (scanned PDF?)")
        return PdfTextResult(success=True, text=text, pages_used=pages_used)
    except Exception as exc:
        logger.exception("PDF text extraction failed")
        return PdfTextResult(success=False, error=str(exc))

