from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import List

from models import SourceResult


class BaseResearchAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(self.name)

    def _fallback_query(self, query: str) -> str:
        cleaned = re.sub(r"\s+", " ", query).strip()
        if not cleaned:
            cleaned = "latest reliable overview"
        self.logger.debug("Fallback query generated: %s", cleaned)
        return cleaned

    def _optimize_quuery(self, query: str) -> str:
        optimized = query.strip()
        if not optimized:
            optimized = self._fallback_query(query)
        if "overview" not in optimized.lower():
            optimized = f"{optimized} overview key facts"
        self.logger.debug("Optimized query generated: %s", optimized)
        return optimized

    def _extract_source_urls(self, items: List[SourceResult]) -> List[str]:
        urls: List[str] = []
        for item in items:
            if item.url and item.url not in urls:
                urls.append(item.url)
        self.logger.debug("Extracted %d source URLs", len(urls))
        return urls

    @abstractmethod
    def invoke(self, query: str):
        raise NotImplementedError
