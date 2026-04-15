from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_anthropic import ChatAnthropic

from agents.llm_factory import create_chat_anthropic
from models import SourceResult


class BaseResearchAgent(ABC):
    def __init__(
        self,
        name: str,
        llm: Optional[ChatAnthropic] = None,
    ) -> None:
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.llm: ChatAnthropic = llm if llm is not None else create_chat_anthropic()

    def _fallback_query(self, query: str) -> str:
        if query is None:
            query = ""
        if not isinstance(query, str):
            try:
                query = str(query)
            except Exception:
                query = ""
                self.logger.warning("Query could not be coerced to string; using empty query.")
        cleaned = re.sub(r"\s+", " ", query).strip()
        if not cleaned:
            cleaned = "latest reliable overview"
        self.logger.debug("Fallback query generated: %s", cleaned)
        return cleaned

    def _optimize_quuery(self, query: str) -> str:
        if query is None:
            query = ""
        if not isinstance(query, str):
            query = str(query)
        optimized = query.strip()
        if not optimized:
            optimized = self._fallback_query(query)
        if "overview" not in optimized.lower():
            optimized = f"{optimized} overview key facts"
        self.logger.debug("Optimized query generated: %s", optimized)
        return optimized

    def _extract_source_urls(self, items: List[SourceResult]) -> List[str]:
        urls: List[str] = []
        if not items:
            return urls
        for item in items:
            try:
                if item.url and item.url not in urls:
                    urls.append(item.url)
            except (TypeError, AttributeError) as exc:
                self.logger.warning("Skipping invalid source item: %s", exc)
                continue
        self.logger.debug("Extracted %d source URLs", len(urls))
        return urls

    @abstractmethod
    def invoke(self, query: str):
        raise NotImplementedError
