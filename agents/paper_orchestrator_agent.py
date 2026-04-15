from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from agents.orchestrator_agent import OrchestratorAgent as FallbackOrchestrator
from agents.paper_summarizer_agent import PaperSummarizerAgent, summarize_abstract_fallback
from models import OrchestrationResult, Paper, PaperSearchResponse, PaperSummary
from paper_ranker import dedupe_papers, rank_papers
from paper_sources.arxiv import search_arxiv
from paper_sources.semantic_scholar import search_semantic_scholar
from pdf_utils import extract_pdf_text, fetch_pdf

logger = logging.getLogger("paper_orchestrator_agent")

TOP_SCORE_THRESHOLD = 0.34
AVG_TOP3_THRESHOLD = 0.27
TITLE_MATCH_THRESHOLD = 0.90
TITLE_LIKE_INTENT_MIN_SCORE = 0.28
TITLE_LIKE_SIMILARITY_MIN = 0.75


class PaperOrchestratorAgent:
    """
    Paper-first orchestration.
    - Search arXiv + Semantic Scholar
    - Dedupe + rank into cards
    - If empty: fallback to web/wiki + critic via existing OrchestratorAgent
    """

    def __init__(self) -> None:
        self.paper_summarizer = PaperSummarizerAgent()
        self.fallback_orchestrator = FallbackOrchestrator()

    @staticmethod
    def _confidence_metrics(scores: List[float]) -> Tuple[float, float]:
        if not scores:
            return 0.0, 0.0
        top = float(scores[0])
        top3 = scores[:3]
        avg3 = float(sum(top3) / len(top3))
        return top, avg3

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()

    def _title_similarity_max(self, query: str, ranked_cards) -> float:
        q = self._normalize_text(query)
        if not q:
            return 0.0
        best = 0.0
        for card in ranked_cards:
            title = self._normalize_text(card.paper.title)
            if not title:
                continue
            if q == title:
                return 1.0
            if q in title or title in q:
                best = max(best, 0.95)
            best = max(best, SequenceMatcher(None, q, title).ratio())
        return best

    def _paper_intent_score(self, query: str) -> int:
        q = self._normalize_text(query)
        if not q:
            return 0
        keywords = {
            "paper",
            "arxiv",
            "model",
            "models",
            "transformer",
            "llm",
            "attention",
            "benchmark",
            "dataset",
            "architecture",
            "algorithm",
            "survey",
            "research",
            "method",
        }
        score = 0
        for k in keywords:
            if k in q:
                score += 1
        # title-like queries with many tokens usually indicate paper lookup intent
        if len(q.split()) >= 5:
            score += 1
        return score

    def _should_fallback(
        self,
        query: str,
        ranked_cards,
        top_score: float,
        avg_top3: float,
        candidates_count: int,
    ) -> Optional[str]:
        if len(ranked_cards) == 0:
            return "no_ranked_cards"

        title_sim = self._title_similarity_max(query, ranked_cards)
        intent_score = self._paper_intent_score(query)
        ranking_good = top_score >= TOP_SCORE_THRESHOLD and avg_top3 >= AVG_TOP3_THRESHOLD
        pdf_count = sum(1 for c in ranked_cards[:5] if c.paper.pdf_url)
        source_quality_good = candidates_count >= 3 and pdf_count >= 1

        # Hybrid gate:
        # 1) strong title match always stays in paper mode
        if title_sim >= TITLE_MATCH_THRESHOLD:
            return None
        # Near-title queries (common for known papers) should not be forced into fallback
        # when signals are close but below strict thresholds.
        if intent_score >= 2 and title_sim >= TITLE_LIKE_SIMILARITY_MIN and top_score >= TITLE_LIKE_INTENT_MIN_SCORE:
            return None
        # 2) high intent + good ranking
        if intent_score >= 2 and ranking_good:
            return None
        # 3) good ranking + source quality
        if ranking_good and source_quality_good:
            return None

        if top_score < TOP_SCORE_THRESHOLD:
            return f"top_score_below_threshold({top_score:.3f}<{TOP_SCORE_THRESHOLD:.2f})"
        if avg_top3 < AVG_TOP3_THRESHOLD:
            return f"avg_top3_below_threshold({avg_top3:.3f}<{AVG_TOP3_THRESHOLD:.2f})"
        return "low_paper_intent_or_source_quality"

    def search(self, topic: str, limit: int = 20) -> Tuple[PaperSearchResponse, Optional[OrchestrationResult]]:
        q = (topic or "").strip()
        if not q:
            return (
                PaperSearchResponse(
                    query="",
                    total_candidates=0,
                    ranked_cards=[],
                    success=False,
                    error="empty topic",
                ),
                None,
            )

        try:
            arxiv = search_arxiv(q, limit=limit)
            s2 = search_semantic_scholar(q, limit=limit)
            candidates = dedupe_papers([*arxiv, *s2])
            ranked_cards = rank_papers(q, candidates, top_k=limit)
            scores = [c.score for c in ranked_cards]
            top_score, avg_top3 = self._confidence_metrics(scores)
            reason = self._should_fallback(
                query=q,
                ranked_cards=ranked_cards,
                top_score=top_score,
                avg_top3=avg_top3,
                candidates_count=len(candidates),
            )
            fallback_triggered = reason is not None
            # Strict policy: do not surface paper cards in low-confidence scenarios.
            visible_cards = ranked_cards if not fallback_triggered else []
            resp = PaperSearchResponse(
                query=q,
                total_candidates=len(candidates),
                ranked_cards=visible_cards,
                top_score=top_score,
                avg_top3_score=avg_top3,
                fallback_triggered=fallback_triggered,
                fallback_reason=reason,
                success=True,
            )
            if fallback_triggered:
                logger.info(
                    "Low-confidence paper result; invoking fallback orchestrator. reason=%s top=%.3f avg3=%.3f",
                    reason,
                    top_score,
                    avg_top3,
                )
                fallback = self.fallback_orchestrator.invoke(q)
                return resp, fallback
            return resp, None
        except Exception as exc:
            logger.exception("Paper search orchestration failed")
            fallback = self.fallback_orchestrator.invoke(q)
            return (
                PaperSearchResponse(
                    query=q,
                    total_candidates=0,
                    ranked_cards=[],
                    top_score=0.0,
                    avg_top3_score=0.0,
                    fallback_triggered=True,
                    fallback_reason="search_exception",
                    success=False,
                    error=str(exc),
                ),
                fallback,
            )

    def summarize(self, paper: Paper) -> PaperSummary:
        """
        Fetch + extract PDF text and summarize using LLM, with abstract fallback.
        """
        pdf_url = (paper.pdf_url or "").strip()
        if not pdf_url:
            return summarize_abstract_fallback(paper)

        fetched = fetch_pdf(pdf_url)
        if not fetched.success or not fetched.path:
            return summarize_abstract_fallback(paper)

        extracted = extract_pdf_text(fetched.path)
        if not extracted.success or not extracted.text:
            return summarize_abstract_fallback(paper)

        return self.paper_summarizer.invoke(paper, extracted.text)

