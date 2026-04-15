from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

from models import Paper, PaperCard

logger = logging.getLogger("paper_ranker")


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def _fingerprint_title(title: str) -> str:
    norm = " ".join(_tokenize(title))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _dedupe_key(paper: Paper) -> str:
    # Priority: DOI → arXiv id → S2 paperId → title fingerprint
    doi = (paper.doi or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    arxiv_id = str(paper.extra.get("arxiv_id") or "").strip().lower()
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    pid = (paper.id or "").strip().lower()
    if pid:
        return f"id:{pid}"
    return f"title:{_fingerprint_title(paper.title)}"


def dedupe_papers(papers: Iterable[Paper]) -> List[Paper]:
    """
    Dedupe papers by DOI/arXivId/id/title-fingerprint.
    Keeps the 'best' candidate when collisions happen (prefers having pdf_url and higher citation_count).
    """
    best_by_key: Dict[str, Paper] = {}

    def better(a: Paper, b: Paper) -> Paper:
        a_pdf = 1 if a.pdf_url else 0
        b_pdf = 1 if b.pdf_url else 0
        a_cit = a.citation_count or 0
        b_cit = b.citation_count or 0
        if a_pdf != b_pdf:
            return a if a_pdf > b_pdf else b
        if a_cit != b_cit:
            return a if a_cit > b_cit else b
        # otherwise prefer longer abstract
        return a if len(a.abstract or "") >= len(b.abstract or "") else b

    for p in papers:
        try:
            key = _dedupe_key(p)
        except Exception:
            key = f"title:{_fingerprint_title(p.title)}"
        if key in best_by_key:
            best_by_key[key] = better(best_by_key[key], p)
        else:
            best_by_key[key] = p

    return list(best_by_key.values())


def _tfidf_cosine_scores(query: str, papers: List[Paper]) -> List[float]:
    """
    Lightweight TF-IDF cosine scoring without sklearn.
    Returns a score per paper in [0, 1].
    """
    q_tokens = _tokenize(query)
    if not q_tokens or not papers:
        return [0.0 for _ in papers]

    docs = []
    for p in papers:
        text = f"{p.title} {p.abstract}"
        docs.append(_tokenize(text))

    # document frequency
    df: Dict[str, int] = {}
    for tokens in docs:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    n = len(docs)
    # idf with smoothing
    idf: Dict[str, float] = {}
    for t, dfi in df.items():
        idf[t] = math.log((n + 1) / (dfi + 1)) + 1.0

    def vec(tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        out: Dict[str, float] = {}
        for t, c in tf.items():
            out[t] = (c / max(1, len(tokens))) * idf.get(t, 0.0)
        return out

    q_vec = vec(q_tokens)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    scores: List[float] = []
    for tokens in docs:
        d_vec = vec(tokens)
        dot = 0.0
        for t, qv in q_vec.items():
            dv = d_vec.get(t)
            if dv:
                dot += qv * dv
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        cos = dot / (q_norm * d_norm)
        scores.append(float(max(0.0, min(1.0, cos))))
    return scores


def _citation_bonus(papers: List[Paper]) -> List[float]:
    # small weight; normalize by log scale
    cits = [p.citation_count or 0 for p in papers]
    max_c = max(cits) if cits else 0
    if max_c <= 0:
        return [0.0 for _ in papers]
    return [math.log1p(c) / math.log1p(max_c) for c in cits]


def rank_papers(query: str, papers: List[Paper], top_k: int = 20) -> List[PaperCard]:
    """
    Rank papers by TF-IDF cosine similarity + small citation bonus.
    Returns PaperCard list sorted descending by score.
    """
    if not papers:
        return []

    base_scores = _tfidf_cosine_scores(query, papers)
    cite_scores = _citation_bonus(papers)

    cards: List[PaperCard] = []
    for p, s_base, s_cite in zip(papers, base_scores, cite_scores):
        score = (0.85 * s_base) + (0.15 * s_cite)
        rationale = "text match" if s_base >= 0.2 else "citation boost" if s_cite >= 0.2 else ""
        cards.append(PaperCard(paper=p, score=round(score, 4), rationale=rationale))

    cards.sort(key=lambda c: c.score, reverse=True)
    return cards[: max(1, min(top_k, len(cards)))]

