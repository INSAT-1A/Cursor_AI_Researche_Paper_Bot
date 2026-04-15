from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, List

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from agents.base_agent import BaseResearchAgent
from models import Paper, PaperSummary

logger = logging.getLogger("paper_summarizer_agent")

# Keep prompts bounded; huge PDF extracts break models and JSON parsers.
_MAX_PAPER_TEXT_CHARS = 45_000


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _coerce_llm_text(response: Any) -> str:
    """
    LangChain model responses may return `AIMessage.content` as a str or a list of blocks.
    """
    content = getattr(response, "content", None)
    if isinstance(response, AIMessage):
        content = response.content

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _extract_json_object(text: str) -> str:
    raw = _strip_code_fences((text or "").strip())
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return raw[start : end + 1]


def _parse_plain_sections(text: str) -> PaperSummary | None:
    """
    Fallback parser if JSON mode fails.
    Expects markers: TLDR:, SIMPLIFIED:, KEY_CONTRIBUTIONS:, LIMITATIONS:, FOLLOWUPS:
    """
    t = text or ""
    if "TLDR:" not in t or "SIMPLIFIED:" not in t:
        return None

    def grab(after: str, until: str | None) -> str:
        i = t.find(after)
        if i == -1:
            return ""
        i += len(after)
        if until is None:
            return t[i:].strip()
        j = t.find(until, i)
        if j == -1:
            return t[i:].strip()
        return t[i:j].strip()

    tldr = grab("TLDR:", "SIMPLIFIED:")
    simplified = grab("SIMPLIFIED:", "KEY_CONTRIBUTIONS:")
    kc = grab("KEY_CONTRIBUTIONS:", "LIMITATIONS:")
    lim = grab("LIMITATIONS:", "FOLLOWUPS:")
    fu = grab("FOLLOWUPS:", None)

    def bullets(block: str) -> List[str]:
        out: List[str] = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*]\s+", "", line)
            if line:
                out.append(line)
        return out

    if not tldr or not simplified:
        return None

    return PaperSummary(
        paper_id="",
        tldr=tldr,
        simplified=simplified,
        key_contributions=bullets(kc),
        limitations=bullets(lim),
        suggested_followups=bullets(fu),
        source_used="pdf",
    )


class PaperSummarizerAgent(BaseResearchAgent):
    def __init__(self) -> None:
        super().__init__(name="paper-summarizer-agent")

    def _build_json_prompt(self, paper: Paper, text: str) -> str:
        return f"""
You are a research assistant. Summarize the research paper for a technical audience, but simplify it so a strong undergraduate can understand.

Return ONLY valid JSON (no markdown fences, no commentary) with exactly these keys:
{{
  "paper_id": "<string>",
  "tldr": "<string>",
  "simplified": "<string>",
  "key_contributions": ["<string>", "..."],
  "limitations": ["<string>", "..."],
  "suggested_followups": ["<string>", "..."],
  "source_used": "pdf"
}}

Constraints:
- JSON must be strictly valid (escape newlines inside strings).
- Be faithful to the paper. If uncertain, say so.
- Keep TL;DR under 6 lines.
- Keep simplified explanation under ~250 words.
- 4-8 key contributions.
- 2-6 limitations.
- 2-6 suggested followups.

Paper metadata:
Title: {paper.title}
Authors: {", ".join([a.name for a in paper.authors])}
Year: {paper.year or ""}
Venue: {paper.venue or ""}

Paper text (may be truncated):
{text}
""".strip()

    def _build_plain_prompt(self, paper: Paper, text: str) -> str:
        return f"""
Summarize the paper using EXACTLY these section headers (plain text, not JSON):

TLDR:
SIMPLIFIED:
KEY_CONTRIBUTIONS:
LIMITATIONS:
FOLLOWUPS:

Rules:
- Under KEY_CONTRIBUTIONS / LIMITATIONS / FOLLOWUPS use bullet lines starting with "- ".
- Keep SIMPLIFIED under ~250 words.

Paper metadata:
Title: {paper.title}
Authors: {", ".join([a.name for a in paper.authors])}

Paper text:
{text}
""".strip()

    def _parse_summary_json(self, raw: str) -> PaperSummary:
        blob = _extract_json_object(raw)
        data = json.loads(blob)
        return PaperSummary.model_validate(data)

    def _structured_invoke(self, paper: Paper, text: str) -> PaperSummary | None:
        try:
            structured = self.llm.with_structured_output(PaperSummary)
        except Exception as exc:
            self.logger.debug("Structured output not available: %s", exc)
            return None

        prompt = self._build_json_prompt(paper, text)
        try:
            out = structured.invoke([HumanMessage(content=prompt)])
            if isinstance(out, PaperSummary):
                out.paper_id = paper.id
                out.source_used = "pdf"
                return out
            if isinstance(out, dict):
                s = PaperSummary.model_validate(out)
                s.paper_id = paper.id
                s.source_used = "pdf"
                return s
        except Exception as exc:
            self.logger.warning("Structured summarization failed: %s", exc)
        return None

    def invoke(self, paper: Paper, paper_text: str) -> PaperSummary:
        self.logger.info("Summarizing paper: %s", paper.id)
        if not os.getenv("ANTHROPIC_API_KEY"):
            return PaperSummary(
                paper_id=paper.id,
                tldr="Cannot summarize with Claude because ANTHROPIC_API_KEY is missing in this process.",
                simplified=(
                    "Set ANTHROPIC_API_KEY in the same terminal where Streamlit is launched, "
                    "then restart the app."
                ),
                key_contributions=[],
                limitations=[],
                suggested_followups=[],
                source_used="pdf",
                error="missing ANTHROPIC_API_KEY",
            )
        text = (paper_text or "").strip()
        if len(text) > _MAX_PAPER_TEXT_CHARS:
            text = text[:_MAX_PAPER_TEXT_CHARS] + "\n\n[TRUNCATED_FOR_MODEL_CONTEXT]"

        # 1) Preferred: native structured output (avoids brittle JSON parsing).
        structured = self._structured_invoke(paper, text)
        if structured is not None:
            return structured

        # 2) JSON prompt + parse.
        try:
            prompt = self._build_json_prompt(paper, text)
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            raw = _coerce_llm_text(resp)
            summary = self._parse_summary_json(raw)
            summary.paper_id = paper.id
            summary.source_used = "pdf"
            return summary
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            self.logger.warning("JSON summarization failed, trying plain-text mode: %s", exc)
            parse_err = str(exc)
        except Exception as exc:
            self.logger.warning("LLM summarization failed, trying plain-text mode: %s", exc)
            parse_err = str(exc)

        # 3) Plain-text sections (more robust).
        try:
            prompt = self._build_plain_prompt(paper, text)
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            raw = _coerce_llm_text(resp)
            parsed = _parse_plain_sections(raw)
            if parsed is not None:
                parsed.paper_id = paper.id
                parsed.source_used = "pdf"
                return parsed
        except Exception as exc:
            self.logger.exception("Plain-text summarization failed: %s", exc)
            plain_err = str(exc)
        else:
            plain_err = ""

        return PaperSummary(
            paper_id=paper.id,
            tldr="Summary generation failed after multiple attempts.",
            simplified=(
                "The model output could not be parsed reliably. "
                "Check logs for details, verify `ANTHROPIC_API_KEY`, and try again."
            ),
            key_contributions=[],
            limitations=[],
            suggested_followups=[],
            source_used="pdf",
            error=f"json_phase={locals().get('parse_err', '')}; plain_phase={locals().get('plain_err', '')}".strip("; "),
        )


def summarize_abstract_fallback(paper: Paper) -> PaperSummary:
    """
    Non-LLM fallback that returns a very small summary based on the abstract.
    Used when PDF fetch/parse fails.
    """
    abstract = (paper.abstract or "").strip()
    if not abstract:
        abstract = "No abstract available."
    return PaperSummary(
        paper_id=paper.id,
        tldr=abstract[:500],
        simplified=abstract[:1200],
        key_contributions=[],
        limitations=[],
        suggested_followups=[],
        source_used="abstract",
        error=None,
    )
