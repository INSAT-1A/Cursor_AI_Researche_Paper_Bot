import hashlib
import logging
import os

import streamlit as st

from agents.paper_orchestrator_agent import PaperOrchestratorAgent
from logging_config import setup_logging
from models import Paper, PaperSearchResponse, PaperSummary


setup_logging()
logger = logging.getLogger("paper_app")


st.set_page_config(
    page_title="Research Paper Bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_paper_orchestrator() -> PaperOrchestratorAgent:
    """One orchestrator per process; avoids slow re-init on every keystroke/rerun."""
    return PaperOrchestratorAgent()


def ensure_state() -> None:
    if "search_response" not in st.session_state:
        st.session_state.search_response = None
    if "fallback_result" not in st.session_state:
        st.session_state.fallback_result = None
    if "selected_paper" not in st.session_state:
        st.session_state.selected_paper = None
    if "paper_summary" not in st.session_state:
        st.session_state.paper_summary = None
    if "paper_topic" not in st.session_state:
        st.session_state.paper_topic = ""
    if "summary_fullscreen" not in st.session_state:
        st.session_state.summary_fullscreen = False


def _widget_key_for_paper(paper_id: str, suffix: str) -> str:
    safe = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:20]
    return f"{suffix}_{safe}"


def render_paper_card(paper: Paper, score: float, rationale: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{paper.title}**")
        author_line = ", ".join([a.name for a in paper.authors[:5]]) or "Unknown authors"
        meta = []
        if paper.year:
            meta.append(str(paper.year))
        if paper.venue:
            meta.append(paper.venue)
        if paper.citation_count is not None:
            meta.append(f"{paper.citation_count} citations")
        st.caption(f"{author_line} · {' · '.join(meta)}")
        # Avoid st.progress(text=...) for compatibility with older Streamlit versions.
        st.progress(min(1.0, max(0.0, score)))
        st.caption(f"Relevance: {score:.3f}{(' · ' + rationale) if rationale else ''}")
        if paper.abstract:
            st.write(paper.abstract[:350] + ("…" if len(paper.abstract) > 350 else ""))
        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.button("Summarize", key=_widget_key_for_paper(paper.id, "sum")):
                st.session_state.selected_paper = paper
                st.session_state.paper_summary = None
        with cols[1]:
            if paper.pdf_url:
                st.link_button("PDF", paper.pdf_url)
        with cols[2]:
            if paper.url:
                st.link_button("Page", paper.url)


def render_summary(summary: PaperSummary, paper: Paper) -> None:
    st.markdown(f"## {paper.title}")
    st.caption(f"Summary source: **{summary.source_used}**")
    if summary.error:
        st.warning(f"Summary warning: {summary.error}")
    st.markdown("### TL;DR")
    st.write(summary.tldr)
    st.markdown("### Simplified explanation")
    st.write(summary.simplified)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### Key contributions")
        if summary.key_contributions:
            for item in summary.key_contributions:
                st.write(f"- {item}")
        else:
            st.write("No contributions extracted.")
    with col2:
        st.markdown("### Limitations")
        if summary.limitations:
            for item in summary.limitations:
                st.write(f"- {item}")
        else:
            st.write("No limitations extracted.")

    st.markdown("### Suggested follow-ups")
    if summary.suggested_followups:
        for item in summary.suggested_followups:
            st.write(f"- {item}")
    else:
        st.write("No follow-ups suggested.")


ensure_state()
orchestrator = get_paper_orchestrator()

with st.sidebar:
    st.title("Research Paper Bot")
    max_results = st.slider("Max results", 5, 30, 15, 1)
    st.caption(
        "Fallback is auto-triggered when paper relevance confidence is low "
        "(top score < 0.34 or avg top-3 < 0.27)."
    )
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY is missing for this Streamlit process. Paper summarization will fail.")

if st.session_state.summary_fullscreen:
    left, right = st.columns([0.001, 1.999], gap="small")
else:
    left, right = st.columns([1.25, 1], gap="large")

if not st.session_state.summary_fullscreen:
    with left:
        st.subheader("Search")
        st.text_input(
            "Research topic",
            key="paper_topic",
            placeholder="e.g., diffusion transformers for image generation",
            help="Type your topic here, then click Search.",
        )
        run = st.button("Search", type="primary")

        if run:
            with st.spinner("Searching arXiv + Semantic Scholar..."):
                topic_value = (st.session_state.get("paper_topic") or "").strip()
                resp, fallback = orchestrator.search(topic_value, limit=max_results)
                st.session_state.search_response = resp
                st.session_state.fallback_result = fallback
                st.session_state.selected_paper = None
                st.session_state.paper_summary = None

        resp: PaperSearchResponse | None = st.session_state.search_response
        if resp and resp.fallback_triggered:
            st.header("Fallback web research")
        else:
            st.header("Technical paper matches")
        if not resp:
            st.info("Enter a topic above and click Search.")
        else:
            if not resp.success:
                st.error(resp.error or "Search failed.")
            metrics = st.columns(3)
            with metrics[0]:
                st.metric("Top score", f"{resp.top_score:.3f}")
            with metrics[1]:
                st.metric("Avg top-3", f"{resp.avg_top3_score:.3f}")
            with metrics[2]:
                st.metric("Fallback", "Yes" if resp.fallback_triggered else "No")
            st.caption(f"Candidates: {resp.total_candidates} · Showing: {len(resp.ranked_cards)}")
            if resp.fallback_triggered:
                st.warning(
                    "Paper confidence is below threshold, so technical papers are hidden "
                    f"and fallback sources are shown. Reason: `{resp.fallback_reason or 'unknown'}`"
                )
            if resp.ranked_cards:
                for card in resp.ranked_cards:
                    render_paper_card(card.paper, card.score, card.rationale)
            else:
                if resp.fallback_triggered:
                    st.info("Technical paper cards are hidden for this query due to low confidence.")
                else:
                    st.warning("No ranked papers found for this topic.")

        if st.session_state.fallback_result is not None:
            st.divider()
            st.subheader("General web fallback analysis (DuckDuckGo + Wikipedia + Critic)")
            fb = st.session_state.fallback_result
            if fb.fallback_synthesis is not None:
                syn = fb.fallback_synthesis
                st.markdown("### Fallback answer")
                st.write(syn.short_answer)
                st.markdown("### Detailed answer")
                st.write(syn.detailed_answer)
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("### Key points")
                    if syn.key_points:
                        for p in syn.key_points:
                            st.write(f"- {p}")
                    else:
                        st.write("No key points available.")
                with cols[1]:
                    st.markdown("### Caveats")
                    if syn.caveats:
                        for c in syn.caveats:
                            st.write(f"- {c}")
                    else:
                        st.write("No caveats available.")
                if syn.sources_used:
                    with st.expander("Synthesis sources"):
                        for s in syn.sources_used:
                            st.write(f"- {s}")
            else:
                st.write(f"**Combined summary:** {fb.combined_summary}")
            st.markdown("**Critic quality verdict:**")
            st.write(fb.critic_report.verdict)
            st.caption(f"Confidence: {fb.critic_report.confidence_score:.2f}")
            if fb.consolidated_urls:
                with st.expander("Fallback source URLs"):
                    for u in fb.consolidated_urls:
                        st.write(f"- {u}")


with right:
    if st.session_state.summary_fullscreen:
        top = st.columns([1, 0.28])
        with top[0]:
            st.header("Summary reader (full screen)")
        with top[1]:
            if st.button("Exit full screen", use_container_width=True):
                st.session_state.summary_fullscreen = False
                st.rerun()
    else:
        st.header("Summary panel")

    paper: Paper | None = st.session_state.selected_paper
    if not paper:
        if st.session_state.fallback_result is not None:
            st.info("Fallback mode is active. Select a paper card to open technical paper summary, or review fallback analysis on the left.")
        else:
            st.info("Click Summarize on a paper card.")
    else:
        if st.session_state.paper_summary is None:
            with st.spinner("Fetching PDF and summarizing..."):
                st.session_state.paper_summary = orchestrator.summarize(paper)

        summary: PaperSummary = st.session_state.paper_summary
        if not st.session_state.summary_fullscreen:
            row = st.columns([1, 0.42])
            with row[1]:
                if st.button("Open full screen", use_container_width=True):
                    st.session_state.summary_fullscreen = True
                    st.rerun()
        render_summary(summary, paper)

