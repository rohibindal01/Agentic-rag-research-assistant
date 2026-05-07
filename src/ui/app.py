"""
app.py — Streamlit frontend for the Agentic RAG Research Assistant.

Run with:
    streamlit run src/ui/app.py
"""


from __future__ import annotations

import json

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.agents.graph import run_agent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic RAG Research Assistant",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    st.markdown("**Model:** Llama 3.3 70B via Groq")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2 (local)")
    st.markdown("**Vector Store:** FAISS")
    st.markdown("**Agent:** LangGraph ReAct + Reflexion")
    st.markdown("---")
    st.markdown("**Tools active:**")
    st.checkbox("📄 RAG Retriever", value=True, disabled=True)
    st.checkbox("🌐 Web Search (Tavily)", value=True, disabled=True)
    st.checkbox("📚 Citation Fetcher", value=True, disabled=True)
    st.markdown("---")
    st.markdown("[GitHub](https://github.com/rohibindal01/agentic-rag-research-assistant)")

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🤖 Agentic RAG Research Assistant")
st.markdown(
    "Ask any research question. The agent will retrieve relevant PDF chunks, "
    "search the web, and fetch academic citations — then synthesise a grounded answer."
)

query = st.text_area(
    "Your research question",
    placeholder="e.g. What are the latest advances in retrieval-augmented generation for scientific literature?",
    height=100,
)

run_btn = st.button("🔍 Run Agent", type="primary", disabled=not query.strip())

if run_btn and query.strip():
    with st.spinner("Agent is thinking… (this may take 15–30 s)"):
        try:
            state = run_agent(query)
        except Exception as exc:
            st.error(f"Agent error: {exc}")
            st.stop()

    # ── Answer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Answer")
    st.markdown(state.get("final_answer", "_No answer generated._"))

    # ── Metrics strip ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{state.get('confidence', 0.0):.0%}")
    col2.metric("Agent Iterations", state.get("iterations", 0))
    col3.metric("Sources Used",
                len(state.get("retrieved_docs", [])) +
                len(state.get("citations", [])) +
                len(state.get("web_results", [])))

    st.markdown("---")

    # ── Source cards ──────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📄 PDF Chunks", "📚 Citations", "🌐 Web Results"])

    with tab1:
        docs = state.get("retrieved_docs", [])
        if docs:
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Chunk {i} — {doc.get('source', 'unknown')} (p.{doc.get('page', '?')}) | score: {doc.get('score', 0):.4f}"):
                    st.write(doc.get("content", ""))
        else:
            st.info("No PDF chunks retrieved.")

    with tab2:
        cites = state.get("citations", [])
        if cites:
            for cite in cites:
                with st.expander(f"📄 {cite.get('title', 'Unknown')} ({cite.get('year', '?')})"):
                    st.markdown(f"**Authors:** {', '.join(cite.get('authors', []))}")
                    st.markdown(f"**DOI:** `{cite.get('doi', 'N/A')}`")
                    st.markdown(f"**Citations:** {cite.get('citation_count', 0)}")
                    st.markdown(f"**Abstract:** {cite.get('abstract', 'N/A')}")
                    if cite.get("url"):
                        st.markdown(f"[Open on Semantic Scholar]({cite['url']})")
        else:
            st.info("No academic citations retrieved.")

    with tab3:
        webs = state.get("web_results", [])
        if webs:
            for web in webs:
                with st.expander(f"🌐 {web.get('title', web.get('url', 'Result'))}"):
                    st.markdown(f"**URL:** {web.get('url', '')}")
                    st.write(web.get("content", ""))
        else:
            st.info("No web results retrieved.")

    # ── Raw state (debug) ──────────────────────────────────────────────────
    with st.expander("🔬 Raw agent state (debug)"):
        debug_state = {k: v for k, v in state.items() if k != "messages"}
        st.json(json.dumps(debug_state, default=str, indent=2))
