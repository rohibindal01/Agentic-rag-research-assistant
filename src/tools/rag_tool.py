"""
rag_tool.py — LangChain tool that performs semantic similarity search
over the persisted FAISS (or Pinecone) vector store.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from configs.embeddings_factory import get_embeddings
from configs.loader import load_config

logger = logging.getLogger(__name__)
cfg = load_config()

# ---------------------------------------------------------------------------
# Vector store loader (lazy singleton)
# ---------------------------------------------------------------------------
_vectorstore: FAISS | None = None


def _get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        index_path = cfg["vectorstore"]["faiss_index_path"]
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'. "
                "Run `python -m src.retrieval.ingest` first."
            )
        embeddings = get_embeddings()
        _vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded from '%s'.", index_path)
    return _vectorstore


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
@tool
def rag_retriever(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most semantically relevant document chunks from the
    ingested PDF corpus using FAISS vector similarity search.

    Parameters
    ----------
    query : str
        The search query derived from the user's research question.
    k : int, optional
        Number of chunks to retrieve (default: 5).

    Returns
    -------
    list[dict]
        Each dict contains:
        - ``content``: the raw text of the chunk
        - ``source``: originating PDF filename
        - ``page``: page number within the PDF
        - ``score``: cosine similarity score (higher = more relevant)
    """
    vs = _get_vectorstore()
    docs_and_scores = vs.similarity_search_with_score(query, k=k)

    results = []
    for doc, score in docs_and_scores:
        results.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "score": round(float(score), 4),
            }
        )

    logger.info("[rag_retriever] Retrieved %d chunks for query: '%s'", len(results), query)
    return results
