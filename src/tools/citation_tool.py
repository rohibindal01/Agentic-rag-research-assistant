"""
citation_tool.py — LangChain tool that fetches academic paper metadata
from the Semantic Scholar API using a paper title or DOI.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,authors,year,abstract,externalIds,url,citationCount"


@tool
def citation_fetcher(query: str) -> list[dict[str, Any]]:
    """
    Fetch academic paper metadata from Semantic Scholar by title keyword or DOI.
    Use this tool to ground answers in peer-reviewed literature and retrieve
    structured citation information.

    Parameters
    ----------
    query : str
        Paper title keywords or a DOI string (e.g. "10.1109/CVPR.2023.xxxxx").
    limit : int, optional
        Maximum number of papers to return (default: 3).

    Returns
    -------
    list[dict]
        Each dict contains:
        - ``title``: paper title
        - ``authors``: list of author names
        - ``year``: publication year
        - ``abstract``: paper abstract (truncated to 500 chars)
        - ``doi``: DOI string if available
        - ``url``: Semantic Scholar paper URL
        - ``citation_count``: number of times the paper has been cited
    """
    logger.info("[citation_fetcher] Fetching citations for: '%s'", query)

    params = {
        "query": query,
        "limit": 3,
        "fields": FIELDS,
    }

    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{SEMANTIC_SCHOLAR_BASE}/paper/search", params=params)
        resp.raise_for_status()
        data = resp.json()

    papers = data.get("data", [])
    results = []

    for paper in papers:
        abstract = paper.get("abstract") or ""
        results.append(
            {
                "title": paper.get("title", ""),
                "authors": [a["name"] for a in paper.get("authors", [])],
                "year": paper.get("year"),
                "abstract": abstract[:500] + ("..." if len(abstract) > 500 else ""),
                "doi": (paper.get("externalIds") or {}).get("DOI", ""),
                "url": paper.get("url", ""),
                "citation_count": paper.get("citationCount", 0),
            }
        )

    logger.info("[citation_fetcher] Found %d papers.", len(results))
    return results
