"""
web_search_tool.py — LangChain tool wrapping the Tavily Search API
for real-time web retrieval within the agent loop.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.tools import tool
from tavily import TavilyClient

logger = logging.getLogger(__name__)

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError("TAVILY_API_KEY is not set in environment variables.")
        _client = TavilyClient(api_key=api_key)
    return _client


@tool
def web_search(query: str) -> list[dict[str, Any]]:
    """
    Search the web in real-time via Tavily and return structured results.
    Use this tool when the PDF corpus does not contain sufficient information,
    or when the question requires recent/up-to-date data.

    Parameters
    ----------
    query : str
        The search query to send to Tavily.
    max_results : int, optional
        Maximum number of results to return (default: 5).

    Returns
    -------
    list[dict]
        Each dict contains:
        - ``title``: page title
        - ``url``: source URL
        - ``content``: snippet / summary of the page content
        - ``score``: Tavily relevance score
    """
    client = _get_client()

    logger.info("[web_search] Searching: '%s'", query)
    response = client.search(
        query=query,
        max_results=5,
        search_depth="advanced",
        include_answer=False,
    )

    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": round(float(item.get("score", 0.0)), 4),
            }
        )

    logger.info("[web_search] Returned %d results.", len(results))
    return results
