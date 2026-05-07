"""tests/test_tools.py"""

from unittest.mock import patch, MagicMock


def test_citation_fetcher_returns_structured_results():
    from src.tools.citation_tool import citation_fetcher

    mock_response = {
        "data": [
            {
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "authors": [{"name": "Patrick Lewis"}, {"name": "Ethan Perez"}],
                "year": 2020,
                "abstract": "We explore RAG models which combine parametric and non-parametric memory.",
                "externalIds": {"DOI": "10.48550/arXiv.2005.11401"},
                "url": "https://www.semanticscholar.org/paper/xxx",
                "citationCount": 4500,
            }
        ]
    }

    with patch("src.tools.citation_tool.httpx.Client") as MockClient:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

        results = citation_fetcher.invoke({"query": "retrieval augmented generation", "limit": 1})

    assert len(results) == 1
    assert results[0]["title"] == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert results[0]["doi"] == "10.48550/arXiv.2005.11401"
    assert results[0]["citation_count"] == 4500


def test_web_search_returns_list():
    from src.tools.web_search_tool import web_search

    mock_results = {
        "results": [
            {"title": "What is RAG?", "url": "https://example.com/rag", "content": "RAG overview.", "score": 0.95}
        ]
    }

    with patch("src.tools.web_search_tool._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.search.return_value = mock_results
        mock_get_client.return_value = mock_client

        results = web_search.invoke({"query": "RAG explained", "max_results": 1})

    assert isinstance(results, list)
    assert results[0]["url"] == "https://example.com/rag"
