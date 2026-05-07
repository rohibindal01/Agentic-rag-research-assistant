"""tests/test_agent.py"""

from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, HumanMessage


def _make_state(**overrides):
    base = {
        "messages": [HumanMessage(content="What is RAG?")],
        "query": "What is RAG?",
        "retrieved_docs": [],
        "web_results": [],
        "citations": [],
        "confidence": 0.0,
        "iterations": 0,
        "final_answer": "",
    }
    return {**base, **overrides}


def test_should_continue_returns_finalize_when_high_confidence():
    from src.agents.nodes import should_continue

    state = _make_state(confidence=0.9, iterations=1)
    assert should_continue(state) == "finalize"


def test_should_continue_returns_tool_caller_when_low_confidence():
    from src.agents.nodes import should_continue

    state = _make_state(confidence=0.3, iterations=1)
    assert should_continue(state) == "tool_caller"


def test_should_continue_returns_finalize_when_max_iterations():
    from src.agents.nodes import should_continue

    state = _make_state(confidence=0.1, iterations=99)
    assert should_continue(state) == "finalize"


def test_finalize_formats_sources():
    from src.agents.nodes import finalize

    state = _make_state(
        final_answer="RAG is a technique that combines retrieval with generation.",
        confidence=0.88,
        iterations=2,
        retrieved_docs=[{"source": "paper.pdf", "page": 3, "content": "...", "score": 0.9}],
        citations=[{"title": "RAG Paper", "doi": "10.1000/test"}],
        web_results=[{"url": "https://example.com", "title": "RAG Overview", "content": "..."}],
    )

    result = finalize(state)
    assert "Sources" in result["final_answer"]
    assert "paper.pdf" in result["final_answer"]
    assert "RAG Paper" in result["final_answer"]
    assert "example.com" in result["final_answer"]
