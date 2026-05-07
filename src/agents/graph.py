"""
graph.py — Builds and compiles the LangGraph agent graph.

Graph topology
--------------
START → router → tool_caller → reflect → (should_continue) → finalize → END
                      ▲                         │
                      └─────────────────────────┘
                         (loop if low confidence)
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.nodes import (
    router,
    tool_caller,
    reflect,
    finalize,
    should_continue,
)


def build_graph() -> StateGraph:
    """
    Construct and compile the agentic RAG graph.

    Returns
    -------
    StateGraph
        A compiled LangGraph ready for invocation.
    """
    builder = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    builder.add_node("router", router)
    builder.add_node("tool_caller", tool_caller)
    builder.add_node("reflect", reflect)
    builder.add_node("finalize", finalize)

    # ── Edges ──────────────────────────────────────────────────────────────
    builder.set_entry_point("router")
    builder.add_edge("router", "tool_caller")
    builder.add_edge("tool_caller", "reflect")

    # Conditional: loop back to tool_caller or proceed to finalize
    builder.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "tool_caller": "tool_caller",
            "finalize": "finalize",
        },
    )

    builder.add_edge("finalize", END)

    return builder.compile()


# Singleton graph instance — import this in app.py and evaluation scripts
graph = build_graph()


def run_agent(query: str) -> dict:
    """
    Convenience wrapper to invoke the agent graph with a plain string query.

    Parameters
    ----------
    query : str
        The user's research question.

    Returns
    -------
    dict
        Final AgentState after the graph completes.
    """
    from langchain_core.messages import HumanMessage

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "retrieved_docs": [],
        "web_results": [],
        "citations": [],
        "confidence": 0.0,
        "iterations": 0,
        "final_answer": "",
    }

    return graph.invoke(initial_state)
