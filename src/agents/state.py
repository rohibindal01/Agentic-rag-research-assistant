"""
state.py — AgentState TypedDict for the LangGraph agent.
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state passed between every node in the LangGraph agent graph.

    Fields
    ------
    messages : list[BaseMessage]
        Full conversation history (HumanMessage, AIMessage, ToolMessage).
        Uses the add_messages reducer so new messages are appended, not replaced.
    query : str
        The original user question (immutable throughout the run).
    retrieved_docs : list[dict]
        Documents returned by the RAG retrieval tool.
    web_results : list[dict]
        Results returned by the web-search tool.
    citations : list[dict]
        Metadata fetched by the citation tool (title, authors, DOI, abstract).
    confidence : float
        Self-assessed confidence score produced by the Reflect node (0.0–1.0).
    iterations : int
        Number of agent loop iterations completed (used to prevent infinite loops).
    final_answer : str
        The grounded, cited answer ready to surface to the user.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    retrieved_docs: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    confidence: float
    iterations: int
    final_answer: str
