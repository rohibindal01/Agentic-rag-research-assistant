"""
nodes.py — Node functions for the LangGraph ReAct + Reflexion agent.

Graph flow
----------
router → tool_caller → reflect → (loop back to tool_caller OR finalize)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from configs.llm_factory import get_llm

from src.agents.state import AgentState
from src.tools.rag_tool import rag_retriever
from src.tools.web_search_tool import web_search
from src.tools.citation_tool import citation_fetcher
from configs.loader import load_prompts, load_config

logger = logging.getLogger(__name__)
cfg = load_config()
prompts = load_prompts()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = get_llm()
llm_with_tools = get_llm(with_tools=[rag_retriever, web_search, citation_fetcher])


# ---------------------------------------------------------------------------
# Node: router
# ---------------------------------------------------------------------------
def router(state: AgentState) -> AgentState:
    """
    Entry node. Decides which tools to call based on the query.
    Appends an AIMessage with tool_calls to state['messages'].
    """
    logger.info("[router] Routing query: %s", state["query"])

    system = SystemMessage(content=prompts["router_system"])
    human = HumanMessage(content=state["query"])

    response: AIMessage = llm_with_tools.invoke([system, human])
    logger.info("[router] Tool calls requested: %s", response.tool_calls)

    return {
        **state,
        "messages": [response],
        "iterations": state.get("iterations", 0),
    }


# ---------------------------------------------------------------------------
# Node: tool_caller
# ---------------------------------------------------------------------------
def tool_caller(state: AgentState) -> AgentState:
    """
    Executes all tool calls requested by the router or reflect nodes.
    Aggregates results into the appropriate state fields.
    """
    last_ai_msg: AIMessage = state["messages"][-1]
    tool_calls = last_ai_msg.tool_calls or []

    retrieved_docs = list(state.get("retrieved_docs") or [])
    web_results = list(state.get("web_results") or [])
    citations = list(state.get("citations") or [])
    tool_messages = []

    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        logger.info("[tool_caller] Calling tool '%s' with args: %s", tool_name, tool_args)

        try:
            if tool_name == "rag_retriever":
                result = rag_retriever.invoke(tool_args)
                retrieved_docs.extend(result)
                content = json.dumps(result)

            elif tool_name == "web_search":
                result = web_search.invoke(tool_args)
                web_results.extend(result)
                content = json.dumps(result)

            elif tool_name == "citation_fetcher":
                result = citation_fetcher.invoke(tool_args)
                citations.extend(result)
                content = json.dumps(result)

            else:
                content = f"Unknown tool: {tool_name}"

        except Exception as exc:
            logger.error("[tool_caller] Tool '%s' failed: %s", tool_name, exc)
            content = f"Tool error: {exc}"

        from langchain_core.messages import ToolMessage
        tool_messages.append(
            ToolMessage(content=content, tool_call_id=call["id"])
        )

    return {
        **state,
        "messages": tool_messages,
        "retrieved_docs": retrieved_docs,
        "web_results": web_results,
        "citations": citations,
        "iterations": state.get("iterations", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node: reflect
# ---------------------------------------------------------------------------
def reflect(state: AgentState) -> AgentState:
    """
    Reflexion node. Scores the current evidence and either:
    - Sets a high confidence score + drafts final_answer  →  proceed to finalize
    - Sets a low confidence score + requests additional tool calls  →  loop back
    """
    max_iterations = cfg["agent"]["max_iterations"]
    confidence_threshold = cfg["agent"]["confidence_threshold"]

    # Build evidence summary for the LLM
    evidence = {
        "retrieved_docs": state.get("retrieved_docs", []),
        "web_results": state.get("web_results", []),
        "citations": state.get("citations", []),
    }

    reflect_prompt = prompts["reflect_system"].format(
        query=state["query"],
        evidence=json.dumps(evidence, indent=2),
        max_iterations=max_iterations,
        current_iteration=state.get("iterations", 0),
    )

    response = llm.invoke([
        SystemMessage(content=reflect_prompt),
        HumanMessage(content="Evaluate the evidence and respond in JSON."),
    ])

    parsed: dict[str, Any] = {}
    try:
        parsed: dict[str, Any] = json.loads(response.content)
        confidence: float = float(parsed.get("confidence", 0.0))
        final_answer: str = parsed.get("answer", "")
        needs_more: bool = parsed.get("needs_more_tools", False)
    except (json.JSONDecodeError, ValueError):
        logger.warning("[reflect] Could not parse LLM JSON; defaulting low confidence.")
        confidence = 0.0
        final_answer = ""
        needs_more = True

    logger.info("[reflect] Confidence=%.2f needs_more=%s iteration=%d",
                confidence, needs_more, state.get("iterations", 0))

    new_state: AgentState = {
        **state,
        "confidence": confidence,
        "final_answer": final_answer,
        "messages": [AIMessage(content=response.content)],
    }

    # If we still need more evidence, append a follow-up tool call request
    if needs_more and state.get("iterations", 0) < max_iterations:
        followup = llm_with_tools.invoke([
            SystemMessage(content=prompts["router_system"]),
            HumanMessage(
                content=f"We need more evidence for: {state['query']}. "
                        f"Gap identified: {parsed.get('gap', 'unknown')}. Call the appropriate tool."
            ),
        ])
        new_state["messages"] = [*new_state["messages"], followup]

    return new_state


# ---------------------------------------------------------------------------
# Node: finalize
# ---------------------------------------------------------------------------
def finalize(state: AgentState) -> AgentState:
    """
    Terminal node. Formats the final grounded answer with source citations.
    """
    sources = []
    for doc in state.get("retrieved_docs", []):
        sources.append(f"[PDF] {doc.get('source', 'unknown')} — {doc.get('page', '')}")
    for cite in state.get("citations", []):
        sources.append(f"[Paper] {cite.get('title', '')} ({cite.get('doi', '')})")
    for web in state.get("web_results", []):
        sources.append(f"[Web] {web.get('url', '')}")

    source_block = "\n".join(f"  • {s}" for s in sources) if sources else "  • No external sources used."

    formatted = (
        f"{state['final_answer']}\n\n"
        f"**Sources**\n{source_block}\n\n"
        f"*Confidence: {state.get('confidence', 0.0):.0%} | "
        f"Iterations: {state.get('iterations', 0)}*"
    )

    logger.info("[finalize] Answer ready. Confidence=%.2f", state.get("confidence", 0.0))
    return {**state, "final_answer": formatted}


# ---------------------------------------------------------------------------
# Conditional edge: should_continue
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> str:
    """
    Edge function called after the reflect node.
    Returns 'finalize' or 'tool_caller'.
    """
    cfg_agent = load_config()["agent"]
    if (
        state.get("confidence", 0.0) >= cfg_agent["confidence_threshold"]
        or state.get("iterations", 0) >= cfg_agent["max_iterations"]
    ):
        return "finalize"
    return "tool_caller"
