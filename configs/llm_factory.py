"""
llm_factory.py — Returns a ChatGroq instance configured from config.yaml.

Supported models (set in configs/config.yaml):
  - llama-3.3-70b-versatile  (recommended — best quality)
  - mixtral-8x7b-32768       (best for long context)
  - gemma2-9b-it             (fastest / lowest latency)

Requires: GROQ_API_KEY environment variable
Get a free key at: https://console.groq.com
"""

from __future__ import annotations
from configs.loader import load_config


def get_llm(with_tools: list | None = None):
    from langchain_groq import ChatGroq

    cfg = load_config()
    model = cfg["llm"]["model"]
    temperature = cfg["llm"]["temperature"]

    llm = ChatGroq(
        model=model,
        temperature=temperature,
    )

    if with_tools:
        return llm.bind_tools(with_tools, parallel_tool_calls=False)
    return llm