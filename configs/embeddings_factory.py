"""
embeddings_factory.py — Returns HuggingFaceEmbeddings from config.yaml.

Uses sentence-transformers locally — no API key required.

Recommended models (set in configs/config.yaml):
  - sentence-transformers/all-MiniLM-L6-v2   (fast, good quality, ~80MB)
  - BAAI/bge-small-en-v1.5                   (better accuracy, ~130MB)
  - BAAI/bge-base-en-v1.5                    (best accuracy, ~440MB)
"""

from __future__ import annotations
from functools import lru_cache
from configs.loader import load_config


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Returns a cached HuggingFaceEmbeddings instance.
    Model is downloaded on first call and cached locally by sentence-transformers.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    cfg = load_config()
    model = cfg["embeddings"]["model"]

    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
