"""
ingest.py — PDF ingestion pipeline.

Steps
-----
1. Load PDF files from an input directory.
2. Split each document into overlapping chunks.
3. Generate embeddings and upsert into FAISS.
4. Persist the FAISS index to disk.

Usage
-----
    python -m src.retrieval.ingest --input_dir data/raw/ --output_dir data/processed/
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from configs.embeddings_factory import get_embeddings
from configs.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ingest_pdfs(input_dir: str, output_dir: str | None = None) -> FAISS:
    """
    Load all PDFs from ``input_dir``, chunk them, embed them, and persist
    the FAISS index.

    Parameters
    ----------
    input_dir : str
        Directory containing raw PDF files.
    output_dir : str, optional
        Directory to save chunked text files (for inspection). If None, skipped.

    Returns
    -------
    FAISS
        The in-memory (and optionally persisted) vector store.
    """
    cfg = load_config()
    chunk_size = cfg["chunking"]["chunk_size"]
    chunk_overlap = cfg["chunking"]["chunk_overlap"]
    index_path = cfg["vectorstore"]["faiss_index_path"]

    pdf_paths = list(Path(input_dir).glob("**/*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in '{input_dir}'.")

    logger.info("Found %d PDF(s) to ingest.", len(pdf_paths))

    # ── Load ──────────────────────────────────────────────────────────────
    all_documents = []
    for pdf_path in pdf_paths:
        logger.info("Loading '%s'...", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        # Attach filename to metadata for citation tracing
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        all_documents.extend(docs)

    logger.info("Loaded %d pages total.", len(all_documents))

    # ── Split ─────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_documents)
    logger.info("Split into %d chunks (size=%d, overlap=%d).",
                len(chunks), chunk_size, chunk_overlap)

    # ── Optional: save chunks as text for inspection ───────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        chunk_file = Path(output_dir) / "chunks.txt"
        with open(chunk_file, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i} | {chunk.metadata} ---\n")
                f.write(chunk.page_content + "\n\n")
        logger.info("Chunks saved to '%s'.", chunk_file)

    # ── Embed & index ─────────────────────────────────────────────────────
    logger.info("Generating embeddings with model '%s'...", cfg["embeddings"]["model"])
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # ── Persist ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    vectorstore.save_local(index_path)
    logger.info("FAISS index saved to '%s'. Ingestion complete ✓", index_path)

    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS vector store.")
    parser.add_argument("--input_dir", default="data/raw/", help="Directory with PDF files.")
    parser.add_argument("--output_dir", default="data/processed/", help="Directory for chunk files.")
    args = parser.parse_args()

    ingest_pdfs(args.input_dir, args.output_dir)
