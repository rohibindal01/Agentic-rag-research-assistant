"""tests/test_ingestion.py"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_ingest_raises_if_no_pdfs():
    from src.retrieval.ingest import ingest_pdfs

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="No PDF files found"):
            ingest_pdfs(tmpdir)


def test_ingest_calls_faiss_save(tmp_path):
    from src.retrieval.ingest import ingest_pdfs

    # Create a dummy PDF placeholder (actual PDF parsing is mocked)
    dummy_pdf = tmp_path / "sample.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    mock_doc = MagicMock()
    mock_doc.page_content = "Test content about RAG systems."
    mock_doc.metadata = {"source": "sample.pdf", "page": 1}

    with (
        patch("src.retrieval.ingest.PyPDFLoader") as MockLoader,
        patch("src.retrieval.ingest.FAISS") as MockFAISS,
        patch("src.retrieval.ingest.get_embeddings"),
    ):
        MockLoader.return_value.load.return_value = [mock_doc]
        mock_vs = MagicMock()
        MockFAISS.from_documents.return_value = mock_vs

        ingest_pdfs(str(tmp_path))

        MockFAISS.from_documents.assert_called_once()
        mock_vs.save_local.assert_called_once()
