"""Unit and pipeline tests for Adaptive RAG Assistant components.
Tests chunking, document loading, TF-IDF baseline, and adaptive retrieval logic.
"""

import pytest
from src.chunker import chunk_document, process_documents
from src.document_loader import load_document, SUPPORTED_EXTENSIONS
from src.adaptive_retrieval import AdaptiveRetriever
from src.baseline import TfidfBaseline


def test_chunking():
    """Test that a long document is chunked into multiple pieces with given size."""
    document = {
        "source": "test.md",
        "text": " ".join(["software"] * 400)
    }
    chunks = chunk_document(
        document,
        chunk_size=180,
        overlap=40
    )
    assert len(chunks) > 1
    assert chunks[0]["source"] == "test.md"
    assert len(chunks[0]["text"].split()) == 180


def test_chunk_metadata():
    """Test that chunk attributes (source, chunk_id) are preserved."""
    document = {
        "source": "README.md",
        "text": "This is a software project documentation."
    }
    chunks = chunk_document(document)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == 0
    assert chunks[0]["source"] == "README.md"
    assert "This is a software project" in chunks[0]["text"]


def test_invalid_chunk_size():
    """Test validation of illegal chunk_size and overlap parameters."""
    document = {"source": "test.md", "text": "sample text"}
    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=100, overlap=-5)


def test_process_documents_multi():
    """Test processing multiple documents and global ID indexing."""
    docs = [
        {"source": "doc1.md", "text": "Doc one content words here."},
        {"source": "doc2.md", "text": "Doc two content words here."}
    ]
    chunks = process_documents(docs, chunk_size=50, overlap=10)
    assert len(chunks) == 2
    assert chunks[0]["global_id"] == 0
    assert chunks[1]["global_id"] == 1
    assert chunks[0]["source"] == "doc1.md"
    assert chunks[1]["source"] == "doc2.md"


def test_tfidf_baseline():
    """Test TF-IDF baseline indexing and keyword search."""
    baseline = TfidfBaseline()
    chunks = [
        {"source": "db.md", "text": "PostgreSQL is the main relational database used for storing users.", "chunk_id": 0},
        {"source": "ui.md", "text": "React and Vite power the single page frontend application.", "chunk_id": 1},
        {"source": "deploy.md", "text": "Deployment is automated on Railway and Docker containers.", "chunk_id": 2}
    ]
    baseline.index(chunks)
    
    # Query for database
    results = baseline.search("PostgreSQL database", top_k=2)
    assert len(results) == 2
    assert results[0]["chunk"]["source"] == "db.md"
    assert results[0]["score"] > 0.0


def test_adaptive_retrieval_modes():
    """Test adaptive retrieval mode selection and thresholds using a mock retriever."""
    class MockRetriever:
        def __init__(self, score):
            self.score = score
        def search(self, query, top_k=1):
            return [{"score": self.score, "chunk": {"source": "mock.md", "text": "mock text", "chunk_id": 0}}]

    # High score -> focused mode (top_k=3)
    adaptive_high = AdaptiveRetriever(MockRetriever(0.75))
    res_high = adaptive_high.retrieve("query")
    assert res_high["mode"] == "focused"
    assert res_high["sufficient_evidence"] is True

    # Medium score -> expanded mode (top_k=6)
    adaptive_mid = AdaptiveRetriever(MockRetriever(0.45))
    res_mid = adaptive_mid.retrieve("query")
    assert res_mid["mode"] == "expanded"
    assert res_mid["sufficient_evidence"] is True

    # Low score -> broad mode (top_k=8)
    adaptive_low = AdaptiveRetriever(MockRetriever(0.20))
    res_low = adaptive_low.retrieve("query")
    assert res_low["mode"] == "broad"
    assert res_low["sufficient_evidence"] is False
