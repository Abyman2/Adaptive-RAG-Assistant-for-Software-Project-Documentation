"""Unit and pipeline tests for Adaptive RAG Assistant components.
Tests chunking, document loading, TF-IDF baseline, directory filtering,
zero-match edge cases, generator abstention, and gold label dataset integrity.
"""

import json
from pathlib import Path
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


def test_directory_filtering(tmp_path):
    """Test that load_directory skips ignored folders like .git, .venv, node_modules."""
    from src.document_loader import load_directory
    
    (tmp_path / "README.md").write_text("Valid project documentation here.", encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config.txt").write_text("git config content", encoding="utf-8")
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "lib.py").write_text("venv code", encoding="utf-8")
    
    docs = load_directory(tmp_path)
    sources = [d["source"] for d in docs]
    assert "README.md" in sources
    assert not any(".git" in s for s in sources)
    assert not any(".venv" in s for s in sources)


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
    
    results = baseline.search("PostgreSQL database", top_k=2)
    assert len(results) == 2
    assert results[0]["chunk"]["source"] == "db.md"
    assert results[0]["score"] > 0.0
    assert results[0]["has_match"] is True


def test_tfidf_zero_match():
    """Test TF-IDF behavior when query words do not exist in corpus."""
    baseline = TfidfBaseline()
    chunks = [
        {"source": "db.md", "text": "PostgreSQL database content.", "chunk_id": 0}
    ]
    baseline.index(chunks)
    
    # Completely non-matching query
    results = baseline.search("Quantum Superposition Entanglement", top_k=1)
    assert len(results) == 1
    assert results[0]["score"] == 0.0
    assert results[0]["has_match"] is False


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


def test_generator_abstention_logic():
    """Verify that AnswerGenerator abstains correctly when sufficient_evidence is False."""
    from src.generator import AnswerGenerator
    
    class DummyGenerator(AnswerGenerator):
        def __init__(self):
            # Bypass heavy neural loading for unit test
            pass

    dummy = DummyGenerator()
    out = dummy.generate("What is the secret key?", [], sufficient_evidence=False)
    assert "does not provide enough information" in out["answer"]
    assert out["grounded"] is False
    assert out["citations"] == []


def test_dataset_integrity():
    """Verify JSON structure and constraints for dev_v2 and held_out_v2."""
    for p in [Path("evaluation/dev_v2.json"), Path("evaluation/held_out_v2.json")]:
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "questions" in data
        for q in data["questions"]:
            assert "id" in q
            assert "category" in q
            assert "question" in q
            assert "answerable" in q
            assert "reference_answer" in q
