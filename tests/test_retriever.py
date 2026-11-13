"""Tests for semantic and adaptive retrieval pipeline.
"""

import pytest
import numpy as np
from src.chunker import process_documents
from src.adaptive_retrieval import AdaptiveRetriever
from src.baseline import TfidfBaseline


def test_baseline_and_adaptive_integration():
    """Verify that document chunks can be indexed by both baseline and adaptive systems."""
    sample_docs = [
        {
            "source": "database.md",
            "text": "PostgreSQL 15 is our primary relational database. Flyway handles schema migrations."
        },
        {
            "source": "frontend.md",
            "text": "The frontend is built with React 18 and Vite. It runs on port 5173."
        },
        {
            "source": "security.md",
            "text": "Spring Security handles authentication with JWT tokens. Role based access control is supported."
        }
    ]
    chunks = process_documents(sample_docs, chunk_size=50, overlap=10)
    assert len(chunks) >= 3

    # Test TF-IDF Baseline
    tfidf = TfidfBaseline()
    tfidf.index(chunks)
    res_tfidf = tfidf.search("PostgreSQL database migrations", top_k=2)
    assert len(res_tfidf) == 2
    assert "database.md" in res_tfidf[0]["chunk"]["source"]

    # Test Adaptive Logic wrapper with mock scores
    class MockSemantic:
        def __init__(self):
            self.chunks = chunks
        def search(self, query, top_k=5):
            # simulate score based on query
            score = 0.65 if "React" in query else 0.15
            return [
                {"rank": i + 1, "score": score, "chunk": c, "method": "mock"}
                for i, c in enumerate(self.chunks[:top_k])
            ]

    mock_sem = MockSemantic()
    adaptive = AdaptiveRetriever(mock_sem)

    # High match test
    res_high = adaptive.retrieve("React port")
    assert res_high["mode"] == "focused"
    assert res_high["top_k"] == 3
    assert res_high["sufficient_evidence"] is True

    # Low match test (abstention check)
    res_low = adaptive.retrieve("completely unrelated quantum physics query")
    assert res_low["mode"] == "broad"
    assert res_low["sufficient_evidence"] is False
