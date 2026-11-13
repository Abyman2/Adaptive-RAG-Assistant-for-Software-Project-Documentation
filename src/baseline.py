"""TF-IDF Keyword Baseline Retriever.
Serves as the non-Transformer baseline to demonstrate value added by semantic embeddings.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TfidfBaseline:
    """Keyword-based retriever using TF-IDF and Cosine Similarity."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.chunks = []
        self.matrix = None

    def index(self, chunks):
        """Index a list of chunks using TF-IDF."""
        if not chunks:
            raise ValueError("No chunks to index")
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=5):
        """Search indexed chunks using TF-IDF cosine similarity.
        
        Args:
            query: Query string.
            top_k: Number of chunks to retrieve.
            
        Returns:
            list of dict: [{"score": float, "chunk": dict, "rank": int, "has_match": bool}, ...]
            Returns empty list [] when no lexical keywords match (all scores 0.0).
        """
        if self.matrix is None or not self.chunks:
            raise ValueError("Index documents before searching")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix).flatten()

        # Identify chunks with strictly positive similarity
        matching_indices = np.where(similarities > 0.0)[0]
        if len(matching_indices) == 0:
            return []

        # Sort only matching chunks by score descending
        sorted_order = np.argsort(similarities[matching_indices])[::-1][:top_k]
        top_indices = matching_indices[sorted_order]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "rank": rank + 1,
                "score": float(similarities[idx]),
                "chunk": self.chunks[idx],
                "method": "tfidf_baseline",
                "has_match": True
            })
        return results
