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
        """
        if self.matrix is None or not self.chunks:
            raise ValueError("Index documents before searching")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix).flatten()

        # Check if any lexical overlap occurred
        has_any_match = not np.all(similarities == 0)

        if has_any_match:
            top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            # When no keywords match, scores are 0.0
            top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(similarities[idx])
            results.append({
                "rank": rank + 1,
                "score": score,
                "chunk": self.chunks[idx],
                "method": "tfidf_baseline",
                "has_match": score > 0.0
            })
        return results
