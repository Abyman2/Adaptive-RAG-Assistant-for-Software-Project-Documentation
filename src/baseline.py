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
            list of dict: [{"score": float, "chunk": dict, "rank": int}, ...]
        """
        if self.matrix is None or not self.chunks:
            raise ValueError("Index documents before searching")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix).flatten()

        # Handle edge case where no terms match (all zeros)
        if np.all(similarities == 0):
            top_indices = np.arange(min(top_k, len(self.chunks)))
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "rank": rank + 1,
                "score": float(similarities[idx]),
                "chunk": self.chunks[idx],
                "method": "tfidf_baseline"
            })
        return results

