"""Semantic Retriever using SentenceTransformer embeddings.
Uses all-MiniLM-L6-v2 to compute 384-dimensional dense semantic vectors.
"""

import numpy as np


class SemanticRetriever:
    """Dense vector retriever using Sentence Transformers."""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.device = device
        try:
            # Try loading from local cache first to avoid network delays
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name, device=device)
        self.chunks = []
        self.embeddings = None

    def index(self, chunks):
        """Index a list of chunks by computing normalized dense embeddings."""
        if not chunks:
            raise ValueError("No document chunks to index")
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def search(self, query, top_k=5):
        """Search chunks by cosine similarity with the query vector.
        
        Args:
            query: Question or search string.
            top_k: Number of chunks to retrieve.
            
        Returns:
            list of dict: [{"score": float, "chunk": dict, "rank": int}, ...]
        """
        if self.embeddings is None or not self.chunks:
            raise ValueError("Index documents before searching")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]

        # Normalized dot product is cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "rank": rank + 1,
                "score": float(similarities[idx]),
                "chunk": self.chunks[idx],
                "method": "dense_semantic"
            })
        return results
