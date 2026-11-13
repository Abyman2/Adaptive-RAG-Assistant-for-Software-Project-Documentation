"""Adaptive Retrieval Module.
Dynamically adjusts retrieval depth (top-K) and confidence thresholds based on
semantic similarity scores and query specificity.
"""

class AdaptiveRetriever:
    """Adaptive wrapper over SemanticRetriever that dynamically sets top-K
    and checks evidence sufficiency.
    """
    
    def __init__(self, retriever, focused_threshold=0.55, expanded_threshold=0.35, min_threshold=0.28):
        self.retriever = retriever
        self.focused_threshold = focused_threshold
        self.expanded_threshold = expanded_threshold
        self.min_threshold = min_threshold

    def retrieve(self, query):
        """Perform adaptive retrieval.
        
        Evaluates top similarity score, determines retrieval mode, retrieves
        the optimal context window, and assesses evidence sufficiency.
        
        Args:
            query: Question string.
            
        Returns:
            dict containing:
                - mode: 'focused' | 'expanded' | 'broad'
                - confidence: float top-1 similarity score
                - top_k: int number of retrieved chunks
                - results: list of retrieved chunks with scores
                - sufficient_evidence: bool whether evidence meets minimum threshold
                - explanation: str explanation of adaptive decision
        """
        initial_results = self.retriever.search(query, top_k=1)
        if not initial_results:
            return {
                "mode": "none",
                "confidence": 0.0,
                "top_k": 0,
                "results": [],
                "sufficient_evidence": False,
                "explanation": "No documents available in index."
            }

        confidence = initial_results[0]["score"]

        if confidence >= self.focused_threshold:
            mode = "focused"
            top_k = 3
            explanation = (
                f"High confidence ({confidence:.2f} >= {self.focused_threshold}). "
                "Retrieving focused top-3 chunks to prevent noise."
            )
        elif confidence >= self.expanded_threshold:
            mode = "expanded"
            top_k = 6
            explanation = (
                f"Moderate confidence ({confidence:.2f} >= {self.expanded_threshold}). "
                "Expanding context to top-6 chunks for comprehensive coverage."
            )
        else:
            mode = "broad"
            top_k = 8
            explanation = (
                f"Low confidence ({confidence:.2f} < {self.expanded_threshold}). "
                "Broadening search to top-8 chunks to locate potential evidence."
            )

        sufficient_evidence = (confidence >= self.min_threshold)
        results = self.retriever.search(query, top_k=top_k)

        return {
            "mode": mode,
            "confidence": confidence,
            "top_k": len(results),
            "results": results,
            "sufficient_evidence": sufficient_evidence,
            "explanation": explanation
        }

