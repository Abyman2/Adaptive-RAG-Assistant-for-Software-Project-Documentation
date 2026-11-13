"""Pretrained Transformer Generator.
Uses google/flan-t5-small to generate grounded answers with citations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AnswerGenerator:
    """Pretrained Seq2Seq LM answer generator grounded on retrieved documentation."""
    
    def __init__(self, model_name="google/flan-t5-small", device="cpu"):
        self.model_name = model_name
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True).to(device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def generate(self, question, retrieved_chunks, sufficient_evidence=True):
        """Generate an answer grounded in retrieved chunks.
        
        Args:
            question: Developer query string.
            retrieved_chunks: List of retrieved chunk dictionaries.
            sufficient_evidence: Boolean flag indicating if retrieval confidence was met.
            
        Returns:
            dict containing:
                - answer: str generated answer
                - citations: list of source document names
                - grounded: bool indicating if answer was grounded on documents
        """
        if not sufficient_evidence or not retrieved_chunks:
            return {
                "answer": "The documentation does not provide enough information to answer this question.",
                "citations": [],
                "grounded": False
            }

        context_parts = []
        citations = []
        for i, chunk_item in enumerate(retrieved_chunks):
            chunk = chunk_item.get("chunk", chunk_item)
            source = chunk.get("source", "unknown")
            chunk_id = chunk.get("chunk_id", i)
            context_parts.append(f"Source: {source} (Section {chunk_id}):\n{chunk.get('text', '')}")
            if source not in citations:
                citations.append(source)

        context = "\n\n".join(context_parts)
        prompt = (
            f"Answer the software engineering question using ONLY the supplied documentation.\n"
            f"If the documentation does not contain the answer, say that the documentation does not provide enough information.\n"
            f"Do not invent file names, functions, commands, or configurations.\n\n"
            f"Documentation:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=2,
                early_stopping=True,
                repetition_penalty=1.2
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if not answer:
            answer = "The documentation does not provide enough information."

        return {
            "answer": answer,
            "citations": citations,
            "grounded": True
        }
