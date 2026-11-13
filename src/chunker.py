"""Document chunker module.
Splits documents into overlapping chunks suitable for embedding models.
"""

def chunk_document(document, chunk_size=180, overlap=40):
    """Split a single document into overlapping word chunks.
    
    Args:
        document: dict with keys "source" and "text"
        chunk_size: number of words per chunk
        overlap: number of overlapping words between consecutive chunks
        
    Returns:
        list of dict: [{"text": str, "source": str, "chunk_id": int, "word_count": int}, ...]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and strictly less than chunk_size")

    words = document["text"].split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            continue
        chunks.append({
            "text": " ".join(chunk_words),
            "source": document["source"],
            "chunk_id": len(chunks),
            "word_count": len(chunk_words)
        })
        if start + chunk_size >= len(words):
            break

    return chunks


def process_documents(documents, chunk_size=180, overlap=40):
    """Process a list of documents into indexed chunks.
    
    Args:
        documents: list of document dicts
        chunk_size: words per chunk
        overlap: word overlap
        
    Returns:
        list of chunk dicts
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        
    # Re-index globally unique chunk IDs
    for idx, c in enumerate(all_chunks):
        c["global_id"] = idx
        
    return all_chunks
