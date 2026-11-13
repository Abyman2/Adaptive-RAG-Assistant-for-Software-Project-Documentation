"""Document chunker module.
Splits documents into overlapping chunks suitable for embedding models
while preserving source filenames, chunk identifiers, and section hierarchy.
"""

def chunk_document(document, chunk_size=100, overlap=25):
    """Split a single document into overlapping word chunks with section tracking.
    
    Args:
        document: dict with keys "source" and "text"
        chunk_size: number of words per chunk
        overlap: number of overlapping words between consecutive chunks
        
    Returns:
        list of dict: [{
            "text": str,
            "source": str,
            "chunk_id": int,
            "word_count": int,
            "sections": list of str
        }, ...]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and strictly less than chunk_size")

    text = document.get("text", "")
    if not text.strip():
        return []

    # Map words to active markdown/text sections
    words = []
    word_to_section = []
    current_section = "Document Root"

    for line in text.split("\n"):
        line_clean = line.strip()
        if line_clean.startswith("#"):
            current_section = line_clean
        for w in line_clean.split():
            words.append(w)
            word_to_section.append(current_section)

    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        
        # Deduplicate sections preserving order
        chunk_sections = list(dict.fromkeys(word_to_section[start:end]))
        
        chunks.append({
            "text": " ".join(chunk_words),
            "source": document["source"],
            "chunk_id": len(chunks),
            "word_count": len(chunk_words),
            "sections": chunk_sections
        })
        if start + chunk_size >= len(words):
            break

    return chunks


def process_documents(documents, chunk_size=100, overlap=25):
    """Process a list of documents into indexed chunks with global IDs.
    
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
