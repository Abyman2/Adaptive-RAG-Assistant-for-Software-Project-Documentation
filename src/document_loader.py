"""Document loader module supporting multiple file formats.
Supports PDF, DOCX, Markdown, Text, and common source code files.
"""

from pathlib import Path
from pypdf import PdfReader
from docx import Document

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".py",
    ".java", ".js", ".jsx", ".ts", ".tsx",
    ".json", ".xml", ".yml", ".yaml",
    ".html", ".css"
}


def load_document(file_path):
    """Load text content from a supported file format.
    
    Args:
        file_path: Path string or Path object to the file.
        
    Returns:
        dict: {"source": filename, "text": extracted_text}
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    elif extension == ".docx":
        document = Document(str(path))
        text = "\n".join(
            paragraph.text for paragraph in document.paragraphs
        )
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    if not text.strip():
        raise ValueError(f"Document {path.name} is empty or unreadable.")

    return {
        "source": path.name,
        "text": text.strip()
    }


def load_directory(directory_path):
    """Load all supported documents from a directory.
    
    Args:
        directory_path: Directory path.
        
    Returns:
        list of dict: [{"source": filename, "text": text}, ...]
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory_path}")

    documents = []
    for file_path in dir_path.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                documents.append(load_document(file_path))
            except Exception as e:
                # Skip invalid or empty files
                continue
    return documents
