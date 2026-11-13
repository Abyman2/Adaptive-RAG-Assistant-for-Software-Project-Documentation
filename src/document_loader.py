"""Document loader module supporting multiple file formats.
Supports PDF, DOCX, Markdown, Text, and common source code files.
Supports directory loading with automated ignore rules for build/cache directories.
"""

from pathlib import Path
from pypdf import PdfReader
from docx import Document

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".py",
    ".java", ".js", ".jsx", ".ts", ".tsx",
    ".json", ".xml", ".yml", ".yaml",
    ".html", ".css", ".rst"
}

IGNORED_DIRECTORIES = {
    ".git", ".venv", "venv", "env", ".env",
    "__pycache__", ".pytest_cache", "node_modules",
    "dist", "build", "target", "out", ".idea",
    ".vscode", ".eggs", "site-packages"
}


def load_document(file_path, base_dir=None):
    """Load text content from a supported file format.
    
    Args:
        file_path: Path string or Path object to the file.
        base_dir: Optional base directory to compute relative paths for citations.
        
    Returns:
        dict: {"source": relative_or_base_name, "text": extracted_text}
    """
    path = Path(file_path).resolve()
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

    # Determine source name (prefer relative path if base_dir provided)
    if base_dir:
        try:
            source_name = str(path.relative_to(Path(base_dir).resolve()))
        except ValueError:
            source_name = path.name
    else:
        source_name = path.name

    return {
        "source": source_name,
        "text": text.strip()
    }


def load_directory(directory_path):
    """Load all supported documents from a directory, skipping build/cache folders.
    
    Args:
        directory_path: Directory path.
        
    Returns:
        list of dict: [{"source": relative_path, "text": text}, ...]
    """
    dir_path = Path(directory_path).resolve()
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory_path}")

    documents = []
    for file_path in dir_path.rglob("*"):
        # Check if any parent folder is in the ignored set
        parts = file_path.relative_to(dir_path).parts
        if any(ignored in parts for ignored in IGNORED_DIRECTORIES):
            continue

        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                documents.append(load_document(file_path, base_dir=dir_path))
            except Exception:
                # Skip unreadable or empty files gracefully
                continue
    return documents
