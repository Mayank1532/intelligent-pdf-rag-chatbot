from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(file_path: str) -> list[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("File must be a PDF.")

    reader = PdfReader(str(path))

    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": str(path), "page": page_number},
        )
        for page_number, page in enumerate(reader.pages)
    ]