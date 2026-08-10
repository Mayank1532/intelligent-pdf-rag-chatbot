from src.chunker import split_documents
from src.pdf_loader import load_pdf


def test_pdf_loading():
    documents = load_pdf("data/raw/sample.pdf")

    assert len(documents) > 0
    assert documents[0].page_content


def test_chunking():
    documents = load_pdf("data/raw/sample.pdf")
    chunks = split_documents(documents)

    assert len(chunks) > 0
    assert all(chunk.page_content for chunk in chunks)