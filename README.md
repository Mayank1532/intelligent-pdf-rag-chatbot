# Intelligent PDF RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions from uploaded PDF documents.

## Objective

Build and understand a complete RAG pipeline:

PDF → Text Extraction → Chunking → Embeddings → Vector Store → Retrieval → LLM → Answer + Sources

## Features

- PDF upload and processing
- Text extraction
- Document chunking
- Hugging Face embeddings
- Chroma vector store
- Similarity-based retrieval
- Local Ollama LLM
- Grounded question answering
- Source/page information
- Streamlit interface
- Basic automated tests

## Technology Stack

- Python 3.12
- UV
- LangChain
- Hugging Face / Sentence Transformers
- Chroma
- Ollama
- Llama 3.2 3B
- Streamlit
- Git/GitHub

## Project Structure

```text
intelligent-pdf-rag-chatbot/
├── app.py
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
├── docs/
│   └── architecture.md
├── src/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm.py
│   └── rag_pipeline.py
├── tests/
│   └── test_rag.py
└── data/
    ├── raw/
    └── processed/