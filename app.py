import streamlit as st

from src.chunker import split_documents
from src.embeddings import create_embeddings
from src.llm import create_llm
from src.pdf_loader import load_pdf
from src.rag_pipeline import create_rag_chain
from src.retriever import get_retriever
from src.vector_store import create_vector_store


st.title("Intelligent PDF RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("data/raw/uploaded.pdf", "wb") as file:
        file.write(uploaded_file.getbuffer())

    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            documents = load_pdf("data/raw/uploaded.pdf")
            chunks = split_documents(documents)
            embeddings = create_embeddings()
            vector_store = create_vector_store(chunks, embeddings)
            retriever = get_retriever(vector_store)
            llm = create_llm()
            rag = create_rag_chain(retriever, llm)

            st.session_state["rag"] = rag

        st.success("PDF processed successfully.")

if "rag" in st.session_state:
    question = st.text_input("Ask a question about the PDF")

    if question:
        with st.spinner("Generating answer..."):
            result = st.session_state["rag"](question)

        st.write(result["answer"])

        st.subheader("Sources")
        for source in result["sources"]:
            st.write(source)