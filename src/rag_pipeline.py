from langchain_core.prompts import ChatPromptTemplate


def create_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(
    """You are a document question-answering assistant.

Answer the question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- If the answer is not in the context, say: "I could not find the answer in the provided document."
- Keep the answer concise and factual.
- Do not invent or assume information.

Context:
{context}

Question:
{question}

Answer:"""
)

    def rag(question: str):
        documents = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in documents)

        response = llm.invoke(
            prompt.format_messages(
                context=context,
                question=question,
            )
        )

        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
            }
            for doc in documents
        ]

        return {
            "answer": response.content,
            "sources": sources,
        }

    return rag