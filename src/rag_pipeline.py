from langchain_core.prompts import ChatPromptTemplate


def create_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the provided context.

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

        return {
            "answer": response.content,
            "sources": documents,
        }

    return rag