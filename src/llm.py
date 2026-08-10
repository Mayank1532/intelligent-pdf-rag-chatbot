from langchain_ollama import ChatOllama


def create_llm():
    return ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )