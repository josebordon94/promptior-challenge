import os

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.loaders import (
    load_promptior_text_docs,
    load_promptior_web_docs,
)

# Environment configuration

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Factory functions

def get_embeddings():
    """Return embeddings based on selected provider."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()

    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=OLLAMA_MODEL)


def get_llm():
    """Return LLM based on selected provider."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
        )

    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=OLLAMA_MODEL)


# Internal RAG builder (shared)


def _build_rag_chain(documents):
    """
    Core RAG pipeline builder.
    Receives documents and returns a runnable LCEL chain.
    """

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are an assistant answering questions ONLY based on the context below.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """
    )

    llm = get_llm()

    return (
        {
            "context": retriever,
            "question": lambda x: x,
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# Public RAG builders

def build_rag_chain_from_text():
    """
    Primary RAG:
    Uses Promptior PDF / text content.
    """
    documents = load_promptior_text_docs()
    return _build_rag_chain(documents)


def build_rag_chain_from_web():
    """
    Secondary RAG (extra points):
    Uses Promptior website scraping.
    """
    documents = load_promptior_web_docs()
    return _build_rag_chain(documents)
