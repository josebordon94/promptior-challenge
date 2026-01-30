import os
import logging
from typing import Literal

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.loaders import (
    load_promptior_text_docs,
    load_promptior_web_docs,
)


# Logging


logger = logging.getLogger(__name__)


# Environment configuration

LLM_PROVIDER: Literal["openai", "ollama"] = os.getenv("LLM_PROVIDER", "openai")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

logger.info(f"Using LLM provider: {LLM_PROVIDER}")

# Factory functions

def get_embeddings():
    if LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()

    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=OLLAMA_MODEL)


def get_llm():
    """Return LLM implementation based on provider."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        logger.info(f"Using OpenAI LLM: {OPENAI_MODEL}")
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    from langchain_ollama import OllamaLLM
    logger.info(f"Using Ollama LLM: {OLLAMA_MODEL}")
    return OllamaLLM(model=OLLAMA_MODEL)


# Internal RAG builder

def _build_rag_chain(documents):
    """Core RAG pipeline builder."""

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

# Public API (lazy + cached)

_RAG_CACHE = {}

def get_rag_chain(source: Literal["text", "web"] = "text"):
    """
    Returns a cached RAG chain.

    source:
    - text → Promptior PDF / text content (primary)
    - web  → Promptior website scraping (secondary / extra points)
    """

    if source in _RAG_CACHE:
        return _RAG_CACHE[source]

    logger.info(f"Building RAG chain | provider={LLM_PROVIDER} | source={source}")

    if source == "text":
        documents = load_promptior_text_docs()
    elif source == "web":
        documents = load_promptior_web_docs()
    else:
        raise ValueError(f"Unknown RAG source: {source}")

    rag_chain = _build_rag_chain(documents)
    _RAG_CACHE[source] = rag_chain

    return rag_chain
