import os

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.loaders import load_promptior_docs


# --------------------------------------------------
# Environment configuration
# --------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# --------------------------------------------------
# Factory functions
# --------------------------------------------------

def get_embeddings():
    """
    Returns the appropriate embeddings implementation
    based on the selected LLM provider.
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()

    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=OLLAMA_MODEL)


def get_llm():
    """
    Returns the appropriate LLM implementation
    based on the selected provider.
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0
        )

    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=OLLAMA_MODEL)


# --------------------------------------------------
# RAG pipeline
# --------------------------------------------------

def build_rag_chain():
    """
    Builds a Retrieval-Augmented Generation (RAG) chain.

    Flow:
    - Load documents
    - Create embeddings
    - Build vector store
    - Retrieve relevant chunks
    - Inject context into prompt
    - Generate answer using selected LLM
    """

    # 1. Load source documents
    documents = load_promptior_docs()

    # 2. Embeddings
    embeddings = get_embeddings()

    # 3. Vector store (in-memory)
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # 4. Retriever (top-k similarity search)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # 5. Prompt template
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

    # 6. LLM
    llm = get_llm()

    # 7. LCEL chain composition
    rag_chain = (
        {
            "context": retriever,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
