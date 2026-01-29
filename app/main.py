from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from langserve import add_routes

from app.rag import build_rag_chain

import os

print("Using LLM Provider: ", os.getenv("LLM_PROVIDER"))

app = FastAPI(
    title="Promptior RAG Chatbot",
    version="1.0",
    description="RAG-based chatbot using LangChain and Ollama"
)

rag_chain = build_rag_chain()

add_routes(
    app,
    rag_chain,
    path="/chat"
)
