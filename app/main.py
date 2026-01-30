from fastapi import FastAPI
from langserve import add_routes

from app.rag import get_rag_chain

import os

print("Using LLM Provider: ", os.getenv("LLM_PROVIDER"))
app = FastAPI(
    title="Promptior RAG Chatbot",
    version="1.0",
    description="RAG-based chatbot using LangChain"
)


rag_chain_text = get_rag_chain(source="text")
rag_chain_web = get_rag_chain(source="web")

add_routes(app, rag_chain_text, path="/chat")
add_routes(app, rag_chain_web, path="/chat-web")

