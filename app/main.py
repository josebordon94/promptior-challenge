from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from langserve import add_routes

from app.rag import build_rag_chain_from_text
from app.rag import build_rag_chain_from_web

import os


print("Using LLM Provider: ", os.getenv("LLM_PROVIDER"))

app = FastAPI(
    title="Promptior RAG Chatbot",
    version="1.0",
    description="RAG-based chatbot using LangChain and Ollama"
)

rag_text = build_rag_chain_from_text()
rag_web = build_rag_chain_from_web()

add_routes(app, rag_text, path="/chat")
add_routes(app, rag_web, path="/chat-web")

