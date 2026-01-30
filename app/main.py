from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langserve import add_routes
from app.rag import get_rag_chain
import os

app = FastAPI(
    title="Promptior RAG Chatbot",
    version="1.0",
    description="RAG-based chatbot using LangChain"
)

templates = Jinja2Templates(directory="app/views")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.on_event("startup")
def startup_event():
    print("Using LLM Provider:", os.getenv("LLM_PROVIDER"))

    rag_chain_text = get_rag_chain(source="text")
    rag_chain_web = get_rag_chain(source="web")

    add_routes(app, rag_chain_text, path="/chat")
    add_routes(app, rag_chain_web, path="/chat-web")
    
@app.get("/", response_class=HTMLResponse)
def chat_ui(request: Request):
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "default_question": "What is Promptior and when was it founded?"
        }
    )
