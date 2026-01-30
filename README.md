# Promptior RAG Chatbot

RAG-based (Retrieval Augmented Generation) chatbot built with **FastAPI** and **LangChain**, supporting multiple LLM providers: **Ollama** (local models) and **OpenAI** (cloud API).

This project demonstrates how to build a simple, local-first AI chatbot that answers questions **only based on provided company documents**, without relying on external APIs.

---

## Project Overview

The goal of this challenge was to design and deploy a chatbot capable of answering questions about Promptior using a Retrieval-Augmented Generation (RAG) architecture, based on LangChain.

My approach was to build a modular RAG pipeline that clearly separates document ingestion, vector retrieval, and answer generation. The solution prioritizes transparency and correctness: the chatbot is explicitly instructed to answer only using the provided sources, avoiding hallucinations.

The implementation loads Promptior-related documents (text-based content extracted from the provided materials, with an optional web-based source for comparison), converts them into vector embeddings, and stores them in an in-memory FAISS vector database. When a question is received, the system retrieves the most relevant document chunks and injects them into a controlled prompt used by the language model.

To keep the solution simple and demo-friendly, the project exposes the RAG chain via a REST API using FastAPI and LangServe.

The final result is a clean, extensible RAG system that can be easily adapted to different data sources, LLM providers, or deployment environments.

---

## Component Diagram

The following diagram illustrates the components involved in the solution and how they interact from the moment a user submits a question until a response is generated.

📎 See diagram: `docs/component-diagram.png`

---

## What this project does

- Loads company information from local documents
- Splits text into chunks
- Converts text into vector embeddings
- Stores embeddings in a vector database (FAISS)
- Retrieves the most relevant chunks for a given question
- Uses either a local LLM (via Ollama) or OpenAI's GPT models to generate answers based only on that context
- Exposes everything through a REST API

---

## Tech Stack

- **Python 3.11**
- **FastAPI** – REST API framework
- **LangChain** – LLM orchestration and RAG pipelines
- **LangServe** – Exposes LangChain chains as HTTP endpoints
- **Ollama** – Runs local LLMs and embeddings (offline)
- **LLaMA 2** – Local language model
- **OpenAI GPT** – Cloud-based models (gpt-4o-mini, gpt-4, gpt-3.5-turbo)
- **FAISS** – Vector similarity search
- **Uvicorn** – ASGI server

---

## Project Structure

```text
promptior-rag-chatbot/
│
├── app/
│   ├── main.py        # FastAPI app entrypoint
│   ├── rag.py         # RAG pipeline definition
│   ├── loaders.py     # Document loading logic
│   └── docs/          # Company documents
│
├── .venv/             # Python virtual environment
├── start.local.sh     # Startup script using local OLLAMA
├── start.openai.sh    # Startup script using OpenAI
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **Python 3.11+**

### For local Ollama

- **Ollama installed** → [https://ollama.com](https://ollama.com)

After installing Ollama, pull the required model:

```bash
ollama pull llama2
```

### For OpenAI

- An openAI key must be provided in .env.openai
- Rename the .env.example and change OPENAI_API_KEY value

---

## Running the project

### Create and activate virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# or
. .venv\Scripts\Activate.ps1   # PowerShell
# or
python3 -m venv .venv
source .venv/bin/activate # Linux

```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API (one command)

#### Using OpenAI

```bash
./start.openai.sh
```

#### Using local OLLAMA

```bash
./start.local.sh
```

The API will be available at:

```
http://127.0.0.1:8000
```

---

## API Usage

### POST /chat/invoke

Example request:

```http
POST http://127.0.0.1:8000/chat/invoke
Content-Type: application/json

"What services does Promptior offer?"
```

Example response:

```json
{
  "output": "Promptior offers GenAI consulting, automation, sales, legal, and customer service solutions..."
}
```

---

## How RAG works (high level)

1. Documents are loaded from `app/docs/`
2. Text is split into chunks
3. Each chunk is converted into a vector embedding
4. Vectors are stored in FAISS (in-memory)
5. When a question arrives:
   - It is embedded
   - Similar chunks are retrieved (top-k)
   - Retrieved text is injected into a prompt
   - LLM generates an answer strictly from that context

---

## Notes

- FAISS is rebuilt on startup (no persistent storage)
- Ideal for demos, interviews, and local experimentation

---

## Possible Improvements

- Persist FAISS index to disk
- Add chat history / memory
- Add authentication
- Dockerize the application

---

## Author

**Jose Bordón**
