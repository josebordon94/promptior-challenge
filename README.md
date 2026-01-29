# Promptior RAG Chatbot

RAG-based (Retrieval Augmented Generation) chatbot built with **FastAPI**, **LangChain**, and **Ollama**, using local LLMs and vector search.

This project demonstrates how to build a simple, local-first AI chatbot that answers questions **only based on provided company documents**, without relying on external APIs.

---

## 🧠 What this project does

- Loads company information from local documents
- Splits text into chunks
- Converts text into vector embeddings
- Stores embeddings in a vector database (FAISS)
- Retrieves the most relevant chunks for a given question
- Uses a local LLM (via Ollama) to generate answers based only on that context
- Exposes everything through a REST API

---

## 🧱 Tech Stack

- **Python 3.11**
- **FastAPI** – REST API framework
- **LangChain** – LLM orchestration and RAG pipelines
- **LangServe** – Exposes LangChain chains as HTTP endpoints
- **Ollama** – Runs local LLMs and embeddings (offline)
- **LLaMA 2** – Local language model
- **FAISS** – Vector similarity search
- **Uvicorn** – ASGI server

---

## 📂 Project Structure

```text
promptior-rag-chatbot/
│
├── app/
│   ├── main.py        # FastAPI app entrypoint
│   ├── rag.py         # RAG pipeline definition
│   ├── loaders.py     # Document loading logic
│   └── docs/          # Company documents (source of truth)
│
├── .venv/             # Python virtual environment
├── start.sh           # Startup script (npm start equivalent)
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites

- **Python 3.11+**
- **Ollama installed** → [https://ollama.com](https://ollama.com)

After installing Ollama, pull the required model:

```bash
ollama pull llama2
```

---

## 🚀 Running the project

### 1️⃣ Create and activate virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# or
. .venv\Scripts\Activate.ps1   # PowerShell
# or
python3 -m venv .venv
source .venv/bin/activate # Linux

```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Start the API (one command)

```bash
./start.sh
```

The API will be available at:

```
http://127.0.0.1:8000
```

---

## 🔌 API Usage

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

## 🧠 How RAG works (high level)

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

## 📌 Notes

- No OpenAI API key is required
- Everything runs **locally**
- FAISS is rebuilt on startup (no persistent storage yet)
- Ideal for demos, interviews, and local experimentation

---

## 🛣️ Possible Improvements

- Persist FAISS index to disk
- Add streaming responses
- Add chat history / memory
- Add authentication
- Dockerize the application

---

## 👤 Author

**Jose Bordón**
